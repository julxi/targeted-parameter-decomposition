"""Estimate training FLOPs for SPD decomposition runs.

Covers configs in batch_commands/{css,full_data,numpy}/. Supports both the 4L-768
and 12L-768 pile_llama_simple_mlp pretrained targets (dispatched by pretrained_model_name).

Uses the 2*m*n*k rule for matmuls; ignores softmax/norm/gelu non-linearities (small).
"""

from pathlib import Path
from typing import Any

import yaml

BATCH_COMMANDS = Path("/home/antoine/Code/SPD/batch_commands")

# --- Target (pretrained) model specs, keyed by wandb run id ---
# t-9d2b8f02: pile_llama_simple_mlp-4L-768
# t-f99617bb: pile_llama_simple_mlp-12L-768
TARGETS = {
    "t-9d2b8f02": dict(
        n_layer=4,
        d_model=768,
        n_head=6,
        head_dim=128,
        n_kv_heads=6,
        d_intermediate=3072,
        vocab_size=50277,
    ),
    "t-f99617bb": dict(
        n_layer=12,
        d_model=768,
        n_head=12,
        head_dim=64,
        n_kv_heads=12,
        d_intermediate=3072,
        vocab_size=50277,
    ),
}


def module_shapes(target: dict[str, int]) -> dict[str, tuple[int, int]]:
    """(d_in, d_out) for each decomposable module suffix."""
    d = target["d_model"]
    h = target["n_head"] * target["head_dim"]
    kv = target["n_kv_heads"] * target["head_dim"]
    dff = target["d_intermediate"]
    return {
        "mlp.c_fc": (d, dff),
        "mlp.down_proj": (dff, d),
        "attn.q_proj": (d, h),
        "attn.k_proj": (d, kv),
        "attn.v_proj": (d, kv),
        "attn.o_proj": (h, d),
    }


def target_for_cfg(cfg: dict[str, Any]) -> dict[str, int]:
    name = cfg["pretrained_model_name"]
    for key, t in TARGETS.items():
        if key in name:
            return t
    raise ValueError(f"no target model spec for {name}")


def target_forward_flops_per_token(target: dict[str, int], seq_len: int) -> int:
    """Forward FLOPs per token for the target model. Matmuls + attention."""
    d = target["d_model"]
    h = target["n_head"] * target["head_dim"]
    kv = target["n_kv_heads"] * target["head_dim"]
    dff = target["d_intermediate"]
    flops_per_layer = (
        2 * d * h  # q_proj
        + 2 * d * kv  # k_proj
        + 2 * d * kv  # v_proj
        + 2 * h * d  # o_proj
        + 2 * 2 * seq_len * h  # QK^T + AV (per token, sums over seq_len)
        + 2 * d * dff  # c_fc
        + 2 * dff * d  # down_proj
    )
    embed = 2 * d * target["vocab_size"]  # unembedding
    return flops_per_layer * target["n_layer"] + embed


def expand_modules(
    module_info: list[dict[str, Any]], target: dict[str, int]
) -> list[dict[str, Any]]:
    """Expand h.*.X into [h.0.X, h.1.X, ...] and attach (d_in, d_out)."""
    shapes = module_shapes(target)
    expanded = []
    for m in module_info:
        pat = m["module_pattern"]
        C = m["C"]
        suffix = pat.split(".", 2)[2]  # e.g. "mlp.c_fc"
        d_in, d_out = shapes[suffix]
        layer_tok = pat.split(".")[1]
        layers = range(target["n_layer"]) if layer_tok == "*" else [int(layer_tok)]
        for _ in layers:
            expanded.append(dict(pattern=pat, C=C, d_in=d_in, d_out=d_out, suffix=suffix))
    return expanded


def component_overhead_flops_per_token(modules: list[dict[str, Any]]) -> int:
    """FLOPs per token for (x @ V) * mask @ U summed over modules.
    Replaces the x @ W matmul in the decomposed forward."""
    return sum(2 * m["C"] * (m["d_in"] + m["d_out"]) for m in modules)


def target_matmul_flops_per_token(modules: list[dict[str, Any]]) -> int:
    """FLOPs per token of the original matmuls being decomposed."""
    return sum(2 * m["d_in"] * m["d_out"] for m in modules)


def decomposed_forward_flops_per_token(
    modules: list[dict[str, Any]], target: dict[str, int], seq_len: int, use_delta: bool
) -> int:
    """Decomposed forward: replace each target W·x with (x @ V) * mask @ U; optionally add
    the delta term (W - U@V)·x which costs one full matmul per module."""
    f = target_forward_flops_per_token(target, seq_len)
    f -= target_matmul_flops_per_token(modules)
    f += component_overhead_flops_per_token(modules)
    if use_delta:
        f += target_matmul_flops_per_token(modules)
    return f


def ci_forward_flops(modules: list[dict[str, Any]], ci_outer: dict[str, Any], seq_len: int) -> int:
    """FLOPs (per sequence) for one CI forward pass.
    Supports global_shared_transformer and global_shared_mlp.
    """
    total_in = sum(m["d_in"] for m in modules)
    total_C = sum(m["C"] for m in modules)
    fn_type = ci_outer.get("fn_type", "global_shared_transformer")

    if fn_type == "global_shared_transformer":
        ci = ci_outer["simple_transformer_ci_cfg"]
        d = ci["d_model"]
        n_blocks = ci["n_blocks"]
        mlp_hidden = ci["mlp_hidden_dim"][0]
        f = 2 * total_in * d  # input projection (per position)
        per_block = (
            4 * 2 * d * d  # Q,K,V,O projections
            + 2 * 2 * seq_len * d  # attention scores + attended values
            + 2 * 2 * d * mlp_hidden  # up + down MLP
        )
        f += n_blocks * per_block
        f += 2 * d * total_C  # output projection
        return f * seq_len
    elif fn_type == "global_shared_mlp":
        hidden_dims = ci_outer["hidden_dims"]
        dims = [total_in] + list(hidden_dims) + [total_C]
        f = sum(2 * dims[i] * dims[i + 1] for i in range(len(dims) - 1))
        return f * seq_len
    else:
        raise ValueError(f"unknown CI fn_type: {fn_type}")


def count_decomp_forwards(loss_configs: list[dict[str, Any]], n_mask_samples: int) -> int:
    """Number of decomposed-model forward passes per training step."""
    n = 0
    for lc in loss_configs:
        cn = lc["classname"]
        if cn == "UnmaskedReconLoss":
            n += 1
        elif cn in (
            "StochasticReconSubsetLoss",
            "StochasticReconLoss",
            "StochasticHiddenActsReconLoss",
        ):
            n += n_mask_samples
        elif cn in ("PersistentPGDReconLoss", "PersistentPGDReconSubsetLoss"):
            n_warmup = lc.get("n_warmup_steps", 0)
            n += n_warmup + 1  # warmup fwd+bwd each, plus final fwd for loss
    return n


def estimate_config(cfg_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(cfg_path.read_text())
    target = target_for_cfg(cfg)
    modules = expand_modules(cfg["module_info"], target)

    seq_len = cfg["task_config"]["max_seq_len"]
    B = cfg["batch_size"]
    steps = cfg["steps"]
    n_mask_samples = cfg.get("n_mask_samples", 1)
    use_delta = cfg.get("use_delta_component", False)
    loss_cfgs = cfg["loss_metric_configs"]
    has_nontarget = cfg.get("nontarget_task_config") is not None

    f_target_tok = target_forward_flops_per_token(target, seq_len)
    f_decomp_tok = decomposed_forward_flops_per_token(modules, target, seq_len, use_delta)
    f_ci_seq = ci_forward_flops(modules, cfg["ci_config"], seq_len)

    n_decomp_fwd = count_decomp_forwards(loss_cfgs, n_mask_samples)

    # --- TARGET batch ---
    target_flops = B * (f_target_tok * seq_len)
    ci_flops = B * f_ci_seq * 3  # CI fwd + bwd ≈ 3x
    decomp_flops = B * (f_decomp_tok * seq_len) * n_decomp_fwd * 3

    step_flops = target_flops + ci_flops + decomp_flops

    # --- NONTARGET batch ---
    nontarget_flops = 0
    if has_nontarget:
        Bn = cfg.get("nontarget_batch_size", B)
        nt_seq = cfg["nontarget_task_config"].get("max_seq_len", seq_len)
        excluded = {"UnmaskedReconLoss", "PersistentPGDReconLoss", "PersistentPGDReconSubsetLoss"}
        nt_loss_cfgs = [lc for lc in loss_cfgs if lc["classname"] not in excluded]
        n_nt_fwd = count_decomp_forwards(nt_loss_cfgs, n_mask_samples)
        f_target_nt = target_forward_flops_per_token(target, nt_seq) * nt_seq
        f_decomp_nt = (
            decomposed_forward_flops_per_token(modules, target, nt_seq, use_delta) * nt_seq
        )
        f_ci_nt = ci_forward_flops(modules, cfg["ci_config"], nt_seq)
        nontarget_flops = Bn * f_target_nt + Bn * f_ci_nt * 3 + Bn * f_decomp_nt * n_nt_fwd * 3
        step_flops += nontarget_flops

    total_flops = step_flops * steps

    return dict(
        name=f"{cfg_path.parent.name}/{cfg_path.stem}",
        n_layer_target=target["n_layer"],
        steps=steps,
        batch_size=B,
        seq_len=seq_len,
        has_nontarget=has_nontarget,
        n_decomp_fwd=n_decomp_fwd,
        target_flops_per_step=target_flops,
        ci_flops_per_step=ci_flops,
        decomp_flops_per_step=decomp_flops,
        nontarget_flops_per_step=nontarget_flops,
        step_flops=step_flops,
        total_flops=total_flops,
    )


def fmt(n: float) -> str:
    for unit, div in [("E", 1e18), ("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.2f}"


# Explicit list of config paths (relative to BATCH_COMMANDS) we want to report on.
CONFIG_GROUPS: list[tuple[str, list[str]]] = [
    (
        "css (4L)",
        [
            "css/config_css_naive.yaml",
            "css/config_css_reference.yaml",
            "css/config_css_alt_seed.yaml",
            "css/config_css_border-radius.yaml",
        ],
    ),
    (
        "full_data (4L)",
        [
            "full_data/config_full_ha.yaml",
            "full_data/config_full_ha_10x.yaml",
            "full_data/config_full_ha_late.yaml",
        ],
    ),
    (
        "numpy_and_pandas (4L)",
        [
            "numpy/numpy_and_pandas/config_numpy.yaml",
            "numpy/numpy_and_pandas/config_pandas.yaml",
            "numpy/numpy_and_pandas/config_numpy_and_pandas.yaml",
            "numpy/numpy_and_pandas/config_numpy_and_pandas_seed1.yaml",
            "numpy/numpy_and_pandas/config_numpy_and_pandas_seed2.yaml",
        ],
    ),
    (
        "numpy reference_4L",
        [
            "numpy/reference_4L/config_numpy_naive.yaml",
            "numpy/reference_4L/config_numpy_reference.yaml",
            "numpy/reference_4L/config_numpy_alt_seed.yaml",
            "numpy/reference_4L/config_numpy_only.yaml",
            "numpy/reference_4L/config_pandas_only.yaml",
        ],
    ),
    (
        "numpy reference_12L",
        [
            "numpy/reference_12L/config_numpy_12L_naive.yaml",
            "numpy/reference_12L/config_numpy_12L_reference.yaml",
            "numpy/reference_12L/config_numpy_12L_alt_seed.yaml",
        ],
    ),
]


def main() -> None:
    all_rows: list[dict[str, Any]] = []
    for group_name, paths in CONFIG_GROUPS:
        rows = [estimate_config(BATCH_COMMANDS / p) for p in paths]
        all_rows.extend(rows)
        print(f"\n=== {group_name} ===")
        print(
            f"{'config':52s}  {'tgtL':>4s}  {'steps':>7s}  {'B':>4s}  {'L':>4s}  "
            f"{'fwds':>5s}  {'step FLOPs':>11s}  {'total FLOPs':>11s}"
        )
        print("-" * 118)
        for r in rows:
            print(
                f"{r['name']:52s}  {r['n_layer_target']:>4d}  {r['steps']:>7d}  "
                f"{r['batch_size']:>4d}  {r['seq_len']:>4d}  {r['n_decomp_fwd']:>5d}  "
                f"{fmt(r['step_flops']):>11s}  {fmt(r['total_flops']):>11s}"
            )

    print("\n=== Per-step breakdown ===")
    print(f"{'config':52s}  {'target':>10s}  {'CI':>10s}  {'decomp':>10s}  {'nontarget':>11s}")
    print("-" * 110)
    for r in all_rows:
        print(
            f"{r['name']:52s}  {fmt(r['target_flops_per_step']):>10s}  "
            f"{fmt(r['ci_flops_per_step']):>10s}  {fmt(r['decomp_flops_per_step']):>10s}  "
            f"{fmt(r['nontarget_flops_per_step']):>11s}"
        )


if __name__ == "__main__":
    main()
