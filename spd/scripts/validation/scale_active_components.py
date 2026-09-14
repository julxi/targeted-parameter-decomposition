"""Scale the active components up/down and measure the effect on the next-token output.

Runs a prompts-based LM decomposition using *only* its active components (delta OFF), with every
active component's mask set to a single amplification factor that is swept from `--min-factor` to
`--max-factor`. "Active" is decided per (prompt, position): a component counts as active wherever
its lower-leaky causal importance exceeds `--ci-thr`. At factor 1.0 this is the binary
active-component reconstruction; smaller/larger factors scale every active component's
contribution down/up uniformly.

At every (prompt, position) the tracked next-token is the original model's top-1 prediction at
that position, then its logit is followed across all factors. Two logits are recorded: the
**pre-RMSnorm** logit (the residual stream projected straight onto the unembed row, skipping the
final `ln_f`) and the standard **post-RMSnorm** logit. The pre-RMSnorm logit exposes
residual-stream magnitude growth that the final RMSNorm would otherwise renormalise away.
Reconstruction quality is the KL divergence to the original model at each position. The original
model's own logits at the same position are recorded as a baseline.

Prompts-based LM tasks only (needs per-position target tokens and well-defined real positions).
Only `LlamaSimpleMLP`-style targets are supported (final norm `ln_f`, unembed `lm_head`).

Usage:
    python -m spd.scripts.validation.scale_active_components <model_path> \
        [--prompts=PATH] [--ci-thr=0.1] \
        [--min-factor=0.1] [--max-factor=2.0] [--n-factors=20] \
        [--output=PATH] [--output-fig=PATH]

Output files (default in the decomposed model's folder):
- `scale_active_components.tsv` — one row per (prompt, position, factor).
- `scale_active_components.png` — pre-RMSnorm logit, post-RMSnorm logit, and KL vs factor, with
  one line per (prompt, position).
"""

import csv
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoTokenizer

from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.scripts.validation.common import (
    build_lm_loader,
    escape_tsv_value,
    is_prompt_task,
    iterate_input_ids,
    load_prompts,
    load_spd_run,
    resolve_task_config,
)
from spd.spd_types import ModelPath

FIELDS = [
    "prompt",
    "pos",
    "factor",
    "n_active",
    "target_token",
    "target_token_str",
    "pre_rmsnorm_logit",
    "post_rmsnorm_logit",
    "kl",
    "orig_pre_rmsnorm_logit",
    "orig_post_rmsnorm_logit",
]


@contextmanager
def _capture_ln_f_input(norm: nn.Module) -> Iterator[dict[str, Tensor]]:
    """Capture the residual stream fed into the final norm (the pre-RMSnorm activations)."""
    captured: dict[str, Tensor] = {}

    def pre_hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
        assert len(args) == 1, f"Expected ln_f to take 1 positional arg, got {len(args)}"
        assert isinstance(args[0], Tensor)
        captured["resid"] = args[0]

    handle = norm.register_forward_pre_hook(pre_hook)
    try:
        yield captured
    finally:
        handle.remove()


def _target_logits(
    pre_resid: Tensor,  # (batch, seq, d_model) — input to ln_f
    norm: nn.Module,
    unembed: Tensor,  # (vocab, d_model)
    target_token: Tensor,  # (batch, seq)
) -> tuple[Tensor, Tensor]:
    """Return per-(prompt, pos) (pre_rmsnorm_logit, post_rmsnorm_logit), each (batch, seq).

    The pre-RMSnorm logit projects the raw residual stream onto the target unembed row; the
    post-RMSnorm logit applies `ln_f` first (the model's actual logit).
    """
    pre = pre_resid.float()  # (batch, seq, d_model)
    post = norm(pre_resid).float()  # (batch, seq, d_model)
    w_target = unembed[target_token].float()  # (batch, seq, d_model)

    pre_logit = (pre * w_target).sum(dim=-1)  # (batch, seq)
    post_logit = (post * w_target).sum(dim=-1)  # (batch, seq)
    return pre_logit, post_logit


def _kl_per_pos(orig_logits: Tensor, other_logits: Tensor) -> Tensor:
    """KL(softmax(orig) || softmax(other)) per (batch, pos)."""
    orig_log_probs = F.log_softmax(orig_logits, dim=-1)
    orig_probs = orig_log_probs.exp()
    other_log_probs = F.log_softmax(other_logits, dim=-1)
    return (orig_probs * (orig_log_probs - other_log_probs)).sum(dim=-1)


def scale_active_components(
    model_path: ModelPath,
    prompts: str | None = None,
    ci_thr: float = 0.1,
    min_factor: float = 0.1,
    max_factor: float = 2.0,
    n_factors: int = 20,
    output: str | None = None,
    output_fig: str | None = None,
) -> tuple[Path, Path]:
    """Sweep the active-component amplification factor and write the per-(prompt, position, factor)
    TSV + a figure of pre/post-RMSnorm next-token logit and KL vs factor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spd_model, config, run_dir = load_spd_run(model_path)
    spd_model = spd_model.to(device)

    assert config.tokenizer_name is not None, "config.tokenizer_name is required"
    target_model = spd_model.target_model
    assert hasattr(target_model, "ln_f") and hasattr(target_model, "lm_head"), (
        "scale_active_components only supports LlamaSimpleMLP-style targets (ln_f + lm_head)"
    )
    norm = target_model.ln_f
    unembed = target_model.lm_head.weight  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(norm, nn.Module) and isinstance(unembed, Tensor)

    task_config = resolve_task_config(config, use_nontarget=False, prompts_override=prompts)
    assert is_prompt_task(task_config), "scale_active_components needs a prompts-based LM task"
    loader = build_lm_loader(task_config, config)

    prompt_texts = load_prompts(config, prompts_override=prompts)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    encoded: Any = tokenizer(prompt_texts)
    last_pos = [len(ids) - 1 for ids in encoded["input_ids"]]

    assert min_factor > 0 and max_factor >= min_factor and n_factors >= 1, (
        f"Invalid factor sweep: min={min_factor}, max={max_factor}, n={n_factors}"
    )
    factors = np.linspace(min_factor, max_factor, n_factors).tolist()

    out_path = Path(output).expanduser() if output else run_dir / "scale_active_components.tsv"
    fig_path = (
        Path(output_fig).expanduser() if output_fig else run_dir / "scale_active_components.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _run(
        spd_model=spd_model,
        config_sampling=config.sampling,
        norm=norm,
        unembed=unembed,
        loader_iter=iterate_input_ids(loader, device),
        prompt_texts=prompt_texts,
        last_pos=last_pos,
        ci_thr=ci_thr,
        factors=factors,
        tokenizer=tokenizer,
        device=device,
    )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} rows to {out_path}")

    _plot(rows, factors, fig_path)
    logger.info(f"Saved figure to {fig_path}")
    return out_path, fig_path


def _run(
    *,
    spd_model: ComponentModel,
    config_sampling: Any,
    norm: nn.Module,
    unembed: Tensor,
    loader_iter: Iterator[Tensor],
    prompt_texts: list[str],
    last_pos: list[int],
    ci_thr: float,
    factors: list[float],
    tokenizer: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run the original model once (for CI, target tokens, baseline logits), then one forward per
    factor with the binary active mask scaled by that factor. Returns the TSV rows."""
    batch = next(loader_iter)
    assert batch.ndim == 2, f"Expected (batch, seq), got {tuple(batch.shape)}"
    batch_size, _ = batch.shape
    assert batch_size == len(prompt_texts) == len(last_pos), (
        f"batch ({batch_size}), prompts ({len(prompt_texts)}), and last_pos ({len(last_pos)}) "
        "must line up — the prompts loader should put one prompt per row in file order"
    )

    with torch.no_grad(), _capture_ln_f_input(norm) as captured:
        orig_out = spd_model(batch, cache_type="input")
        orig_logits = orig_out.output
        assert isinstance(orig_logits, Tensor)
        orig_pre_resid = captured["resid"]
        ci_outputs = spd_model.calc_causal_importances(
            pre_weight_acts=orig_out.cache, sampling=config_sampling
        )

        # Per-(prompt, pos) target token: the original model's top-1 prediction at that position.
        target_token = orig_logits.argmax(dim=-1)  # (batch, seq)

        orig_pre_logit, orig_post_logit = _target_logits(
            orig_pre_resid, norm, unembed, target_token
        )

        # Binary per-(prompt, pos, component) active mask, shared across factors. Inactive
        # components and the delta are off, so the model runs on active components only.
        active_masks = {
            name: (ci_outputs.lower_leaky[name] > ci_thr).to(orig_pre_resid.dtype)
            for name in spd_model.module_to_c
        }
        # Active-component count at each (prompt, pos) is constant across factors.
        n_active_grid = torch.zeros(batch.shape, device=device)  # (batch, seq)
        for m in active_masks.values():
            n_active_grid += m.sum(dim=-1)
        n_active = n_active_grid.int().cpu().tolist()

        rows_t = torch.arange(batch_size, device=device)
        pos_t = torch.tensor(last_pos, device=device)
        n_active_last = int(n_active_grid[rows_t, pos_t].sum().item())
        logger.info(
            f"{n_active_last} active component-firings across {batch_size} prompts "
            f"at their last position (ci_thr={ci_thr})"
        )

        orig_pre = orig_pre_logit.cpu().tolist()
        orig_post = orig_post_logit.cpu().tolist()
        target_token_list = target_token.cpu().tolist()

        rows: list[dict[str, Any]] = []
        for factor in tqdm(factors, desc="factors"):
            masks = {name: m * factor for name, m in active_masks.items()}
            mask_infos = make_mask_infos(masks, weight_deltas_and_masks=None)
            with _capture_ln_f_input(norm) as cap:
                dec_logits = spd_model(batch, mask_infos=mask_infos)
                assert isinstance(dec_logits, Tensor)
                dec_pre_resid = cap["resid"]

            pre_logit, post_logit = _target_logits(dec_pre_resid, norm, unembed, target_token)
            kl = _kl_per_pos(orig_logits, dec_logits)  # (batch, seq)
            pre_l = pre_logit.cpu().tolist()
            post_l = post_logit.cpu().tolist()
            kl_l = kl.cpu().tolist()

            for b in range(batch_size):
                for pos in range(last_pos[b] + 1):
                    rows.append(
                        {
                            "prompt": b,
                            "pos": pos,
                            "factor": factor,
                            "n_active": n_active[b][pos],
                            "target_token": target_token_list[b][pos],
                            "target_token_str": escape_tsv_value(
                                tokenizer.decode([target_token_list[b][pos]])
                            ),
                            "pre_rmsnorm_logit": pre_l[b][pos],
                            "post_rmsnorm_logit": post_l[b][pos],
                            "kl": kl_l[b][pos],
                            "orig_pre_rmsnorm_logit": orig_pre[b][pos],
                            "orig_post_rmsnorm_logit": orig_post[b][pos],
                        }
                    )
    return rows


def _plot(rows: list[dict[str, Any]], factors: list[float], fig_path: Path) -> None:
    """Grid of panels: three rows (pre-RMSnorm logit, post-RMSnorm logit, KL) and one column per
    (prompt, position). The original model's per-(prompt, position) logit is drawn as a horizontal
    baseline in each logit panel."""
    x = np.array(factors)

    by_line: dict[tuple[int, int], dict[float, dict[str, Any]]] = {}
    for r in rows:
        by_line.setdefault((r["prompt"], r["pos"]), {})[r["factor"]] = r
    lines = sorted(by_line)

    panels = [
        ("pre_rmsnorm_logit", "orig_pre_rmsnorm_logit", "pre-RMSnorm logit", False),
        ("post_rmsnorm_logit", "orig_post_rmsnorm_logit", "post-RMSnorm logit", False),
        ("kl", None, "KL(orig ‖ scaled)", True),
    ]
    fig, axes = plt.subplots(
        len(panels), len(lines), figsize=(3 * len(lines), 9), sharex=True, squeeze=False
    )
    for col, line in enumerate(lines):
        by_factor = by_line[line]
        first = by_factor[factors[0]]
        for row, (key, baseline_key, ylabel, log_y) in enumerate(panels):
            ax = axes[row][col]
            y = np.array([by_factor[f][key] for f in factors])
            ax.plot(x, y, color="C0", marker="o", ms=3, label="scaled")
            if baseline_key is not None:
                ax.axhline(first[baseline_key], color="C3", ls="--", label="original model")
            ax.axvline(1.0, color="gray", ls=":", lw=1, label="factor = 1.0")
            ax.grid(alpha=0.3)
            if log_y:
                ax.set_yscale("log")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                prompt, pos = line
                ax.set_title(
                    f"prompt {prompt} · pos {pos}\n→ {first['target_token_str']!r}", fontsize=8
                )
            if row == len(panels) - 1:
                ax.set_xlabel("factor")
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle("Scaling active components vs. next-token output")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire(scale_active_components)
