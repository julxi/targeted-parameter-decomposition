"""Compare attention U/V components across two SPD decompositions (position-dependent).

Compares components that are active at the same sequence position, producing per-position
scatter plots grouped by (layer, proj_type).

Usage:
    uv run python -m spd.scripts.compare_decompositions.compare_attn_uv \
        "wandb:entity/project/runs/run_id_1" \
        "wandb:entity/project/runs/run_id_2" \
        --prompt "The cat sat on the mat" \
        --ci_threshold 0.1 \
        --output_dir /tmp/compare_decompositions_uv
"""

from collections import defaultdict
from pathlib import Path

import fire
import torch

from spd.scripts.compare_decompositions.utils import (
    PositionUVComparisonResult,
    compare_attention_heads_uv_by_position,
    compute_ci,
    get_active_component_indices_per_position,
    get_attn_info,
    get_tokenizer,
    load_decomposition,
    parse_attn_module_path,
    plot_comparison_summary_uv,
    plot_head_comparisons_uv,
)


@torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
def main(
    wandb_path_a: str,
    wandb_path_b: str,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
    prompt: str,
    ci_threshold: float = 0.1,
    device: str = "cuda",
    output_dir: str = "/tmp/compare_decompositions_uv",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load models
    decomp_a = load_decomposition(wandb_path_a, label_a, device)
    decomp_b = load_decomposition(wandb_path_b, label_b, device)
    print(f"Run A: {decomp_a.label}")
    print(f"Run B: {decomp_b.label}")

    tokenizer = get_tokenizer(decomp_a)
    n_heads, n_kv_heads, d_head = get_attn_info(decomp_a.model)
    print(f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, d_head={d_head}")

    # Compute CI & find active components per position
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    assert isinstance(input_ids, torch.Tensor)
    tokens = input_ids.to(device)
    token_strs = [tokenizer.decode(t) for t in tokens[0]]
    print(f"Prompt: {prompt!r}  (tokens: {tokens.shape[1]})")
    print(f"Tokens: {token_strs}")

    ci_a = compute_ci(decomp_a.model, tokens, decomp_a.run_info.config.sampling)
    ci_b = compute_ci(decomp_b.model, tokens, decomp_b.run_info.config.sampling)

    active_a = get_active_component_indices_per_position(ci_a, ci_threshold)
    active_b = get_active_component_indices_per_position(ci_b, ci_threshold)

    print(f"\nActive attention components per position (CI > {ci_threshold}):")
    all_attn_paths = sorted(set(active_a) | set(active_b))
    for path in all_attn_paths:
        parsed = parse_attn_module_path(path)
        if parsed is None:
            continue
        layer_idx, proj_type = parsed
        positions_a = active_a.get(path, {})
        positions_b = active_b.get(path, {})
        shared_pos = sorted(set(positions_a) & set(positions_b))
        if not shared_pos:
            continue
        print(f"  L{layer_idx}.{proj_type}:")
        for pos in shared_pos:
            n_a = len(positions_a.get(pos, []))
            n_b = len(positions_b.get(pos, []))
            tok = token_strs[pos] if pos < len(token_strs) else "?"
            print(f"    pos {pos} '{tok}': {n_a} ({decomp_a.label}), {n_b} ({decomp_b.label})")

    # Compare
    results = compare_attention_heads_uv_by_position(
        decomp_a, decomp_b, active_a, active_b, token_strs
    )
    print(f"\n{len(results)} position-level comparisons")

    # Group results by (layer_idx, proj_type), one figure per group
    grouped: dict[str, list[PositionUVComparisonResult]] = defaultdict(list)
    for r in results:
        grouped[f"L{r.layer_idx}_{r.proj_type}"].append(r)

    for key in sorted(grouped):
        fig = plot_head_comparisons_uv(grouped[key])
        path = out / f"{key}_uv.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")

    # Summary
    if results:
        fig = plot_comparison_summary_uv(results)
        path = out / "summary_uv.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")

        print(f"\n{'Layer.Proj':<20} {'Pos':>4} {'Token':<10} {'U best':>8} {'V best':>8}")
        print("-" * 54)
        for r in results:
            u_per_pair = r.u_cos_sim.abs().mean(dim=-1)
            v_per_pair = r.v_cos_sim.abs().mean(dim=-1)
            u_best_a = u_per_pair.max(dim=1).values.mean().item()
            u_best_b = u_per_pair.max(dim=0).values.mean().item()
            v_best_a = v_per_pair.max(dim=1).values.mean().item()
            v_best_b = v_per_pair.max(dim=0).values.mean().item()
            u_mean = (u_best_a + u_best_b) / 2
            v_mean = (v_best_a + v_best_b) / 2
            label = f"L{r.layer_idx}.{r.proj_type}"
            print(f"{label:<20} {r.seq_pos:>4} {r.token_str:<10} {u_mean:>8.4f} {v_mean:>8.4f}")


if __name__ == "__main__":
    fire.Fire(main)
