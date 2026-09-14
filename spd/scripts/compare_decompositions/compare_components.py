"""Compare active components across two SPD decompositions.

For each shared module_path, computes pairwise cosine similarities between active
components from both runs and plots heatmaps.

Usage:
    uv run python -m spd.scripts.compare_decompositions.compare_components \
        "wandb:entity/project/runs/run_id_1" \
        "wandb:entity/project/runs/run_id_2" \
        --prompt "The cat sat on the mat" \
        --ci_threshold 0.1 \
        --output_dir /tmp/compare_decompositions_components
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from spd.scripts.compare_decompositions.utils import (
    compute_ci,
    compute_pairwise_cosine_sim,
    get_active_component_indices_per_position,
    get_component_weight,
    get_tokenizer,
    load_decomposition,
)

TILE_SIZE = 0.5  # inches per tile
LABEL_PAD = 1.5  # inches padding for axis labels/titles
COLORBAR_WIDTH_RATIO = 2  # gridspec width ratio units for colorbar column


def plot_component_heatmaps(
    sim_matrices: dict[str, dict[int, torch.Tensor]],
    active_indices_a: dict[str, dict[int, list[int]]],
    active_indices_b: dict[str, dict[int, list[int]]],
    token_strs: list[str],
    label_a: str,
    label_b: str,
) -> Figure:
    """Plot a 2D grid of heatmaps: rows=positions, columns=modules.

    Each cell shows pairwise cosine similarity between active components at that
    (position, module) pair. Tile sizes are uniform across all subplots.
    """
    sorted_modules = sorted(sim_matrices)
    all_positions = sorted({pos for mod in sorted_modules for pos in sim_matrices[mod]})
    assert len(all_positions) > 0

    # Compute max Nb per module column and max Na per position row for uniform tiles
    col_widths = []
    for mod in sorted_modules:
        max_nb = max(
            (sim_matrices[mod][pos].shape[1] for pos in all_positions if pos in sim_matrices[mod]),
            default=1,
        )
        col_widths.append(max_nb)

    row_heights = []
    for pos in all_positions:
        max_na = max(
            (sim_matrices[mod][pos].shape[0] for mod in sorted_modules if pos in sim_matrices[mod]),
            default=1,
        )
        row_heights.append(max_na)

    width_ratios = col_widths + [COLORBAR_WIDTH_RATIO]
    fig_width = sum(col_widths) * TILE_SIZE + LABEL_PAD * (len(sorted_modules) + 1)
    fig_height = sum(row_heights) * TILE_SIZE + LABEL_PAD * (len(all_positions) + 1)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=len(all_positions),
        ncols=len(sorted_modules) + 1,
        width_ratios=width_ratios,
        height_ratios=row_heights,
        wspace=0.4,
        hspace=0.5,
    )

    im = None
    for row_idx, pos in enumerate(all_positions):
        tok = token_strs[pos] if pos < len(token_strs) else f"pos{pos}"
        for col_idx, mod in enumerate(sorted_modules):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if pos not in sim_matrices[mod]:
                ax.set_visible(False)
                continue

            sim_np = sim_matrices[mod][pos].cpu().numpy()
            im = ax.imshow(sim_np, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

            indices_a = active_indices_a[mod][pos]
            indices_b = active_indices_b[mod][pos]
            ax.set_xticks(range(len(indices_b)))
            ax.set_xticklabels([str(i) for i in indices_b], fontsize=7)
            ax.set_yticks(range(len(indices_a)))
            ax.set_yticklabels([str(i) for i in indices_a], fontsize=7)

            if row_idx == 0:
                ax.set_title(mod, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"pos {pos} '{tok}'\n{label_a} comps", fontsize=8)
            if row_idx == len(all_positions) - 1:
                ax.set_xlabel(f"{label_b} comps", fontsize=8)

    fig.suptitle("Component Cosine Similarity", fontsize=14)
    assert im is not None
    cbar_ax = fig.add_subplot(gs[:, -1])
    fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")
    return fig


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
    output_dir: str = "/tmp/compare_decompositions_components",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    decomp_a = load_decomposition(wandb_path_a, label_a, device)
    decomp_b = load_decomposition(wandb_path_b, label_b, device)
    print(f"Run A: {decomp_a.label}")
    print(f"Run B: {decomp_b.label}")

    tokenizer = get_tokenizer(decomp_a)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    assert isinstance(input_ids, torch.Tensor)
    tokens = input_ids.to(device)
    token_strs = [tokenizer.decode(t) for t in tokens[0]]
    print(f"Prompt: {prompt!r}  (tokens: {tokens.shape[1]})")

    ci_a = compute_ci(decomp_a.model, tokens, decomp_a.run_info.config.sampling)
    ci_b = compute_ci(decomp_b.model, tokens, decomp_b.run_info.config.sampling)

    active_a = get_active_component_indices_per_position(ci_a, ci_threshold)
    active_b = get_active_component_indices_per_position(ci_b, ci_threshold)

    print(f"\nActive components per position (CI > {ci_threshold}):")
    for path in sorted(set(active_a) | set(active_b)):
        positions_a = active_a.get(path, {})
        positions_b = active_b.get(path, {})
        all_pos = sorted(set(positions_a) | set(positions_b))
        for pos in all_pos:
            n_a = len(positions_a.get(pos, []))
            n_b = len(positions_b.get(pos, []))
            tok = token_strs[pos] if pos < len(token_strs) else f"pos{pos}"
            print(f"  {path} pos {pos} '{tok}': {n_a} ({decomp_a.label}), {n_b} ({decomp_b.label})")

    # Compute sim matrices: module_path -> pos -> (Na, Nb) tensor
    sim_matrices: dict[str, dict[int, torch.Tensor]] = {}
    shared_modules = sorted(p for p in active_a if p in active_b)
    for module_path in shared_modules:
        shared_positions = sorted(set(active_a[module_path]) & set(active_b[module_path]))
        if not shared_positions:
            continue
        pos_sims: dict[int, torch.Tensor] = {}
        for pos in shared_positions:
            indices_a = active_a[module_path][pos]
            indices_b = active_b[module_path][pos]
            weights_a = torch.stack(
                [get_component_weight(decomp_a.model, module_path, i).flatten() for i in indices_a]
            )
            weights_b = torch.stack(
                [get_component_weight(decomp_b.model, module_path, i).flatten() for i in indices_b]
            )
            pos_sims[pos] = compute_pairwise_cosine_sim(weights_a, weights_b)
        sim_matrices[module_path] = pos_sims

    if not sim_matrices:
        print("No shared modules with active components in both runs at any position.")
        return

    n_cells = sum(len(pos_sims) for pos_sims in sim_matrices.values())
    print(f"\n{len(sim_matrices)} shared modules, {n_cells} (module, position) cells")

    fig = plot_component_heatmaps(
        sim_matrices, active_a, active_b, token_strs, decomp_a.label, decomp_b.label
    )
    path = out / "component_similarity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")

    # Print summary table
    print(f"\n{'Module':<50} {'Pos':>4} {'Token':<10} {'Best A→B':>10} {'Best B→A':>10}")
    print("-" * 88)
    for module_path in sorted(sim_matrices):
        for pos in sorted(sim_matrices[module_path]):
            sim = sim_matrices[module_path][pos].abs()
            best_a_to_b = sim.max(dim=1).values.mean().item()
            best_b_to_a = sim.max(dim=0).values.mean().item()
            tok = token_strs[pos] if pos < len(token_strs) else f"pos{pos}"
            print(f"{module_path:<50} {pos:>4} {tok:<10} {best_a_to_b:>10.4f} {best_b_to_a:>10.4f}")


if __name__ == "__main__":
    fire.Fire(main)
