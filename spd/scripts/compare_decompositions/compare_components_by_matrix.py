"""Compare active components across two SPD decompositions, one image per module.

For each shared module_path, produces one PNG with positions arranged horizontally.
Tiles are uniform squares sized by the component counts.

Usage:
    uv run python -m spd.scripts.compare_decompositions.compare_components_by_matrix \
        "wandb:entity/project/runs/run_id_1" \
        "wandb:entity/project/runs/run_id_2" \
        --prompt "The cat sat on the mat" \
        --ci_threshold 0.1 \
        --output_dir /tmp/compare_decompositions_by_matrix
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from spd.scripts.compare_decompositions.utils import (
    compute_ci,
    compute_pairwise_cosine_sim,
    get_active_component_indices_per_position,
    get_component_uv,
    get_component_weight,
    get_tokenizer,
    load_decomposition,
)

TILE_SIZE = 0.5  # inches per tile

# Layout constants (inches)
LEFT_MARGIN = 1.2
RIGHT_MARGIN = 0.3
TOP_MARGIN = 0.8
BOTTOM_MARGIN = 0.6
COL_GAP = 0.6
ROW_GAP = 0.5
CBAR_GAP = 0.3
CBAR_WIDTH = 0.15
ROW_LABEL_X = 0.15
NORM_ROW_HEIGHT = 1.5

ROW_LABELS = ["Full", "V", "U", "||U||·||V||"]


def plot_matrix_heatmaps(
    module_path: str,
    pos_sims_full: dict[int, torch.Tensor],
    pos_sims_v: dict[int, torch.Tensor],
    pos_sims_u: dict[int, torch.Tensor],
    active_a: dict[int, list[int]],
    active_b: dict[int, list[int]],
    pos_norms: dict[int, list[tuple[str, int, float]]],
    token_strs: list[str],
    label_a: str,
    label_b: str,
) -> Figure:
    """3-row heatmap grid (Full/V/U) + 4th row of component norm bar charts.

    Uses manually positioned axes so every tile is exactly TILE_SIZE inches.
    """
    sorted_positions = sorted(pos_sims_full)
    all_sims = [pos_sims_full, pos_sims_v, pos_sims_u]
    n_cols = len(sorted_positions)
    n_heatmap_rows = 3

    col_nb = [pos_sims_full[pos].shape[1] for pos in sorted_positions]
    max_na = max(pos_sims_full[pos].shape[0] for pos in sorted_positions)
    row_height = max_na * TILE_SIZE

    total_data_width = sum(nb * TILE_SIZE for nb in col_nb) + COL_GAP * (n_cols - 1)
    heatmap_height = row_height * n_heatmap_rows + ROW_GAP * (n_heatmap_rows - 1)
    total_data_height = heatmap_height + ROW_GAP + NORM_ROW_HEIGHT

    fig_width = LEFT_MARGIN + total_data_width + CBAR_GAP + CBAR_WIDTH + RIGHT_MARGIN
    fig_height = TOP_MARGIN + total_data_height + BOTTOM_MARGIN

    fig = plt.figure(figsize=(fig_width, fig_height))

    # x-start of each column (inches from left edge)
    col_x: list[float] = []
    x = LEFT_MARGIN
    for nb in col_nb:
        col_x.append(x)
        x += nb * TILE_SIZE + COL_GAP

    # y-top of each heatmap row (inches from bottom edge)
    row_y_top: list[float] = []
    y = fig_height - TOP_MARGIN
    for _ in range(n_heatmap_rows):
        row_y_top.append(y)
        y -= row_height + ROW_GAP

    # 4th row (norm bar charts) top
    norm_row_y_top = y

    im = None
    for row_idx, sims_dict in enumerate(all_sims):
        for col_idx, pos in enumerate(sorted_positions):
            sim = sims_dict[pos]
            na, nb = sim.shape

            w_in = nb * TILE_SIZE
            h_in = na * TILE_SIZE
            x_in = col_x[col_idx]
            y_in = row_y_top[row_idx] - h_in  # top-align within row

            ax = fig.add_axes(
                (x_in / fig_width, y_in / fig_height, w_in / fig_width, h_in / fig_height)
            )
            sim_np = sim.cpu().numpy()
            im = ax.imshow(sim_np, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

            indices_a = active_a[pos]
            indices_b = active_b[pos]
            ax.set_xticks(range(len(indices_b)))
            ax.set_xticklabels([str(i) for i in indices_b], fontsize=7)
            ax.set_yticks(range(len(indices_a)))
            ax.set_yticklabels([str(i) for i in indices_a], fontsize=7)

            if row_idx == 0:
                tok = token_strs[pos] if pos < len(token_strs) else f"pos{pos}"
                ax.set_title(f"pos {pos} '{tok}'", fontsize=9)

            if col_idx == 0:
                ax.set_ylabel(label_a, fontsize=8)
            if row_idx == 2:
                ax.set_xlabel(label_b, fontsize=8)

    # 4th row: norm bar charts per position (shared y-range)
    rdbu = plt.get_cmap("RdBu_r")
    color_a = rdbu(0.95)  # red
    color_b = rdbu(0.05)  # blue
    max_norm = max(norm for pos in sorted_positions for _, _, norm in pos_norms[pos])
    norm_ylim = max_norm * 1.2

    for col_idx, pos in enumerate(sorted_positions):
        entries = pos_norms[pos]
        n_bars = len(entries)
        col_w = col_nb[col_idx] * TILE_SIZE
        x_in = col_x[col_idx]
        y_in = norm_row_y_top - NORM_ROW_HEIGHT

        ax = fig.add_axes(
            (x_in / fig_width, y_in / fig_height, col_w / fig_width, NORM_ROW_HEIGHT / fig_height)
        )
        bar_x = np.arange(n_bars)
        bar_labels = [f"{label}:{idx}" for label, idx, _ in entries]
        bar_values = [norm for _, _, norm in entries]
        bar_colors = [color_a if label == label_a else color_b for label, _, _ in entries]
        ax.bar(bar_x, bar_values, color=bar_colors)
        ax.set_xticks(bar_x)
        ax.set_xticklabels(bar_labels, fontsize=7, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(0, norm_ylim)

    # Bold row labels on the left
    for row_idx, row_label in enumerate(ROW_LABELS[:n_heatmap_rows]):
        y_center = row_y_top[row_idx] - row_height / 2
        fig.text(
            ROW_LABEL_X / fig_width,
            y_center / fig_height,
            row_label,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="center",
            rotation=90,
        )
    # Norm row label
    fig.text(
        ROW_LABEL_X / fig_width,
        (norm_row_y_top - NORM_ROW_HEIGHT / 2) / fig_height,
        ROW_LABELS[3],
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center",
        rotation=90,
    )

    fig.suptitle(module_path, fontsize=12, y=(fig_height - 0.2) / fig_height)

    assert im is not None
    cbar_x = col_x[-1] + col_nb[-1] * TILE_SIZE + CBAR_GAP
    cbar_bottom = row_y_top[-1] - row_height
    cbar_ax = fig.add_axes(
        (
            cbar_x / fig_width,
            cbar_bottom / fig_height,
            CBAR_WIDTH / fig_width,
            heatmap_height / fig_height,
        )
    )
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
    output_dir: str = "/tmp/compare_decompositions_by_matrix",
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

    shared_modules = sorted(p for p in active_a if p in active_b)
    saved_count = 0

    print(f"\n{'Module':<50} {'Pos':>4} {'Token':<10} {'Best A→B':>10} {'Best B→A':>10}")
    print("-" * 88)

    for module_path in shared_modules:
        shared_positions = sorted(set(active_a[module_path]) & set(active_b[module_path]))
        if not shared_positions:
            continue

        pos_sims_full: dict[int, torch.Tensor] = {}
        pos_sims_v: dict[int, torch.Tensor] = {}
        pos_sims_u: dict[int, torch.Tensor] = {}
        pos_norms: dict[int, list[tuple[str, int, float]]] = {}
        for pos in shared_positions:
            indices_a = active_a[module_path][pos]
            indices_b = active_b[module_path][pos]
            weights_a = torch.stack(
                [get_component_weight(decomp_a.model, module_path, i).flatten() for i in indices_a]
            )
            weights_b = torch.stack(
                [get_component_weight(decomp_b.model, module_path, i).flatten() for i in indices_b]
            )
            pos_sims_full[pos] = compute_pairwise_cosine_sim(weights_a, weights_b)

            uv_a = [get_component_uv(decomp_a.model, module_path, i) for i in indices_a]
            uv_b = [get_component_uv(decomp_b.model, module_path, i) for i in indices_b]

            v_a = torch.stack([v for _, v in uv_a])
            v_b = torch.stack([v for _, v in uv_b])
            pos_sims_v[pos] = compute_pairwise_cosine_sim(v_a, v_b)

            u_a = torch.stack([u for u, _ in uv_a])
            u_b = torch.stack([u for u, _ in uv_b])
            pos_sims_u[pos] = compute_pairwise_cosine_sim(u_a, u_b)

            # Norms: sorted by label then component index
            entries: list[tuple[str, int, float]] = []
            for i, (u, v) in zip(indices_a, uv_a, strict=True):
                entries.append((decomp_a.label, i, u.norm().item() * v.norm().item()))
            for i, (u, v) in zip(indices_b, uv_b, strict=True):
                entries.append((decomp_b.label, i, u.norm().item() * v.norm().item()))
            entries.sort(key=lambda e: (e[0], e[1]))
            pos_norms[pos] = entries

        fig = plot_matrix_heatmaps(
            module_path,
            pos_sims_full,
            pos_sims_v,
            pos_sims_u,
            active_a[module_path],
            active_b[module_path],
            pos_norms,
            token_strs,
            decomp_a.label,
            decomp_b.label,
        )
        sanitized = module_path.replace(".", "_").replace("/", "_")
        path = out / f"{sanitized}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_count += 1
        print(f"  Saved {path}")

        # Summary rows for this module
        for pos in sorted(pos_sims_full):
            sim = pos_sims_full[pos].abs()
            best_a_to_b = sim.max(dim=1).values.mean().item()
            best_b_to_a = sim.max(dim=0).values.mean().item()
            tok = token_strs[pos] if pos < len(token_strs) else f"pos{pos}"
            print(f"{module_path:<50} {pos:>4} {tok:<10} {best_a_to_b:>10.4f} {best_b_to_a:>10.4f}")

    if saved_count == 0:
        print("No shared modules with active components in both runs at any position.")
    else:
        print(f"\nSaved {saved_count} images to {out}")


if __name__ == "__main__":
    fire.Fire(main)
