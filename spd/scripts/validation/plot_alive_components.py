"""Plot a CI heatmap of alive components across prompts and positions.

For each decomposed module, shows only the components whose CI exceeds `--ci-thr` on at least
one (prompt, position). Components are sorted by the earliest position they activate at in each
prompt. Subplots are arranged as a grid with rows = layers and columns = matrix types.

Prompts-based LM tasks only — the y-axis of the heatmap is the decoded `(prompt, position)`
token, which has no analogue for dataset-based or completeness tasks.

Usage:
    python -m spd.scripts.validation.plot_alive_components <model_path> \
        [--ci-thr=0.01] [--prompts=PATH] [--output=PATH]
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from transformers import AutoTokenizer

from spd.log import logger
from spd.scripts.validation.common import (
    build_lm_loader,
    is_prompt_task,
    iterate_input_ids,
    load_spd_run,
    parse_module_name,
    resolve_task_config,
)
from spd.spd_types import ModelPath


def plot_alive_components(
    model_path: ModelPath,
    ci_thr: float = 0.01,
    prompts: str | None = None,
    output: str | None = None,
) -> Path:
    """Render a grid of CI heatmaps (one per decomposed module) showing only alive components."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spd_model, config, run_dir = load_spd_run(model_path)
    spd_model = spd_model.to(device)

    task_config = resolve_task_config(config, use_nontarget=False, prompts_override=prompts)
    assert is_prompt_task(task_config), "plot_alive_components only supports prompts-based LM tasks"
    assert config.tokenizer_name is not None, "config.tokenizer_name is required"

    loader = build_lm_loader(task_config, config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    input_ids = next(iter(iterate_input_ids(loader, device)))
    n_prompts, seq_len = input_ids.shape
    logger.info(f"Loaded {n_prompts} prompts, seq_len={seq_len}")

    with torch.no_grad():
        output_with_cache = spd_model(input_ids, cache_type="input")
        ci = spd_model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=config.sampling,
        ).lower_leaky

    module_info: dict[str, tuple[int, str]] = {
        module_path: parse_module_name(module_path) for module_path in ci
    }
    all_layers = sorted({layer for layer, _ in module_info.values()})
    seen_matrices: list[str] = []
    for _, matrix_name in module_info.values():
        if matrix_name not in seen_matrices:
            seen_matrices.append(matrix_name)

    layer_to_row = {layer: i for i, layer in enumerate(all_layers)}
    matrix_to_col = {name: i for i, name in enumerate(seen_matrices)}
    n_rows = len(all_layers)
    n_cols = len(seen_matrices)
    logger.info(f"Grid: {n_rows} layers × {n_cols} matrix types")

    y_labels: list[str] = []
    for prompt_idx in range(n_prompts):
        for pos in range(seq_len):
            token_id = input_ids[prompt_idx, pos].item()
            token_str = tokenizer.decode([token_id])
            y_labels.append(f"{pos}: {token_str}")

    n_y = n_prompts * seq_len
    subplot_data: dict[tuple[int, int], tuple[np.ndarray, torch.Tensor]] = {}
    for module_path, ci_tensor in ci.items():
        layer_idx, matrix_name = module_info[module_path]
        row = layer_to_row[layer_idx]
        col = matrix_to_col[matrix_name]

        ci_flat = ci_tensor.reshape(n_y, -1).cpu()
        alive_mask = (ci_flat > ci_thr).any(dim=0)
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if len(alive_indices) == 0:
            continue

        # Sort alive components by (first-active-pos in prompt 0, prompt 1, ...) so that
        # components firing earliest sit on the left of the heatmap.
        active_per_pos = ci_tensor[:, :, alive_indices] > ci_thr  # (n_prompts, seq_len, n_alive)
        sort_keys: list[tuple[int, ...]] = []
        for i in range(len(alive_indices)):
            key = []
            for p in range(n_prompts):
                active_positions = active_per_pos[p, :, i].nonzero(as_tuple=True)[0]
                first_active = int(active_positions[0]) if len(active_positions) > 0 else seq_len
                key.append(first_active)
            sort_keys.append(tuple(key))
        sort_order = sorted(range(len(alive_indices)), key=lambda i: sort_keys[i])
        alive_sorted = alive_indices[sort_order]
        heatmap_data = ci_flat[:, alive_sorted].numpy()
        subplot_data[(row, col)] = (heatmap_data, alive_sorted)

    total_alive = 0
    total_components = 0
    logger.info(f"Alive components (CI > {ci_thr}):")
    for module_path in ci:
        layer_idx, matrix_name = module_info[module_path]
        row = layer_to_row[layer_idx]
        col = matrix_to_col[matrix_name]
        n_components = ci[module_path].shape[-1]
        total_components += n_components
        n_alive = subplot_data[(row, col)][0].shape[1] if (row, col) in subplot_data else 0
        total_alive += n_alive
        logger.info(f"  Layer {layer_idx} / {matrix_name}: {n_alive} / {n_components}")
    logger.info(f"  Total: {total_alive} / {total_components}")

    tile_inches = 0.16
    margin_left = 1.2
    margin_top = 0.5
    margin_bottom = 0.3
    margin_right = 0.8
    gap_x = 0.4
    gap_y = 0.4

    col_widths = {}
    for col in range(n_cols):
        max_alive = max(
            (subplot_data[(r, col)][0].shape[1] for r in range(n_rows) if (r, col) in subplot_data),
            default=1,
        )
        col_widths[col] = max_alive * tile_inches
    row_height = n_y * tile_inches

    fig_width = margin_left + sum(col_widths.values()) + gap_x * (n_cols - 1) + margin_right
    fig_height = margin_bottom + n_rows * row_height + gap_y * (n_rows - 1) + margin_top

    fig = plt.figure(figsize=(fig_width, fig_height))
    norm = Normalize(vmin=0, vmax=1)
    axes_grid: dict[tuple[int, int], plt.Axes] = {}

    for row in range(n_rows):
        for col in range(n_cols):
            x = margin_left + sum(col_widths[c] for c in range(col)) + gap_x * col
            y = margin_bottom + (n_rows - 1 - row) * (row_height + gap_y)
            ax = fig.add_axes(
                (
                    x / fig_width,
                    y / fig_height,
                    col_widths[col] / fig_width,
                    row_height / fig_height,
                )
            )
            axes_grid[(row, col)] = ax

            if (row, col) in subplot_data:
                heatmap_data, alive_sorted = subplot_data[(row, col)]
                ax.imshow(heatmap_data, aspect="equal", cmap="RdPu", norm=norm)
                ax.set_xticks(range(len(alive_sorted)))
                ax.set_xticklabels(alive_sorted.numpy(), fontsize=8, rotation=90)
                ax.set_yticks(range(len(y_labels)))
                ax.set_yticklabels(y_labels if col == 0 else [], fontsize=8)
                spine_lw = ax.spines["bottom"].get_linewidth()
                for p in range(1, n_prompts):
                    ax.axhline(p * seq_len - 0.5, color="gray", linewidth=spine_lw, alpha=1)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

    for matrix_name, col in matrix_to_col.items():
        axes_grid[(0, col)].set_title(matrix_name, fontsize=10, fontweight="bold", pad=10)

    label_x = (margin_left * 0.3) / fig_width
    for layer, row in layer_to_row.items():
        ax = axes_grid[(row, 0)]
        ax_pos = ax.get_position()
        label_y = ax_pos.y0 + ax_pos.height / 2
        fig.text(
            label_x,
            label_y,
            f"Layer {layer}",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
        )

    cbar_x = (margin_left + sum(col_widths.values()) + gap_x * (n_cols - 1) + 0.3) / fig_width
    cbar_ax = fig.add_axes((cbar_x, 0.15, 0.015 / fig_width * 3 * 1.8, 0.7))
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="RdPu"), cax=cbar_ax, label="Causal importance"
    )

    out_path = Path(output).expanduser() if output else run_dir / "alive_components.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved to {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(plot_alive_components)
