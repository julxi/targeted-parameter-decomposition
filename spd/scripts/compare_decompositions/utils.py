"""Utilities for comparing attention components across SPD decompositions.

Loads two decomposition runs, computes CI on a prompt, identifies active components,
splits component weights by attention head, and computes pairwise cosine similarity.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from matplotlib.figure import Figure
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from spd.configs import SamplingType
from spd.models.component_model import CIOutputs, ComponentModel, SPDRunInfo
from spd.models.components import LinearComponents


@dataclass
class DecompositionInfo:
    model: ComponentModel
    run_info: SPDRunInfo
    label: str


@dataclass
class ProjComparisonResult:
    layer_idx: int
    proj_type: str  # "q_proj", "k_proj", "v_proj", "o_proj"
    label_a: str
    label_b: str
    active_indices_a: list[int]
    active_indices_b: list[int]
    # sim_tensor[i, j] is (n_split, n_split) cosine sim between
    # head h1 of comp active_indices_a[i] vs head h2 of comp active_indices_b[j]
    sim_tensor: Float[Tensor, "Na Nb n_split_a n_split_b"]


def label_from_path(path: str) -> str:
    """Derive a label from a wandb/local path via SPDRunInfo.checkpoint_path."""
    run_info = SPDRunInfo.from_path(path)
    return run_info.checkpoint_path.parent.name


def load_decomposition(
    wandb_path: str, label: str | None = None, device: str = "cuda"
) -> DecompositionInfo:
    run_info = SPDRunInfo.from_path(wandb_path)
    if label is None:
        label = run_info.checkpoint_path.parent.name
    model = ComponentModel.from_run_info(run_info)
    model.to(device).eval()
    return DecompositionInfo(model=model, run_info=run_info, label=label)


def get_tokenizer(decomp: DecompositionInfo) -> PreTrainedTokenizerBase:
    tokenizer_name = decomp.run_info.config.tokenizer_name
    assert tokenizer_name is not None, "Config has no tokenizer_name"
    return AutoTokenizer.from_pretrained(tokenizer_name)


@torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
def compute_ci(
    model: ComponentModel,
    tokens: Tensor,
    sampling: SamplingType,
) -> dict[str, Float[Tensor, "... C"]]:
    out = model(tokens, cache_type="input")
    ci: CIOutputs = model.calc_causal_importances(
        pre_weight_acts=out.cache,
        sampling=sampling,
        detach_inputs=True,
    )
    return ci.lower_leaky


def get_active_component_indices(
    ci_dict: dict[str, Float[Tensor, "... C"]],
    threshold: float,
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for module_path, ci in ci_dict.items():
        # Max CI across all non-component dims (batch, seq, etc.)
        max_ci = ci.amax(dim=tuple(range(ci.ndim - 1)))
        active_mask = max_ci > threshold
        result[module_path] = sorted(active_mask.nonzero(as_tuple=True)[0].tolist())
    return result


def get_active_component_indices_per_position(
    ci_dict: dict[str, Float[Tensor, "... C"]],
    threshold: float,
) -> dict[str, dict[int, list[int]]]:
    """Return {module_path: {seq_pos: [active_component_indices]}}.

    For each position, finds components where CI > threshold (max over batch dim).
    Expects CI shape (batch, seq, C) or (seq, C).
    """
    result: dict[str, dict[int, list[int]]] = {}
    for module_path, ci in ci_dict.items():
        assert ci.ndim >= 2, f"Expected CI with at least 2 dims, got {ci.shape}"
        # Collapse all dims except seq and C: max over batch dims
        ci_flat = ci.amax(dim=tuple(range(ci.ndim - 2))) if ci.ndim > 2 else ci
        seq_len, _n_components = ci_flat.shape
        pos_dict: dict[int, list[int]] = {}
        for pos in range(seq_len):
            active_mask = ci_flat[pos] > threshold
            indices = sorted(active_mask.nonzero(as_tuple=True)[0].tolist())
            if indices:
                pos_dict[pos] = indices
        if pos_dict:
            result[module_path] = pos_dict
    return result


_ATTN_PATTERN = re.compile(r"h\.(\d+)\.attn\.(q_proj|k_proj|v_proj|o_proj)")


def get_attn_info(model: ComponentModel) -> tuple[int, int, int]:
    """Returns (n_heads, n_kv_heads, d_head) from the target model's first attention module."""
    for _name, module in model.target_model.named_modules():
        if (
            hasattr(module, "n_head")
            and hasattr(module, "n_key_value_heads")
            and hasattr(module, "head_dim")
        ):
            n_heads = module.n_head
            n_kv_heads = module.n_key_value_heads
            d_head = module.head_dim
            assert isinstance(n_heads, int)
            assert isinstance(n_kv_heads, int)
            assert isinstance(d_head, int)
            return n_heads, n_kv_heads, d_head
    raise ValueError("Could not find attention module with n_head/n_key_value_heads/head_dim")


def parse_attn_module_path(path: str) -> tuple[int, str] | None:
    m = _ATTN_PATTERN.search(path)
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


def get_component_weight(
    model: ComponentModel, module_path: str, idx: int
) -> Float[Tensor, "d_in d_out"]:
    comp = model.components[module_path]
    assert isinstance(comp, LinearComponents)
    # V[:, idx:idx+1] @ U[idx:idx+1, :] -> (d_in, d_out)
    return comp.V[:, idx : idx + 1] @ comp.U[idx : idx + 1, :]


def split_weight_by_head(
    weight: Float[Tensor, "d_in d_out"],
    n_split_heads: int,
    proj_type: str,
) -> Float[Tensor, "n_heads flat_dim"]:
    """Split a component weight matrix by attention head, returning flattened per-head vectors."""
    d_in, d_out = weight.shape
    if proj_type == "o_proj":
        # O proj: d_in = n_heads * d_head, split input dim
        d_head = d_in // n_split_heads
        # (n_heads, d_head, d_out)
        reshaped = weight.reshape(n_split_heads, d_head, d_out)
        return reshaped.reshape(n_split_heads, -1)
    else:
        # Q/K/V proj: d_out = n_split_heads * d_head, split output dim
        d_head = d_out // n_split_heads
        # (d_in, n_heads, d_head) -> (n_heads, d_in, d_head)
        reshaped = weight.reshape(d_in, n_split_heads, d_head).permute(1, 0, 2)
        return reshaped.reshape(n_split_heads, -1)


def compute_pairwise_cosine_sim(
    weights_a: Float[Tensor, "Na flat"],
    weights_b: Float[Tensor, "Nb flat"],
) -> Float[Tensor, "Na Nb"]:
    a_norm = F.normalize(weights_a, dim=-1)
    b_norm = F.normalize(weights_b, dim=-1)
    return a_norm @ b_norm.T


def compare_attention_heads(
    decomp_a: DecompositionInfo,
    decomp_b: DecompositionInfo,
    active_a: dict[str, list[int]],
    active_b: dict[str, list[int]],
) -> list[ProjComparisonResult]:
    n_heads, n_kv_heads, _d_head = get_attn_info(decomp_a.model)

    shared_modules = [
        p for p in active_a if p in active_b and parse_attn_module_path(p) is not None
    ]

    results: list[ProjComparisonResult] = []
    for module_path in sorted(shared_modules):
        parsed = parse_attn_module_path(module_path)
        assert parsed is not None
        layer_idx, proj_type = parsed

        indices_a = active_a[module_path]
        indices_b = active_b[module_path]
        if not indices_a or not indices_b:
            continue

        n_split = n_heads if proj_type in ("q_proj", "o_proj") else n_kv_heads

        # Get per-component weights split by head: (N_active, n_split, flat_dim)
        stacked_a = torch.stack(
            [
                split_weight_by_head(
                    get_component_weight(decomp_a.model, module_path, idx), n_split, proj_type
                )
                for idx in indices_a
            ]
        )
        stacked_b = torch.stack(
            [
                split_weight_by_head(
                    get_component_weight(decomp_b.model, module_path, idx), n_split, proj_type
                )
                for idx in indices_b
            ]
        )

        # For each pair (i, j), compute (n_split, n_split) cosine sim
        na, nb = len(indices_a), len(indices_b)
        sim_tensor = torch.zeros(na, nb, n_split, n_split)
        for i in range(na):
            for j in range(nb):
                sim_tensor[i, j] = compute_pairwise_cosine_sim(stacked_a[i], stacked_b[j])

        results.append(
            ProjComparisonResult(
                layer_idx=layer_idx,
                proj_type=proj_type,
                label_a=decomp_a.label,
                label_b=decomp_b.label,
                active_indices_a=indices_a,
                active_indices_b=indices_b,
                sim_tensor=sim_tensor,
            )
        )

    return results


def plot_head_comparisons(result: ProjComparisonResult) -> Figure:
    """Plot a grid of heatmaps for a single ProjComparisonResult.

    Grid: rows = components from run A, cols = components from run B.
    Each subplot: (n_split, n_split) heatmap with head indices on both axes.
    """
    na = len(result.active_indices_a)
    nb = len(result.active_indices_b)
    n_split = result.sim_tensor.shape[2]

    fig, axes = plt.subplots(na, nb, figsize=(3 * nb + 1, 3 * na + 1), squeeze=False)

    im = None
    for i in range(na):
        for j in range(nb):
            ax = axes[i][j]
            sim_np = result.sim_tensor[i, j].cpu().numpy()
            im = ax.imshow(sim_np, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            ax.set_xticks(range(n_split))
            ax.set_xticklabels(range(n_split), fontsize=6)
            ax.set_yticks(range(n_split))
            ax.set_yticklabels(range(n_split), fontsize=6)

            if i == 0:
                ax.set_title(f"{result.label_b} C{result.active_indices_b[j]}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{result.label_a} C{result.active_indices_a[i]}", fontsize=8)
            if i == na - 1:
                ax.set_xlabel("Head", fontsize=7)

    fig.suptitle(f"Layer {result.layer_idx} {result.proj_type}", fontsize=14)
    fig.tight_layout()
    assert im is not None
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Cosine Similarity")
    return fig


@dataclass
class PositionUVComparisonResult:
    layer_idx: int
    proj_type: str  # "q_proj", "k_proj", "v_proj", "o_proj"
    seq_pos: int
    token_str: str
    label_a: str
    label_b: str
    active_indices_a: list[int]
    active_indices_b: list[int]
    u_cos_sim: Float[Tensor, "Na Nb n_split"]  # per-head U cosine similarity
    v_cos_sim: Float[Tensor, "Na Nb n_split"]  # per-head V cosine similarity
    magnitude: Float[
        Tensor, "Na Nb n_split"
    ]  # per-head magnitude: mean of ||U_h||*||V|| or ||U||*||V_h||


def get_component_uv(model: ComponentModel, module_path: str, idx: int) -> tuple[Tensor, Tensor]:
    """Returns (U[idx], V[:, idx]) — raw U and V vectors."""
    comp = model.components[module_path]
    assert isinstance(comp, LinearComponents)
    return comp.U[idx], comp.V[:, idx]


def compare_attention_heads_uv_by_position(
    decomp_a: DecompositionInfo,
    decomp_b: DecompositionInfo,
    active_a: dict[str, dict[int, list[int]]],
    active_b: dict[str, dict[int, list[int]]],
    token_strs: list[str],
) -> list[PositionUVComparisonResult]:
    """Compare UV components at each position where both runs have active components."""
    n_heads, n_kv_heads, _d_head = get_attn_info(decomp_a.model)

    shared_modules = [
        p for p in active_a if p in active_b and parse_attn_module_path(p) is not None
    ]

    results: list[PositionUVComparisonResult] = []
    for module_path in sorted(shared_modules):
        parsed = parse_attn_module_path(module_path)
        assert parsed is not None
        layer_idx, proj_type = parsed

        shared_positions = sorted(set(active_a[module_path]) & set(active_b[module_path]))

        n_split = n_heads if proj_type in ("q_proj", "o_proj") else n_kv_heads

        for pos in shared_positions:
            indices_a = active_a[module_path][pos]
            indices_b = active_b[module_path][pos]
            na, nb = len(indices_a), len(indices_b)

            u_cos_sim = torch.zeros(na, nb, n_split)
            v_cos_sim = torch.zeros(na, nb, n_split)
            magnitude = torch.zeros(na, nb, n_split)

            for i, idx_a in enumerate(indices_a):
                u_a, v_a = get_component_uv(decomp_a.model, module_path, idx_a)
                for j, idx_b in enumerate(indices_b):
                    u_b, v_b = get_component_uv(decomp_b.model, module_path, idx_b)

                    if proj_type == "o_proj":
                        v_a_heads = v_a.reshape(n_split, -1)
                        v_b_heads = v_b.reshape(n_split, -1)
                        v_cos_sim[i, j] = F.cosine_similarity(v_a_heads, v_b_heads, dim=-1)
                        u_scalar = F.cosine_similarity(u_a.unsqueeze(0), u_b.unsqueeze(0)).squeeze()
                        u_cos_sim[i, j] = u_scalar.expand(n_split)
                        mag_a = v_a_heads.norm(dim=-1) * u_a.norm()
                        mag_b = v_b_heads.norm(dim=-1) * u_b.norm()
                        magnitude[i, j] = (mag_a + mag_b) / 2
                    else:
                        u_a_heads = u_a.reshape(n_split, -1)
                        u_b_heads = u_b.reshape(n_split, -1)
                        u_cos_sim[i, j] = F.cosine_similarity(u_a_heads, u_b_heads, dim=-1)
                        v_scalar = F.cosine_similarity(v_a.unsqueeze(0), v_b.unsqueeze(0)).squeeze()
                        v_cos_sim[i, j] = v_scalar.expand(n_split)
                        mag_a = u_a_heads.norm(dim=-1) * v_a.norm()
                        mag_b = u_b_heads.norm(dim=-1) * v_b.norm()
                        magnitude[i, j] = (mag_a + mag_b) / 2

            results.append(
                PositionUVComparisonResult(
                    layer_idx=layer_idx,
                    proj_type=proj_type,
                    seq_pos=pos,
                    token_str=token_strs[pos] if pos < len(token_strs) else f"pos{pos}",
                    label_a=decomp_a.label,
                    label_b=decomp_b.label,
                    active_indices_a=indices_a,
                    active_indices_b=indices_b,
                    u_cos_sim=u_cos_sim,
                    v_cos_sim=v_cos_sim,
                    magnitude=magnitude,
                )
            )

    return results


def plot_head_comparisons_uv(results_for_proj: list[PositionUVComparisonResult]) -> Figure:
    """Position-grouped scatter plot for results sharing the same (layer_idx, proj_type).

    Each position gets a Na*Nb subplot grid placed side by side. Positions are separated
    visually with separator lines. Dot size is normalized globally across all positions.
    """
    assert len(results_for_proj) > 0
    r0 = results_for_proj[0]
    n_split = r0.u_cos_sim.shape[2]

    # Global magnitude normalization
    mag_max = max(r.magnitude.max().item() for r in results_for_proj)
    if mag_max == 0:
        mag_max = 1.0

    n_positions = len(results_for_proj)
    na_per_pos = [len(r.active_indices_a) for r in results_for_proj]
    nb_per_pos = [len(r.active_indices_b) for r in results_for_proj]

    total_cols = sum(nb_per_pos)
    max_rows = max(na_per_pos)

    fig, all_axes = plt.subplots(
        max_rows,
        total_cols,
        figsize=(3 * total_cols + 1, 3 * max_rows + 1),
        squeeze=False,
    )

    head_x = np.arange(n_split)
    col_offset = 0

    for pos_idx, r in enumerate(results_for_proj):
        na = len(r.active_indices_a)
        nb = len(r.active_indices_b)

        for i in range(max_rows):
            for j in range(nb):
                ax = all_axes[i][col_offset + j]
                if i >= na:
                    ax.set_visible(False)
                    continue

                sizes = (r.magnitude[i, j].cpu().numpy() / mag_max) * 200
                u_vals = r.u_cos_sim[i, j].cpu().numpy()
                v_vals = r.v_cos_sim[i, j].cpu().numpy()

                ax.scatter(head_x, u_vals, s=sizes, c="red", alpha=0.7, label="U", zorder=3)
                ax.scatter(head_x, v_vals, s=sizes, c="blue", alpha=0.7, label="V", zorder=3)
                ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                ax.set_ylim(-1.1, 1.1)
                ax.set_xticks(head_x)
                ax.set_xticklabels(head_x, fontsize=6)
                ax.set_box_aspect(1)

                if i == 0:
                    title = f"pos {r.seq_pos} '{r.token_str}'\n"
                    title += f"{r.label_b} C{r.active_indices_b[j]}"
                    ax.set_title(title, fontsize=8)
                if j == 0 and col_offset == 0:
                    ax.set_ylabel(f"{r.label_a} C{r.active_indices_a[i]}", fontsize=8)
                elif j == 0:
                    ax.set_ylabel(f"C{r.active_indices_a[i]}", fontsize=8)
                if i == max_rows - 1 or (i == na - 1):
                    ax.set_xlabel("Head", fontsize=7)

        # Draw a vertical separator line between position groups
        if pos_idx < n_positions - 1 and col_offset + nb < total_cols:
            sep_x = (col_offset + nb) / total_cols
            fig.add_artist(
                plt.Line2D(
                    [sep_x, sep_x],
                    [0.02, 0.92],
                    transform=fig.transFigure,
                    color="gray",
                    linewidth=1,
                    linestyle="--",
                )
            )

        col_offset += nb

    fig.suptitle(f"Layer {r0.layer_idx} {r0.proj_type} (U/V)", fontsize=14)
    fig.tight_layout()

    # Legend from first visible axes
    for ax in all_axes.flat:
        if ax.get_visible():
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    ncol=2,
                    fontsize=8,
                    bbox_to_anchor=(0.5, -0.02),
                )
                break

    return fig


def plot_comparison_summary_uv(results: list[PositionUVComparisonResult]) -> Figure:
    """Bar chart with two bars per module (U and V), averaging best-match across positions."""
    grouped: dict[str, list[PositionUVComparisonResult]] = defaultdict(list)
    for r in results:
        grouped[f"L{r.layer_idx}.{r.proj_type}"].append(r)

    labels: list[str] = []
    u_best_matches: list[float] = []
    v_best_matches: list[float] = []

    for key in sorted(grouped):
        pos_results = grouped[key]
        u_bests: list[float] = []
        v_bests: list[float] = []

        for r in pos_results:
            u_per_pair = r.u_cos_sim.abs().mean(dim=-1)  # (Na, Nb)
            v_per_pair = r.v_cos_sim.abs().mean(dim=-1)

            u_best_a = u_per_pair.max(dim=1).values.mean().item()
            u_best_b = u_per_pair.max(dim=0).values.mean().item()
            v_best_a = v_per_pair.max(dim=1).values.mean().item()
            v_best_b = v_per_pair.max(dim=0).values.mean().item()

            u_bests.append((u_best_a + u_best_b) / 2)
            v_bests.append((v_best_a + v_best_b) / 2)

        labels.append(key)
        u_best_matches.append(sum(u_bests) / len(u_bests))
        v_best_matches.append(sum(v_bests) / len(v_bests))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, u_best_matches, width, color="red", alpha=0.7, label="U")
    ax.bar(x + width / 2, v_best_matches, width, color="blue", alpha=0.7, label="V")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=8, ha="right")
    ax.set_ylabel("Mean Best-Match |cos sim|")
    ax.set_title("U/V Component Similarity Summary (averaged across positions)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_comparison_summary(results: list[ProjComparisonResult]) -> Figure:
    """Bar chart of max |cos sim| and mean best-match per (layer, proj)."""
    labels: list[str] = []
    mean_best_matches: list[float] = []

    for r in results:
        # Collapse head dims: take max |cos sim| across heads for each (comp_a, comp_b) pair
        # sim_tensor: (Na, Nb, n_split, n_split)
        per_pair_max = r.sim_tensor.abs().amax(dim=(-2, -1))  # (Na, Nb)

        # Best match for each component in A (max over B), then average
        best_a = per_pair_max.max(dim=1).values.mean().item()
        best_b = per_pair_max.max(dim=0).values.mean().item()
        mean_best = (best_a + best_b) / 2

        labels.append(f"L{r.layer_idx}.{r.proj_type}")
        mean_best_matches.append(mean_best)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 4))
    x = np.arange(len(labels))
    ax.bar(x, mean_best_matches, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=8, ha="right")
    ax.set_ylabel("Mean Best-Match |cos sim|")
    ax.set_title("Component Similarity Summary")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
