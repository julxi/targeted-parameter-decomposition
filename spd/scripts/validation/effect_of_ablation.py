"""Measure the per-position KL divergence induced by ablating one component at a time.

By default, writes the full per-(component, prompt, pos) KL + per-(prompt, pos) orig-prediction
TSVs. With `--summary-only`, streams batches through running-aggregate state per component
(t-digest for quantile, running sum/max/count) and writes a single summary TSV instead — skipping
the intermediate full-data file. Summary-only mode does not support prompt exclusion; use the
two-step `effect_of_ablation + summarize_nontarget` pipeline for the targeted-decomposition flow
that needs task-A/B-based exclusion.

Summary-only mode also supports checkpoint/resume via `--checkpoint-every=N`: after every N
processed batches the streaming state (t-digest centroids + running totals) is saved atomically
next to the summary TSV as `<summary>.ckpt`. If that file exists on startup it is loaded and the
data loader is fast-forwarded past the already-processed batches (requires a deterministic
loader). The checkpoint is deleted once the final summary TSV is written.

Usage:
    python -m spd.scripts.validation.effect_of_ablation <model_path> <components_tsv> \
        [--n-batches=1] [--nontarget] [--prompts=PATH] [--batch-size=N] \
        [--output-kl=PATH] [--output-orig=PATH]

    python -m spd.scripts.validation.effect_of_ablation <model_path> <components_tsv> \
        --summary-only [--quantile=0.99] [--output-summary=PATH] \
        [--checkpoint-every=N] \
        [--n-batches=1] [--nontarget] [--prompts=PATH] [--batch-size=N]

Output files:
- Full mode:
  - `effect_of_ablation[_nontarget].tsv` — one row per (component, prompt, pos) with columns
    `layer, matrix, component, prompt, pos, kl`.
  - `orig_predictions[_nontarget].tsv` — one row per (prompt, pos) with columns
    `prompt, pos, token, token_str, orig_pred, orig_pred_str, orig_prob`.
- Summary-only mode:
  - `effect_of_ablation[_nontarget]_summary.tsv` — one row per component with columns
    `layer, matrix, component, n_positions, mean_kl, kl_q{pct}, max_kl`.
"""

import csv
import pickle
from collections.abc import Callable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
import torch.nn.functional as F
from pytdigest import TDigest
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import WeightDeltaAndMask, make_mask_infos
from spd.scripts.validation.common import (
    build_lm_loader,
    build_module_lookup,
    escape_tsv_value,
    is_prompt_task,
    iterate_input_ids,
    load_spd_run,
    resolve_task_config,
)
from spd.spd_types import ModelPath

KL_FIELDS = ["layer", "matrix", "component", "prompt", "pos", "kl"]
ORIG_FIELDS = [
    "prompt",
    "pos",
    "token",
    "token_str",
    "orig_pred",
    "orig_pred_str",
    "orig_prob",
]


def _load_components(
    components_path: Path, module_lookup: dict[tuple[int, str], str]
) -> list[tuple[str, int, int, str]]:
    """Return [(module_name, layer, component, matrix), ...] rows."""
    rows: list[tuple[str, int, int, str]] = []
    with components_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for record in reader:
            layer = int(record["layer"])
            matrix = record["matrix"]
            component = int(record["component"])
            key = (layer, matrix)
            assert key in module_lookup, (
                f"No decomposed module matches layer={layer}, matrix={matrix}. "
                f"Available: {sorted(module_lookup.keys())}"
            )
            rows.append((module_lookup[key], layer, component, matrix))
    return rows


def _build_baseline_masks(
    module_to_c: dict[str, int],
    batch_shape: tuple[int, ...],
    device: torch.device,
) -> dict[str, Tensor]:
    """All-ones component masks, one tensor per module. Callers mutate a single entry in place
    to ablate a specific component and restore it afterwards."""
    return {name: torch.ones((*batch_shape, C), device=device) for name, C in module_to_c.items()}


def _build_full_delta_masks(
    weight_deltas: dict[str, Tensor],
    batch_shape: tuple[int, ...],
    device: torch.device,
) -> dict[str, WeightDeltaAndMask]:
    """Delta enabled everywhere (mask=1) for every decomposed module."""
    delta_mask = torch.ones(batch_shape, device=device)
    return {name: (weight_deltas[name], delta_mask) for name in weight_deltas}


def _make_decoder(tokenizer: PreTrainedTokenizer) -> Callable[[int], str]:
    """Return a memoised, TSV-safe single-token decoder."""
    cache: dict[int, str] = {}

    def decode(tid: int) -> str:
        s = cache.get(tid)
        if s is None:
            s = escape_tsv_value(tokenizer.decode([tid]))
            cache[tid] = s
        return s

    return decode


def effect_of_ablation(
    model_path: ModelPath,
    components_path: str,
    n_batches: int = 1,
    nontarget: bool = False,
    prompts: str | None = None,
    split: str | None = None,
    batch_size: int | None = None,
    summary_only: bool = False,
    quantile: float = 0.99,
    checkpoint_every: int = 0,
    output_kl: str | None = None,
    output_orig: str | None = None,
    output_summary: str | None = None,
) -> tuple[Path, Path] | Path:
    """Write per-(component, prompt, pos) KL TSVs (default) or a per-component summary TSV.

    Returns `(kl_path, orig_path)` in full mode or `summary_path` in summary-only mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spd_model, config, run_dir = load_spd_run(model_path)
    spd_model = spd_model.to(device)

    assert config.tokenizer_name is not None, "config.tokenizer_name is required"

    task_config = resolve_task_config(
        config, use_nontarget=nontarget, prompts_override=prompts, split_override=split
    )
    loader = build_lm_loader(task_config, config, batch_size_override=batch_size)
    single_batch = is_prompt_task(task_config)

    module_lookup = build_module_lookup(spd_model.target_module_paths)
    components_file = Path(components_path).expanduser()
    components = _load_components(components_file, module_lookup)

    n_to_run = 1 if single_batch else n_batches
    iterator = iterate_input_ids(loader, device)

    # Delta weights are fixed (target_weight - component_weight); compute once.
    weight_deltas = spd_model.calc_weight_deltas()

    if summary_only:
        assert 0.0 < quantile < 1.0, f"--quantile must be in (0, 1), got {quantile}"
        assert checkpoint_every >= 0, f"--checkpoint-every must be >= 0, got {checkpoint_every}"
        summary_default = (
            "effect_of_ablation_nontarget_summary.tsv"
            if nontarget
            else "effect_of_ablation_summary.tsv"
        )
        summary_path = (
            Path(output_summary).expanduser() if output_summary else run_dir / summary_default
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        total_prompts, seq_len = _run_summary(
            spd_model=spd_model,
            iterator=iterator,
            n_to_run=n_to_run,
            components=components,
            weight_deltas=weight_deltas,
            device=device,
            summary_path=summary_path,
            quantile=quantile,
            checkpoint_every=checkpoint_every,
        )
        logger.info(
            f"Saw {total_prompts} prompts of {seq_len} positions each "
            f"(total: {total_prompts * seq_len})"
        )
        return summary_path

    assert checkpoint_every == 0, "--checkpoint-every is only supported with --summary-only"

    kl_default = "effect_of_ablation_nontarget.tsv" if nontarget else "effect_of_ablation.tsv"
    orig_default = "orig_predictions_nontarget.tsv" if nontarget else "orig_predictions.tsv"
    kl_path = Path(output_kl).expanduser() if output_kl else run_dir / kl_default
    orig_path = Path(output_orig).expanduser() if output_orig else run_dir / orig_default
    for p in (kl_path, orig_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    decode = _make_decoder(AutoTokenizer.from_pretrained(config.tokenizer_name))
    total_prompts, seq_len = _run(
        spd_model=spd_model,
        iterator=iterator,
        n_to_run=n_to_run,
        components=components,
        weight_deltas=weight_deltas,
        decode=decode,
        device=device,
        kl_path=kl_path,
        orig_path=orig_path,
    )
    logger.info(
        f"Saw {total_prompts} prompts of {seq_len} positions each "
        f"(total: {total_prompts * seq_len})"
    )
    return kl_path, orig_path


def _run(
    *,
    spd_model: ComponentModel,
    iterator: Iterator[Tensor],
    n_to_run: int,
    components: list[tuple[str, int, int, str]],
    weight_deltas: dict[str, Tensor],
    decode: Callable[[int], str],
    device: torch.device,
    kl_path: Path,
    orig_path: Path,
) -> tuple[int, int]:
    """Stream per-(component, prompt, pos) KL and per-(prompt, pos) orig rows to the two TSVs.

    `component_masks` and `mask_infos` are built lazily from the first batch's shape and reused
    across all subsequent batches — the DataLoader uses `drop_last=True` / `StaticBatchLoader`,
    so every batch has the same shape. Returns (total_prompts, seq_len).
    """
    total_prompts = 0
    seq_len = 0
    batches_done = 0
    component_masks: dict[str, Tensor] | None = None
    mask_infos: Any = None
    first_shape: tuple[int, ...] | None = None
    with (
        kl_path.open("w", newline="") as kl_f,
        orig_path.open("w", newline="") as orig_f,
        torch.no_grad(),
    ):
        kl_writer = csv.DictWriter(kl_f, fieldnames=KL_FIELDS, delimiter="\t")
        orig_writer = csv.DictWriter(orig_f, fieldnames=ORIG_FIELDS, delimiter="\t")
        kl_writer.writeheader()
        orig_writer.writeheader()

        for batch_idx, batch in zip(tqdm(range(n_to_run), desc="batches"), iterator, strict=False):
            batches_done = batch_idx + 1
            assert batch.ndim == 2, f"Expected (batch, seq), got {tuple(batch.shape)}"
            batch_size, seq_len = batch.shape
            if component_masks is None:
                first_shape = (batch_size, seq_len)
                component_masks = _build_baseline_masks(spd_model.module_to_c, first_shape, device)
                delta_masks = _build_full_delta_masks(weight_deltas, first_shape, device)
                mask_infos = make_mask_infos(component_masks, weight_deltas_and_masks=delta_masks)
            assert (batch_size, seq_len) == first_shape, (
                f"Batch shape changed mid-pass: got {(batch_size, seq_len)}, expected {first_shape}"
            )
            total_prompts += batch_size
            batch_cpu = batch.cpu().tolist()

            orig_logits = spd_model(batch)
            assert isinstance(orig_logits, Tensor)
            orig_log_probs = F.log_softmax(orig_logits, dim=-1)
            orig_probs = orig_log_probs.exp()
            orig_prob_max, orig_pred = orig_probs.max(dim=-1)
            orig_pred_cpu = orig_pred.cpu().tolist()
            orig_prob_cpu = orig_prob_max.cpu().tolist()
            prompt_offset = batch_idx * batch_size

            for b in range(batch_size):
                prompt_idx = prompt_offset + b
                for t in range(seq_len):
                    token_id = batch_cpu[b][t]
                    orig_id = orig_pred_cpu[b][t]
                    orig_writer.writerow(
                        {
                            "prompt": prompt_idx,
                            "pos": t,
                            "token": token_id,
                            "token_str": decode(token_id),
                            "orig_pred": orig_id,
                            "orig_pred_str": decode(orig_id),
                            "orig_prob": orig_prob_cpu[b][t],
                        }
                    )

            for module_name, layer, component, matrix in tqdm(
                components, desc=f"components (batch {batch_idx})", leave=False
            ):
                mask_tensor = component_masks[module_name]
                mask_tensor[..., component] = 0.0
                ablated_logits = spd_model(batch, mask_infos=mask_infos)
                mask_tensor[..., component] = 1.0
                assert isinstance(ablated_logits, Tensor)
                ablated_log_probs = F.log_softmax(ablated_logits, dim=-1)
                kl = (orig_probs * (orig_log_probs - ablated_log_probs)).sum(dim=-1)
                kl_cpu = kl.cpu().tolist()

                for b in range(batch_size):
                    prompt_idx = prompt_offset + b
                    for t in range(seq_len):
                        kl_writer.writerow(
                            {
                                "layer": layer,
                                "matrix": matrix,
                                "component": component,
                                "prompt": prompt_idx,
                                "pos": t,
                                "kl": kl_cpu[b][t],
                            }
                        )
    if batches_done < n_to_run:
        logger.warning(
            f"Loader exhausted after {batches_done}/{n_to_run} batches (dataset too small?)"
        )
    return total_prompts, seq_len


class _ComponentStats:
    """Streaming per-component aggregate: t-digest (for quantile) + running sum/max/count."""

    __slots__ = ("digest", "n", "sum", "max")

    def __init__(self) -> None:
        self.digest = TDigest()
        self.n = 0
        self.sum = 0.0
        self.max = 0.0

    def update(self, kls: np.ndarray) -> None:
        self.digest.update(kls)
        self.n += kls.size
        self.sum += float(kls.sum())
        self.max = max(self.max, float(kls.max()))

    def to_state(self) -> dict[str, Any]:
        return {
            "centroids": self.digest.get_centroids(),
            "n": self.n,
            "sum": self.sum,
            "max": self.max,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "_ComponentStats":
        obj = cls()
        obj.digest = TDigest.of_centroids(state["centroids"])
        obj.n = state["n"]
        obj.sum = state["sum"]
        obj.max = state["max"]
        return obj


def _checkpoint_path_for(summary_path: Path) -> Path:
    return summary_path.with_suffix(summary_path.suffix + ".ckpt")


def _save_checkpoint(
    ckpt_path: Path,
    stats: dict[tuple[int, str, int], _ComponentStats],
    batches_done: int,
    total_prompts: int,
    seq_len: int,
    component_keys: list[tuple[int, str, int]],
) -> None:
    """Atomically persist streaming state so the run can resume after an interruption."""
    payload = {
        "version": 1,
        "batches_done": batches_done,
        "total_prompts": total_prompts,
        "seq_len": seq_len,
        "component_keys": component_keys,
        "stats": {k: stats[k].to_state() for k in component_keys},
    }
    tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(ckpt_path)


def _load_checkpoint(
    ckpt_path: Path, expected_keys: list[tuple[int, str, int]]
) -> tuple[dict[tuple[int, str, int], _ComponentStats], int, int, int]:
    with ckpt_path.open("rb") as f:
        payload = pickle.load(f)
    assert payload["version"] == 1, f"Unknown checkpoint version {payload['version']}"
    assert payload["component_keys"] == expected_keys, (
        f"Checkpoint components don't match current components_tsv "
        f"({len(payload['component_keys'])} vs {len(expected_keys)} rows)"
    )
    stats = {k: _ComponentStats.from_state(s) for k, s in payload["stats"].items()}
    return stats, payload["batches_done"], payload["total_prompts"], payload["seq_len"]


def _run_summary(
    *,
    spd_model: ComponentModel,
    iterator: Iterator[Tensor],
    n_to_run: int,
    components: list[tuple[str, int, int, str]],
    weight_deltas: dict[str, Tensor],
    device: torch.device,
    summary_path: Path,
    quantile: float,
    checkpoint_every: int,
) -> tuple[int, int]:
    """Stream batches through per-component running stats, write one summary TSV at the end.

    Returns (total_prompts, seq_len). Schema matches `summarize_nontarget.py`'s output:
    `layer, matrix, component, n_positions, mean_kl, kl_q{pct}, max_kl`.

    When `checkpoint_every > 0`, the streaming state is saved every N processed batches to
    `<summary>.ckpt` and loaded on startup if present, allowing interrupted runs to resume.
    """
    component_keys = [(layer, matrix, component) for _, layer, component, matrix in components]
    stats: dict[tuple[int, str, int], _ComponentStats] = {
        k: _ComponentStats() for k in component_keys
    }

    ckpt_path = _checkpoint_path_for(summary_path) if checkpoint_every > 0 else None
    start_batch = 0
    total_prompts = 0
    seq_len = 0
    if ckpt_path is not None and ckpt_path.exists():
        stats, start_batch, total_prompts, seq_len = _load_checkpoint(ckpt_path, component_keys)
        assert start_batch <= n_to_run, (
            f"Checkpoint already covers {start_batch} batches but --n-batches={n_to_run}; "
            f"increase --n-batches or delete {ckpt_path}"
        )
        logger.info(f"Resuming from checkpoint at batch {start_batch}/{n_to_run}")

    batches_done = start_batch
    component_masks: dict[str, Tensor] | None = None
    mask_infos: Any = None
    first_shape: tuple[int, ...] | None = None

    with torch.no_grad():
        if start_batch > 0:
            for _ in tqdm(
                islice(iterator, start_batch), total=start_batch, desc="skipping (resume)"
            ):
                pass

        remaining = n_to_run - start_batch
        for offset, batch in zip(tqdm(range(remaining), desc="batches"), iterator, strict=False):
            batch_idx = start_batch + offset
            batches_done = batch_idx + 1
            assert batch.ndim == 2, f"Expected (batch, seq), got {tuple(batch.shape)}"
            batch_size, seq_len = batch.shape
            if component_masks is None:
                first_shape = (batch_size, seq_len)
                component_masks = _build_baseline_masks(spd_model.module_to_c, first_shape, device)
                delta_masks = _build_full_delta_masks(weight_deltas, first_shape, device)
                mask_infos = make_mask_infos(component_masks, weight_deltas_and_masks=delta_masks)
            assert (batch_size, seq_len) == first_shape, (
                f"Batch shape changed mid-pass: got {(batch_size, seq_len)}, expected {first_shape}"
            )
            total_prompts += batch_size

            orig_logits = spd_model(batch)
            assert isinstance(orig_logits, Tensor)
            orig_log_probs = F.log_softmax(orig_logits, dim=-1)
            orig_probs = orig_log_probs.exp()

            for module_name, layer, component, matrix in tqdm(
                components, desc=f"components (batch {batch_idx})", leave=False
            ):
                mask_tensor = component_masks[module_name]
                mask_tensor[..., component] = 0.0
                ablated_logits = spd_model(batch, mask_infos=mask_infos)
                mask_tensor[..., component] = 1.0
                assert isinstance(ablated_logits, Tensor)
                ablated_log_probs = F.log_softmax(ablated_logits, dim=-1)
                kl = (orig_probs * (orig_log_probs - ablated_log_probs)).sum(dim=-1)
                stats[(layer, matrix, component)].update(kl.flatten().cpu().numpy())

            if ckpt_path is not None and batches_done % checkpoint_every == 0:
                _save_checkpoint(
                    ckpt_path, stats, batches_done, total_prompts, seq_len, component_keys
                )

    if batches_done < n_to_run:
        logger.warning(
            f"Loader exhausted after {batches_done}/{n_to_run} batches (dataset too small?)"
        )

    pct = round(quantile * 100)
    quantile_col = f"kl_q{pct}"
    fieldnames = ["layer", "matrix", "component", "n_positions", "mean_kl", quantile_col, "max_kl"]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for (layer, matrix, component), s in sorted(stats.items()):
            assert s.n > 0, f"Component {(layer, matrix, component)} saw no positions"
            writer.writerow(
                {
                    "layer": layer,
                    "matrix": matrix,
                    "component": component,
                    "n_positions": s.n,
                    "mean_kl": s.sum / s.n,
                    quantile_col: float(s.digest.inverse_cdf(quantile)),  # pyright: ignore[reportArgumentType]
                    "max_kl": s.max,
                }
            )
    if ckpt_path is not None and ckpt_path.exists():
        ckpt_path.unlink()
    logger.info(f"Summarised {len(stats)} components (quantile={quantile})")
    return total_prompts, seq_len


if __name__ == "__main__":
    fire.Fire(effect_of_ablation)
