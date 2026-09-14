"""Find (component_A, component_B) pairs that are good candidates for U-vector swapping.

A pair `(a_comp, b_comp)` in the same `(layer, matrix)` is a candidate iff:
- `a_comp` is important for task A only: KL at task A's target > `--high-kl` AND KL at task B's
  target < `--low-kl`.
- `b_comp` is important for task B only: KL at task B's target > `--high-kl` AND KL at task A's
  target < `--low-kl`.
- Neither component disrupts the nontarget distribution much: nontarget KL quantile (from the
  summary file) < `--low-kl` for both.

Pairs are sorted by the smallest log-ratio margin across the six conditions (larger = more
robustly satisfies the criteria); there is no top-k filter — every pair that passes is written.

Reads:
- target KL + orig-predictions files (produced by `effect_of_ablation`) for the task-specific KLs
  (and asserts the original model actually predicts the target token at each task position);
- a nontarget summary TSV (produced by `summarize_nontarget`).

Usage:
    python -m spd.scripts.validation.find_swap_candidates <model_path> \
        <effect_target_kl_tsv> <effect_target_orig_tsv> <nontarget_summary_tsv> \
        --task-a='{"prompt": "import numpy as", "target": " np"}' \
        --task-b='{"prompt": "import pandas as", "target": " pd"}' \
        [--high-kl=0.5] [--low-kl=0.1] [--prompts=PATH] [--output=PATH]
"""

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
from tqdm import tqdm
from transformers import AutoTokenizer

from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.scripts.validation.common import (
    TaskSpec,
    load_prompts,
    resolve_task,
)
from spd.spd_types import ModelPath

ComponentKey = tuple[int, str, int]  # (layer, matrix, component)
MatrixKey = tuple[int, str]
_KL_FLOOR = 1e-12


@dataclass
class PairScore:
    layer: int
    matrix: str
    a_comp: int
    b_comp: int
    a_on_kl: float  # A's KL on task A (its intended task; must exceed high_kl)
    a_off_kl: float  # A's KL on task B (must stay below low_kl)
    b_on_kl: float  # B's KL on task B
    b_off_kl: float  # B's KL on task A
    a_nontarg_kl: float
    b_nontarg_kl: float
    margin: float  # min log-ratio margin across all six cutoffs (larger = more robust)


def _assert_orig_predicts_target(orig_path: Path, task: TaskSpec) -> None:
    """Sanity check: the original model's top-1 prediction at the task position is the target token."""
    with orig_path.open() as f:
        for record in csv.DictReader(f, delimiter="\t"):
            if int(record["prompt"]) != task.prompt_idx or int(record["pos"]) != task.last_pos:
                continue
            orig_pred = int(record["orig_pred"])
            assert orig_pred == task.target_token_id, (
                f"Task {task.name}: orig_predictions row at prompt={task.prompt_idx}, "
                f"pos={task.last_pos} has orig_pred={orig_pred}, but target_token_id="
                f"{task.target_token_id}. The original model doesn't predict the target token "
                "— aborting."
            )
            return
    raise AssertionError(
        f"Task {task.name}: no row for (prompt={task.prompt_idx}, pos={task.last_pos}) in "
        f"{orig_path}"
    )


def _read_target_kls(
    kl_path: Path, orig_path: Path, task_a: TaskSpec, task_b: TaskSpec
) -> tuple[dict[ComponentKey, float], dict[ComponentKey, float]]:
    """Return per-component KL at each task's target position, for tasks A and B."""
    _assert_orig_predicts_target(orig_path, task_a)
    _assert_orig_predicts_target(orig_path, task_b)

    kls_a: dict[ComponentKey, float] = {}
    kls_b: dict[ComponentKey, float] = {}
    with kl_path.open() as f:
        for record in tqdm(csv.DictReader(f, delimiter="\t"), desc="target kl TSV"):
            prompt = int(record["prompt"])
            pos = int(record["pos"])
            for task, bucket in ((task_a, kls_a), (task_b, kls_b)):
                if prompt != task.prompt_idx or pos != task.last_pos:
                    continue
                key: ComponentKey = (
                    int(record["layer"]),
                    record["matrix"],
                    int(record["component"]),
                )
                bucket[key] = float(record["kl"])
    return kls_a, kls_b


_QUANTILE_COL_RE = re.compile(r"^kl_q(\d+)$")


def _read_nontarget_summary(summary_path: Path) -> tuple[dict[ComponentKey, float], int]:
    """Read the summary TSV; return (side_effects, quantile_pct).

    The quantile percent is read from the `kl_q<pct>` column name written by `summarize_nontarget`.
    """
    side_effects: dict[ComponentKey, float] = {}
    with summary_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        assert reader.fieldnames is not None, f"Empty summary file: {summary_path}"
        quantile_cols = [c for c in reader.fieldnames if _QUANTILE_COL_RE.match(c)]
        assert len(quantile_cols) == 1, (
            f"Expected exactly one `kl_q<pct>` column in {summary_path}, got {quantile_cols}"
        )
        quantile_col = quantile_cols[0]
        match = _QUANTILE_COL_RE.match(quantile_col)
        assert match is not None
        quantile_pct = int(match.group(1))

        for record in reader:
            key: ComponentKey = (
                int(record["layer"]),
                record["matrix"],
                int(record["component"]),
            )
            side_effects[key] = float(record[quantile_col])
    return side_effects, quantile_pct


def _task_specific_pool(
    on_task_kl: dict[ComponentKey, float],
    off_task_kl: dict[ComponentKey, float],
    nontarg_kl: dict[ComponentKey, float],
    high_kl: float,
    low_kl: float,
) -> dict[ComponentKey, tuple[float, float, float]]:
    """Return components that pass the per-task cutoffs, with their (on, off, nontarg) KLs."""
    pool: dict[ComponentKey, tuple[float, float, float]] = {}
    for key, on_kl in on_task_kl.items():
        off_kl = off_task_kl.get(key, 0.0)
        nontarg = nontarg_kl.get(key, 0.0)
        if on_kl > high_kl and off_kl < low_kl and nontarg < low_kl:
            pool[key] = (on_kl, off_kl, nontarg)
    return pool


def _find_candidate_pairs(
    targ_kl_a: dict[ComponentKey, float],
    targ_kl_b: dict[ComponentKey, float],
    nontarg_kl: dict[ComponentKey, float],
    high_kl: float,
    low_kl: float,
) -> list[PairScore]:
    """Same-matrix (a, b) pairs where each component passes its side's cutoffs, sorted by margin."""
    a_pool = _task_specific_pool(targ_kl_a, targ_kl_b, nontarg_kl, high_kl, low_kl)
    b_pool = _task_specific_pool(targ_kl_b, targ_kl_a, nontarg_kl, high_kl, low_kl)

    a_by_matrix: dict[MatrixKey, list[ComponentKey]] = {}
    b_by_matrix: dict[MatrixKey, list[ComponentKey]] = {}
    for key in a_pool:
        a_by_matrix.setdefault((key[0], key[1]), []).append(key)
    for key in b_pool:
        b_by_matrix.setdefault((key[0], key[1]), []).append(key)

    pairs: list[PairScore] = []
    for matrix_key in a_by_matrix.keys() & b_by_matrix.keys():
        layer, matrix = matrix_key
        for a_key in a_by_matrix[matrix_key]:
            a_on, a_off, a_nontarg = a_pool[a_key]
            for b_key in b_by_matrix[matrix_key]:
                if a_key == b_key:
                    continue
                b_on, b_off, b_nontarg = b_pool[b_key]
                margin = min(
                    math.log(a_on / high_kl),
                    math.log(b_on / high_kl),
                    math.log(low_kl / max(a_off, _KL_FLOOR)),
                    math.log(low_kl / max(b_off, _KL_FLOOR)),
                    math.log(low_kl / max(a_nontarg, _KL_FLOOR)),
                    math.log(low_kl / max(b_nontarg, _KL_FLOOR)),
                )
                pairs.append(
                    PairScore(
                        layer=layer,
                        matrix=matrix,
                        a_comp=a_key[2],
                        b_comp=b_key[2],
                        a_on_kl=a_on,
                        a_off_kl=a_off,
                        b_on_kl=b_on,
                        b_off_kl=b_off,
                        a_nontarg_kl=a_nontarg,
                        b_nontarg_kl=b_nontarg,
                        margin=margin,
                    )
                )

    pairs.sort(key=lambda p: p.margin, reverse=True)
    return pairs


def _write_pairs(pairs: list[PairScore], quantile_pct: int, out_path: Path) -> None:
    a_nontarg_col = f"a_nontarg_kl_q{quantile_pct}"
    b_nontarg_col = f"b_nontarg_kl_q{quantile_pct}"
    fieldnames = [
        "rank",
        "layer",
        "matrix",
        "a_comp",
        "b_comp",
        "a_on_kl",
        "a_off_kl",
        "b_on_kl",
        "b_off_kl",
        a_nontarg_col,
        b_nontarg_col,
        "margin",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for rank, p in enumerate(pairs, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "layer": p.layer,
                    "matrix": p.matrix,
                    "a_comp": p.a_comp,
                    "b_comp": p.b_comp,
                    "a_on_kl": p.a_on_kl,
                    "a_off_kl": p.a_off_kl,
                    "b_on_kl": p.b_on_kl,
                    "b_off_kl": p.b_off_kl,
                    a_nontarg_col: p.a_nontarg_kl,
                    b_nontarg_col: p.b_nontarg_kl,
                    "margin": p.margin,
                }
            )


def find_swap_candidates(
    model_path: ModelPath,
    effect_target_kl_path: str,
    effect_target_orig_path: str,
    nontarget_summary_path: str,
    task_a: str | dict[str, Any],
    task_b: str | dict[str, Any],
    high_kl: float = 0.5,
    low_kl: float = 0.1,
    prompts: str | None = None,
    output: str | None = None,
) -> Path:
    assert high_kl > low_kl, f"--high-kl ({high_kl}) must exceed --low-kl ({low_kl})"

    run_info = SPDRunInfo.from_path(model_path)
    config = run_info.config
    run_dir = run_info.checkpoint_path.parent
    assert config.tokenizer_name is not None, "config.tokenizer_name is required"

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    prompt_texts = load_prompts(config, prompts)

    spec_a, _, _ = resolve_task("A", task_a, prompt_texts, tokenizer)
    spec_b, _, _ = resolve_task("B", task_b, prompt_texts, tokenizer)

    target_kl = Path(effect_target_kl_path).expanduser()
    target_orig = Path(effect_target_orig_path).expanduser()
    summary_path = Path(nontarget_summary_path).expanduser()

    targ_kl_a, targ_kl_b = _read_target_kls(target_kl, target_orig, spec_a, spec_b)
    assert targ_kl_a, f"No target-ablation rows found for task A (prompt_idx={spec_a.prompt_idx})"
    assert targ_kl_b, f"No target-ablation rows found for task B (prompt_idx={spec_b.prompt_idx})"

    nontarg_kl, quantile_pct = _read_nontarget_summary(summary_path)
    pairs = _find_candidate_pairs(targ_kl_a, targ_kl_b, nontarg_kl, high_kl, low_kl)
    logger.info(f"Found {len(pairs)} candidate pairs (high_kl={high_kl}, low_kl={low_kl})")

    out_path = Path(output).expanduser() if output else run_dir / "swap_candidates.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pairs(pairs, quantile_pct, out_path)
    return out_path


if __name__ == "__main__":
    fire.Fire(find_swap_candidates)
