"""Aggregate the nontarget ablation output into a per-component summary.

Reads the per-(component, prompt, pos) KL file and the per-(prompt, pos) orig-predictions file
produced by `effect_of_ablation --nontarget`, applies task-based prompt exclusion (any nontarget
prompt containing either task A's or task B's tokenised prompt as a contiguous sub-sequence is
dropped), and writes a small per-component TSV with summary statistics (count, mean, quantile,
max of KL). Also prints the excluded prompts and the nontarget-monitor alerts to stdout.

This is consumed by `find_swap_candidates`, which would otherwise have to re-scan the (large)
nontarget KL file every time its ranking parameters change.

Usage:
    python -m spd.scripts.validation.summarize_nontarget <model_path> \
        <effect_nontarget_kl_tsv> <effect_nontarget_orig_tsv> \
        --task-a='{"prompt": "import numpy as", "target": " np"}' \
        --task-b='{"prompt": "import pandas as", "target": " pd"}' \
        [--quantile=0.99] [--prompts=PATH] [--output=PATH]
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.scripts.validation.common import (
    TaskSpec,
    contains_subsequence,
    load_prompts,
    resolve_task,
)
from spd.spd_types import ModelPath

ComponentKey = tuple[int, str, int]
_CONTEXT_SIZE = 5


@dataclass
class NontargetHit:
    task_name: str
    prompt: int
    pos: int
    context_token_ids: list[int]
    orig_pred: int
    orig_prob: float


def _scan_orig(
    orig_path: Path, task_a: TaskSpec, task_b: TaskSpec
) -> tuple[set[int], dict[tuple[int, int], NontargetHit]]:
    """Find prompts to exclude (contain a target prompt as a sub-sequence) and per-position alerts."""
    tokens_per_prompt: dict[int, dict[int, int]] = {}
    hits: dict[tuple[int, int], NontargetHit] = {}
    target_to_task_name = {
        task_a.target_token_id: task_a.name,
        task_b.target_token_id: task_b.name,
    }
    with orig_path.open() as f:
        for record in tqdm(csv.DictReader(f, delimiter="\t"), desc="nontarget orig TSV"):
            prompt = int(record["prompt"])
            pos = int(record["pos"])
            tokens_per_prompt.setdefault(prompt, {})[pos] = int(record["token"])
            orig_pred = int(record["orig_pred"])
            matched_task = target_to_task_name.get(orig_pred)
            if matched_task is None:
                continue
            pos_key = (prompt, pos)
            if pos_key in hits:
                continue
            pos_to_tok = tokens_per_prompt[prompt]
            context_ids = [
                pos_to_tok[p] for p in range(max(0, pos - _CONTEXT_SIZE), pos) if p in pos_to_tok
            ]
            hits[pos_key] = NontargetHit(
                task_name=matched_task,
                prompt=prompt,
                pos=pos,
                context_token_ids=context_ids,
                orig_pred=orig_pred,
                orig_prob=float(record["orig_prob"]),
            )

    excluded: set[int] = set()
    for prompt, pos_to_tok in tokens_per_prompt.items():
        seq = [pos_to_tok[p] for p in sorted(pos_to_tok)]
        if contains_subsequence(seq, task_a.prompt_token_ids) or contains_subsequence(
            seq, task_b.prompt_token_ids
        ):
            excluded.add(prompt)
    return excluded, hits


def _collect_kl_values(kl_path: Path, excluded: set[int]) -> dict[ComponentKey, list[float]]:
    values: dict[ComponentKey, list[float]] = {}
    with kl_path.open() as f:
        for record in tqdm(csv.DictReader(f, delimiter="\t"), desc="nontarget kl TSV"):
            prompt = int(record["prompt"])
            if prompt in excluded:
                continue
            key: ComponentKey = (
                int(record["layer"]),
                record["matrix"],
                int(record["component"]),
            )
            values.setdefault(key, []).append(float(record["kl"]))
    return values


def _report_alerts(
    excluded: set[int],
    hits: dict[tuple[int, int], NontargetHit],
    tokenizer: PreTrainedTokenizer,
) -> None:
    if excluded:
        print(
            f"[nontarget-monitor] excluding {len(excluded)} nontarget prompt(s) that contain "
            f"a target prompt as a token sub-sequence: {sorted(excluded)}"
        )
    else:
        print("[nontarget-monitor] no nontarget prompts contain a target prompt as a sub-sequence.")

    if not hits:
        print("[nontarget-monitor] no nontarget positions predict a target token.")
        return

    print(
        f"[nontarget-monitor] {len(hits)} nontarget position(s) where the original model "
        f"predicts one of the target tokens:"
    )
    for hit in sorted(hits.values(), key=lambda h: (h.prompt, h.pos)):
        ctx = tokenizer.decode(hit.context_token_ids)
        orig_tok = tokenizer.decode([hit.orig_pred])
        in_excluded = " [in-excluded-prompt]" if hit.prompt in excluded else ""
        print(
            f"[nontarget-hit] task={hit.task_name} prompt={hit.prompt} pos={hit.pos} "
            f"{ctx!r} -> orig: {orig_tok!r} ({hit.orig_prob:.3f}){in_excluded}"
        )


def summarize_nontarget(
    model_path: ModelPath,
    effect_nontarget_kl_path: str,
    effect_nontarget_orig_path: str,
    task_a: str | dict[str, Any],
    task_b: str | dict[str, Any],
    quantile: float = 0.99,
    prompts: str | None = None,
    output: str | None = None,
) -> Path:
    assert 0.0 < quantile < 1.0, f"--quantile must be in (0, 1), got {quantile}"

    run_info = SPDRunInfo.from_path(model_path)
    config = run_info.config
    run_dir = run_info.checkpoint_path.parent
    assert config.tokenizer_name is not None, "config.tokenizer_name is required"

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    prompt_texts = load_prompts(config, prompts)

    spec_a, _, _ = resolve_task("A", task_a, prompt_texts, tokenizer)
    spec_b, _, _ = resolve_task("B", task_b, prompt_texts, tokenizer)

    kl_path = Path(effect_nontarget_kl_path).expanduser()
    orig_path = Path(effect_nontarget_orig_path).expanduser()

    excluded, hits = _scan_orig(orig_path, spec_a, spec_b)
    _report_alerts(excluded, hits, tokenizer)

    values_by_key = _collect_kl_values(kl_path, excluded)

    pct = round(quantile * 100)
    quantile_col = f"kl_q{pct}"
    fieldnames = ["layer", "matrix", "component", "n_positions", "mean_kl", quantile_col, "max_kl"]

    out_path = Path(output).expanduser() if output else run_dir / "nontarget_summary.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for (layer, matrix, component), vs in sorted(values_by_key.items()):
            arr = np.asarray(vs, dtype=np.float32)
            writer.writerow(
                {
                    "layer": layer,
                    "matrix": matrix,
                    "component": component,
                    "n_positions": len(vs),
                    "mean_kl": float(arr.mean()),
                    quantile_col: float(np.quantile(arr, quantile)),
                    "max_kl": float(arr.max()),
                }
            )

    logger.info(f"Summarised {len(values_by_key)} components (quantile={quantile})")
    return out_path


if __name__ == "__main__":
    fire.Fire(summarize_nontarget)
