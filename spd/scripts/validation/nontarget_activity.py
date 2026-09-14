"""Summarize alive components' activity on the harvested (nontarget) dataset.

Reads the pre-computed harvest data for a decomposition and joins it with an
`alive_components.tsv` produced by `find_alive_components.py`. Writes:
- `nontarget_activity.tsv`: firing stats per alive component.
- `nontarget_activity_sequences.jsonl`: one JSON line per (component, activation example),
  with the input token window, firing mask, per-position activations, and decoded text.

Usage:
    python -m spd.scripts.validation.nontarget_activity <model_path> <alive_components_tsv> \
        [--harvest-subrun-id=ID] [--output-summary=PATH] [--output-sequences=PATH]
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import fire
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentData, ComponentSummary
from spd.harvest.storage import CorrelationStorage
from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.scripts.validation.common import build_module_lookup
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path


def _resolve_run_id(model_path: ModelPath, run_info: SPDRunInfo) -> str:
    """Extract the wandb run_id from the path, the dir name, or a '<project>-<run_id>' dir name.

    Locally-trained runs live in `SPD_OUT_DIR/spd/<run_id>/` (e.g. `s-429ea112`) — the dir name
    is the run_id. Cached wandb downloads live in `SPD_OUT_DIR/runs/<project>-<run_id>/` (e.g.
    `spd-s-429ea112`) — strip the project prefix.
    """
    try:
        _, _, run_id = parse_wandb_run_path(str(model_path))
        return run_id
    except ValueError:
        pass

    dir_name = run_info.checkpoint_path.parent.name
    try:
        _, _, run_id = parse_wandb_run_path(dir_name)
        return run_id
    except ValueError:
        pass

    project, sep, run_id = dir_name.partition("-")
    assert sep == "-" and project and run_id, (
        f"Cannot derive run_id from directory name {dir_name!r}; expected a bare run_id "
        f"(e.g. 's-abc12345') or '<project>-<run_id>'"
    )
    return run_id


def _open_harvest(run_id: str, subrun_id: str | None) -> HarvestRepo:
    if subrun_id is not None:
        return HarvestRepo(decomposition_id=run_id, subrun_id=subrun_id, readonly=True)
    repo = HarvestRepo.open_most_recent(decomposition_id=run_id, readonly=True)
    assert repo is not None, f"No harvest data found for run_id={run_id}"
    return repo


@dataclass
class AliveRow:
    layer: int
    matrix: str
    component: int
    module_name: str  # reconstructed module path used to build the harvest key


def _load_alive_components(path: Path, module_lookup: dict[tuple[int, str], str]) -> list[AliveRow]:
    rows: list[AliveRow] = []
    with path.open() as f:
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
            rows.append(
                AliveRow(
                    layer=layer,
                    matrix=matrix,
                    component=component,
                    module_name=module_lookup[key],
                )
            )
    return rows


def _write_summary(
    rows: list[AliveRow],
    summaries: dict[str, ComponentSummary],
    correlations: CorrelationStorage | None,
    out_path: Path,
) -> int:
    fieldnames = [
        "layer",
        "matrix",
        "component",
        "firing_density",
        "n_firings",
        "n_tokens_seen",
        "mean_ci",
        "mean_component_activation",
    ]
    n_written = 0
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            key = f"{row.module_name}:{row.component}"
            summary = summaries.get(key)
            if summary is None:
                logger.warning(f"No harvest entry for {key}; skipping")
                continue
            n_firings: int | str = ""
            n_tokens_seen: int | str = ""
            if correlations is not None and key in correlations.key_to_idx:
                idx = correlations.key_to_idx[key]
                n_firings = int(correlations.count_i[idx].item())
                n_tokens_seen = correlations.count_total

            writer.writerow(
                {
                    "layer": row.layer,
                    "matrix": row.matrix,
                    "component": row.component,
                    "firing_density": summary.firing_density,
                    "n_firings": n_firings,
                    "n_tokens_seen": n_tokens_seen,
                    "mean_ci": summary.mean_activations.get("causal_importance", ""),
                    "mean_component_activation": summary.mean_activations.get(
                        "component_activation", ""
                    ),
                }
            )
            n_written += 1
    return n_written


def _write_sequences(
    rows: list[AliveRow],
    components: dict[str, ComponentData],
    tokenizer: PreTrainedTokenizer,
    out_path: Path,
) -> int:
    n_written = 0
    with out_path.open("w") as f:
        for row in rows:
            key = f"{row.module_name}:{row.component}"
            comp = components.get(key)
            if comp is None:
                continue
            for example_idx, example in enumerate(comp.activation_examples):
                record = {
                    "layer": row.layer,
                    "matrix": row.matrix,
                    "component": row.component,
                    "example_idx": example_idx,
                    "token_ids": example.token_ids,
                    "firings": example.firings,
                    "activations": example.activations,
                    "text": tokenizer.decode(example.token_ids),
                }
                f.write(json.dumps(record) + "\n")
                n_written += 1
    return n_written


def nontarget_activity(
    model_path: ModelPath,
    alive_components_path: str,
    harvest_subrun_id: str | None = None,
    output_summary: str | None = None,
    output_sequences: str | None = None,
) -> tuple[Path, Path]:
    """Join alive components with harvest data and write summary + sequences files."""
    run_info = SPDRunInfo.from_path(model_path)
    run_dir = run_info.checkpoint_path.parent
    run_id = _resolve_run_id(model_path, run_info)

    assert run_info.config.tokenizer_name is not None, (
        "config.tokenizer_name is required to decode harvested token windows"
    )
    tokenizer = AutoTokenizer.from_pretrained(run_info.config.tokenizer_name)

    harvest = _open_harvest(run_id, harvest_subrun_id)

    summaries = harvest.get_summary()
    assert summaries, "Harvest data contains no components"
    module_lookup = build_module_lookup(sorted({s.layer for s in summaries.values()}))

    alive_rows = _load_alive_components(Path(alive_components_path).expanduser(), module_lookup)
    correlations = harvest.get_correlations()

    summary_path = (
        Path(output_summary).expanduser() if output_summary else run_dir / "nontarget_activity.tsv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_summary(alive_rows, summaries, correlations, summary_path)

    keys = [f"{row.module_name}:{row.component}" for row in alive_rows]
    components = harvest.get_components_bulk(keys)

    sequences_path = (
        Path(output_sequences).expanduser()
        if output_sequences
        else run_dir / "nontarget_activity_sequences.jsonl"
    )
    sequences_path.parent.mkdir(parents=True, exist_ok=True)
    _write_sequences(alive_rows, components, tokenizer, sequences_path)

    return summary_path, sequences_path


if __name__ == "__main__":
    fire.Fire(nontarget_activity)
