"""Shared helpers for the validation scripts."""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from spd.configs import CompletenessTaskConfig, Config, LMTaskConfig, ResidMLPTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.completeness.models import CompletenessTargetRunInfo, CopyTaskDataset
from spd.experiments.lm.prompts_dataset import (
    StaticBatchLoader,
    create_prompts_data_loader,
    read_prompts_file,
)
from spd.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.spd_types import ModelPath
from spd.utils.data_utils import DatasetGeneratedDataLoader

TaskConfig = LMTaskConfig | CompletenessTaskConfig | ResidMLPTaskConfig
DataLoaderT = DataLoader[Any] | StaticBatchLoader | DatasetGeneratedDataLoader[Any]


def load_spd_run(path: ModelPath) -> tuple[ComponentModel, Config, Path]:
    """Load the ComponentModel, its config, and the directory the checkpoint lives in."""
    run_info = SPDRunInfo.from_path(path)
    spd_model = ComponentModel.from_run_info(run_info)
    spd_model.eval()
    return spd_model, run_info.config, run_info.checkpoint_path.parent


def resolve_task_config(
    config: Config,
    use_nontarget: bool,
    prompts_override: str | None = None,
    split_override: str | None = None,
) -> TaskConfig:
    """Return the target or nontarget task config, erroring if nontarget is missing.

    Supports `LMTaskConfig`, `CompletenessTaskConfig`, and `ResidMLPTaskConfig`. `prompts_override`
    and `split_override` are LM-specific and error out for the non-LM tasks.
    """
    if use_nontarget:
        assert config.nontarget_task_config is not None, (
            "--nontarget was passed but the config has no nontarget_task_config"
        )
        task_config = config.nontarget_task_config
    else:
        task_config = config.task_config

    assert isinstance(task_config, (LMTaskConfig, CompletenessTaskConfig, ResidMLPTaskConfig)), (
        f"Validation scripts only support LM, completeness, or resid_mlp tasks, "
        f"got {type(task_config).__name__}"
    )

    if isinstance(task_config, (CompletenessTaskConfig, ResidMLPTaskConfig)):
        kind = "completeness" if isinstance(task_config, CompletenessTaskConfig) else "resid_mlp"
        assert prompts_override is None, f"--prompts is LM-only; not supported for {kind}"
        assert split_override is None, f"--split is LM-only; not supported for {kind}"
        return task_config

    assert prompts_override is None or split_override is None, (
        "--prompts and --split are mutually exclusive"
    )

    if prompts_override is not None:
        task_config = task_config.model_copy(
            update={"prompts_file": prompts_override, "dataset_name": None}
        )

    if split_override is not None:
        task_config = task_config.model_copy(
            update={"eval_data_split": split_override, "prompts_file": None}
        )
    return task_config


def is_prompt_task(task_config: TaskConfig) -> bool:
    """Prompt-based LM tasks run exactly one batch containing every prompt. Completeness and
    resid_mlp are not prompt-based."""
    if isinstance(task_config, (CompletenessTaskConfig, ResidMLPTaskConfig)):
        return False
    return task_config.prompts_file is not None


def build_lm_loader(
    task_config: TaskConfig, config: Config, batch_size_override: int | None = None
) -> DataLoaderT:
    """Build a DataLoader for the given task.

    For LM dataset-based tasks, `batch_size_override` replaces `config.batch_size` if set. For
    prompts-based LM tasks the override is ignored — the batch always contains every prompt. For
    completeness tasks, a `CopyTaskDataset` is constructed from the target model's train config
    (read from `config.pretrained_model_path`) and wrapped in a `DatasetGeneratedDataLoader`. For
    resid_mlp tasks, a `ResidMLPDataset` is built from the target model's `n_features` and the
    train run's `synced_inputs`.
    """
    if isinstance(task_config, CompletenessTaskConfig):
        assert config.pretrained_model_path is not None, (
            "completeness task needs config.pretrained_model_path to read vocab_size / eq_token"
        )
        target_run_info = CompletenessTargetRunInfo.from_path(config.pretrained_model_path)
        model_config = target_run_info.config.completeness_model_config
        dataset = CopyTaskDataset(
            vocab_size=model_config.vocab_size,
            eq_token=model_config.eq_token,
            device="cpu",  # iterate_input_ids moves to target device
        )
        batch_size = batch_size_override if batch_size_override is not None else config.batch_size
        return DatasetGeneratedDataLoader(dataset, batch_size=batch_size)

    if isinstance(task_config, ResidMLPTaskConfig):
        assert config.pretrained_model_path is not None, (
            "resid_mlp task needs config.pretrained_model_path to read n_features / synced_inputs"
        )
        target_run_info = ResidMLPTargetRunInfo.from_path(config.pretrained_model_path)
        target_model = ResidMLP.from_run_info(target_run_info)
        dataset = ResidMLPDataset(
            n_features=target_model.config.n_features,
            feature_probability=task_config.feature_probability,
            device="cpu",  # iterate_input_ids moves to target device
            calc_labels=False,
            label_type=None,
            act_fn_name=None,
            label_fn_seed=None,
            label_coeffs=None,
            data_generation_type=task_config.data_generation_type,
            synced_inputs=target_run_info.config.synced_inputs,
            active_indices=task_config.active_indices,
        )
        batch_size = batch_size_override if batch_size_override is not None else config.batch_size
        return DatasetGeneratedDataLoader(dataset, batch_size=batch_size)

    assert config.tokenizer_name is not None, "LM tasks need config.tokenizer_name"

    if task_config.prompts_file is not None:
        prompts = read_prompts_file(Path(task_config.prompts_file).expanduser())
        loader, _ = create_prompts_data_loader(
            prompts=prompts,
            tokenizer_name=config.tokenizer_name,
            max_seq_len=task_config.max_seq_len,
            loss_positions=task_config.loss_positions,
            batch_size=len(prompts),
        )
        return loader

    assert task_config.dataset_name is not None
    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
        seed=task_config.dataset_seed,
    )
    loader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=batch_size_override if batch_size_override is not None else config.batch_size,
        buffer_size=task_config.buffer_size,
    )
    return loader


def iterate_input_ids(loader: DataLoaderT, device: torch.device) -> Iterator[Tensor]:
    """Yield input token tensors from a single epoch — no cycling.

    Handles HF-style dict batches (`batch["input_ids"]`) and tuple batches from
    `DatasetGeneratedDataLoader` where the first element is the input tokens.
    """
    for batch in loader:
        if isinstance(batch, dict):
            assert "input_ids" in batch, f"Expected 'input_ids' key, got {list(batch.keys())}"
            yield batch["input_ids"].to(device)
        elif isinstance(batch, tuple):
            assert isinstance(batch[0], Tensor), (
                f"Expected Tensor first element, got {type(batch[0])}"
            )
            yield batch[0].to(device)
        else:
            raise AssertionError(f"Unsupported batch type: {type(batch)}")


_LAYER_IN_NAME = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")


def parse_module_name(module_name: str) -> tuple[int, str]:
    """Extract (layer, matrix) from a module name. Falls back to (-1, module_name)."""
    match = _LAYER_IN_NAME.search(module_name)
    if match is None:
        return -1, module_name
    layer = int(match.group(1))
    matrix = module_name[match.end() :]
    return layer, matrix


def build_module_lookup(module_paths: list[str]) -> dict[tuple[int, str], str]:
    """Map (layer, matrix) back to the full module path."""
    lookup: dict[tuple[int, str], str] = {}
    for path in module_paths:
        key = parse_module_name(path)
        assert key not in lookup, (
            f"Duplicate (layer, matrix) key {key} from modules {lookup.get(key)} and {path}"
        )
        lookup[key] = path
    return lookup


@dataclass
class TaskSpec:
    """Resolved task: prompt index in the prompts file, its tokenisation, and the target token id."""

    name: str
    prompt_idx: int
    prompt_token_ids: list[int]
    last_pos: int
    target_token_id: int


def load_prompts(config: Config, prompts_override: str | None = None) -> list[str]:
    """Read the prompts file referenced by the config (or an override) into a list of strings."""
    assert isinstance(config.task_config, LMTaskConfig) and config.task_config.prompts_file, (
        "Expected a prompts-based LM task_config"
    )
    prompts_path = Path(prompts_override or config.task_config.prompts_file).expanduser()
    return [ln.strip() for ln in prompts_path.read_text().splitlines() if ln.strip()]


def resolve_task(
    name: str, raw: Any, prompts: list[str], tokenizer: PreTrainedTokenizer
) -> tuple[TaskSpec, str, str]:
    """Parse a `--task-*` CLI arg (JSON dict or dict literal) into a TaskSpec.

    Returns `(spec, prompt_text, target_text)` where the text strings are useful for logging.
    Asserts the prompt exists exactly once in the prompts file and the target tokenises to
    exactly one token.
    """
    data = json.loads(raw) if isinstance(raw, str) else raw
    assert isinstance(data, dict) and "prompt" in data and "target" in data, (
        f"--task-{name} must be a JSON dict with 'prompt' and 'target' keys, got {raw!r}"
    )
    prompt_text, target_text = str(data["prompt"]), str(data["target"])

    matches = [i for i, p in enumerate(prompts) if p == prompt_text]
    assert len(matches) == 1, (
        f"Task {name}: expected exactly one prompt matching {prompt_text!r} in the prompts file, "
        f"found {len(matches)}"
    )

    prompt_encoded: Any = tokenizer(prompt_text)
    prompt_ids: list[int] = prompt_encoded["input_ids"]

    target_encoded: Any = tokenizer(target_text, add_special_tokens=False)
    target_ids: list[int] = target_encoded["input_ids"]
    assert len(target_ids) == 1, (
        f"Task {name}: target {target_text!r} must tokenize to exactly one token, got {target_ids}"
    )

    spec = TaskSpec(
        name=name,
        prompt_idx=matches[0],
        prompt_token_ids=prompt_ids,
        last_pos=len(prompt_ids) - 1,
        target_token_id=target_ids[0],
    )
    return spec, prompt_text, target_text


def contains_subsequence(haystack: list[int], needle: list[int]) -> bool:
    """Return True if `needle` appears as a contiguous sub-sequence in `haystack`."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def escape_tsv_value(s: str) -> str:
    """Backslash-escape characters that break naive TSV splitting.

    Python's `csv` module already quotes tabs/newlines/quotes with `QUOTE_MINIMAL`, but tools like
    `cut`, `awk`, or a human eyeballing the file treat the quoted form as broken. Replacing the
    offending chars with `\\t`, `\\n`, `\\r` (and escaping backslashes so the transformation is
    reversible) keeps the file splittable by tab everywhere.
    """
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
