"""Targeted CI heatmap metric for visualizing causal importances on target vs nontarget data."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor
from transformers import AutoTokenizer

from spd.configs import Config, LMTaskConfig, ResidMLPTaskConfig, TMSTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.plotting import plot_targeted_ci_heatmaps
from spd.utils.general_utils import extract_batch_data


class TargetedCIHeatmap(Metric):
    """Visualize causal importances comparing target vs nontarget data.

    Generates controlled target inputs for visualization and fetches nontarget
    data from the nontarget_eval_iterator.

    For LM: Target inputs are the prompts from the prompts_file.
    For TMS/ResidMLP: Target inputs have one row per active_index (single feature active).
    """

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "figures"

    def __init__(
        self,
        model: ComponentModel,
        run_config: Config,
        device: str,
        n_nontarget_examples: int,
        nontarget_eval_iterator: Iterator[
            Int[Tensor, "..."] | tuple[Float[Tensor, "..."], Float[Tensor, "..."]]
        ],
    ) -> None:
        self.model = model
        self.run_config = run_config
        self.device = device
        self.n_nontarget_examples = n_nontarget_examples
        self.nontarget_eval_iterator = nontarget_eval_iterator

    @override
    def update(self, *, ci: CIOutputs, **_: Any) -> None:
        pass

    @override
    def compute(self) -> dict[str, Image.Image]:
        target_cis, target_labels = self._compute_target_cis()
        nontarget_cis, nontarget_labels = self._compute_nontarget_cis()

        img = plot_targeted_ci_heatmaps(
            target_cis=target_cis,
            nontarget_cis=nontarget_cis,
            n_nontarget_examples=self.n_nontarget_examples,
            target_labels=target_labels,
            nontarget_labels=nontarget_labels,
        )
        return {"targeted_ci_heatmap": img}

    def _compute_cis_from_batch(self, batch: Tensor) -> dict[str, Float[Tensor, "... C"]]:
        batch = batch.to(self.device)

        with torch.no_grad():
            pre_weight_acts = self.model(batch, cache_type="input").cache
            ci = self.model.calc_causal_importances(
                pre_weight_acts=pre_weight_acts,
                detach_inputs=False,
                sampling=self.run_config.sampling,
            )

        return {name: vals.detach().cpu() for name, vals in ci.lower_leaky.items()}

    def _compute_target_cis(self) -> tuple[dict[str, Float[Tensor, "... C"]], list[str]]:
        task_config = self.run_config.task_config

        match task_config:
            case LMTaskConfig():
                batch, labels = self._generate_lm_target_batch(task_config)
            case TMSTaskConfig() | ResidMLPTaskConfig():
                batch, labels = self._generate_toy_model_target_batch(task_config)
            case _:
                raise ValueError(f"Unsupported task config type: {type(task_config)}")

        cis = self._compute_cis_from_batch(batch)
        return cis, labels

    def _compute_nontarget_cis(self) -> tuple[dict[str, Float[Tensor, "... C"]], list[str]]:
        collected_batches: list[Tensor] = []
        n_collected = 0

        while n_collected < self.n_nontarget_examples:
            batch_raw = next(self.nontarget_eval_iterator)
            batch = extract_batch_data(batch_raw).to(self.device)
            collected_batches.append(batch)
            n_collected += batch.shape[0]

        batch = torch.cat(collected_batches, dim=0)[: self.n_nontarget_examples]
        cis = self._compute_cis_from_batch(batch)

        task_config = self.run_config.task_config
        if isinstance(task_config, LMTaskConfig):
            labels = self._tokens_to_labels(batch)
        else:
            labels = self._tensor_to_labels(batch)

        return cis, labels

    # --- Target data generation ---

    def _generate_lm_target_batch(self, task_config: LMTaskConfig) -> tuple[Tensor, list[str]]:
        if task_config.prompts_file is not None:
            tokens = self._load_target_from_prompts_file(task_config)
        else:
            assert task_config.dataset_name is not None, (
                "LM targeted mode requires prompts_file or dataset_name in task_config"
            )
            tokens = self._load_target_from_dataset(task_config)

        labels = self._tokens_to_labels(tokens)
        return tokens, labels

    def _load_target_from_prompts_file(self, task_config: LMTaskConfig) -> Tensor:
        assert task_config.prompts_file is not None
        prompts_path = Path(task_config.prompts_file)
        assert prompts_path.exists(), f"Prompts file not found: {prompts_path}"

        tokenizer = self._get_tokenizer()
        prompts = prompts_path.read_text().strip().split("\n")
        prompts = [p.strip() for p in prompts if p.strip()]

        encoded = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=task_config.max_seq_len,
            return_tensors="pt",
        )
        return encoded["input_ids"]

    def _load_target_from_dataset(self, task_config: LMTaskConfig) -> Tensor:
        """Load a batch of target examples from a HuggingFace dataset."""
        assert task_config.dataset_name is not None
        dataset_config = DatasetConfig(
            name=task_config.dataset_name,
            hf_tokenizer_path=self.run_config.tokenizer_name,
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
            batch_size=self.n_nontarget_examples,
            buffer_size=task_config.buffer_size,
        )
        batch = next(iter(loader))
        return extract_batch_data(batch)

    def _generate_toy_model_target_batch(
        self,
        task_config: TMSTaskConfig | ResidMLPTaskConfig,
    ) -> tuple[Float[Tensor, "batch n_features"], list[str]]:
        n_features = self._get_n_features()
        active_indices = task_config.active_indices
        assert active_indices is not None, (
            "Targeted mode requires active_indices to be set in task_config"
        )

        batch = torch.zeros(len(active_indices), n_features)
        labels = []
        for i, idx in enumerate(active_indices):
            batch[i, idx] = 1.0
            labels.append(f"feat {idx}")

        return batch, labels

    def _get_n_features(self) -> int:
        target_model = self.model.target_model
        model_config = getattr(target_model, "config", None)
        assert model_config is not None, "Target model must have a config attribute"
        n_features = getattr(model_config, "n_features", None)
        assert isinstance(n_features, int), "Target model must have config.n_features as int"
        return n_features

    # --- Label generation ---

    def _get_tokenizer(self) -> Any:
        assert self.run_config.tokenizer_name is not None
        tokenizer = AutoTokenizer.from_pretrained(self.run_config.tokenizer_name)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _tokens_to_labels(self, tokens: Tensor) -> list[str]:
        tokenizer = self._get_tokenizer()
        labels = []
        for batch_idx in range(tokens.shape[0]):
            for pos_idx in range(tokens.shape[1]):
                token_id = int(tokens[batch_idx, pos_idx].item())
                token_str = tokenizer.decode([token_id])
                token_str = token_str.replace("\n", "\\n").replace("\t", "\\t")
                if len(token_str) > 10:
                    token_str = token_str[:8] + ".."
                labels.append(f"{batch_idx}:{token_str}")
        return labels

    def _tensor_to_labels(self, batch: Tensor) -> list[str]:
        labels = []
        for row_idx in range(batch.shape[0]):
            row = batch[row_idx]
            active = (row != 0).nonzero(as_tuple=True)[0].tolist()
            if len(active) == 0:
                label = "none"
            elif len(active) <= 3:
                label = ",".join(str(i) for i in active)
            else:
                label = f"{active[0]}..{active[-1]}"
            labels.append(f"{row_idx}:{label}")
        return labels
