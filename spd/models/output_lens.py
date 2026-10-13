"""Output lenses: what the reconstruction losses on target data compare (configs.TargetLens).

Inside `ComponentModel.output_lens(lens)`, ComponentModel.forward returns the lens's view of the
forward pass instead of the model output, for the masked and the original forward alike. run_spd and
eval enter it for target batches only (`make_target_output_lens` builds the lens per batch);
nontarget batches always compare the model output. All losses then select the decomposed positions
(`position_mask`) from the lens output as they would from logits, so no loss needs to know about
lenses.
"""

import re
from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import NextTokenLens, ResidualLens, StimulusLens, TargetLens
from spd.utils.general_utils import STIMULUS_END_KEY


@dataclass(frozen=True)
class ResidualOutputLens:
    """The output of module `module_path` (a decoder layer: the residual stream after it), each
    channel divided by `channel_std`, in fp32. Shape (batch, pos, d_model)."""

    module_path: str
    channel_std: Float[Tensor, " d_model"]

    def apply(self, captured: Tensor) -> Float[Tensor, "batch pos d_model"]:
        assert captured.ndim == 3 and captured.shape[-1] == self.channel_std.shape[0], (
            captured.shape,
            self.channel_std.shape,
        )
        # fp32: the squared error of bf16 residuals (channel values up to ~1e3 in Qwen's massive
        # activation channel) would lose the small differences the loss is about.
        return captured.float() / self.channel_std


@dataclass(frozen=True)
class StimulusOutputLens:
    """The model output (logits) at each row's `stimulus_end`, repeated at every position of that
    row, so that selecting the decomposed positions reads the stimulus end for each of them.
    Shape (batch, pos, vocab)."""

    stimulus_end: Int[Tensor, " batch"]

    def apply(self, logits: Float[Tensor, "batch pos vocab"]) -> Float[Tensor, "batch pos vocab"]:
        batch, pos, _ = logits.shape
        assert self.stimulus_end.shape == (batch,), (self.stimulus_end.shape, batch)
        assert int(self.stimulus_end.max()) < pos, "stimulus end beyond the batch width"
        at_end = logits[torch.arange(batch, device=logits.device), self.stimulus_end]
        return at_end[:, None, :].expand(batch, pos, at_end.shape[-1])


OutputLens = ResidualOutputLens | StimulusOutputLens


def residual_lens_module_path(lens_cfg: ResidualLens) -> str:
    """Hugging Face decoder naming (Qwen2, Llama): the residual stream after decoder layer L is the
    first output of `model.layers.L`."""
    return f"model.layers.{lens_cfg.layer}"


def assert_lens_after_decomposed_layers(lens_cfg: ResidualLens, module_paths: list[str]) -> None:
    """A lens at or before a decomposed layer cannot see (all of) what the components write."""
    layers = [re.search(r"layers\.(\d+)\.", path) for path in module_paths]
    assert all(m is not None for m in layers), f"cannot parse a layer index from {module_paths}"
    max_layer = max(int(m.group(1)) for m in layers if m is not None)
    assert lens_cfg.layer > max_layer, (
        f"residual lens layer {lens_cfg.layer} must come after the last decomposed layer {max_layer}"
    )


def make_target_output_lens(
    lens_cfg: TargetLens,
    batch_raw: Any,
    residual_channel_std: Float[Tensor, " d_model"] | None,
    device: str,
) -> OutputLens | None:
    """The lens for one target batch; None for the next-token lens (the plain model output).
    `residual_channel_std` is required for, and only for, the residual lens."""
    match lens_cfg:
        case NextTokenLens():
            assert residual_channel_std is None
            return None
        case ResidualLens():
            assert residual_channel_std is not None
            return ResidualOutputLens(residual_lens_module_path(lens_cfg), residual_channel_std)
        case StimulusLens():
            assert residual_channel_std is None
            assert isinstance(batch_raw, dict) and STIMULUS_END_KEY in batch_raw, (
                "stimulus lens needs batches built with a stimulus"
            )
            return StimulusOutputLens(batch_raw[STIMULUS_END_KEY].to(device))
