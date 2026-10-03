"""Reconstruction loss eval metrics for targeted decomposition.

TargetReconLoss: evaluates on target eval data (delta=0 for component masks, delta=1 for delta-only).
NontargetReconLoss: evaluates on nontarget eval data (delta=1 for all mask types).
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

if TYPE_CHECKING:
    from spd.configs import Config

from spd.configs import SamplingType
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel, OutputWithCache
from spd.models.components import (
    WeightDeltaAndMask,
    make_mask_infos,
    route_only_selected_positions,
)
from spd.routing import AllLayersRouter
from spd.utils.component_utils import calc_ci_l_zero, calc_stochastic_component_mask_info
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import (
    PositionMask,
    calc_sum_recon_loss_lm_at_positions,
    extract_batch_data,
    extract_position_mask,
    select_positions,
)


def _compute_recon_losses(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
    output_loss_type: Literal["mse", "kl"],
    sampling: SamplingType,
    rounding_threshold: float,
    force_delta: float,
    position_mask: PositionMask | None,
) -> dict[str, tuple[Float[Tensor, ""], int]]:
    """Compute reconstruction losses under 4 masking strategies, over the positions selected by
    `position_mask` (all if None).

    Returns {key: (sum_loss, n_examples)} for: rounded, CImasked, stochastic, delta_only.
    """
    ci_sample = next(iter(ci.values()))
    leading_dims = ci_sample.shape[:-1]
    device = ci_sample.device

    def _make_weight_deltas_and_masks(
        delta_val: float,
    ) -> dict[str, WeightDeltaAndMask]:
        return {
            layer: (weight_deltas[layer], torch.full(leading_dims, delta_val, device=device))
            for layer in ci
        }

    def _forward_loss(
        mask_infos: dict[str, Any],
    ) -> tuple[Float[Tensor, ""], int]:
        out = model(batch, mask_infos=route_only_selected_positions(mask_infos, position_mask))
        return calc_sum_recon_loss_lm_at_positions(
            pred=out, target=target_out, loss_type=output_loss_type, position_mask=position_mask
        )

    results: dict[str, tuple[Float[Tensor, ""], int]] = {}

    # 1) rounded: CI thresholded to binary
    rounded_masks = {k: (v > rounding_threshold).float() for k, v in ci.items()}
    mask_infos = make_mask_infos(
        rounded_masks, weight_deltas_and_masks=_make_weight_deltas_and_masks(force_delta)
    )
    results["rounded_recon_loss"] = _forward_loss(mask_infos)

    # 2) CImasked: raw CI values as masks
    mask_infos = make_mask_infos(
        ci, weight_deltas_and_masks=_make_weight_deltas_and_masks(force_delta)
    )
    results["CImasked_recon_loss"] = _forward_loss(mask_infos)

    # 3) stochastic: stochastic masks sampled from CI
    stoch_mask_infos = calc_stochastic_component_mask_info(
        causal_importances=ci,
        component_mask_sampling=sampling,
        weight_deltas=weight_deltas,
        router=AllLayersRouter(),
        force_delta=force_delta,
    )
    results["stochastic_recon_loss"] = _forward_loss(stoch_mask_infos)

    # 4) delta_only: component masks=0, delta=1
    zero_masks = {k: torch.zeros_like(v) for k, v in ci.items()}
    mask_infos = make_mask_infos(
        zero_masks, weight_deltas_and_masks=_make_weight_deltas_and_masks(1.0)
    )
    results["delta_only_recon_loss"] = _forward_loss(mask_infos)

    return results


class TargetReconLoss(Metric):
    """Reconstruction loss on target eval data under 4 masking strategies.

    delta=0 for rounded/CImasked/stochastic, delta=1 for delta_only.
    """

    metric_section: ClassVar[str] = "target"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        output_loss_type: Literal["mse", "kl"],
        sampling: SamplingType,
        rounding_threshold: float,
    ) -> None:
        self.model = model
        self.device = device
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type
        self.sampling: SamplingType = sampling
        self.rounding_threshold = rounding_threshold
        self._keys = [
            "rounded_recon_loss",
            "CImasked_recon_loss",
            "stochastic_recon_loss",
            "delta_only_recon_loss",
        ]
        self.sum_losses = {k: torch.tensor(0.0, device=device) for k in self._keys}
        self.n_examples = {k: torch.tensor(0, device=device) for k in self._keys}

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        position_mask: PositionMask | None,
        **_: Any,
    ) -> None:
        with torch.no_grad():
            results = _compute_recon_losses(
                model=self.model,
                batch=batch,
                target_out=target_out,
                ci=ci.lower_leaky,
                weight_deltas=weight_deltas,
                output_loss_type=self.output_loss_type,
                sampling=self.sampling,
                rounding_threshold=self.rounding_threshold,
                force_delta=0.0,
                position_mask=position_mask,
            )
        for k in self._keys:
            sum_loss, n = results[k]
            self.sum_losses[k] += sum_loss
            self.n_examples[k] += n

    @override
    def compute(self) -> dict[str, Float[Tensor, ""]]:
        out: dict[str, Float[Tensor, ""]] = {}
        for k in self._keys:
            sum_loss = all_reduce(self.sum_losses[k], op=ReduceOp.SUM)
            n = all_reduce(self.n_examples[k], op=ReduceOp.SUM)
            out[k] = sum_loss / n
        return out


class NontargetReconLoss(Metric):
    """Reconstruction loss on nontarget eval data under 4 masking strategies.

    delta=1 for all mask types (nontarget forces full delta reconstruction).
    update() is a no-op; compute() fetches nontarget batches directly.
    """

    metric_section: ClassVar[str] = "nontarget"

    def __init__(
        self,
        model: ComponentModel,
        run_config: "Config",
        device: str,
        rounding_threshold: float,
        n_nontarget_batches: int,
        nontarget_eval_iterator: Iterator[
            Int[Tensor, "..."] | tuple[Float[Tensor, "..."], Float[Tensor, "..."]]
        ],
    ) -> None:
        self.model = model
        self.run_config = run_config
        self.device = device
        self.rounding_threshold = rounding_threshold
        self.n_nontarget_batches = n_nontarget_batches
        self.nontarget_eval_iterator = nontarget_eval_iterator

    @override
    def update(self, **_: Any) -> None:
        pass

    @override
    def compute(self) -> dict[str, Float[Tensor, ""]]:
        keys = [
            "rounded_recon_loss",
            "CImasked_recon_loss",
            "stochastic_recon_loss",
            "delta_only_recon_loss",
        ]
        sum_losses = {k: torch.tensor(0.0, device=self.device) for k in keys}
        n_examples = {k: torch.tensor(0, device=self.device) for k in keys}
        total_l0_sum = 0.0

        weight_deltas = self.model.calc_weight_deltas()

        for _ in range(self.n_nontarget_batches):
            batch_raw = next(self.nontarget_eval_iterator)
            batch = extract_batch_data(batch_raw).to(self.device)
            position_mask = extract_position_mask(batch_raw)
            if position_mask is not None:
                position_mask = position_mask.to(self.device)

            with torch.no_grad():
                target_output: OutputWithCache = self.model(batch, cache_type="input")
                ci = self.model.calc_causal_importances(
                    pre_weight_acts=target_output.cache,
                    detach_inputs=False,
                    sampling=self.run_config.sampling,
                )

                threshold = self.run_config.ci_alive_threshold
                batch_l0 = sum(
                    calc_ci_l_zero(select_positions(layer_ci, position_mask), threshold)
                    for layer_ci in ci.lower_leaky.values()
                )
                total_l0_sum += batch_l0

                results = _compute_recon_losses(
                    model=self.model,
                    batch=batch,
                    target_out=target_output.output,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas,
                    output_loss_type=self.run_config.output_loss_type,
                    sampling=self.run_config.sampling,
                    rounding_threshold=self.rounding_threshold,
                    force_delta=1.0,
                    position_mask=position_mask,
                )

            for k in keys:
                s, n = results[k]
                sum_losses[k] += s
                n_examples[k] += n

        out: dict[str, Float[Tensor, ""]] = {}
        for k in keys:
            s = all_reduce(sum_losses[k], op=ReduceOp.SUM)
            n = all_reduce(n_examples[k], op=ReduceOp.SUM)
            out[k] = s / n

        avg_l0 = all_reduce(torch.tensor(total_l0_sum, device=self.device), op=ReduceOp.SUM)
        avg_l0_count = all_reduce(
            torch.tensor(self.n_nontarget_batches, device=self.device), op=ReduceOp.SUM
        )
        out["total_l0"] = avg_l0 / avg_l0_count

        return out
