from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor
from torch.distributed import ReduceOp

if TYPE_CHECKING:
    from spd.configs import Config

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel, OutputWithCache
from spd.plotting import plot_mean_component_cis_both_scales
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import (
    PositionMask,
    extract_batch_data,
    extract_position_mask,
    select_positions,
)


class CIMeanPerComponent(Metric):
    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "figures"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        run_config: "Config | None" = None,
        n_nontarget_batches: int = 1,
        nontarget_eval_iterator: Iterator[
            Int[Tensor, "..."] | tuple[Float[Tensor, "..."], Float[Tensor, "..."]]
        ]
        | None = None,
    ) -> None:
        self.model = model
        self.run_config = run_config
        self.n_nontarget_batches = n_nontarget_batches
        self.nontarget_eval_iterator = nontarget_eval_iterator
        self.components = model.components
        self.component_ci_sums: dict[str, Tensor] = {
            module_name: torch.zeros(model.module_to_c[module_name], device=device)
            for module_name in self.components
        }
        self.examples_seen: dict[str, Tensor] = {
            module_name: torch.tensor(0, device=device) for module_name in self.components
        }

    @override
    def update(self, *, ci: CIOutputs, position_mask: PositionMask | None, **_: Any) -> None:
        for module_name, ci_vals in ci.lower_leaky.items():
            ci_vals = select_positions(ci_vals, position_mask)
            n_leading_dims = ci_vals.ndim - 1
            n_examples = ci_vals.shape[:n_leading_dims].numel()

            self.examples_seen[module_name] += n_examples

            leading_dim_idxs = tuple(range(n_leading_dims))
            self.component_ci_sums[module_name] += ci_vals.sum(dim=leading_dim_idxs)

    def _compute_nontarget_cis(self) -> dict[str, Tensor]:
        assert self.nontarget_eval_iterator is not None
        assert self.run_config is not None

        device = next(iter(self.component_ci_sums.values())).device
        ci_sums: dict[str, Tensor] = {
            module_name: torch.zeros_like(self.component_ci_sums[module_name])
            for module_name in self.components
        }
        examples_seen: dict[str, Tensor] = {
            module_name: torch.tensor(0, device=device) for module_name in self.components
        }

        for _ in range(self.n_nontarget_batches):
            batch_raw = next(self.nontarget_eval_iterator)
            batch = extract_batch_data(batch_raw).to(device)
            position_mask = extract_position_mask(batch_raw)
            if position_mask is not None:
                position_mask = position_mask.to(device)

            with torch.no_grad():
                output: OutputWithCache = self.model(batch, cache_type="input")
                ci = self.model.calc_causal_importances(
                    pre_weight_acts=output.cache,
                    detach_inputs=False,
                    sampling=self.run_config.sampling,
                )

            for module_name, ci_vals in ci.lower_leaky.items():
                ci_vals = select_positions(ci_vals, position_mask)
                n_leading_dims = ci_vals.ndim - 1
                n_examples = ci_vals.shape[:n_leading_dims].numel()
                examples_seen[module_name] += n_examples
                leading_dim_idxs = tuple(range(n_leading_dims))
                ci_sums[module_name] += ci_vals.sum(dim=leading_dim_idxs)

        mean_cis = {}
        for module_name in self.components:
            s = all_reduce(ci_sums[module_name], op=ReduceOp.SUM)
            n = all_reduce(examples_seen[module_name], op=ReduceOp.SUM)
            mean_cis[module_name] = s / n
        return mean_cis

    @override
    def compute(self) -> dict[str, Image.Image]:
        mean_component_cis = {}
        for module_name in self.components:
            summed_ci = all_reduce(self.component_ci_sums[module_name], op=ReduceOp.SUM)
            examples_reduced = all_reduce(self.examples_seen[module_name], op=ReduceOp.SUM)
            mean_component_cis[module_name] = summed_ci / examples_reduced

        img_linear, img_log = plot_mean_component_cis_both_scales(mean_component_cis)

        out: dict[str, Image.Image] = {
            "ci_mean_per_component": img_linear,
            "ci_mean_per_component_log": img_log,
        }

        if self.nontarget_eval_iterator is not None:
            nontarget_cis = self._compute_nontarget_cis()
            nt_linear, nt_log = plot_mean_component_cis_both_scales(nontarget_cis)
            out["nontarget_ci_mean_per_component"] = nt_linear
            out["nontarget_ci_mean_per_component_log"] = nt_log

        return out
