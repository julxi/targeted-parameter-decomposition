"""Weight magnitude per component, sorted by mean CI, colored by max CI."""

from typing import Any, ClassVar, override

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import LinearComponents
from spd.plotting import plot_weight_magnitude
from spd.utils.general_utils import PositionMask, select_positions

C = None  # jaxtyping placeholder


class WeightMagnitude(Metric):
    """Weight magnitude per component, sorted by mean CI over eval inputs.

    For each layer:
    - X-axis: Components sorted by mean CI (descending)
    - Y-axis: Weight magnitude (||V|| * ||U||) per component
    - Color: Max CI over eval inputs per component

    CI values are accumulated from the shared eval loop batches via update(), only at the positions
    selected by the batch's position mask (padding and other unselected positions are excluded).
    Each batch is flattened to (positions, C) before accumulation, because prompt batches are
    trimmed to their longest sample and so differ in width.
    """

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "figures"

    def __init__(self, model: ComponentModel) -> None:
        self.model = model
        self.accumulated_cis: dict[str, list[Float[Tensor, "... C"]]] = {}

    @override
    def update(self, *, ci: CIOutputs, position_mask: PositionMask | None, **_: Any) -> None:
        for name, vals in ci.lower_leaky.items():
            selected = select_positions(vals.detach(), position_mask)
            self.accumulated_cis.setdefault(name, []).append(
                selected.reshape(-1, selected.shape[-1]).cpu()
            )

    @override
    def compute(self) -> dict[str, Image.Image]:
        assert self.accumulated_cis, "No CI data accumulated"

        weight_magnitudes: dict[str, Float[Tensor, C]] = {}
        for layer_name, component in self.model.components.items():
            if not isinstance(component, LinearComponents):
                continue
            v_norms = component.V.norm(dim=0)  # [C]
            u_norms = component.U.norm(dim=1)  # [C]
            weight_magnitudes[layer_name] = (v_norms * u_norms).detach().cpu()

        max_cis: dict[str, Float[Tensor, C]] = {}
        mean_cis: dict[str, Float[Tensor, C]] = {}
        for layer_name, ci_tensors in self.accumulated_cis.items():
            ci_flat = torch.cat(ci_tensors, dim=0)  # [N, C]
            max_cis[layer_name] = ci_flat.max(dim=0).values  # [C]
            mean_cis[layer_name] = ci_flat.mean(dim=0)  # [C]

        img = plot_weight_magnitude(
            weight_magnitudes=weight_magnitudes,
            max_cis=max_cis,
            mean_cis=mean_cis,
        )
        return {"weight_magnitude": img}
