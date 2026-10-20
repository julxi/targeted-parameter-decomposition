"""Orthogonality penalty between a sum A of the components' writes and the rest y - A of each
decomposed module's output (ComponentOrthogonalityLossConfig). Topic:
convos/julian/26-10-16_faithful_targeting_LOG.md.

Per decomposed module and selected position, with x the module's input on the original model:
  y = W x                         (original output)
  A = sum_c m_c U_c (V_c x)       (delta off), with m_c = CI_c (state "ci_weighted", Exp 4d)
                                  or m_c = 1 (state "all_components", Exp 4e: then y - A = delta x)
  penalty = (<A, y - A> / |y|^2)^2
<A, y - A> = 0 means A lies on the sphere with diameter y: A is the orthogonal projection of y onto
A's own direction, so |A| <= |y|, |y - A| <= |y|, and the part of A orthogonal to y is at most
|y| / 2. A = 0 and A = y both satisfy it; only the target loss pushes components on.

All components together (one condition per position and module), intentionally not one condition
per component: Julian's choice, so that the components may interact (e.g. partly cancel each other)
as long as their sum is a projection of y. With "all_components" the CIs get no gradient from this
term; it acts on U and V only.

A and y come from the cached inputs of the original forward, one module at a time, not from a
masked forward through all modules: this is the same A as in the cancellation test of the Exp 4
analysis, and it costs no extra forward pass. Euclidean inner product in the module's output basis;
the per-position division by |y|^2 makes positions with very large outputs (massive activations)
count like any other.
"""

from typing import Any, ClassVar, Literal, override

import einops
import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import LinearComponents
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import PositionMask, select_positions

OrthogonalityState = Literal["ci_weighted", "all_components"]


def _per_position_terms(
    model: ComponentModel,
    module_name: str,
    x: Float[Tensor, "... d_in"],
    ci: Float[Tensor, "... C"],
    state: OrthogonalityState,
) -> tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
    """(<A, y - A> / |y|^2, |A| / |y|) per position, computed in fp32."""
    components = model.components[module_name]
    assert isinstance(components, LinearComponents), type(components)
    assert components.bias is None, f"{module_name}: biased modules not supported"
    # fp32 and no autocast: <A, y> - |A|^2 is a difference of two similar numbers near the sphere
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        y = einops.einsum(
            x, model.target_weight(module_name).float(), "... d_in, d_out d_in -> ... d_out"
        )
        acts = einops.einsum(x, components.V.float(), "... d_in, d_in C -> ... C")
        match state:
            case "ci_weighted":
                acts = acts * ci.float()
            case "all_components":
                pass
        a = einops.einsum(acts, components.U.float(), "... C, C d_out -> ... d_out")
        y_sq = y.pow(2).sum(-1)
        assert (y_sq > 0).all(), f"{module_name}: zero original output at a selected position"
        inner = ((a * (y - a)).sum(-1)) / y_sq
        norm_ratio = a.norm(dim=-1) / y_sq.sqrt()
    return inner, norm_ratio


def component_orthogonality_loss(
    model: ComponentModel,
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"]],
    ci: dict[str, Float[Tensor, "... C"]],
    position_mask: PositionMask | None,
    state: OrthogonalityState,
) -> Float[Tensor, ""]:
    """Mean over selected positions and decomposed modules of (<A, y - A> / |y|^2)^2."""
    assert ci, "Empty ci"
    total = torch.tensor(0.0, device=next(iter(ci.values())).device)
    n = 0
    for name, layer_ci in ci.items():
        inner, _ = _per_position_terms(
            model,
            name,
            select_positions(pre_weight_acts[name], position_mask),
            select_positions(layer_ci, position_mask),
            state,
        )
        total = total + inner.pow(2).sum()
        n += inner.numel()
    assert n > 0, "no selected positions"
    return total / n


class ComponentOrthogonalityLoss(Metric):
    """Eval version of component_orthogonality_loss. Besides the penalty it reports, per module,
    the mean signed <A, y - A> / |y|^2 (negative: A reaches outside the sphere, e.g. too large;
    positive: inside) and the mean |A| / |y|, so a small penalty from A = 0 is told apart from one
    with A on the sphere."""

    slow: ClassVar[bool] = False
    metric_section: ClassVar[str] = "loss"

    def __init__(self, model: ComponentModel, device: str, state: OrthogonalityState) -> None:
        self.model = model
        self.device = device
        self.state: OrthogonalityState = state
        self.sums: dict[str, Tensor] = {}

    @override
    def update(
        self,
        *,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"]],
        ci: CIOutputs,
        position_mask: PositionMask | None,
        **_: Any,
    ) -> None:
        for name, layer_ci in ci.lower_leaky.items():
            inner, norm_ratio = _per_position_terms(
                self.model,
                name,
                select_positions(pre_weight_acts[name], position_mask),
                select_positions(layer_ci, position_mask),
                self.state,
            )
            for key, val in (
                (f"penalty/{name}", inner.pow(2).sum()),
                (f"inner/{name}", inner.sum()),
                (f"norm_ratio/{name}", norm_ratio.sum()),
                (f"n/{name}", torch.tensor(float(inner.numel()), device=self.device)),
            ):
                self.sums[key] = self.sums.get(key, torch.tensor(0.0, device=self.device)) + (
                    val.detach()
                )

    @override
    def compute(self) -> dict[str, Float[Tensor, ""]]:
        assert self.sums, "No data accumulated"
        keys = sorted(self.sums)
        reduced = dict(
            zip(
                keys,
                all_reduce(torch.stack([self.sums[k] for k in keys]), op=ReduceOp.SUM),
                strict=True,
            )
        )
        names = sorted(k.removeprefix("n/") for k in keys if k.startswith("n/"))
        class_name = type(self).__name__
        out: dict[str, Float[Tensor, ""]] = {}
        for name in names:
            n = reduced[f"n/{name}"]
            for stat in ("penalty", "inner", "norm_ratio"):
                out[f"{class_name}/{stat}/{name}"] = reduced[f"{stat}/{name}"] / n
        n_total = torch.stack([reduced[f"n/{name}"] for name in names]).sum()
        penalty_total = torch.stack([reduced[f"penalty/{name}"] for name in names]).sum()
        out[class_name] = penalty_total / n_total
        return out
