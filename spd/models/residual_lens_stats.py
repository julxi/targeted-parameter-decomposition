"""Per-channel statistics for the residual lens (configs.ResidualLens, spd/models/output_lens.py).

Separate from output_lens.py because it runs a ComponentModel, which itself imports the lenses.
"""

from collections.abc import Iterable
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from spd.configs import ResidualLens
from spd.models.component_model import ComponentModel
from spd.models.output_lens import ResidualOutputLens, residual_lens_module_path
from spd.utils.general_utils import (
    bf16_autocast,
    extract_batch_data,
    extract_position_mask,
    select_positions,
)


def residual_channel_std(
    model: ComponentModel,
    batches: Iterable[Any],
    lens_cfg: ResidualLens,
    device: str,
    autocast_bf16: bool,
) -> Float[Tensor, " d_model"]:
    """Per-channel standard deviation of the residual stream after `lens_cfg.layer` on the original
    model, over the decomposed positions (`position_mask`) of `batches` (pass one epoch of the
    target training data)."""
    module_path = residual_lens_module_path(lens_cfg)
    # Qwen2/Llama decoder layers normalise their input with `input_layernorm` (one weight per
    # residual channel).
    d_model = model.target_model.get_parameter(f"{module_path}.input_layernorm.weight").shape[0]
    unit = ResidualOutputLens(module_path, torch.ones(d_model, device=device))
    total = torch.zeros(d_model, dtype=torch.float64, device=device)
    total_sq = torch.zeros(d_model, dtype=torch.float64, device=device)
    n = 0
    with torch.no_grad(), bf16_autocast(enabled=autocast_bf16), model.output_lens(unit):
        for batch_raw in batches:
            position_mask = extract_position_mask(batch_raw)
            assert position_mask is not None, "residual lens needs a position mask"
            out = model(extract_batch_data(batch_raw).to(device))
            resid = select_positions(out, position_mask.to(device)).double()
            total += resid.sum(0)
            total_sq += (resid**2).sum(0)
            n += resid.shape[0]
    assert n > 1, f"too few decomposed positions ({n}) for a standard deviation"
    var = (total_sq - total**2 / n) / (n - 1)
    std = var.clamp_min(0).sqrt().float()
    assert bool((std > 0).all()), f"{int((std == 0).sum())} residual channels never vary"
    return std
