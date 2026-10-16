"""StochasticHiddenActsReconLoss with a position mask and `relative` (configs.py): only the
selected positions count, and with `relative` each position's squared error is divided by the
squared norm of that position's original module output."""

import torch
from transformers import AutoModelForCausalLM

from spd.configs import GlobalCiConfig
from spd.metrics.hidden_acts_recon_loss import calc_hidden_acts_mse
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos, route_only_selected_positions
from spd.utils.module_utils import ModulePathInfo

TINY_QWEN = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
C = 4


def _model() -> ComponentModel:
    target = AutoModelForCausalLM.from_pretrained(TINY_QWEN, dtype=torch.float32).eval()
    target.requires_grad_(False)
    torch.manual_seed(0)
    return ComponentModel(
        target_model=target,
        module_path_info=[
            ModulePathInfo(module_path=f"model.layers.{i}.mlp.down_proj", C=C) for i in (0, 1)
        ],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        sigmoid_type="leaky_hard",
        pretrained_model_output_attr="logits",
    )


def test_relative_error_at_selected_positions_only() -> None:
    model = _model()
    ids = torch.randint(1, 1000, (2, 6), generator=torch.Generator().manual_seed(0))
    position_mask = torch.zeros(2, 6, dtype=torch.bool)
    position_mask[0, [2, 4]] = True
    position_mask[1, 5] = True
    with torch.no_grad():
        target = model(ids, cache_type="output").cache
    masks = {
        name: torch.rand(2, 6, C, generator=torch.Generator().manual_seed(1))
        for name in model.target_module_paths
    }
    with torch.no_grad():
        per_module, _ = calc_hidden_acts_mse(
            model, ids, make_mask_infos(masks), target, position_mask, relative=True
        )
        # by hand: masked forward with components substituted at the selected positions only
        pred = model(
            ids,
            mask_infos=route_only_selected_positions(make_mask_infos(masks), position_mask),
            cache_type="output",
        ).cache
    for name in model.target_module_paths:
        y, p = target[name][position_mask], pred[name][position_mask]
        expected = ((p - y).pow(2).sum(-1) / y.pow(2).sum(-1)).sum()
        total, n = per_module[name]
        assert n == 3, n
        torch.testing.assert_close(total, expected)
        assert total / n > 1e-3, "masks too close to the identity to test anything"
    # Unselected positions run on the original weights: the first decomposed module's output
    # there is unchanged (later modules see earlier positions' changes through attention).
    first = "model.layers.0.mlp.down_proj"
    torch.testing.assert_close(pred[first][~position_mask], target[first][~position_mask])


def test_all_off_gives_relative_error_one() -> None:
    """Components off and no delta: the module outputs zero at the selected positions, which is
    relative error 1 by definition."""
    model = _model()
    ids = torch.randint(1, 1000, (2, 6), generator=torch.Generator().manual_seed(0))
    position_mask = torch.zeros(2, 6, dtype=torch.bool)
    position_mask[:, 3:] = True
    with torch.no_grad():
        target = model(ids, cache_type="output").cache
        zeros = {name: torch.zeros(2, 6, C) for name in model.target_module_paths}
        per_module, _ = calc_hidden_acts_mse(
            model, ids, make_mask_infos(zeros), target, position_mask, relative=True
        )
    for total, n in per_module.values():
        assert n == 6
        torch.testing.assert_close(total / n, torch.tensor(1.0))


def test_stochastic_loss_uses_the_position_mask() -> None:
    """The training loss equals the mean relative error at the selected positions under the same
    stochastic masks (same seed); a loss that ignored the mask would average over all positions."""
    from spd.metrics.hidden_acts_recon_loss import stochastic_hidden_acts_recon_loss
    from spd.routing import AllLayersRouter
    from spd.utils.component_utils import calc_stochastic_component_mask_info

    model = _model()
    ids = torch.randint(1, 1000, (2, 6), generator=torch.Generator().manual_seed(0))
    position_mask = torch.zeros(2, 6, dtype=torch.bool)
    position_mask[:, 4:] = True
    ci = {
        name: torch.rand(2, 6, C, generator=torch.Generator().manual_seed(2))
        for name in model.target_module_paths
    }
    deltas = model.calc_weight_deltas()
    with torch.no_grad():
        torch.manual_seed(3)
        loss = stochastic_hidden_acts_recon_loss(
            model, "continuous", 1, ids, ci, deltas, position_mask, relative=True
        )
        target = model(ids, cache_type="output").cache
        torch.manual_seed(3)
        mask_infos = calc_stochastic_component_mask_info(
            ci, "continuous", deltas, AllLayersRouter()
        )
        per_module, _ = calc_hidden_acts_mse(
            model, ids, mask_infos, target, position_mask, relative=True
        )
        all_positions, _ = calc_hidden_acts_mse(model, ids, mask_infos, target, None, relative=True)
    expected = sum(t for t, _ in per_module.values()) / sum(n for _, n in per_module.values())
    torch.testing.assert_close(loss, expected)
    ignored = sum(t for t, _ in all_positions.values()) / sum(n for _, n in all_positions.values())
    assert abs(float(ignored) - float(expected)) > 1e-3 * float(expected), (
        "test cannot tell the two apart"
    )
