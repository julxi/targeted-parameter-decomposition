"""ComponentOrthogonalityLoss (configs.py): (<A, y - A> / |y|^2)^2 per decomposed module and
selected position, with y the original output and A the CI-weighted components' output (state
"ci_weighted") or the unweighted sum of all components (state "all_components", y - A = delta x),
both on the original input to the module."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from spd.configs import GlobalCiConfig
from spd.metrics.orthogonality_loss import (
    ComponentOrthogonalityLoss,
    OrthogonalityState,
    component_orthogonality_loss,
)
from spd.models.component_model import CIOutputs, ComponentModel
from spd.utils.module_utils import ModulePathInfo

TINY_QWEN = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
MODULES = [f"model.layers.{i}.mlp.down_proj" for i in (0, 1)]
C = 8  # = d_out of the tiny model's down_proj, so the components can factorise W exactly


def _model() -> ComponentModel:
    target = AutoModelForCausalLM.from_pretrained(TINY_QWEN, dtype=torch.float32).eval()
    target.requires_grad_(False)
    torch.manual_seed(0)
    return ComponentModel(
        target_model=target,
        module_path_info=[ModulePathInfo(module_path=m, C=C) for m in MODULES],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        sigmoid_type="leaky_hard",
        pretrained_model_output_attr="logits",
    )


def _inputs(model: ComponentModel) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    ids = torch.randint(1, 1000, (2, 6), generator=torch.Generator().manual_seed(0))
    position_mask = torch.zeros(2, 6, dtype=torch.bool)
    position_mask[0, [2, 4]] = True
    position_mask[1, 5] = True
    with torch.no_grad():
        acts = model(ids, cache_type="input").cache
    return acts, position_mask


def test_matches_hand_computation_at_selected_positions() -> None:
    model = _model()
    acts, position_mask = _inputs(model)
    ci = {m: torch.rand(2, 6, C, generator=torch.Generator().manual_seed(1)) for m in MODULES}
    with torch.no_grad():
        loss = component_orthogonality_loss(model, acts, ci, position_mask, "ci_weighted")
        loss_all = component_orthogonality_loss(model, acts, ci, None, "ci_weighted")
        expected = []
        for m in MODULES:
            x = acts[m][position_mask]
            y = model.target_model.get_submodule(m)(x)
            a = model.components[m](x, mask=ci[m][position_mask])
            expected.append(((a * (y - a)).sum(-1) / y.pow(2).sum(-1)).pow(2))
    torch.testing.assert_close(loss, torch.cat(expected).mean())
    assert loss > 1e-4, "random components too close to the sphere to test anything"
    assert abs(float(loss_all) - float(loss)) > 1e-3 * float(loss), (
        "test cannot tell a loss that ignores the position mask apart"
    )


def test_all_components_is_the_inner_product_with_delta() -> None:
    """state all_components: A = sum of all components' writes, so y - A = delta x and the loss is
    the mean of (<C x, delta x> / |y|^2)^2, whatever the CIs are."""
    model = _model()
    acts, position_mask = _inputs(model)
    ci = {m: torch.rand(2, 6, C, generator=torch.Generator().manual_seed(1)) for m in MODULES}
    ci_other = {m: torch.rand(2, 6, C, generator=torch.Generator().manual_seed(2)) for m in MODULES}
    deltas = model.calc_weight_deltas()
    with torch.no_grad():
        loss = component_orthogonality_loss(model, acts, ci, position_mask, "all_components")
        loss_other_ci = component_orthogonality_loss(
            model, acts, ci_other, position_mask, "all_components"
        )
        loss_weighted = component_orthogonality_loss(model, acts, ci, position_mask, "ci_weighted")
        expected = []
        for m in MODULES:
            x = acts[m][position_mask]
            y = model.target_model.get_submodule(m)(x)
            a = model.components[m](x, mask=torch.ones(x.shape[0], C))
            d = x @ deltas[m].T
            torch.testing.assert_close(y - a, d, atol=1e-5, rtol=1e-4)
            expected.append(((a * d).sum(-1) / y.pow(2).sum(-1)).pow(2))
    torch.testing.assert_close(loss, torch.cat(expected).mean())
    torch.testing.assert_close(loss, loss_other_ci)
    assert loss > 1e-4, "random components too close to the sphere to test anything"
    assert abs(float(loss_weighted) - float(loss)) > 1e-3 * float(loss), (
        "test cannot tell the two states apart"
    )


@pytest.mark.parametrize("state", ["ci_weighted", "all_components"])
@pytest.mark.parametrize("t", [0.0, 0.5, 1.0, 2.0, -1.0])
def test_scaled_reconstruction(t: float, state: OrthogonalityState) -> None:
    """Components = t * W with CI 1: A = t y, so <A, y - A> / |y|^2 = t (1 - t). Zero for A = 0 and
    A = y (both on the sphere with diameter y), 1/4 at its centre y / 2, negative outside."""
    model = _model()
    for m in MODULES:
        w = model.target_weight(m)  # (d_out = C, d_in)
        assert w.shape[0] == C
        model.components[m].V.data = t * w.T.clone()
        model.components[m].U.data = torch.eye(C)
    acts, position_mask = _inputs(model)
    ones = {m: torch.ones(2, 6, C) for m in MODULES}
    with torch.no_grad():
        loss = component_orthogonality_loss(model, acts, ones, position_mask, state)
        metric = ComponentOrthogonalityLoss(model, device="cpu", state=state)
        metric.update(
            pre_weight_acts=acts,
            ci=CIOutputs(lower_leaky=ones, upper_leaky=ones, pre_sigmoid=ones),
            position_mask=position_mask,
        )
        out = metric.compute()
    expected = (t * (1 - t)) ** 2
    torch.testing.assert_close(loss, torch.tensor(expected), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out["ComponentOrthogonalityLoss"], loss)
    for m in MODULES:
        torch.testing.assert_close(
            out[f"ComponentOrthogonalityLoss/inner/{m}"],
            torch.tensor(t * (1 - t)),
            atol=1e-5,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            out[f"ComponentOrthogonalityLoss/norm_ratio/{m}"],
            torch.tensor(abs(t)),
            atol=1e-5,
            rtol=1e-4,
        )


def test_gradient_reaches_components_and_ci() -> None:
    model = _model()
    acts, position_mask = _inputs(model)
    ci = {
        m: torch.rand(2, 6, C, generator=torch.Generator().manual_seed(1)).requires_grad_(True)
        for m in MODULES
    }
    component_orthogonality_loss(model, acts, ci, position_mask, "ci_weighted").backward()
    for m in MODULES:
        v_grad, ci_grad = model.components[m].V.grad, ci[m].grad
        assert v_grad is not None and v_grad.abs().sum() > 0
        assert ci_grad is not None
        assert ci_grad[~position_mask].abs().sum() == 0, "unselected positions got gradient"
        assert ci_grad[position_mask].abs().sum() > 0


def test_all_components_gradient_reaches_components_not_ci() -> None:
    model = _model()
    acts, position_mask = _inputs(model)
    ci = {
        m: torch.rand(2, 6, C, generator=torch.Generator().manual_seed(1)).requires_grad_(True)
        for m in MODULES
    }
    loss = component_orthogonality_loss(model, acts, ci, position_mask, "all_components")
    loss.backward()
    for m in MODULES:
        u_grad, v_grad = model.components[m].U.grad, model.components[m].V.grad
        assert u_grad is not None and u_grad.abs().sum() > 0
        assert v_grad is not None and v_grad.abs().sum() > 0
        assert ci[m].grad is None, "all_components must not depend on the CIs"
