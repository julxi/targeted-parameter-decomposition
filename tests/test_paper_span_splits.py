"""Unit checks of the model-anchored splits and the clamped swap of the behavioural tests
(spd/experiments/lm/honesty_targeted_decomposition/paper_interventions.py; topic
convos/julian/26-11-15_open_research_LOG.md): the algebra of each split on small random tensors."""

import pytest
import torch

from spd.experiments.lm.honesty_targeted_decomposition.paper_interventions import (
    BatchContext,
    ComponentSplit,
    edit_fn,
    parse_intervention,
    part_of,
    span_split,
)

M = "model.layers.11.mlp.down_proj"
D_IN, D_OUT, K = 40, 12, 3


def _setup() -> tuple[ComponentSplit, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    W = {M: torch.randn(D_OUT, D_IN)}
    base = ComponentSplit(
        "active", {M: torch.randn(K, D_OUT)}, {M: torch.randn(D_IN, K)}, ("L11:0", "L11:1", "L11:2")
    )
    x = torch.randn(2, 5, D_IN)
    return base, W, x, x @ W[M].T


def test_span_splits_are_the_stated_restrictions_of_w():
    base, W, x, y = _setup()
    qv = torch.linalg.qr(base.V[M])[0]
    qu = torch.linalg.qr(base.U[M].T)[0]
    basis = torch.linalg.qr(torch.randn(D_IN, 8))[0]  # stands in for the data-aligned basis
    diff = torch.linalg.qr(torch.randn(D_OUT, D_OUT))[0]
    expected = {
        "readspan": (x @ qv @ qv.T) @ W[M].T,
        "writespan": y @ qu @ qu.T,
        "bothspan": ((x @ qv @ qv.T) @ W[M].T) @ qu @ qu.T,
        "pcaspan": (x @ basis[:, -K:] @ basis[:, -K:].T) @ W[M].T,
        "diffspan": y @ diff[:, -K:] @ diff[:, -K:].T,
    }
    for kind, want in expected.items():
        sp = span_split(kind, base, W, {M: basis}, 0, {M: diff})  # pyright: ignore[reportArgumentType]
        p, r = part_of(sp, "P", M, x, y), part_of(sp, "R", M, x, y)
        assert torch.allclose(p, want, atol=1e-4), kind
        assert torch.allclose(p + r, y, atol=1e-4), kind


def test_randspan_is_a_seeded_projection():
    base, W, x, y = _setup()
    a, b = span_split("randspan", base, W, None, 3), span_split("randspan", base, W, None, 3)
    assert torch.equal(a.U[M], b.U[M])
    q = a.U[M]
    assert torch.allclose(q @ q.T, torch.eye(K), atol=1e-5)
    assert torch.allclose(part_of(a, "P", M, x, y), y @ q.T @ q, atol=1e-4)
    assert not torch.allclose(a.U[M], span_split("randspan", base, W, None, 4).U[M])


def test_dead_components_are_dropped_from_the_span():
    base, W, x, y = _setup()
    U, V = base.U[M].clone(), base.V[M].clone()
    U[2], V[:, 2] = 0, 0
    dead = ComponentSplit("all", {M: U}, {M: V}, base.components)
    sp = span_split("writespan", dead, W, None, 0)
    assert sp.U[M].shape[0] == K - 1
    qu = torch.linalg.qr(U[:2].T)[0]
    assert torch.allclose(part_of(sp, "P", M, x, y), y @ qu @ qu.T, atol=1e-4)


@pytest.mark.parametrize("part", ["P", "R"])
def test_swap_clamped_holds_the_other_part_at_its_clean_value(part: str):
    base, W, x, y = _setup()
    x_twin, x_now = torch.randn_like(x), torch.randn_like(x)  # x_now: the edited run's input
    y_twin, y_now = x_twin @ W[M].T, x_now @ W[M].T
    dec = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]).bool()
    other = "R" if part == "P" else "P"
    ctx = BatchContext(
        dec, {}, torch.ones(2),
        {M: part_of(base, part, M, x, y)}, {M: y}, {M: part_of(base, part, M, x_twin, y_twin)},  # pyright: ignore[reportArgumentType]
    )  # fmt: skip
    out = edit_fn(base, part, parse_intervention("swap_clamped"), ctx, None)(M, x_now, y_now)  # pyright: ignore[reportArgumentType]
    assert out is not None
    want = part_of(base, part, M, x_twin, y_twin) + part_of(base, other, M, x, y)  # pyright: ignore[reportArgumentType]
    assert torch.allclose(out[dec], want[dec], atol=1e-4)
    assert torch.equal(out[~dec], y_now[~dec])
    # the plain swap keeps the CURRENT other part instead
    out2 = edit_fn(base, part, parse_intervention("swap"), ctx, None)(M, x_now, y_now)  # pyright: ignore[reportArgumentType]
    assert out2 is not None
    want2 = part_of(base, part, M, x_twin, y_twin) + part_of(base, other, M, x_now, y_now)  # pyright: ignore[reportArgumentType]
    assert torch.allclose(out2[dec], want2[dec], atol=1e-4)
