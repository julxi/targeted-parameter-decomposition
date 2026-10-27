"""Tests of the shared paper-analysis code (spd/experiments/lm/honesty_targeted_decomposition/
paper_common.py): statement sets, metrics, and the down_proj edit that every intervention uses."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

from spd.experiments.lm.honesty_targeted_decomposition.paper_common import (
    EXCLUDED_OOD,
    GROUND_TRUTH_OOD,
    Basis,
    Batch,
    bootstrap_intervals,
    edited_outputs,
    family_of,
    form_of,
    load_statements,
    smallest_reaching,
    state_metrics,
)
from spd.models.component_model import ComponentModel

TRAIN_DIRS = [
    f"data/tiu/{f}_{lab}/v3"
    for f in ("animal_class", "cities", "neg_cities", "larger_than")
    for lab in ("true", "false")
]


def test_family_and_form() -> None:
    assert family_of("neg_cities_de_false") == "neg_cities_de"
    assert form_of("neg_cities_de") == "negated"
    assert form_of("smaller_than") == "comparison"
    assert form_of("cities_conj") == "conjunction"
    assert form_of("facts_disj") == "disjunction"
    assert form_of("inventors_de") == "affirmative"
    with pytest.raises(AssertionError):
        family_of("cities")


def test_ood_statements() -> None:
    st = load_statements("ood", TRAIN_DIRS)
    families = set(st.families)
    assert len(families) == 24
    assert not families & set(EXCLUDED_OOD)
    assert set(GROUND_TRUTH_OOD) <= families
    assert st.labels.any() and not st.labels.all()
    assert all(st.datasets[i].endswith("_true") == st.labels[i] for i in range(len(st)))
    assert st.german[st.families == "neg_cities_de"].all()
    assert not st.german[st.families == "facts"].any()


def test_state_metrics() -> None:
    labels = np.array([True, True, False, False])
    base = np.array([4.0, 2.0, -2.0, -4.0])
    score = np.array([2.0, -1.0, -1.0, -2.0])  # one true statement flips
    kl_off = np.full(4, 2.0)
    m = state_metrics(score, np.full(4, 0.5), base, kl_off, labels, np.ones(4))
    assert m["kl_recovered"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["agreement"] == pytest.approx(0.75)
    assert m["sep_share"] == pytest.approx((0.5 - (-1.5)) / 6.0)


def test_bootstrap_intervals_cover_estimate_and_are_deterministic() -> None:
    rng = np.random.default_rng(0)
    families = np.repeat(np.array(["a", "b"]), 50)
    values = rng.normal(size=100)

    def fn(idx: np.ndarray) -> dict[str, float]:
        return {"mean": float(values[idx].mean())}

    lo, hi = bootstrap_intervals(families, fn, 200, seed=1)["mean"]
    assert lo < values.mean() < hi
    assert bootstrap_intervals(families, fn, 200, seed=1)["mean"] == (lo, hi)


def test_smallest_reaching() -> None:
    assert smallest_reaching({1: 0.1, 4: 0.96, 2: 0.97, 8: 0.99}, 0.95) == 2
    with pytest.raises(AssertionError):
        smallest_reaching({1: 0.1}, 0.95)


def test_edited_outputs() -> None:
    """y' = d_on (y - C) + sum_c mask_c (v_c . x) u_c at decomposed positions, y elsewhere, for a basis
    that does not span W (so y - C != 0 and both terms are exercised)."""
    torch.manual_seed(0)
    d_in, d_out, n_c = 6, 5, 3
    lin = nn.Linear(d_in, d_out, bias=False)
    target = nn.Sequential()
    target.add_module("down_proj", lin)
    model = cast(ComponentModel, cast(object, SimpleNamespace(target_model=target)))
    basis = Basis(
        "t", {"down_proj": torch.randn(n_c, d_out)}, {"down_proj": torch.randn(d_in, n_c)}
    )
    x = torch.randn(2, 4, d_in)
    dec = torch.tensor([[True, True, False, False], [False, True, True, True]])
    batch = Batch(np.arange(2), torch.zeros(2, 4, dtype=torch.long), dec, torch.zeros(2))
    y = lin(x).detach()
    acts = x @ basis.V["down_proj"]
    C = acts @ basis.U["down_proj"]
    mask = torch.rand(2, 4, n_c)
    for d_on in (True, False):
        with (
            torch.no_grad(),
            edited_outputs(model, basis, batch, lambda *_, d_on=d_on: (mask, d_on)),
        ):
            out = target(x)
        expected = (acts * mask) @ basis.U["down_proj"] + (y - C if d_on else 0)
        assert torch.allclose(out[dec], expected[dec], atol=1e-5)
        assert torch.equal(out[~dec], y[~dec])
    # the hook is removed on exit
    assert torch.equal(target(x), lin(x))
