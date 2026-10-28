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
    auc_columns,
    bootstrap_intervals,
    cap_per_dataset,
    control_rows,
    edited_outputs,
    family_of,
    form_of,
    ground_truth_ood,
    load_statements,
    projected_out,
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


def test_ground_truth_ood_keeps_capped_draws() -> None:
    """Restricting after a cap keeps exactly the capped statements of the 13 files, so capped runs
    before and after the cut are comparable per family."""
    capped = cap_per_dataset(load_statements("ood", TRAIN_DIRS), 4, seed=0)
    gt = ground_truth_ood(capped)
    assert set(gt.families) == set(GROUND_TRUTH_OOD)
    keep = np.isin(capped.families, GROUND_TRUTH_OOD)
    assert gt.texts == [t for t, k in zip(capped.texts, keep, strict=True) if k]
    assert (gt.labels == capped.labels[keep]).all()


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
    assert m["flip_1_to_0"] == pytest.approx(0.5)
    assert m["flip_0_to_1"] == pytest.approx(0.0)


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


def test_projected_out() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(6, 5, bias=False)
    target = nn.Sequential()
    target.add_module("down_proj", lin)
    model = cast(ComponentModel, cast(object, SimpleNamespace(target_model=target)))
    q = torch.linalg.qr(torch.randn(5, 2))[0]
    x = torch.randn(2, 3, 6)
    dec = torch.tensor([[True, False, True], [False, True, True]])
    batch = Batch(np.arange(2), torch.zeros(2, 3, dtype=torch.long), dec, torch.zeros(2))
    y = lin(x).detach()
    with torch.no_grad(), projected_out(model, batch, {"down_proj": q}):
        out = target(x)
    assert torch.allclose(out[dec] @ q, torch.zeros(int(dec.sum()), 2), atol=1e-5)
    assert torch.allclose(out[dec], y[dec] - (y[dec] @ q) @ q.T, atol=1e-5)
    assert torch.equal(out[~dec], y[~dec])
    # columns must be orthonormal
    with pytest.raises(AssertionError), projected_out(model, batch, {"down_proj": 2 * q}):
        pass


def test_cap_per_dataset() -> None:
    st = load_statements("ood", TRAIN_DIRS)
    capped = cap_per_dataset(st, 30, seed=0)
    counts = {d: int((capped.datasets == d).sum()) for d in set(st.datasets)}
    assert all(c == min(30, int((st.datasets == d).sum())) for d, c in counts.items())
    assert cap_per_dataset(st, 30, seed=0).texts == capped.texts  # deterministic
    assert all(capped.datasets[i].endswith("_true") == capped.labels[i] for i in range(len(capped)))


def test_control_rows_are_a_nested_cap_on_ood_only() -> None:
    """The control repetitions' rows: on OOD at most N per dataset, a subset of the statements the
    main conditions use (also after a flat --ood_cap), and the same rows as cap_per_dataset picks;
    without the option, or on test, all statements."""
    st = load_statements("ood", TRAIN_DIRS)
    for main in (st, cap_per_dataset(st, 200, seed=0)):
        rows = control_rows(main, 100, seed=0)
        assert len(set(rows.tolist())) == len(rows) and rows.max() < len(main)
        assert np.all(np.diff(rows) > 0)  # original order
        sub = [main.texts[i] for i in rows]
        assert sub == cap_per_dataset(main, 100, seed=0).texts
        for d in set(main.datasets):
            n_d = int((main.datasets == d).sum())
            assert int((main.datasets[rows] == d).sum()) == min(100, n_d)
    assert len(control_rows(st, 100, seed=0)) < len(st)
    assert np.array_equal(control_rows(st, None, seed=0), np.arange(len(st)))
    test = load_statements("test", TRAIN_DIRS)
    assert np.array_equal(control_rows(test, 100, seed=0), np.arange(len(test)))


def test_auc_columns_matches_sklearn() -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.random(60) < 0.4
    X = np.column_stack([rng.normal(size=60) + y, rng.integers(0, 3, 60), -y.astype(float)])
    expected = np.array([float(roc_auc_score(y, X[:, j])) for j in range(3)])
    assert np.allclose(auc_columns(X, y), expected)


def test_twin_pairs_and_twin_test() -> None:
    from spd.experiments.lm.honesty_targeted_decomposition.paper_manipulation import (
        twin_pairs,
        twin_test,
    )

    st = load_statements("val", TRAIN_DIRS)
    pairs = twin_pairs(st)
    assert len(pairs) == int(
        (st.forms == "negated").sum()
    )  # every negated val statement has its twin
    for i, j in pairs[:20]:
        assert st.forms[j] == "affirmative" and st.families[i] == "neg_" + st.families[j]
        assert st.labels[i] != st.labels[j]  # "X is not Y" and "X is Y" have opposite truth
    base = np.where(st.labels, 3.0, -3.0)  # a model that is always right
    r = twin_test(base, base, pairs)
    assert r["share_follows_twin"] == 0.0 and r["share_follows_twin_original"] == 0.0
    ignored = base.copy()
    ignored[pairs[:, 0]] = base[pairs[:, 1]]  # an edit that makes the model ignore "not"
    assert twin_test(ignored, base, pairs)["share_follows_twin"] == 1.0


def test_directions_recover_planted_direction() -> None:
    from spd.experiments.lm.honesty_targeted_decomposition.paper_directions import directions

    rng = np.random.default_rng(0)
    d_true = np.zeros(20)
    d_true[3] = 1.0
    y = rng.random(400) < 0.5
    X = rng.normal(size=(400, 20)) + 2.0 * np.outer(y, d_true)
    dirs, _ = directions(X[:300], y[:300], X[300:], y[300:])
    assert dirs["mm"] @ d_true > 0.95 and dirs["lr"] @ d_true > 0.9
    assert np.isclose(np.linalg.norm(dirs["lr"]), 1.0)
