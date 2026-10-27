"""Manipulation analyses of one basis (paper topic convos/julian/26-10-22_tpd_truth_paper_LOG.md, frozen
plan D3, section 4 as decided in the entry *manipulation: A plus full exploratory tables*). Shared code
and conventions: paper_common.py; bases as in paper_termination.py. Reads paper_termination.py's JSON
(the val single-removal ranking, the candidates) and paper_correlation.py's JSON (the val-selective
components, K2) of the same run and basis.

Fixed core (reported in full, whatever comes out):
- Components: the TOP top-ranked by single-removal separation drop on val (T1), each alone and as one
  group; and the label-selective group = the K2 val-selective components of the coefficient readout
  (all forms; the coefficient, not the CI, so the group is defined the same way for every basis).
  For a decomposition with an active set also N_RANDOM random active components, each alone (the
  random control; the data-aligned basis and the untrained decomposition are their own runs of this
  script, with components chosen by the same rule).
- Operations at every decomposed position: reverse (mask -1) and amplify (mask 2); removal (0) is T1.
- On test and OOD under both prompts: per family, verdict flips (1->0, 0->1, relative to the model's
  own verdict), accuracy, separation share, digit mass.
- One prediction fixed in advance, the paired-twin test: under an operation, does a negated test
  statement's verdict follow its affirmative twin's original verdict ("the 'not' is ignored")?
  Reported per condition: share of negated statements whose sign equals the twin's original sign
  (original model: near 0) and the correlation of the scores.
Exploratory (browsed on val only; anything interesting becomes a hypothesis tested on test/OOD
before it counts as a finding): the wide table, every active-set component reversed alone on val
under the training prompt, per family and label. Activity profiles come from paper_correlation.py.

Output in --out: manipulation_<run id>_<basis>.json, .npz (per-statement scores per condition).

Usage: python paper_manipulation.py --run <s-id or checkpoint> --basis decomposition --out <dir>
       --termination <json> --correlation <json> [--ood_cap N]
"""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import partial
from pathlib import Path
from typing import Any, Literal

import fire
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig, StimulusLens
from spd.experiments.lm.honesty_targeted_decomposition.paper_common import (
    Batch,
    Statements,
    StatementSplit,
    cap_per_dataset,
    edited_outputs,
    load_statements,
    make_basis,
    make_batches,
    maybe_cap,
    original_pass,
    run_edit,
    sorted_modules,
    summarize_state,
    verdict_prompts,
)
from spd.experiments.lm.honesty_targeted_decomposition.probe_ci import WANDB_PREFIX
from spd.models.component_model import ComponentModel, SPDRunInfo

TOP = 10
N_RANDOM = 10
OPERATIONS = {"reverse": -1.0, "amplify": 2.0}
N_BOOT = 1000
# negated phrasing -> affirmative phrasing; every negated tiu statement contains exactly one
# (from the steering topic's mask_search.py)
NEGATIONS = ((" does not mean ", " means "), (" does not have ", " has "), (" is not ", " is "))

Component = tuple[str, int]


def scale_mask(
    scale: dict[str, Tensor], m: str, _bt: Batch, acts: Tensor, _y: Tensor
) -> tuple[Tensor, bool]:
    """Full model with each component's write multiplied by scale[m][c] (1 = unchanged)."""
    return scale[m].expand_as(acts), True


def twin_pairs(st: Statements) -> np.ndarray:
    """(n_pairs, 2) indices (negated statement, its affirmative twin) within one statement set."""
    index = {t: i for i, t in enumerate(st.texts)}
    pairs = []
    for i in np.flatnonzero(st.forms == "negated"):
        hits = [(a, b) for a, b in NEGATIONS if a in st.texts[i]]
        assert len(hits) == 1, (st.texts[i], hits)
        aff = st.texts[i].replace(*hits[0])
        if aff in index:
            pairs.append((i, index[aff]))
    return np.array(pairs, dtype=int).reshape(-1, 2)


def twin_test(score: np.ndarray, base: np.ndarray, pairs: np.ndarray) -> dict[str, float]:
    """Does the negated statement's verdict under the edit follow its twin's ORIGINAL verdict?"""
    neg, aff = pairs[:, 0], pairs[:, 1]
    return {
        "share_follows_twin": float(((score[neg] > 0) == (base[aff] > 0)).mean()),
        "share_follows_twin_original": float(((base[neg] > 0) == (base[aff] > 0)).mean()),
        "corr_with_twin": float(np.corrcoef(score[neg], base[aff])[0, 1]),
    }


def main(
    run: str,
    basis: Literal["decomposition", "data_aligned"],
    out: str,
    termination: str,
    correlation: str,
    ood_cap: int | None = None,
    n_per_dataset: int | None = None,
    batch_size: int = 64,
    device: str = "cuda",
    seed: int = 0,
) -> None:
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(seed)
    run_info = SPDRunInfo.from_path(run if Path(run).exists() else WANDB_PREFIX + run)
    run_id = Path(run).parent.name if Path(run).exists() else run
    config = run_info.config
    lens = config.target_lens
    assert isinstance(lens, StimulusLens) and lens.tokenize_jointly, lens
    task = config.task_config
    assert isinstance(task, LMTaskConfig) and task.prepared_datasets is not None, task
    model = ComponentModel.from_run_info(run_info).to(device).eval()
    assert config.tokenizer_name is not None
    tok = AutoTokenizer.from_pretrained(config.tokenizer_name)
    one, zero = tok.encode("1", add_special_tokens=False), tok.encode("0", add_special_tokens=False)
    assert len(one) == 1 and len(zero) == 1, (one, zero)
    digits = (one[0], zero[0])
    autocast = config.autocast_bf16
    modules = sorted_modules(model)
    layer = {m: int(m.split(".")[2]) for m in modules}
    prompts = verdict_prompts(lens, tok, task.max_seq_len)

    def batches_of(texts: list[str], prompt_i: int) -> list[Batch]:
        return make_batches(texts, tok, task.loss_positions, prompts[prompt_i], batch_size, device)

    bas = make_basis(
        basis, model, run_id, task.prepared_datasets, lambda tx: batches_of(tx, 0), autocast, seed
    )
    n_c = {m: bas.U[m].shape[0] for m in modules}
    by_name = {f"L{layer[m]}:{c}": (m, c) for m in modules for c in range(n_c[m])}

    term = json.loads(Path(termination).read_text())
    corr = json.loads(Path(correlation).read_text())
    for name, d in (("termination", term), ("correlation", corr)):
        assert d["run"] == run, (name, d["run"], run)
        assert d["basis"] == bas.name, (name, d["basis"], bas.name)
    top: list[Component] = [by_name[nm] for nm in term["ranked"][:TOP]]
    selective: list[Component] = sorted(
        {by_name[nm] for form in corr["K2_coef"].values() for nm in form["selective"]}
    )
    has_active = bool(term["candidates_are_active_set"])
    active: list[Component] = [by_name[nm] for nm in term["candidates"]] if has_active else []
    print(f"top {TOP}: {term['ranked'][:TOP]}; label-selective group: {len(selective)}; "
          f"active set: {len(active) if has_active else 'none'}")  # fmt: skip

    def scaled(
        comps: list[Component], factor: float
    ) -> Callable[[int, Batch], AbstractContextManager[None]]:
        scale = {m: torch.ones(n_c[m], device=device) for m in modules}
        for m, c in comps:
            scale[m][c] = factor
        return lambda _ib, bt: edited_outputs(model, bas, bt, partial(scale_mask, scale))

    conditions: dict[str, Callable[[int, Batch], AbstractContextManager[None]]] = {}
    for op, factor in OPERATIONS.items():
        for mc in top:
            conditions[f"{op} {_name(mc, layer)}"] = scaled([mc], factor)
        conditions[f"{op} top{TOP} group"] = scaled(top, factor)
        if selective:
            conditions[f"{op} selective group"] = scaled(selective, factor)
        if has_active:
            pool = [mc for mc in active if mc not in top]
            for i in rng.choice(len(pool), size=min(N_RANDOM, len(pool)), replace=False):
                conditions[f"{op} random {_name(pool[i], layer)}"] = scaled([pool[i]], factor)
    print(f"{len(conditions)} core conditions per set and prompt", flush=True)

    summary: dict[str, Any] = {
        "run": run, "basis": bas.name, "top": term["ranked"][:TOP],
        "selective_group": [_name(mc, layer) for mc in selective], "sets": {},
    }  # fmt: skip
    arrays: dict[str, np.ndarray] = {}
    split_list: tuple[StatementSplit, ...] = ("test", "ood")
    for split in split_list:
        st = maybe_cap(load_statements(split, task.prepared_datasets), n_per_dataset, seed)
        if split == "ood" and ood_cap is not None:
            st = cap_per_dataset(st, ood_cap, seed)
        pairs = twin_pairs(st) if split == "test" else None
        arrays[f"{split}_labels"], arrays[f"{split}_datasets"] = st.labels, st.datasets
        summary["sets"][split] = {}
        for pi, prompt in enumerate(prompts):
            batches = batches_of(st.texts, pi)
            orig = original_pass(model, batches, autocast, None)
            base = (orig.base_lp[:, digits[0]] - orig.base_lp[:, digits[1]]).numpy()
            all_off = lambda _ib, bt: edited_outputs(  # noqa: E731
                model, bas, bt, lambda _m, _b, acts, _y: (0 * acts, False)
            )
            _, kl_off, _ = run_edit(model, batches, orig, all_off, digits, autocast)
            res: dict[str, Any] = {"n": len(st), "conditions": {}}
            for name, edit in conditions.items():
                s, k, ms = run_edit(model, batches, orig, edit, digits, autocast)
                arrays[f"{split}_{prompt.name}_{name}_score"] = s.astype(np.float32)
                n_boot = N_BOOT if name.endswith("group") else 0
                r = summarize_state(s, k, ms, base, kl_off, st.labels, st.families, n_boot, seed)
                if pairs is not None and len(pairs):
                    r["twin_test"] = twin_test(s, base, pairs)
                res["conditions"][name] = r
                m_ = r["macro"]
                print(f"  {split}/{prompt.name} {name:28s} 1->0 {m_['flip_1_to_0']:.3f} "
                      f"0->1 {m_['flip_0_to_1']:.3f} sep {m_['sep_share']:.3f} "
                      f"digit {m_['digit_mass']:.3f}"
                      + (f" twin {r['twin_test']['share_follows_twin']:.3f}" if "twin_test" in r else ""),
                      flush=True)  # fmt: skip
            summary["sets"][split][prompt.name] = res

    # ---- exploratory wide table: every active component reversed alone, val, training prompt ----
    if has_active:
        val = maybe_cap(load_statements("val", task.prepared_datasets), n_per_dataset, seed)
        batches = batches_of(val.texts, 0)
        orig = original_pass(model, batches, autocast, None)
        base = (orig.base_lp[:, digits[0]] - orig.base_lp[:, digits[1]]).numpy()
        all_off = lambda _ib, bt: edited_outputs(  # noqa: E731
            model, bas, bt, lambda _m, _b, acts, _y: (0 * acts, False)
        )
        _, kl_off, _ = run_edit(model, batches, orig, all_off, digits, autocast)
        wide = {}
        for i, mc in enumerate(active):
            s, k, ms = run_edit(model, batches, orig, scaled([mc], -1.0), digits, autocast)
            arrays[f"wide_val_{_name(mc, layer)}_score"] = s.astype(np.float32)
            r = summarize_state(s, k, ms, base, kl_off, val.labels, val.families, 0, seed)
            wide[_name(mc, layer)] = {"pooled": r["pooled"], "per_family": r["per_family"]}
            if i % 50 == 0:
                print(f"  wide table {i}/{len(active)}", flush=True)
        summary["wide_val_reverse"] = wide
        arrays["wide_val_base_score"] = base
        arrays["wide_val_labels"], arrays["wide_val_datasets"] = val.labels, val.datasets

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"manipulation_{run_id}_{basis}"
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    np.savez(out_dir / f"{stem}.npz", **arrays)  # pyright: ignore[reportArgumentType]


def _name(mc: Component, layer: dict[str, int]) -> str:
    return f"L{layer[mc[0]]}:{mc[1]}"


if __name__ == "__main__":
    fire.Fire(main)
