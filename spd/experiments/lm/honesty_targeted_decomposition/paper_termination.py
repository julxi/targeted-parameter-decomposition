"""Termination (necessity) analyses of one basis (paper topic convos/julian/26-10-22_tpd_truth_paper_LOG.md,
frozen plan D3, section 2; T0 = all components removed is in paper_recovery.py). Shared code and
conventions: paper_common.py.

Bases (--basis):
  decomposition  the run's own components: a trained final run, or the untrained steps-0 checkpoint
  data_aligned   the untrained data-aligned control, built from the run's target model (v_c = top-C
                 principal directions of the MLP input on train statements, u_c = W v_c); the run only
                 supplies model, data and decomposed positions
Candidates: the run's functional active set from paper_recovery.py's JSON (--recovery). Without an
active set (untrained decomposition, data-aligned basis): every component. No CI threshold is used.

T1  each candidate removed alone from the full model (mask 0, all other components and the remainder
    kept), at every decomposed position: on val under the training prompt for every candidate (the
    ranking used by T2 and by the manipulation script), and for an active set also on test.
T2  the top-k candidates by val separation drop removed together, k in K_GRID (and --extra_k, e.g. the
    trained runs' active-set sizes, so controls can be compared at the same k) up to all candidates;
    on test and OOD under both prompts. With an active set also N_RANDOM random size-k subsets of it
    (the size-matched random control).
T3  (A8; --a8, needs an active set) one layer at a time, everything else original: remove from the
    ORIGINAL MLP output its projection onto span(U) of (a) all C write vectors of the layer, (b) the
    active set's write vectors in that layer [both decided: C: comment under *D3 proposal*, question 3];
    controls of the same dimension: N_RAND_SUB random subspaces and the top principal directions of y
    (uncentred second moment over the val decomposed positions). Test and OOD, both prompts.

Output in --out: termination_<run id>_<basis>.json (summaries; T1 tables), .npz (per-statement scores
and KLs of every condition, set and prompt).

Usage: python paper_termination.py --run <s-id or checkpoint> --basis decomposition --out <dir>
       [--recovery <recovery json>] [--a8] [--extra_k=57,63] [--sets test,ood] [--ood_cap N]
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
    Basis,
    Batch,
    Original,
    StatementSplit,
    cap_per_dataset,
    edited_outputs,
    load_statements,
    make_basis,
    make_batches,
    maybe_cap,
    original_pass,
    projected_out,
    run_edit,
    sorted_modules,
    state_metrics,
    summarize_state,
    verdict_prompts,
)
from spd.experiments.lm.honesty_targeted_decomposition.probe_ci import WANDB_PREFIX
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.general_utils import bf16_autocast

K_GRID = (1, 2, 5, 10, 20, 50, 100, 200)
N_RANDOM = 5
N_RAND_SUB = 3
N_BOOT = 1000

Component = tuple[str, int]


def removal_mask(
    remove: dict[str, Tensor], m: str, _bt: Batch, acts: Tensor, _y: Tensor
) -> tuple[Tensor, bool]:
    """Full model with the components in `remove` (module -> 0/1 vector over C) at mask 0."""
    return (1 - remove[m]).expand_as(acts), True


def main(
    run: str,
    basis: Literal["decomposition", "data_aligned"],
    out: str,
    recovery: str | None = None,
    a8: bool = False,
    extra_k: tuple[int, ...] | int = (),
    sets: str = "test,ood",
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
    train_prompt = prompts[0]

    def batches_of(texts: list[str], prompt_i: int) -> list[Batch]:
        return make_batches(texts, tok, task.loss_positions, prompts[prompt_i], batch_size, device)

    bas = make_basis(
        basis, model, run_id, task.prepared_datasets, lambda tx: batches_of(tx, 0), autocast, seed
    )
    n_c = {m: bas.U[m].shape[0] for m in modules}
    name_of = {(m, c): f"L{layer[m]}:{c}" for m in modules for c in range(n_c[m])}
    by_name = {v: k for k, v in name_of.items()}

    active: list[Component] | None = None
    if recovery is not None:
        assert basis == "decomposition", "an active set belongs to a decomposition"
        rec = json.loads(Path(recovery).read_text())
        assert rec["run"] == run, (rec["run"], run)
        if rec["active_set"] is not None:
            active = [by_name[nm] for nm in rec["active_set"]["components"]]
    candidates = active if active is not None else list(name_of)
    print(f"basis {bas.name}: {len(candidates)} candidates ("
          f"{'active set' if active is not None else 'all components'})")  # fmt: skip
    assert not a8 or active is not None, "A8 needs an active set"

    def onehot(comps: list[Component]) -> dict[str, Tensor]:
        v = {m: torch.zeros(n_c[m], device=device) for m in modules}
        for m, c in comps:
            v[m][c] = 1.0
        return v

    def removal(comps: list[Component]) -> Callable[[int, Batch], AbstractContextManager[None]]:
        remove = onehot(comps)
        return lambda _ib, bt: edited_outputs(model, bas, bt, partial(removal_mask, remove))

    def nothing_removed() -> Callable[[int, Batch], AbstractContextManager[None]]:
        return removal([])

    def all_off(_ib: int, bt: Batch) -> AbstractContextManager[None]:
        return edited_outputs(model, bas, bt, lambda _m, _b, acts, _y: (0 * acts, False))

    # ---- T1 on val: ranking ----
    val = maybe_cap(load_statements("val", task.prepared_datasets), n_per_dataset, seed)
    val_batches = batches_of(val.texts, 0)
    val_orig = original_pass(model, val_batches, autocast, None)
    base_val = score_of(val_orig, digits)
    _, kl_off_val, _ = run_edit(model, val_batches, val_orig, all_off, digits, autocast)
    full_val = run_edit(model, val_batches, val_orig, nothing_removed(), digits, autocast)
    assert ((full_val[0] > 0) == (base_val > 0)).mean() > 0.99, "full state does not reproduce"
    t1_val: dict[str, dict[str, float]] = {}
    for i, comp in enumerate(candidates):
        s, k, ms = run_edit(model, val_batches, val_orig, removal([comp]), digits, autocast)
        t1_val[name_of[comp]] = state_metrics(s, k, base_val, kl_off_val, val.labels, ms)
        if i % 50 == 0:
            print(f"  T1 val {i}/{len(candidates)}", flush=True)
    ranked = sorted(
        candidates, key=lambda mc: t1_val[name_of[mc]]["sep_share"]
    )  # largest drop first
    print("T1 val, 15 largest separation drops: " + ", ".join(
        f"{name_of[mc]} {t1_val[name_of[mc]]['sep_share']:.3f}" for mc in ranked[:15]))  # fmt: skip

    ks = sorted({k for k in K_GRID if k < len(ranked)} | {len(ranked)}
                | set((extra_k,) if isinstance(extra_k, int) else extra_k))  # fmt: skip
    assert max(ks) <= len(ranked), (ks, len(ranked))

    # ---- conditions on test / OOD ----
    conditions: dict[str, Callable[[int, Batch], AbstractContextManager[None]]] = {}
    for k in ks:
        conditions[f"T2 top{k}"] = removal(ranked[:k])
        if active is not None and k < len(active):
            for r in range(N_RANDOM):
                draw = [active[i] for i in rng.choice(len(active), size=k, replace=False)]
                conditions[f"T2 rand{k} #{r}"] = removal(draw)
    if a8:
        assert active is not None
        a8_conditions(conditions, model, bas, val_batches, active, layer, autocast, rng)
    print(f"{len(conditions)} conditions per set and prompt", flush=True)

    summary: dict[str, Any] = {
        "run": run, "basis": bas.name, "candidates_are_active_set": active is not None,
        "candidates": [name_of[c] for c in candidates],
        "ranked": [name_of[c] for c in ranked], "ks": ks, "t1_val": t1_val, "sets": {},
    }  # fmt: skip
    arrays: dict[str, np.ndarray] = {}
    split_list: list[StatementSplit] = [s for s in sets.split(",")]  # pyright: ignore[reportAssignmentType]
    assert set(split_list) <= {"test", "ood"}, split_list
    for split in split_list:
        st = maybe_cap(load_statements(split, task.prepared_datasets), n_per_dataset, seed)
        if split == "ood" and ood_cap is not None:
            st = cap_per_dataset(st, ood_cap, seed)
        arrays[f"{split}_labels"], arrays[f"{split}_datasets"] = st.labels, st.datasets
        summary["sets"][split] = {}
        for pi, prompt in enumerate(prompts):
            batches = batches_of(st.texts, pi)
            orig = original_pass(model, batches, autocast, None)
            base = score_of(orig, digits)
            _, kl_off, _ = run_edit(model, batches, orig, all_off, digits, autocast)
            res: dict[str, Any] = {"n": len(st), "conditions": {}}
            conds = dict(conditions)
            if active is not None and split == "test" and prompt is train_prompt:
                conds.update({f"T1 {name_of[c]}": removal([c]) for c in active})
            for name, edit in conds.items():
                s, k, ms = run_edit(model, batches, orig, edit, digits, autocast)
                arrays[f"{split}_{prompt.name}_{name}_score"] = s.astype(np.float32)
                arrays[f"{split}_{prompt.name}_{name}_kl"] = k.astype(np.float32)
                n_boot = 0 if name.startswith("T1 ") else N_BOOT
                res["conditions"][name] = summarize_state(
                    s, k, ms, base, kl_off, st.labels, st.families, n_boot, seed
                )
                r = res["conditions"][name]["macro"]
                print(f"  {split}/{prompt.name} {name:22s} sep {r['sep_share']:.3f} "
                      f"agree {r['agreement']:.3f} 1->0 {r['flip_1_to_0']:.3f} "
                      f"0->1 {r['flip_0_to_1']:.3f}", flush=True)  # fmt: skip
            summary["sets"][split][prompt.name] = res

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"termination_{run_id}_{bas.name if basis == 'data_aligned' else 'decomposition'}"
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    np.savez(out_dir / f"{stem}.npz", **arrays)  # pyright: ignore[reportArgumentType]


def score_of(orig: Original, digits: tuple[int, int]) -> np.ndarray:
    return (orig.base_lp[:, digits[0]] - orig.base_lp[:, digits[1]]).numpy()


def orthonormal(cols: Tensor) -> Tensor:
    """Orthonormal basis (d, k) of the span of the k columns; asserts full rank."""
    q, r = torch.linalg.qr(cols)
    diag = r.diagonal().abs()
    assert int((diag > 1e-6 * diag.max()).sum()) == cols.shape[1], "rank-deficient span"
    return q


def a8_conditions(
    conditions: dict[str, Callable[[int, Batch], AbstractContextManager[None]]],
    model: ComponentModel,
    bas: Basis,
    val_batches: list[Batch],
    active: list[Component],
    layer: dict[str, int],
    autocast: bool,
    rng: np.random.Generator,
) -> None:
    """Adds the A8 conditions (module docstring, T3) to `conditions`."""
    device = val_batches[0].ids.device
    second = {m: torch.zeros(u.shape[1], u.shape[1], device=device) for m, u in bas.U.items()}
    for bt in val_batches:
        with bf16_autocast(enabled=autocast):
            cache = model(bt.ids, cache_type="input").cache
        for m in bas.U:
            y = cache[m][bt.dec].float() @ model.target_weight(m).float().T
            second[m] += y.T @ y
    gen = torch.Generator(device=device).manual_seed(int(rng.integers(2**31)))
    for m in bas.U:
        evecs = torch.linalg.eigh(second[m])[1]  # ascending eigenvalues
        in_layer = [c for mm, c in active if mm == m]
        spans = {"all": bas.U[m].T}
        if in_layer:
            spans["active"] = bas.U[m][in_layer].T
        for span_name, cols in spans.items():
            k = cols.shape[1]
            subs = {f"span{span_name}": orthonormal(cols), f"pca{span_name}": evecs[:, -k:]}
            for r in range(N_RAND_SUB):
                g = torch.randn(cols.shape[0], k, device=device, generator=gen)
                subs[f"rand{span_name} #{r}"] = orthonormal(g)
            for sub_name, q in subs.items():
                conditions[f"A8 L{layer[m]} {sub_name}"] = lambda _ib, bt, q=q, m=m: projected_out(
                    model, bt, {m: q}
                )


if __name__ == "__main__":
    fire.Fire(main)
