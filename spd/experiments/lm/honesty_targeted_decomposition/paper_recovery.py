"""Recovery, the functional active set, the inactive components' role and the CI distribution of one
decomposition (paper topic convos/julian/26-10-22_tpd_truth_paper_LOG.md, frozen plan D3: section 1
*Recovery*, T0 of *Termination*, *active vs inactive components*, the valley check). Runs on trained
decompositions and on the untrained one (steps: 0); the data-aligned basis has no CIs, so it has no
recovery. Shared code and conventions: paper_common.py.

1. Active set, on val under the training prompt [decided: C: comment under *D3 plan, revision 2*]:
   rank all components by their mean CI over the decomposed positions; keep only the top m (mask = CI,
   the others 0, delta off); the active set S is the smallest such set whose KL recovered reaches
   TARGET_SHARE of what all components recover. The curve over m is saved. S is chosen on val and
   then used unchanged on test and OOD. No CI threshold defines "active" anywhere [decided: C:
   comment, same entry]. If all components together recover nothing (expected for the untrained
   decomposition), there is no active set and the S-dependent states are skipped.
2. CI values at every decomposed position of val, saved for the valley check (float16) and printed as
   histograms: is there a minimum between a mode near 0 and a mode near 1?
3. Per statement set (test, OOD) and prompt (training, second), the states
     full                    all components + delta (sanity: must reproduce the original)
     components_ci           mask = CI, delta off (recovery)
     active_ci               only S at mask = CI, delta off
     mean_mask               mask = CI + (1 - CI) / 2, delta off: the training's expected mask
     delta_only              all components removed (T0: are they together necessary?)
     all_off                 the decomposed MLP outputs set to 0
     full_without_inactive   full model with the components outside S removed
   with KL recovered, accuracy (ground truth), agreement (model's verdict), separation share, digit
   mass; pooled, macro-averaged over families with bootstrap intervals, and per family. Also the sum
   of CIs per statement (counts without a cutoff) and per layer, at the decomposed positions, the
   CI-weighted writes of S (A) and of the rest (I) and the cancellation diagnostics (relative to |y|):
   |A|, |I|, cos(A, I), |C|, |D|, <C, D> / |y|^2.

Output in --out: recovery_<run id>.json (all summaries), recovery_<run id>.npz (per-statement scores
and KLs per set, prompt and state), ci_val_<run id>.npz (the val CI values).

Usage: python paper_recovery.py --run <s-id or checkpoint path> --out <dir>
       [--sets test,ood] [--n_per_dataset N] (N: a quick check on N statements per dataset)
"""

import json
from contextlib import AbstractContextManager
from functools import partial
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig, StimulusLens
from spd.experiments.lm.honesty_targeted_decomposition.paper_common import (
    Basis,
    Batch,
    Original,
    Statements,
    StatementSplit,
    cap_per_dataset,
    decomposition_basis,
    edited_outputs,
    load_statements,
    make_batches,
    original_pass,
    run_edit,
    smallest_reaching,
    sorted_modules,
    summarize_state,
    verdict_prompts,
)
from spd.experiments.lm.honesty_targeted_decomposition.probe_ci import WANDB_PREFIX
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.general_utils import bf16_autocast

TARGET_SHARE = 0.95
M_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
STATES = (
    "full", "components_ci", "active_ci", "mean_mask", "delta_only", "all_off",
    "full_without_inactive",
)  # fmt: skip
NEEDS_ACTIVE_SET = ("active_ci", "full_without_inactive")
N_BOOT = 1000


def maybe_cap(st: Statements, n_per_dataset: int | None, seed: int) -> Statements:
    return st if n_per_dataset is None else cap_per_dataset(st, n_per_dataset, seed)


def main(
    run: str,
    out: str,
    sets: str = "test,ood",
    n_per_dataset: int | None = None,
    batch_size: int = 64,
    device: str = "cuda",
    seed: int = 0,
) -> None:
    torch.set_grad_enabled(False)
    run_info = SPDRunInfo.from_path(run if Path(run).exists() else WANDB_PREFIX + run)
    # name by run id, never by the checkpoint file name (all final checkpoints share it; FEEDBACK.md)
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
    one, zero = one[0], zero[0]
    modules = sorted_modules(model)
    basis = decomposition_basis(model, run_id)
    n_c = {m: basis.U[m].shape[0] for m in modules}
    layer = {m: int(m.split(".")[2]) for m in modules}
    flat = [(m, c) for m in modules for c in range(n_c[m])]  # global component order
    prompts = verdict_prompts(lens, tok, task.max_seq_len)
    split_list: list[StatementSplit] = [s for s in sets.split(",")]  # pyright: ignore[reportAssignmentType]
    assert set(split_list) <= {"test", "ood"}, split_list

    digits = (one, zero)

    def pass1(batches: list[Batch]) -> Original:
        return original_pass(model, batches, config.autocast_bf16, config.sampling)

    def run_state(
        batches: list[Batch], p1: Original, state: str, keep: dict[str, Tensor] | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Verdict score, KL(original || state) and digit mass per statement."""
        cis = p1.cis
        assert cis is not None

        def edit(ib: int, bt: Batch) -> AbstractContextManager[None]:
            return edited_outputs(model, basis, bt, partial(state_mask, state, cis[ib], keep))

        return run_edit(model, batches, p1, edit, digits, config.autocast_bf16)

    def keep_top(order: list[int], m_top: int) -> dict[str, Tensor]:
        keep = {m: torch.zeros(n_c[m], device=device) for m in modules}
        for i in order[:m_top]:
            m, c = flat[i]
            keep[m][c] = 1.0
        return keep

    # ---- 1./2. val: ranking, active-set curve, CI values ----
    val = maybe_cap(load_statements("val", task.prepared_datasets), n_per_dataset, seed)
    train_prompt = prompts[0]
    val_batches = make_batches(
        val.texts, tok, task.loss_positions, train_prompt, batch_size, device
    )
    val_p1 = pass1(val_batches)
    val_cis = val_p1.cis
    assert val_cis is not None
    ci_sum = torch.zeros(len(flat), device=device)
    n_pos = 0
    ci_values = {m: [] for m in modules}
    for ib, bt in enumerate(val_batches):
        n_pos += int(bt.dec.sum())
        ci_sum += torch.cat([val_cis[ib][m][bt.dec].sum(0) for m in modules])
        for m in modules:
            ci_values[m].append(val_cis[ib][m][bt.dec].half().cpu())
    mean_ci = (ci_sum / n_pos).cpu().numpy()
    order = [int(i) for i in np.argsort(-mean_ci, kind="stable")]
    _, kl_off_val, _ = run_state(val_batches, val_p1, "all_off", None)
    _, kl_all_val, _ = run_state(val_batches, val_p1, "components_ci", None)
    rec_all = 1 - kl_all_val.mean() / kl_off_val.mean()
    print(
        f"val ({len(val)} statements): all components (mask = CI, delta off) recover {rec_all:.4f}"
    )

    curve: dict[int, float] = {}

    def curve_at(m_top: int) -> float:
        if m_top not in curve:
            _, k, _ = run_state(val_batches, val_p1, "active_ci", keep_top(order, m_top))
            curve[m_top] = 1 - float(k.mean()) / float(kl_off_val.mean())
            print(f"  top {m_top:4d} by mean CI: KL recovered {curve[m_top]:.4f}", flush=True)
        return curve[m_top]

    active: dict[str, Any] | None = None
    if rec_all <= 0:
        print(f"no active set: all components together recover {rec_all:.4f} <= 0")
    else:
        target = TARGET_SHARE * rec_all
        for m_top in (*M_GRID, len(flat)):
            if m_top <= len(flat) and curve_at(m_top) >= target:
                break
        # bisection between the last grid point below the target and the first reaching it
        lo = max([m for m, v in curve.items() if v < target], default=0)
        hi = smallest_reaching(curve, target)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if curve_at(mid) >= target:
                hi = mid
            else:
                lo = mid
        m_star = smallest_reaching(curve, target)
        names = [f"L{layer[flat[i][0]]}:{flat[i][1]}" for i in order[:m_star]]
        active = {
            "m": m_star,
            "target": target,
            "kl_recovered_all": rec_all,
            "kl_recovered_active": curve[m_star],
            "components": names,
            "curve": {str(k): v for k, v in sorted(curve.items())},
        }
        print(
            f"active set: {m_star} components reach {curve[m_star]:.4f} >= {target:.4f}: {names[:20]}"
        )
    keep_s = keep_top(order, active["m"]) if active is not None else None

    ci_cat = {m: torch.cat(ci_values[m]).numpy() for m in modules}
    print(
        "\nCI values at the decomposed val positions (valley check): counts per bin, all layers pooled"
    )
    allv = np.concatenate([v.astype(np.float32).ravel() for v in ci_cat.values()])
    print(f"  exactly 0: {(allv == 0).sum()}, exactly 1: {(allv == 1).sum()}, total {allv.size}")
    lin = np.histogram(allv[(allv > 0) & (allv < 1)], bins=20, range=(0, 1))[0]
    print("  (0, 1) in 20 linear bins: " + " ".join(str(int(x)) for x in lin))
    logb = np.histogram(allv[(allv > 0) & (allv < 1)], bins=np.logspace(-6, 0, 13))[0]
    print("  (1e-6, 1) in 12 log bins: " + " ".join(str(int(x)) for x in logb))

    # ---- 3. test / OOD under both prompts ----
    summary: dict[str, Any] = {
        "run": run,
        "n_per_dataset": n_per_dataset,
        "target_share": TARGET_SHARE,
        "active_set": active,
        "mean_ci_val": {f"L{layer[m]}:{c}": float(mean_ci[i]) for i, (m, c) in enumerate(flat)},
        "sets": {},
    }
    arrays: dict[str, np.ndarray] = {}
    for split in split_list:
        st = maybe_cap(load_statements(split, task.prepared_datasets), n_per_dataset, seed)
        arrays[f"{split}_labels"] = st.labels
        arrays[f"{split}_datasets"] = st.datasets
        summary["sets"][split] = {}
        for prompt in prompts:
            batches = make_batches(st.texts, tok, task.loss_positions, prompt, batch_size, device)
            p1 = pass1(batches)
            base = (p1.base_lp[:, one] - p1.base_lp[:, zero]).numpy()
            base_mass = (p1.base_lp[:, one].exp() + p1.base_lp[:, zero].exp()).numpy()
            res: dict[str, Any] = {
                "n": len(st),
                "original": {
                    "accuracy": float(((base > 0) == st.labels).mean()),
                    "digit_mass": float(base_mass.mean()),
                },
                "states": {},
            }
            runs = {}
            for state in STATES:
                if state in NEEDS_ACTIVE_SET and keep_s is None:
                    continue
                runs[state] = run_state(batches, p1, state, keep_s)
                arrays[f"{split}_{prompt.name}_{state}_score"] = runs[state][0]
                arrays[f"{split}_{prompt.name}_{state}_kl"] = runs[state][1]
            arrays[f"{split}_{prompt.name}_original_score"] = base
            full_score = runs["full"][0]
            assert ((full_score > 0) == (base > 0)).mean() > 0.99, "full state does not reproduce"
            print(f"{split}/{prompt.name}: full state max |score - original| "
                  f"{np.abs(full_score - base).max():.3f}")  # fmt: skip
            kl_off = runs["all_off"][1]
            for state, (score, kl, mass) in runs.items():
                res["states"][state] = summarize_state(
                    score, kl, mass, base, kl_off, st.labels, st.families, N_BOOT, seed
                )
                r = res["states"][state]
                print(
                    f"  {state:22s} KL rec. pooled {r['pooled']['kl_recovered']:7.3f} macro "
                    f"{r['macro']['kl_recovered']:7.3f} [{r['macro_interval']['kl_recovered'][0]:.3f}, "
                    f"{r['macro_interval']['kl_recovered'][1]:.3f}] | acc {r['macro']['accuracy']:.3f} "
                    f"agree {r['macro']['agreement']:.3f} sep {r['macro']['sep_share']:.3f} "
                    f"digit {r['macro']['digit_mass']:.3f}",
                    flush=True,
                )
            res["ci_sum_per_statement"], res["layers"] = ci_counts_and_norms(
                model, basis, batches, p1, keep_s, len(st), config.autocast_bf16
            )
            arrays[f"{split}_{prompt.name}_ci_sum"] = np.array(
                res["ci_sum_per_statement"]["values"]
            )
            del res["ci_sum_per_statement"]["values"]
            print(f"  sum of CIs per statement: mean {res['ci_sum_per_statement']['mean']:.2f}")
            for m, row in res["layers"].items():
                print(f"  {m}: " + " ".join(f"{k} {v:.3f}" for k, v in row.items()))
            summary["sets"][split][prompt.name] = res

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"recovery_{run_id}.json").write_text(json.dumps(summary, indent=1))
    np.savez(out_dir / f"recovery_{run_id}.npz", **arrays)  # pyright: ignore[reportArgumentType]
    np.savez(
        out_dir / f"ci_val_{run_id}.npz",
        **{f"L{layer[m]}": v for m, v in ci_cat.items()},  # pyright: ignore[reportArgumentType]
        mean_ci=mean_ci,
    )


def state_mask(
    state: str,
    ci: dict[str, Tensor],
    keep: dict[str, Tensor] | None,
    m: str,
    _bt: Batch,
    acts: Tensor,
    _y: Tensor,
) -> tuple[Tensor, bool]:
    """MaskFn of one verdict state (module docstring): per-component mask and whether delta stays."""
    ones = torch.ones_like(acts)
    match state:
        case "full":
            return ones, True
        case "components_ci":
            return ci[m], False
        case "active_ci":
            assert keep is not None
            return ci[m] * keep[m], False
        case "mean_mask":
            return ci[m] + (1 - ci[m]) / 2, False
        case "delta_only":
            return 0 * ones, True
        case "all_off":
            return 0 * ones, False
        case "full_without_inactive":
            assert keep is not None
            return ones * keep[m], True
        case _:
            raise AssertionError(state)


def ci_counts_and_norms(
    model: ComponentModel,
    basis: Basis,
    batches: list[Batch],
    p1: Original,
    keep: dict[str, Tensor] | None,
    n: int,
    autocast: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Sum of CIs per statement (over decomposed positions and layers) and, per layer, the CI-weighted
    writes of the active set (A) and of the other components (I) with the cancellation diagnostics.
    Norm ratios are mean |.| over positions divided by mean |y|, as in verdict_lens_analysis.py."""
    assert p1.cis is not None
    ci_sum = np.zeros(n)
    acc: dict[str, dict[str, list[Tensor]]] = {m: {} for m in basis.U}
    for ib, bt in enumerate(batches):
        with bf16_autocast(enabled=autocast):
            o = model(bt.ids, cache_type="input")
        for m in basis.U:
            ci = p1.cis[ib][m]
            ci_sum[bt.rows] += (ci * bt.dec[..., None]).sum((1, 2)).cpu().numpy()
            x = o.cache[m][bt.dec].float()
            acts, ci_d = x @ basis.V[m], ci[bt.dec]
            k = keep[m] if keep is not None else torch.zeros_like(ci_d[0])
            A = (acts * ci_d * k) @ basis.U[m]
            I = (acts * ci_d * (1 - k)) @ basis.U[m]  # noqa: E741
            C = acts @ basis.U[m]
            y = x @ model.target_weight(m).float().T
            D = y - C
            for name, v in (
                ("y", y.norm(dim=-1)), ("A", A.norm(dim=-1)), ("I", I.norm(dim=-1)),
                ("C", C.norm(dim=-1)), ("D", D.norm(dim=-1)),
                ("cos_AI", F.cosine_similarity(A, I, dim=-1)),
                ("inner_CD", (C * D).sum(-1) / y.pow(2).sum(-1)),
            ):  # fmt: skip
                acc[m].setdefault(name, []).append(v.cpu())
    layers = {}
    for m, d in acc.items():
        cat = {k: torch.cat(v) for k, v in d.items()}
        y_mean = float(cat["y"].mean())
        row = {f"{k}/y": float(cat[k].mean()) / y_mean for k in ("A", "I", "C", "D")}
        if keep is None:
            row = {k: v for k, v in row.items() if k in ("C/y", "D/y")}
        else:
            row["cos_AI"] = float(cat["cos_AI"].nanmean())
        row["inner_CD"] = float(cat["inner_CD"].mean())
        layers[f"L{m.split('.')[2]}"] = row
    return {"mean": float(ci_sum.mean()), "values": ci_sum.tolist()}, layers


if __name__ == "__main__":
    fire.Fire(main)
