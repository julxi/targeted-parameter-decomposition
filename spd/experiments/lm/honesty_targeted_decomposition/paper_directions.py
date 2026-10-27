"""Truth directions vs the decomposition's writes (paper topic convos/julian/26-10-22_tpd_truth_paper_LOG.md,
frozen plan D3, section 5, Julian's comment under *D3 proposal*). Shared code and conventions:
paper_common.py; bases as in paper_termination.py.

Directions, per decomposed layer lay, in the residual stream after layer lay (where that layer's MLP
writes) at the statement's last token, training prompt: mass-mean (mean true - mean false, unit) and
logistic regression (standardised features, L2 strength chosen on val accuracy from LOGREG_CS, the
coefficients mapped back to raw space, unit; as A4, ~/spd_out/26-10-22_tpd_truth_paper/mm_vs_lr.py),
both fit on train. Because the statement comes first, the residuals at the statement tokens equal
those of the bare statement except at the last token ('.\\n' instead of '.').

Measured on test, at the statement's last token, per layer and direction d:
- per component: cos(u_c, d); baseline: the |cos| of N_RANDOM_VECTORS random directions.
- share of the truth gap: [mean over true - mean over false of (X . d)] / [the same for y], for
  X = A (the CI-weighted writes of all components; for the data-aligned basis all writes at mask 1),
  I = C - A (the rest of the components' writes), D = y - C (the remainder), and each component's
  CI-weighted write alone (A + I + D = y, so the three parts sum to 1).
- link to function (with --termination): the val single-removal drop (T1) of each candidate next to
  its gap share and cos, and their Spearman correlation over the candidates.
- with --remove_directions (basis-independent, so needed once per run): the verdict on test when d
  alone is projected out of the ORIGINAL MLP output at the decomposed positions, per layer and at all
  layers together; compare with removing the components (paper_termination.py, T2).

Output in --out: directions_<run id>_<basis>.json, .npz (the directions, per-component tables).

Usage: python paper_directions.py --run <s-id or checkpoint> --basis decomposition --out <dir>
       [--termination <json>] [--remove_directions]
"""

import json
from pathlib import Path
from typing import Any, Literal

import fire
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig, StimulusLens
from spd.experiments.lm.honesty_targeted_decomposition.paper_common import (
    Basis,
    Batch,
    StatementSplit,
    load_statements,
    make_basis,
    make_batches,
    maybe_cap,
    original_pass,
    projected_out,
    run_edit,
    sorted_modules,
    summarize_state,
    verdict_prompts,
)
from spd.experiments.lm.honesty_targeted_decomposition.probe_ci import WANDB_PREFIX
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.general_utils import bf16_autocast

LOGREG_CS = (1e-4, 1e-3, 1e-2, 0.1, 1.0)
N_RANDOM_VECTORS = 1000
N_BOOT = 1000


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def gap(proj: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean over true minus mean over false statements, along axis 0."""
    return proj[labels].mean(0) - proj[~labels].mean(0)


def last_token_parts(
    model: ComponentModel,
    bas: Basis,
    batches: list[Batch],
    n: int,
    with_ci: bool,
    sampling: Literal["continuous", "binomial"],
    autocast: bool,
) -> dict[str, np.ndarray]:
    """At each statement's last token, per decomposed layer: the residual after that layer ("resid_<layer>"), the MLP
    output y, the CI-weighted writes A, all writes C (mask 1), and the per-component CI-weighted
    coefficients ("coef_L", (n, C)). For the data-aligned basis CI = 1, so A = C."""
    modules = list(bas.U)
    layers = {m: int(m.split(".")[2]) for m in modules}
    resid: dict[int, Tensor] = {}
    handles = []
    for lay in layers.values():

        def grab(
            _mod: torch.nn.Module,
            _args: tuple[Tensor, ...],
            output: Tensor | tuple[Tensor, ...],
            lay: int = lay,
        ) -> None:
            resid[lay] = (output[0] if isinstance(output, tuple) else output).float()

        handles.append(
            model.target_model.get_submodule(f"model.layers.{lay}").register_forward_hook(grab)
        )
    out: dict[str, list[np.ndarray]] = {}
    try:
        for bt in batches:
            with bf16_autocast(enabled=autocast):
                o = model(bt.ids, cache_type="input")
                ci = (
                    model.calc_causal_importances(
                        pre_weight_acts=o.cache, detach_inputs=True, sampling=sampling
                    ).lower_leaky
                    if with_ci
                    else None
                )
            idx = torch.arange(bt.dec.shape[1], device=bt.dec.device).expand_as(bt.dec)
            last = (bt.dec * (idx + 1)).max(1).values - 1
            r = torch.arange(len(last), device=last.device)
            for m, lay in layers.items():
                x = o.cache[m][r, last].float()  # (b, d_in)
                acts = x @ bas.V[m]
                coef = acts * ci[m][r, last].float() if ci is not None else acts
                parts = {
                    f"resid_{lay}": resid[lay][r, last],
                    f"y_{lay}": x @ model.target_weight(m).float().T,
                    f"A_{lay}": coef @ bas.U[m],
                    f"C_{lay}": acts @ bas.U[m],
                    f"coef_{lay}": coef,
                }
                for k, v in parts.items():
                    out.setdefault(k, []).append(v.cpu().numpy())
    finally:
        for h in handles:
            h.remove()
    res = {k: np.concatenate(v) for k, v in out.items()}
    assert all(len(v) == n for v in res.values())
    return res


def directions(
    tr: np.ndarray, ytr: np.ndarray, va: np.ndarray, yva: np.ndarray
) -> tuple[dict[str, np.ndarray], float]:
    """Mass-mean and logistic-regression directions (unit, raw space) and the chosen L2 strength."""
    best: tuple[float, float, np.ndarray] | None = None
    for c in LOGREG_CS:
        sc = StandardScaler().fit(tr)
        lr = LogisticRegression(C=c, max_iter=5000).fit(sc.transform(tr), ytr)
        acc = float((lr.predict(sc.transform(va)) == yva).mean())
        if best is None or acc > best[0]:
            best = (acc, c, unit(lr.coef_[0] / sc.scale_))
    assert best is not None
    return {"mm": unit(gap(tr, ytr)), "lr": best[2]}, best[1]


def main(
    run: str,
    basis: Literal["decomposition", "data_aligned"],
    out: str,
    termination: str | None = None,
    remove_directions: bool = False,
    n_per_dataset: int | None = None,
    batch_size: int = 64,
    device: str = "cuda",
    seed: int = 0,
) -> None:
    torch.set_grad_enabled(False)
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
    layers = [int(m.split(".")[2]) for m in modules]
    prompt = verdict_prompts(lens, tok, task.max_seq_len)[0]

    def batches_of(texts: list[str]) -> list[Batch]:
        return make_batches(texts, tok, task.loss_positions, prompt, batch_size, device)

    splits: tuple[StatementSplit, ...] = ("train", "val", "test")
    sets = {
        s: maybe_cap(load_statements(s, task.prepared_datasets), n_per_dataset, seed)
        for s in splits
    }
    bas = make_basis(basis, model, run_id, task.prepared_datasets, batches_of, autocast, seed)
    names = {
        lay: [f"L{lay}:{c}" for c in range(bas.U[m].shape[0])]
        for m, lay in zip(modules, layers, strict=True)
    }
    P = {
        s: last_token_parts(
            model,
            bas,
            batches_of(st.texts),
            len(st),
            basis == "decomposition",
            config.sampling,
            autocast,
        )  # fmt: skip
        for s, st in sets.items()
    }
    rng = np.random.default_rng(seed)
    d_model = P["test"][f"y_{layers[0]}"].shape[1]
    rand = rng.normal(size=(N_RANDOM_VECTORS, d_model))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    t1 = None
    if termination is not None:
        term = json.loads(Path(termination).read_text())
        assert term["run"] == run and term["basis"] == bas.name, (term["run"], term["basis"])
        t1 = {nm: 1 - v["sep_share"] for nm, v in term["t1_val"].items()}

    yte = sets["test"].labels
    summary: dict[str, Any] = {"run": run, "basis": bas.name, "layers": {}}
    arrays: dict[str, np.ndarray] = {}
    dirs: dict[int, dict[str, np.ndarray]] = {}
    for m, lay in zip(modules, layers, strict=True):
        dirs[lay], c_best = directions(
            P["train"][f"resid_{lay}"],
            sets["train"].labels,
            P["val"][f"resid_{lay}"],
            sets["val"].labels,
        )
        U = bas.U[m].cpu().numpy()
        U_unit = U / np.linalg.norm(U, axis=1, keepdims=True)
        row: dict[str, Any] = {
            "logreg_C": c_best,
            "cos_mm_lr": float(dirs[lay]["mm"] @ dirs[lay]["lr"]),
        }
        y, A, C = (P["test"][f"{k}_{lay}"] for k in ("y", "A", "C"))
        for dn, d in dirs[lay].items():
            arrays[f"dir_{dn}_L{lay}"] = d
            cos = U_unit @ d
            gy = gap(y @ d, yte)
            per_comp = gap(P["test"][f"coef_{lay}"] * (U @ d)[None, :], yte) / gy
            arrays[f"cos_{dn}_L{lay}"], arrays[f"gapshare_{dn}_L{lay}"] = cos, per_comp
            r: dict[str, Any] = {
                "gap_y": float(gy),
                "share_A": float(gap(A @ d, yte) / gy),
                "share_I": float(gap((C - A) @ d, yte) / gy),
                "share_D": float(gap((y - C) @ d, yte) / gy),
                "max_abs_cos": float(np.abs(cos).max()),
                "mean_abs_cos": float(np.abs(cos).mean()),
                "random_mean_abs_cos": float(np.abs(rand @ d).mean()),
                "random_max_abs_cos_per_C": float(
                    np.mean(
                        [
                            np.abs(rand[i : i + len(cos)] @ d).max()
                            for i in range(0, N_RANDOM_VECTORS - len(cos) + 1, len(cos))
                        ]
                    )
                ),  # fmt: skip
                "top_gap_share": sorted(
                    zip(names[lay], per_comp.tolist(), strict=True), key=lambda kv: -abs(kv[1])
                )[:10],
            }
            if t1 is not None:
                cand = [i for i, nm in enumerate(names[lay]) if nm in t1]
                if len(cand) >= 3:
                    drop = np.array([t1[names[lay][i]] for i in cand])
                    r["spearman_t1_drop_vs_abs_gapshare"] = float(
                        spearmanr(drop, np.abs(per_comp[cand])).statistic  # pyright: ignore[reportAttributeAccessIssue]
                    )
                    r["t1_candidates"] = {
                        names[lay][i]: {
                            "t1_drop": float(t1[names[lay][i]]),
                            "cos": float(cos[i]),
                            "gap_share": float(per_comp[i]),
                        }  # fmt: skip
                        for i in cand
                    }
            row[dn] = r
            print(f"L{lay} {dn}: gap {gy:7.2f} shares A {r['share_A']:+.3f} I {r['share_I']:+.3f} "
                  f"D {r['share_D']:+.3f} | max|cos| {r['max_abs_cos']:.3f} (random max over "
                  f"{len(cos)}: {r['random_max_abs_cos_per_C']:.3f})", flush=True)  # fmt: skip
        summary["layers"][f"L{lay}"] = row

    if remove_directions:
        st = sets["test"]
        batches = batches_of(st.texts)
        orig = original_pass(model, batches, autocast, None)
        base = (orig.base_lp[:, digits[0]] - orig.base_lp[:, digits[1]]).numpy()

        # KL reference: all decomposed MLP outputs off at the statement tokens (as everywhere else),
        # written as the projection onto the whole output space
        q_full = {m: torch.eye(bas.U[m].shape[1], device=device) for m in modules}
        _, kl_off, _ = run_edit(model, batches, orig, lambda _i, bt: projected_out(model, bt, q_full),
                                digits, autocast)  # fmt: skip
        summary["direction_removal"] = {}
        for dn in ("mm", "lr"):
            qs = {
                m: torch.tensor(dirs[lay][dn], device=device, dtype=torch.float32)[:, None]
                for m, lay in zip(modules, layers, strict=True)
            }
            conds = {f"{dn} L{lay}": {m: qs[m]} for m, lay in zip(modules, layers, strict=True)}
            conds[f"{dn} all layers"] = qs
            for name, q in conds.items():
                s, k, ms = run_edit(model, batches, orig,
                                    lambda _i, bt, q=q: projected_out(model, bt, q), digits, autocast)  # fmt: skip
                r = summarize_state(s, k, ms, base, kl_off, st.labels, st.families, N_BOOT, seed)
                summary["direction_removal"][name] = r
                print(f"  remove {name:14s} sep {r['macro']['sep_share']:.3f} 1->0 "
                      f"{r['macro']['flip_1_to_0']:.3f} 0->1 {r['macro']['flip_0_to_1']:.3f}", flush=True)  # fmt: skip

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"directions_{run_id}_{basis}"
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    np.savez(out_dir / f"{stem}.npz", **arrays)  # pyright: ignore[reportArgumentType]


if __name__ == "__main__":
    fire.Fire(main)
