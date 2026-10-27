"""Correlation analyses of one basis (paper topic convos/julian/26-10-22_tpd_truth_paper_LOG.md, frozen
plan D3, section 3: K0-K3). Shared code and conventions: paper_common.py. Bases as in
paper_termination.py (--basis decomposition for a trained or the untrained run, data_aligned).

Per statement and component, under the training prompt, two readouts (as the arm A probes,
probe_ci.py): the value at the statement's last token and the mean over the statement tokens. Values:
  coef  the component's coefficient in the forward pass: CI_c (v_c . x) for a decomposition, v_c . x
        for the data-aligned basis (which has no CIs; its mask is 1). Comparable across bases.
  ci    the CI itself (decompositions only).
Plus the residual stream after the last decomposed layer (the activation baseline of K0).

K0  probes over ALL features of a kind (Julian's comment under *D3 proposal*, section 3: shown as the
    reason probe accuracy is not tPD-specific): logistic regression on standardised features (default
    L2, as probe_ci.py), fit on train, scored on test and OOD; targets the ground-truth label and the
    model's own verdict.
K1  sparsity: AUC of every single component within each family (label and verdict; saved in full for
    browsing), with the readout chosen on train; probes on the top-k components by their train
    selectivity (k in K_SPARSE), scored on test and OOD. Comparable across bases at the same k.
K2  transfer: components selective on val within a sentence form (|macro AUC - 0.5| >= SELECT, the
    readout chosen on train) keep their sign on the OOD families of the matching form (negated ->
    negated; affirmative -> affirmative, conjunction, disjunction) with |AUC - 0.5| >= TRANSFER. Share
    of (component, OOD family) pairs that transfer; ground-truth label on GROUND_TRUTH_OOD only,
    verdict on all 24 files.
K3  truth vs verdict: on the test and OOD statements where the model's verdict contradicts the label,
    in the forms matching a K2-selective component, does the component follow the label (oriented AUC
    against the label > 0.5) or the verdict (< 0.5)? On those statements label = not verdict, so one
    AUC answers both.
Thresholds SELECT and TRANSFER are part of the frozen plan (K2 as drafted in *D3 proposal*).

Output in --out: correlation_<run id>_<basis>.json, correlation_<run id>_<basis>.npz (all features
and the single-component AUC tables).

Usage: python paper_correlation.py --run <s-id or checkpoint> --basis decomposition --out <dir>
       [--ood_cap N]
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import fire
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig, StimulusLens
from spd.experiments.lm.honesty_targeted_decomposition.paper_common import (
    GROUND_TRUTH_OOD,
    Basis,
    Batch,
    Statements,
    StatementSplit,
    auc_columns,
    cap_per_dataset,
    load_statements,
    make_basis,
    make_batches,
    maybe_cap,
    sorted_modules,
    verdict_logprobs,
    verdict_prompts,
)
from spd.experiments.lm.honesty_targeted_decomposition.probe_ci import WANDB_PREFIX
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.general_utils import bf16_autocast

K_SPARSE = (1, 2, 5, 10)
SELECT = 0.2
TRANSFER = 0.1
READOUTS = ("last", "mean")
SPLITS: tuple[StatementSplit, ...] = ("train", "val", "test", "ood")
# OOD forms a val form's selectivity is expected to carry over to (K2)
TRANSFER_FORMS = {
    "negated": ("negated",),
    "affirmative": ("affirmative", "conjunction", "disjunction"),
}


def statement_readouts(v: Tensor, dec: Tensor) -> dict[str, np.ndarray]:
    """(b, pos, k) values -> per statement the value at the last decomposed position (the
    statement's last token) and the mean over the decomposed positions, each (b, k)."""
    idx = torch.arange(dec.shape[1], device=dec.device).expand_as(dec)
    last = (dec * (idx + 1)).max(1).values - 1
    assert bool(dec[torch.arange(len(last), device=dec.device), last].all())
    d = dec.float()[..., None]
    return {
        "last": v[torch.arange(len(last), device=dec.device), last].cpu().numpy(),
        "mean": ((v * d).sum(1) / d.sum(1)).cpu().numpy(),
    }


def extract(
    model: ComponentModel,
    bas: Basis,
    batches: list[Batch],
    n: int,
    with_ci: bool,
    sampling: Literal["continuous", "binomial"],
    autocast: bool,
    digits: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Per statement: coef/ci readouts (n, K), residual readouts (n, d) after the last decomposed
    layer, and the verdict score."""
    modules = list(bas.U)
    last_layer = f"model.layers.{max(int(m.split('.')[2]) for m in modules)}"
    resid: list[Tensor] = []

    def grab(
        _mod: torch.nn.Module, _args: tuple[Tensor, ...], output: Tensor | tuple[Tensor, ...]
    ) -> None:
        resid.append((output[0] if isinstance(output, tuple) else output).float())

    handle = model.target_model.get_submodule(last_layer).register_forward_hook(grab)
    feats: dict[str, list[np.ndarray]] = {}
    score = np.zeros(n)
    try:
        for bt in batches:
            resid.clear()
            with bf16_autocast(enabled=autocast):
                o = model(bt.ids, cache_type="input")
                ci = (
                    model.calc_causal_importances(
                        pre_weight_acts=o.cache, detach_inputs=True, sampling=sampling
                    ).lower_leaky
                    if with_ci
                    else None
                )
            lp = verdict_logprobs(o.output, bt.end)
            score[bt.rows] = (lp[:, digits[0]] - lp[:, digits[1]]).cpu().numpy()
            coef = [o.cache[m].float() @ bas.V[m] for m in modules]
            if ci is not None:
                cis = [ci[m].float() for m in modules]
                coef = [a * c for a, c in zip(coef, cis, strict=True)]
                for k, v in statement_readouts(torch.cat(cis, -1), bt.dec).items():
                    feats.setdefault(f"ci_{k}", []).append(v)
            for k, v in statement_readouts(torch.cat(coef, -1), bt.dec).items():
                feats.setdefault(f"coef_{k}", []).append(v)
            assert len(resid) == 1, len(resid)
            for k, v in statement_readouts(resid[0], bt.dec).items():
                feats.setdefault(f"resid_{k}", []).append(v)
    finally:
        handle.remove()
    out: dict[str, np.ndarray] = {k: np.concatenate(v).astype(np.float32) for k, v in feats.items()}
    out["score"] = score
    return out


def macro_auc(X: np.ndarray, y: np.ndarray, families: np.ndarray) -> np.ndarray:
    """Per column: AUC within each family, averaged over families (families need both classes)."""
    fams = [f for f in sorted(set(families)) if 0 < y[families == f].sum() < (families == f).sum()]
    assert fams, "no family with both classes"
    return np.mean([auc_columns(X[families == f], y[families == f]) for f in fams], axis=0)


def probe_scores(
    Xtr: np.ndarray, ytr: np.ndarray, tests: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> dict[str, dict[str, float]]:
    """Fit on train, score each test set: pooled accuracy and AUC, family-macro accuracy."""
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    probe.fit(Xtr, ytr)
    res = {}
    for name, (X, y, fam) in tests.items():
        pred, p = probe.predict(X), probe.predict_proba(X)[:, 1]
        res[name] = {
            "accuracy": float((pred == y).mean()),
            "auc": float(roc_auc_score(y, p)),
            "macro_accuracy": float(
                np.mean([(pred == y)[fam == f].mean() for f in sorted(set(fam))])
            ),
        }
    return res


def main(
    run: str,
    basis: Literal["decomposition", "data_aligned"],
    out: str,
    ood_cap: int | None = None,
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
    prompt = verdict_prompts(lens, tok, task.max_seq_len)[0]  # correlation: training prompt only

    def batches_of(texts: list[str]) -> list[Batch]:
        return make_batches(texts, tok, task.loss_positions, prompt, batch_size, device)

    sets: dict[str, Statements] = {
        s: maybe_cap(load_statements(s, task.prepared_datasets), n_per_dataset, seed)
        for s in SPLITS
    }
    if ood_cap is not None:
        sets["ood"] = cap_per_dataset(sets["ood"], ood_cap, seed)
    bas = make_basis(basis, model, run_id, task.prepared_datasets, batches_of, autocast, seed)
    names = [f"L{m.split('.')[2]}:{c}" for m in modules for c in range(bas.U[m].shape[0])]
    with_ci = basis == "decomposition"
    F = {
        s: extract(
            model, bas, batches_of(st.texts), len(st), with_ci, config.sampling, autocast, digits
        )  # fmt: skip
        for s, st in sets.items()
    }
    verdict = {s: F[s]["score"] > 0 for s in sets}
    for s, st in sets.items():
        print(f"{s}: {len(st)} statements, model accuracy {(verdict[s] == st.labels).mean():.3f}")
    strict = np.isin(sets["ood"].families, GROUND_TRUTH_OOD)
    kinds = ["coef", *(["ci"] if with_ci else [])]
    summary: dict[str, Any] = {"run": run, "basis": bas.name, "components": names}
    arrays: dict[str, np.ndarray] = {f"{s}_{k}": v for s in sets for k, v in F[s].items()}

    def tests_for(get: Callable[[str], np.ndarray], target: str) -> dict[str, Any]:
        """Probe on the features get(split): fit on train; test, and OOD (ground-truth label: the
        GROUND_TRUTH_OOD files only; verdict: all 24)."""
        y = {s: sets[s].labels if target == "label" else verdict[s] for s in sets}
        ood_rows = strict if target == "label" else np.ones(len(sets["ood"]), bool)
        tests = {
            "test": (get("test"), y["test"], sets["test"].families),
            "ood": (get("ood")[ood_rows], y["ood"][ood_rows], sets["ood"].families[ood_rows]),
        }
        return probe_scores(get("train"), y["train"], tests)

    # ---- K0 ----
    summary["K0"] = {
        f"{kind}_{r}": {
            t: tests_for(lambda s, fs=f"{kind}_{r}": F[s][fs], t) for t in ("label", "verdict")
        }
        for kind in [*kinds, "resid"]
        for r in READOUTS
    }
    for fs, v in summary["K0"].items():
        print(f"K0 {fs:10s} label test {v['label']['test']['accuracy']:.3f} ood "
              f"{v['label']['ood']['accuracy']:.3f} | verdict test {v['verdict']['test']['accuracy']:.3f} "
              f"ood {v['verdict']['ood']['accuracy']:.3f}")  # fmt: skip

    # ---- K1-K3 per value kind ----
    for kind in kinds:
        tr = sets["train"]
        auc_tr = {r: macro_auc(F["train"][f"{kind}_{r}"], tr.labels, tr.families) for r in READOUTS}
        best_r = np.where(np.abs(auc_tr["last"] - 0.5) >= np.abs(auc_tr["mean"] - 0.5), 0, 1)

        def chosen(s: str, best_r: np.ndarray = best_r, kind: str = kind) -> np.ndarray:
            """(n, K) values of each component in the readout chosen on train."""
            return np.where(best_r[None, :] == 0, F[s][f"{kind}_last"], F[s][f"{kind}_mean"])

        for s in sets:
            arrays[f"{s}_{kind}_chosen"] = chosen(s)
        sel_tr = np.abs(np.where(best_r == 0, auc_tr["last"], auc_tr["mean"]) - 0.5)
        order = np.argsort(-sel_tr, kind="stable")
        k1: dict[str, Any] = {
            "readout": [READOUTS[i] for i in best_r],
            "train_selectivity": sel_tr.tolist(),
        }
        # single-component AUC per family on test and OOD (wide table, saved as arrays)
        for s in ("test", "ood"):
            st = sets[s]
            fams = sorted(set(st.families))
            for target, y in (("label", st.labels), ("verdict", verdict[s])):
                tab = np.full((len(fams), len(names)), np.nan)
                for i, f in enumerate(fams):
                    rows = st.families == f
                    if 0 < y[rows].sum() < rows.sum():
                        tab[i] = auc_columns(chosen(s)[rows], y[rows])
                arrays[f"K1_{kind}_{s}_{target}_auc"] = tab
                arrays[f"K1_{kind}_{s}_families"] = np.array(fams)
        k1["sparse_probes"] = {}
        for k in K_SPARSE:
            cols = order[:k]
            k1["sparse_probes"][str(k)] = {
                "components": [names[i] for i in cols],
                **{
                    t: tests_for(lambda s, cols=cols, chosen=chosen: chosen(s)[:, cols], t)
                    for t in ("label", "verdict")
                },
            }
            r = k1["sparse_probes"][str(k)]
            print(f"K1 {kind} top-{k:2d} probe: label test {r['label']['test']['macro_accuracy']:.3f} "
                  f"ood {r['label']['ood']['macro_accuracy']:.3f} | verdict test "
                  f"{r['verdict']['test']['macro_accuracy']:.3f} ood "
                  f"{r['verdict']['ood']['macro_accuracy']:.3f}")  # fmt: skip
        summary[f"K1_{kind}"] = k1
        summary[f"K2_{kind}"], summary[f"K3_{kind}"] = transfer_and_dissociation(
            sets, verdict, chosen, names, strict
        )
        k2 = summary[f"K2_{kind}"]
        print(f"K2 {kind}: " + "; ".join(
            f"{form}: {len(v['selective'])} selective, transfer label {v['share_label']:.3f} "
            f"verdict {v['share_verdict']:.3f}" for form, v in k2.items()))  # fmt: skip
        k3 = summary[f"K3_{kind}"]
        print(f"K3 {kind}: " + "; ".join(
            f"{form}: n {v['n']}, follows label {v['share_follow_label']}" for form, v in k3.items()))  # fmt: skip

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"correlation_{run_id}_{basis}"
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    np.savez(out_dir / f"{stem}.npz", names=np.array(names), **arrays)  # pyright: ignore[reportArgumentType]


def transfer_and_dissociation(
    sets: dict[str, Statements],
    verdict: dict[str, np.ndarray],
    chosen: Callable[[str], np.ndarray],
    names: list[str],
    strict: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """K2 and K3 (module docstring) for one value kind; chosen(split) -> (n, K) values."""
    val, ood, test = sets["val"], sets["ood"], sets["test"]
    k2: dict[str, Any] = {}
    k3: dict[str, Any] = {}
    for form, targets in TRANSFER_FORMS.items():
        in_form = val.forms == form
        a_val = macro_auc(chosen("val")[in_form], val.labels[in_form], val.families[in_form])
        selective = np.flatnonzero(np.abs(a_val - 0.5) >= SELECT)
        sign = np.sign(a_val[selective] - 0.5)
        ood_rows = np.isin(ood.forms, targets)
        pairs: dict[str, list[bool]] = {"label": [], "verdict": []}
        detail = []
        for f in sorted(set(ood.families[ood_rows])):
            rows = ood.families == f
            for target, y in (("label", ood.labels), ("verdict", verdict["ood"])):
                if target == "label" and not strict[rows].all():
                    continue
                if not 0 < y[rows].sum() < rows.sum():
                    continue
                a = auc_columns(chosen("ood")[rows][:, selective], y[rows])
                ok = (np.sign(a - 0.5) == sign) & (np.abs(a - 0.5) >= TRANSFER)
                pairs[target] += ok.tolist()
                detail.append({"family": f, "target": target, "auc": a.tolist()})
        k2[form] = {
            "selective": [names[i] for i in selective],
            "val_auc": a_val[selective].tolist(),
            "share_label": float(np.mean(pairs["label"])) if pairs["label"] else float("nan"),
            "share_verdict": float(np.mean(pairs["verdict"])) if pairs["verdict"] else float("nan"),
            "n_pairs": {t: len(v) for t, v in pairs.items()},
            "detail": detail,
        }
        # K3: statements of the matching forms where the verdict contradicts the label
        X, y, wrong = [], [], []
        for s, st in (("test", test), ("ood", ood)):
            rows = np.isin(st.forms, (form, *targets)) & (verdict[s] != st.labels)
            X.append(chosen(s)[rows][:, selective])
            y.append(st.labels[rows])
            wrong.append(int(rows.sum()))
        Xd, yd = np.concatenate(X), np.concatenate(y)
        if len(selective) and 0 < yd.sum() < len(yd):
            oriented = np.where(sign > 0, auc_columns(Xd, yd), 1 - auc_columns(Xd, yd))
            k3[form] = {
                "n": int(len(yd)),
                "n_by_set": dict(zip(("test", "ood"), wrong, strict=True)),
                "oriented_auc": oriented.tolist(),
                "share_follow_label": float((oriented > 0.5).mean()),
            }
        else:
            k3[form] = {"n": int(len(yd)), "share_follow_label": None,
                        "reason": "no selective components or one class only"}  # fmt: skip
    return k2, k3


if __name__ == "__main__":
    fire.Fire(main)
