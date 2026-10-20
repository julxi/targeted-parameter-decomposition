"""Verdict and cancellation analysis of a decomposition trained with the statement-first verdict lens
(Exp 4a-4e: "<statement>\\nThe amount of truth in this statement is ", the model answers '1' or '0';
topics convos/julian/26-10-10_bolder_experiments_LOG.md, entry *Exp 4a/4b*, and
convos/julian/26-10-16_faithful_targeting_LOG.md). Reads the run's own config for the data, the
lens and the decomposed positions; writes <out>/verdict_<run id>_<split>.{json,npz}.

Verdict states: the decomposed down_proj outputs are replaced at the decomposed positions only,
everything else runs on the original weights.
  full           all components + delta (= the original weights through the decomposition)
  components     components with CI > ROUNDING_THRESHOLD at mask 1, the rest 0, delta off
  components_ci  components at mask = CI, delta off (the 4e tuning objective)
  delta_only     delta alone
  all_off        the decomposed MLP outputs set to 0 there
Per state: verdict score s = log P('1') - log P('0') after the final space; accuracy, AUC,
separation share (mean s true minus false, relative to the original), digit mass,
KL(original || state) over the vocabulary, and kl_recovered = 1 - KL / KL(all_off).

Cancellation test per decomposed module, at the decomposed positions, with x the module's input on
the original model: y = W x = A + I + D, A = active components (rounded CI), I = inactive
components, D = delta x, C = A + I = all components (so y - C = D). Reported relative to |y|:
|y - A|, cos(A, y), |A|, |I|, |D|, |C|, cos(I, D), cos(A, I), <C, D> / |y|^2 (the 4e penalty's
inner product; 0 when C is the projection of y onto C's direction), <A, y - A> / |y|^2 (the 4d
one), the foreign part |A - proj_y A| / |y|. The same for the CI-weighted split A_ci = sum_c CI_c
write_c, I_ci = C - A_ci. Norm ratios are mean |.| over positions divided by mean |y|, as in the
earlier Exp 4 analyses, so they are comparable with those.

Usage:
    python verdict_lens_analysis.py --run s-<id> --split val [--out <dir>] [--n_per_dataset N]
"""

import json
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from sklearn.metrics import roc_auc_score
from torch import Tensor
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig, StimulusLens
from spd.experiments.lm.honesty_targeted_decomposition.probe_ci import (
    WANDB_PREFIX,
    load_split_with_labels,
)
from spd.experiments.lm.prompts_dataset import load_prompts_dataset, trim_to_loss_positions
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import (
    ComponentsMaskInfo,
    LinearComponents,
    make_mask_infos,
    route_only_selected_positions,
)
from spd.utils.general_utils import POSITION_MASK_KEY, STIMULUS_END_KEY, bf16_autocast

# Same rule as the eval metric TargetReconLoss (rounding_threshold 0.01 in the Exp 4 configs)
ROUNDING_THRESHOLD = 0.01
STATES = ["full", "components", "components_ci", "delta_only", "all_off"]
# per-position quantities collected per module; norms are kept raw and divided by mean |y| at the end
PER_POSITION_KEYS = [
    "y", "A", "I", "D", "C", "A_ci", "I_ci",
    "rel_err", "rel_err_ci", "cos_Ay", "cos_Aci_y", "cos_Cy", "cos_ID", "cos_AI", "cos_AI_ci",
    "inner_C", "inner_A", "inner_A_ci", "foreign", "foreign_ci", "n_active", "frac_mid", "ptype",
]  # fmt: skip


def _cos(a: Float[Tensor, "n d"], b: Float[Tensor, "n d"]) -> list[float]:
    return F.cosine_similarity(a, b, dim=-1).tolist()


def _module_parts(
    x: Float[Tensor, "n d_in"],
    comp: LinearComponents,
    delta: Float[Tensor, "d_out d_in"],
    ci: Float[Tensor, "n C"],
) -> dict[str, list[float]]:
    """Per-position cancellation numbers of one module (fp32), keyed as PER_POSITION_KEYS."""
    x = x.float()
    acts = x @ comp.V.float()
    on = (ci > ROUNDING_THRESHOLD).float()
    U = comp.U.float()
    # A / I / C / D as in the docstring and the topic LOGs
    A, I, A_ci = (acts * on) @ U, (acts * (1 - on)) @ U, (acts * ci) @ U  # noqa: E741
    C = A + I
    D = x @ delta.float().T
    y = C + D
    y_sq = y.pow(2).sum(-1)
    assert (y_sq > 0).all(), "zero MLP output at a decomposed position"
    yn = y_sq.sqrt()

    def inner(a: Tensor, b: Tensor) -> list[float]:
        return ((a * b).sum(-1) / y_sq).tolist()

    def foreign(a: Tensor) -> list[float]:
        proj = ((a * y).sum(-1) / y_sq)[:, None] * y
        return ((a - proj).norm(dim=-1) / yn).tolist()

    n_on = on.sum(-1)
    mid = ((ci > ROUNDING_THRESHOLD) & (ci < 0.99)).sum(-1).float()
    I_ci = C - A_ci
    return {
        **{k: v.norm(dim=-1).tolist() for k, v in (("y", y), ("A", A), ("I", I), ("D", D),
                                                    ("C", C), ("A_ci", A_ci), ("I_ci", I_ci))},
        "rel_err": ((y - A).norm(dim=-1) / yn).tolist(),
        "rel_err_ci": ((y - A_ci).norm(dim=-1) / yn).tolist(),
        "cos_Ay": _cos(A, y),
        "cos_Aci_y": _cos(A_ci, y),
        "cos_Cy": _cos(C, y),
        "cos_ID": _cos(I, D),
        "cos_AI": _cos(A, I),
        "cos_AI_ci": _cos(A_ci, I_ci),
        "inner_C": inner(C, D),
        "inner_A": inner(A, y - A),
        "inner_A_ci": inner(A_ci, y - A_ci),
        "foreign": foreign(A),
        "foreign_ci": foreign(A_ci),
        "n_active": n_on.tolist(),
        # share of the active CIs that are not ~1: where the rounded and CI-weighted states differ
        "frac_mid": (mid / n_on.clamp(min=1)).tolist(),
    }  # fmt: skip


def main(
    run: str,
    split: str,
    out: str = "verdict_lens_out",
    n_per_dataset: int | None = None,
    batch_size: int = 64,
    device: str = "cuda",
) -> None:
    assert split in ("val", "test"), f"split must be val or test, got {split!r}"
    torch.set_grad_enabled(False)
    run_info = SPDRunInfo.from_path(run if Path(run).exists() else WANDB_PREFIX + run)
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
    modules = sorted(model.target_module_paths, key=lambda m: int(m.split(".")[2]))
    n_c = {m: model.module_to_c[m] for m in modules}
    deltas = model.calc_weight_deltas()

    texts, labels, names_list = load_split_with_labels(task.prepared_datasets, split)
    names = np.array(names_list)
    if n_per_dataset is not None:
        keep = np.concatenate(
            [np.flatnonzero(names == nm)[:n_per_dataset] for nm in sorted(set(names_list))]
        )
        texts, labels, names = [texts[i] for i in keep], labels[keep], names[keep]
    n = len(texts)
    data = load_prompts_dataset(texts, tok, task.max_seq_len, task.loss_positions, lens)[:]
    first_end = int(data[STIMULUS_END_KEY][0])
    print(
        f"run {run}, {split} split: {n} statements; example tokens "
        f"{[tok.decode([i]) for i in data['input_ids'][0][: first_end + 1]]}, "
        f"decomposed {data[POSITION_MASK_KEY][0].nonzero().flatten().tolist()}"
    )
    batches = []
    for b in range(0, n, batch_size):
        batch = trim_to_loss_positions({k: v[b : b + batch_size] for k, v in data.items()})
        batches.append(
            (
                np.arange(b, min(b + batch_size, n)),
                batch["input_ids"].to(device),
                batch[POSITION_MASK_KEY].to(device),
                batch[STIMULUS_END_KEY].to(device),
            )
        )

    # pass 1: original output, CIs, cancellation parts
    lp_orig_batches: list[Tensor] = []
    ci_per_batch: list[dict[str, Tensor]] = []
    parts = {m: {k: [] for k in PER_POSITION_KEYS} for m in modules}
    for _rows, ids, dec, end in batches:
        with bf16_autocast(enabled=config.autocast_bf16):
            o = model(ids, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=o.cache, detach_inputs=True, sampling=config.sampling
            )
        lp_orig_batches.append(
            torch.log_softmax(
                o.output[torch.arange(len(end), device=device), end].float(), -1
            ).cpu()
        )
        ci_per_batch.append({m: ci.lower_leaky[m].float() for m in modules})
        # position type: 0 = sequence position 0 (the statement's first token), 2 = the row's last
        # decomposed position (the statement end), 1 = in between
        idx = torch.arange(ids.shape[1], device=device).expand_as(dec)
        last_dec = (dec * (idx + 1)).max(1).values - 1
        ptype = torch.where(idx == 0, 0, torch.where(idx == last_dec[:, None], 2, 1))[dec]
        for m in modules:
            comp = model.components[m]
            assert isinstance(comp, LinearComponents), type(comp)
            for k, v in _module_parts(
                o.cache[m][dec], comp, deltas[m], ci_per_batch[-1][m][dec]
            ).items():
                parts[m][k] += v
            parts[m]["ptype"] += ptype.tolist()

    lp_orig = torch.cat(lp_orig_batches)  # batches cover rows 0..n-1 in order
    assert lp_orig.shape[0] == n

    def infos_for(
        state: str, ids: Tensor, dec: Tensor, i_batch: int
    ) -> dict[str, ComponentsMaskInfo]:
        b, pos = ids.shape
        match state:
            case "components":
                masks = {
                    m: (c > ROUNDING_THRESHOLD).float() for m, c in ci_per_batch[i_batch].items()
                }
            case "components_ci":
                masks = ci_per_batch[i_batch]
            case "full" | "delta_only" | "all_off":
                val = 1.0 if state == "full" else 0.0
                masks = {m: torch.full((b, pos, n_c[m]), val, device=device) for m in modules}
            case _:
                raise AssertionError(state)
        d_on = float(state in ("full", "delta_only"))
        infos = make_mask_infos(
            masks,
            weight_deltas_and_masks={
                m: (deltas[m], torch.full((b, pos), d_on, device=device)) for m in modules
            },
        )
        return route_only_selected_positions(infos, dec)

    scores = {"original": (lp_orig[:, one] - lp_orig[:, zero]).numpy()}
    mass = {"original": (lp_orig[:, one].exp() + lp_orig[:, zero].exp()).numpy()}
    kl = {"original": np.zeros(n)}
    for state in STATES:
        s_, m_, k_ = np.zeros(n), np.zeros(n), np.zeros(n)
        for i_batch, (rows, ids, dec, end) in enumerate(batches):
            with bf16_autocast(enabled=config.autocast_bf16):
                logits = model(ids, mask_infos=infos_for(state, ids, dec, i_batch))
            lp = torch.log_softmax(logits[torch.arange(len(end), device=device), end].float(), -1)
            s_[rows] = (lp[:, one] - lp[:, zero]).cpu().numpy()
            m_[rows] = (lp[:, one].exp() + lp[:, zero].exp()).cpu().numpy()
            p = lp_orig[rows].to(device)
            k_[rows] = (p.exp() * (p - lp)).sum(-1).cpu().numpy()
        scores[state], mass[state], kl[state] = s_, m_, k_

    def gap(s: np.ndarray) -> float:
        return float(s[labels].mean() - s[~labels].mean())

    neg = np.array([nm.startswith("neg_") for nm in names])
    summary: dict[str, Any] = {"run": run, "split": split, "n": n, "states": {}, "cancellation": {}}
    sep0, kl_off = gap(scores["original"]), float(kl["all_off"].mean())
    print(f"\nverdict (s = log P('1') - log P('0')); original separation {sep0:.2f}")
    print(
        "state         | acc   aff   neg   | AUC   | sep share | share '1' | digit mass | KL    | KL recovered"
    )
    for st in ["original", *STATES]:
        s = scores[st]
        pred = s > 0
        row = {
            "accuracy": float((pred == labels).mean()),
            "acc_aff": float((pred == labels)[~neg].mean()),
            "acc_neg": float((pred == labels)[neg].mean()),
            "auc": float(roc_auc_score(labels, s)),
            "sep_share": gap(s) / sep0,
            "share_1": float(pred.mean()),
            "digit_mass": float(mass[st].mean()),
            "kl": float(kl[st].mean()),
            "kl_recovered": 1 - float(kl[st].mean()) / kl_off,
            "per_dataset": {
                nm: float((pred == labels)[names == nm].mean()) for nm in sorted(set(names_list))
            },
        }
        summary["states"][st] = row
        print(
            f"{st:13s} | {row['accuracy']:.3f} {row['acc_aff']:.3f} {row['acc_neg']:.3f} | "
            f"{row['auc']:.3f} | {row['sep_share']:9.2f} | {row['share_1']:9.3f} | "
            f"{row['digit_mass']:10.3f} | {row['kl']:.3f} | {row['kl_recovered']:12.3f}"
        )

    norm_keys = ("A", "I", "D", "C", "A_ci", "I_ci")
    mean_keys = [k for k in PER_POSITION_KEYS if k not in ("y", "ptype", *norm_keys)]
    pooled: dict[str, list[float]] = {k: [] for k in mean_keys}
    for m in modules:
        s = parts[m]
        y_mean = float(np.mean(s["y"]))
        row = {
            "y_norm": y_mean,
            **{f"{k}_over_y": float(np.mean(s[k])) / y_mean for k in norm_keys},
            **{k: float(np.mean(s[k])) for k in mean_keys},
            **{
                f"rms_{k}": float(np.sqrt(np.mean(np.square(s[k]))))
                for k in ("inner_C", "inner_A", "inner_A_ci")
            },
        }
        summary["cancellation"][m] = row
        for k in mean_keys:
            pooled[k] += s[k]
    summary["cancellation"]["pooled"] = {
        **{k: float(np.mean(v)) for k, v in pooled.items()},
        **{
            f"rms_{k}": float(np.sqrt(np.mean(np.square(pooled[k]))))
            for k in ("inner_C", "inner_A", "inner_A_ci")
        },
    }
    cols = [
        ("active", "n_active", "6.2f"), ("|y-A|", "rel_err", "6.3f"), ("cos(A,y)", "cos_Ay", "8.3f"),
        ("|A|", "A_over_y", "6.2f"), ("|I|", "I_over_y", "6.2f"), ("|D|", "D_over_y", "6.2f"),
        ("|C|", "C_over_y", "6.2f"), ("cos(I,D)", "cos_ID", "8.3f"), ("cos(A,I)", "cos_AI", "8.3f"),
        ("<C,D>", "inner_C", "8.2f"), ("rms<C,D>", "rms_inner_C", "8.2f"),
    ]  # fmt: skip
    cols_ci = [
        ("|y-Aci|", "rel_err_ci", "7.3f"), ("cos(Aci,y)", "cos_Aci_y", "10.3f"),
        ("|Aci|", "A_ci_over_y", "6.2f"), ("|Ici|", "I_ci_over_y", "6.2f"),
        ("cos(Aci,Ici)", "cos_AI_ci", "12.3f"), ("<Aci,y-Aci>", "inner_A_ci", "11.3f"),
        ("foreign_ci", "foreign_ci", "10.3f"), ("CIs mid", "frac_mid", "7.3f"),
    ]  # fmt: skip
    for title, cs in (("rounded CIs; norms relative to |y|, inner products / |y|^2", cols),
                      ("CI-weighted (A_ci = sum CI_c write_c, I_ci = C - A_ci)", cols_ci)):  # fmt: skip
        print(f"\ncancellation test at the decomposed positions, {title}")
        print("module | " + " | ".join(h for h, _, _ in cs))
        for m in modules:
            r = summary["cancellation"][m]
            print(f"L{m.split('.')[2]:>5s} | " + " | ".join(f"{r[k]:{f}}" for _, k, f in cs))

    # by position type, pooled over layers; 'massive' = |y| > 5x the layer's median |y|
    groups: dict[str, dict[str, list[np.ndarray]]] = {}
    group_keys = (
        "y",
        "A",
        "A_ci",
        "rel_err",
        "rel_err_ci",
        "cos_Ay",
        "cos_Cy",
        "inner_C",
        "n_active",
    )
    for m in modules:
        s = {k: np.array(v) for k, v in parts[m].items()}
        massive = s["y"] > 5 * np.median(s["y"])
        for gname, sel in (("first token", s["ptype"] == 0), ("middle tokens", s["ptype"] == 1),
                           ("end token", s["ptype"] == 2), ("massive", massive),
                           ("not massive", ~massive)):  # fmt: skip
            g = groups.setdefault(gname, {k: [] for k in group_keys})
            for k in group_keys:
                g[k].append(s[k][sel])
    summary["by_position"] = {}
    print("\nby position type (pooled over layers)")
    print(
        "group         | n      | mean |y| | |A| / |Aci| | |y-A|  |y-Aci| | cos(A,y) cos(C,y) | <C,D> | active"
    )
    for gname, g in groups.items():
        cat = {k: np.concatenate(v) for k, v in g.items()}
        if len(cat["y"]) == 0:
            continue
        row = {"n": len(cat["y"]), **{k: float(np.mean(v)) for k, v in cat.items()}}
        summary["by_position"][gname] = row
        print(
            f"{gname:13s} | {row['n']:6d} | {row['y']:8.1f} | {row['A']:5.1f} {row['A_ci']:5.1f} | "
            f"{row['rel_err']:6.3f} {row['rel_err_ci']:6.3f} | {row['cos_Ay']:8.3f} {row['cos_Cy']:8.3f} | "
            f"{row['inner_C']:5.2f} | {row['n_active']:6.2f}"
        )
    act_per_stmt = sum(np.sum(parts[m]["n_active"]) for m in modules) / n
    summary["active_per_statement"] = float(act_per_stmt)
    print(
        f"active components per statement (summed over decomposed positions and layers): {act_per_stmt:.1f}"
    )

    # name by run id (the WandB id, or a local checkpoint's run directory), never by the checkpoint
    # file name: every run's final checkpoint has the same name (FEEDBACK.md, 26-10-16)
    run_id = Path(run).parent.name if Path(run).exists() else run
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"verdict_{run_id}_{split}.json").write_text(json.dumps(summary, indent=1))
    arrays = {
        "labels": labels,
        "names": names,
        **{f"score_{k}": v for k, v in scores.items()},
        **{f"kl_{k}": v for k, v in kl.items()},
    }
    np.savez(out_dir / f"verdict_{run_id}_{split}.npz", **arrays)  # pyright: ignore[reportArgumentType]


if __name__ == "__main__":
    fire.Fire(main)
