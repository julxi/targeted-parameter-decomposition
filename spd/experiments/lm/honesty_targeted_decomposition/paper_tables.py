"""Offline tables and figures of the paper analyses (topic convos/julian/26-10-29_paper_writing_LOG.md,
stage 1a; the analyses themselves: convos/julian/26-10-22_tpd_truth_paper_LOG.md, frozen plan D3).

Reads the outputs of paper_{recovery,termination,correlation,manipulation,directions}.py for the 6
final decompositions, the 2 untrained ones and the data-aligned basis, plus the finals' val readout
of phase C (side effects: Pile KL with delta only), and writes, CPU only:
  tables.md            the frozen plan's tables, every run and control side by side
  details.md           per-family recovery, single-component conditions, per-layer tables
  wide_val_reverse_<run>.csv   exploratory: every active component reversed alone, val
  figs/*.png
Nothing here changes a number: every value comes from the analysis outputs, except the pooled
bootstrap intervals of KL recovered (recomputed from the per-statement npz arrays, the same
resampling as paper_common.bootstrap_intervals) and the separation shift (sep_shift; added after the
outputs were read, descriptive, next to every separation share whose per-statement scores are
stored; each such stored share is recomputed and asserted equal). Before writing, the script recomputes a sample of
stored summaries from the npz arrays and asserts that they agree.

Aggregation: the frozen plan macro-averages over families. Macro KL recovered is dominated by families
whose all-off KL is small (paper topic, entry *final chains checked after the session break*), and the
choice is open (this topic, Julian's call), so KL recovered is shown pooled AND macro, and the
per-family values are in details.md. Separation share and flip rates are macro, as stored.

Usage: python paper_tables.py --phase_d <dir with out/ and recovery/> --phase_c <rounds dir> --out <dir>
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spd.experiments.lm.honesty_targeted_decomposition.paper_common import (
    family_of,
    form_of,
    kl_recovered,
    state_metrics,
)
from spd.experiments.lm.honesty_targeted_decomposition.paper_correlation import SELECT, TRANSFER

N_BOOT = 1000
SETS = (("test", "train"), ("test", "second"), ("ood", "train"), ("ood", "second"))
LAYERS = tuple(range(11, 18))
# categorical slots 1-5 of the dataviz reference palette, in fixed order; seed 2 dashed
COLOR = {
    "plain": "#2a78d6",
    "recon": "#eb6834",
    "ortho": "#1baf7a",
    "data-aligned": "#eda100",
    "untrained": "#e87ba4",
}
GRAY = "#8a8a85"


@dataclass(frozen=True)
class Obj:
    """One analysed basis: a final run, an untrained decomposition or the data-aligned basis."""

    method: str  # plain | recon | ortho | untrained | data-aligned
    seed: int
    run_id: str
    basis: str  # decomposition | data_aligned

    @property
    def label(self) -> str:
        return self.method if self.method == "data-aligned" else f"{self.method} s{self.seed}"

    @property
    def trained(self) -> bool:
        return self.method in ("plain", "recon", "ortho")

    @property
    def stem(self) -> str:
        return f"{self.run_id}_{self.basis}"


FINALS = [
    Obj("plain", 1, "s-7152548c", "decomposition"),
    Obj("plain", 2, "s-46efc525", "decomposition"),
    Obj("recon", 1, "s-eeac663f", "decomposition"),
    Obj("recon", 2, "s-edc17041", "decomposition"),
    Obj("ortho", 1, "s-0c2fa0d7", "decomposition"),
    Obj("ortho", 2, "s-4186f973", "decomposition"),
]
UNTRAINED = [
    Obj("untrained", 1, "s-d5c56fb0", "decomposition"),
    Obj("untrained", 2, "s-25d2cca7", "decomposition"),
]
# built from plain s1's target model; the target model is the same for every run, so it is the one
# data-aligned basis of the paper
DATA_ALIGNED = Obj("data-aligned", 0, "s-7152548c", "data_aligned")
DECOMPS = FINALS + UNTRAINED
ALL = DECOMPS + [DATA_ALIGNED]


# ---- formatting ----


def f(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def fs(x: float | None, nd: int = 3) -> str:
    """Signed, for the shift."""
    return "—" if x is None or np.isnan(x) else f"{x:+.{nd}f}"


def fci(x: float, ci: tuple[float, float] | list[float], nd: int = 3) -> str:
    return f"{f(x, nd)} [{f(ci[0], nd)}, {f(ci[1], nd)}]"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    assert all(len(r) == len(header) for r in rows), (header, [len(r) for r in rows])

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    out = [line(header), "|" + "---|" * len(header)] + [line(r) for r in rows]
    return "\n".join(out) + "\n"


def style(o: Obj) -> dict[str, Any]:
    return {"color": COLOR[o.method], "linestyle": "--" if o.seed == 2 else "-", "lw": 2}


# ---- loading ----


def load_json(p: Path) -> Any:
    assert p.exists(), p
    return json.loads(p.read_text())


def families(datasets: np.ndarray) -> np.ndarray:
    return np.array([family_of(str(d)) for d in datasets])


def pooled_kl_interval(
    kl: np.ndarray, kl_off: np.ndarray, fam: np.ndarray, seed: int
) -> tuple[float, float]:
    """95% bootstrap interval of the pooled KL recovered, resampling within family (as
    paper_common.bootstrap_intervals does for the macro averages)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(fam == g) for g in sorted(set(fam))]
    draws = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        draws.append(kl_recovered(kl[idx], kl_off[idx]))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


# ---- separation shift (added in stage 1a, after the outputs were read; descriptive, not in the
# frozen plan) ----

# summaries annotated with a shift, per output kind; every one had its stored sep_share recomputed
SHIFT_CHECKED: dict[str, int] = {}


def sep_shift(score: np.ndarray, base_score: np.ndarray, labels: np.ndarray) -> float:
    """Signed move of the midpoint between the true and the false class mean of the verdict score, in
    units of the original true-false gap; > 0 = toward '1'. With sep_share it fixes both class means:
    the flips mix a closing gap with a shift of both classes."""
    sep0 = base_score[labels].mean() - base_score[~labels].mean()
    mid = (score[labels].mean() + score[~labels].mean()) / 2
    mid0 = (base_score[labels].mean() + base_score[~labels].mean()) / 2
    return float((mid - mid0) / sep0)


def add_shift(
    summary: dict[str, Any], score: np.ndarray, base: np.ndarray, labels: np.ndarray,
    fam: np.ndarray, kind: str,
) -> None:  # fmt: skip
    """Adds 'shift' to the pooled, per-family and (if stored) macro metrics of one stored summary, and
    'abs_shift' (macro of |shift|; the signed macro can cancel across families) to the macro. First
    recomputes sep_share from the same arrays and asserts it equals the stored value, so the arrays
    are the ones the summary was made from."""
    assert len(score) == len(base) == len(labels) == len(fam), (kind, len(score), len(base))
    score, base = score.astype(np.float64), base.astype(np.float64)

    def share(idx: np.ndarray) -> float:
        lab = labels[idx]
        s, b = score[idx], base[idx]
        return float((s[lab].mean() - s[~lab].mean()) / (b[lab].mean() - b[~lab].mean()))

    every = np.arange(len(score))
    stored = summary["pooled"]["sep_share"]
    assert np.isclose(share(every), stored, rtol=1e-4, atol=1e-5), (kind, share(every), stored)
    summary["pooled"]["shift"] = sep_shift(score, base, labels)
    per = {}
    for g in sorted(set(fam)):
        idx = np.flatnonzero(fam == g)
        stored = summary["per_family"][g]["sep_share"]
        assert np.isclose(share(idx), stored, rtol=1e-4, atol=1e-5), (kind, g, share(idx), stored)
        per[g] = sep_shift(score[idx], base[idx], labels[idx])
        summary["per_family"][g]["shift"] = per[g]
    assert set(per) == set(summary["per_family"]), (
        kind,
        sorted(per),
        sorted(summary["per_family"]),
    )
    if "macro" in summary:
        summary["macro"]["shift"] = float(np.nanmean(list(per.values())))
        summary["macro"]["abs_shift"] = float(np.nanmean(np.abs(list(per.values()))))
    SHIFT_CHECKED[kind] = SHIFT_CHECKED.get(kind, 0) + 1


def load_with_shift(phase_d: Path, kind: str, o: Obj) -> dict[str, Any]:
    """The stored json of one analysis output, with add_shift applied to every summary whose
    per-statement scores are in the npz: recovery states; termination and manipulation conditions
    (controls on the capped OOD subset with that subset's reference pass) and the main conditions
    restricted to that subset; manipulation's val wide table. Not covered (no per-statement scores
    stored): termination's t1_val ranking, the directions' direction removal."""
    if kind == "recovery":
        j = load_json(phase_d / "recovery" / f"recovery_{o.run_id}.json")
        z = np.load(phase_d / "recovery" / f"recovery_{o.run_id}.npz")
    else:
        j = load_json(phase_d / "out" / f"{kind}_{o.stem}.json")
        z = np.load(phase_d / "out" / f"{kind}_{o.stem}.npz")
    for split, prompt in SETS:
        lab, fam = z[f"{split}_labels"], families(z[f"{split}_datasets"])
        ref = f"{split}_{prompt}_"
        r = j["sets"][split][prompt]
        if kind == "recovery":
            for state, v in r["states"].items():
                # states that need an active set are None for the untrained decompositions
                if v is None:
                    assert j["active_set"] is None, (o.label, state)
                    continue
                add_shift(v, z[ref + f"{state}_score"], z[ref + "original_score"], lab, fam, kind)
            continue
        rows = z[f"{split}_control_rows"]
        for name, v in r["conditions"].items():
            s = z[ref + f"{name}_score"]
            if len(s) == len(lab):
                add_shift(v, s, z[ref + "original_score"], lab, fam, kind)
            else:  # a control run on the capped OOD subset only, against that subset's own pass
                assert len(s) == len(rows) < len(lab), (o.label, name, len(s), len(rows))
                base = z[ref + "control_rows_original_score"]
                add_shift(v, s, base, lab[rows], fam[rows], kind)
        for name, v in r["main_on_control_rows"].items():
            s = z[ref + f"{name}_score"]
            add_shift(v, s[rows], z[ref + "original_score"][rows], lab[rows], fam[rows], kind)
    # the wide table exists for the trained runs only (it needs an active set)
    if kind == "manipulation" and o.trained:
        assert "wide_val_reverse" in j, o.label
        lab, fam = z["wide_val_labels"], families(z["wide_val_datasets"])
        for name, v in j["wide_val_reverse"].items():
            add_shift(v, z[f"wide_val_{name}_score"], z["wide_val_base_score"], lab, fam, kind)
    return j


# ---- consistency checks: stored summaries vs per-statement arrays ----


def check_recomputation(phase_d: Path) -> list[str]:
    """Recompute stored summaries of every script from its npz arrays; assert agreement."""
    log = []
    for o in DECOMPS:
        rec = load_json(phase_d / "recovery" / f"recovery_{o.run_id}.json")
        z = np.load(phase_d / "recovery" / f"recovery_{o.run_id}.npz")
        for split, prompt in SETS:
            for state in ("components_ci", "delta_only"):
                kl = z[f"{split}_{prompt}_{state}_kl"]
                stored = rec["sets"][split][prompt]["states"][state]["pooled"]["kl_recovered"]
                mine = kl_recovered(kl, z[f"{split}_{prompt}_all_off_kl"])
                assert abs(mine - stored) < 1e-9, (o.label, split, prompt, state, mine, stored)
    log.append(
        "recovery: pooled KL recovered of components_ci and delta_only, 8 runs x 4 sets: equal"
    )
    # the all-off state (decomposed MLP outputs 0) does not depend on the basis
    offs = [
        np.load(phase_d / "recovery" / f"recovery_{o.run_id}.npz")["test_train_all_off_kl"]
        for o in DECOMPS
    ]
    dev = max(float(np.abs(a - offs[0]).max() / offs[0].mean()) for a in offs)
    log.append(f"recovery: all-off KL across the 8 runs, max |diff| / mean = {dev:.2e}")
    for o in ALL:
        t = load_json(phase_d / "out" / f"termination_{o.stem}.json")
        z = np.load(phase_d / "out" / f"termination_{o.stem}.npz")
        lab, fam = z["test_labels"], families(z["test_datasets"])
        for k in (t["ks"][0], t["ks"][-1]):
            name = f"T2 top{k}"
            s, kl, ms = (
                z[f"test_train_{name}_{x}"].astype(np.float64) for x in ("score", "kl", "mass")
            )
            base, off = z["test_train_original_score"], z["test_train_all_off_kl"]
            mine = np.nanmean(
                [
                    state_metrics(s[fam == g], kl[fam == g], base[fam == g], off[fam == g],
                                  lab[fam == g], ms[fam == g])["sep_share"]
                    for g in sorted(set(fam))
                ]
            )  # fmt: skip
            stored = t["sets"]["test"]["train"]["conditions"][name]["macro"]["sep_share"]
            assert abs(mine - stored) < 1e-5, (o.label, name, mine, stored)  # npz is float32
    log.append(
        "termination: macro separation share of T2 at the smallest and largest k, 9 bases: equal"
    )
    for o in ALL:
        mj = load_json(phase_d / "out" / f"manipulation_{o.stem}.json")
        z = np.load(phase_d / "out" / f"manipulation_{o.stem}.npz")
        name = "reverse top10 group"
        s = z[f"test_train_{name}_score"].astype(np.float64)
        base = z["test_train_original_score"]
        mine = float((s[base > 0] <= 0).mean())
        stored = mj["sets"]["test"]["train"]["conditions"][name]["pooled"]["flip_1_to_0"]
        assert abs(mine - stored) < 1e-9, (o.label, mine, stored)
    log.append("manipulation: pooled 1->0 flips of 'reverse top10 group', 9 bases: equal")
    return log


# ---- sections ----


def side_effects_section(phase_c: Path) -> str:
    rows = []
    for rnd in ("final_plain", "final_rest"):
        for v in load_json(phase_c / rnd / "summary.json").values():
            o = next(x for x in FINALS if x.run_id == v["run"])
            rows.append(
                (FINALS.index(o), [o.label, o.run_id, f(v["kl_rec"]), f(v["kl_rec2"]),
                 f(v["delta_only"]), f(v["pile_delta"]), f(v["pgd"], 2), f(v["l0"], 1)])
            )  # fmt: skip
    rows.sort()
    return (
        "## 0. Finals on val and side effects (phase C readout)\n\n"
        "From the finals' val readout (`verdict_lens_analysis.py`, val, training prompt), not phase D. "
        "Pile Δ-only KL = KL on Pile text with all components removed (side effects off-target); "
        "PGD = adversarial-mask loss; L0 = mean active components per token on the target data "
        "(CI > 0 rounding of the eval metric; not used as a definition of 'active').\n\n"
        + md_table(
            [
                "run",
                "id",
                "comps-only KL rec. (val)",
                "2nd prompt",
                "Δ-only KL rec.",
                "Pile Δ-only KL",
                "PGD",
                "L0",
            ],
            [r for _, r in rows],
        )  # fmt: skip
    )


def recovery_section(phase_d: Path, figs: Path) -> tuple[str, str, str]:
    """Returns (main tables, details, the T0 table that belongs to termination)."""
    rec = {o.run_id: load_with_shift(phase_d, "recovery", o) for o in DECOMPS}
    z = {o.run_id: np.load(phase_d / "recovery" / f"recovery_{o.run_id}.npz") for o in DECOMPS}
    hdr = ["run"] + [f"{s}/{p}" for s, p in SETS]

    def state_rows(
        state: str, metric: str, with_pooled_ci: bool
    ) -> tuple[list[list[str]], list[list[str]]]:
        pooled, macro = [], []
        for o in DECOMPS:
            rp, rm = [o.label], [o.label]
            for split, prompt in SETS:
                r = rec[o.run_id]["sets"][split][prompt]["states"][state]
                if with_pooled_ci:
                    zz = z[o.run_id]
                    ci = pooled_kl_interval(
                        zz[f"{split}_{prompt}_{state}_kl"], zz[f"{split}_{prompt}_all_off_kl"],
                        families(zz[f"{split}_datasets"]), seed=0,
                    )  # fmt: skip
                    rp.append(fci(r["pooled"][metric], ci))
                else:
                    rp.append(f(r["pooled"][metric]))
                rm.append(fci(r["macro"][metric], r["macro_interval"][metric]))
            pooled.append(rp)
            macro.append(rm)
        return pooled, macro

    orig = [["original model: accuracy"] + [
        f(rec[FINALS[0].run_id]["sets"][s][p]["original"]["accuracy"]) for s, p in SETS]]  # fmt: skip
    ns = [["n statements"] + [str(rec[FINALS[0].run_id]["sets"][s][p]["n"]) for s, p in SETS]]
    kp, km = state_rows("components_ci", "kl_recovered", True)
    ap, am = state_rows("components_ci", "agreement", False)
    md = (
        "## 1. Recovery: components only (mask = CI, Δ off)\n\n"
        "KL recovered = 1 − KL(original ‖ state) / KL(original ‖ decomposed MLP outputs off), at the "
        "verdict digit; 1 = original verdict distribution, 0 = as bad as the MLPs off. Test = tiu v3 "
        "test (in-distribution, 10 families); OOD = all 24 OOD files; `train` = the training verdict "
        "prompt, `second` = the held-out prompt. 95% bootstrap intervals (1,000 resamples within "
        "family). The untrained decompositions are the floor.\n\n"
        + md_table(hdr, ns + orig)
        + "\n**KL recovered, pooled over statements**\n\n" + md_table(hdr, kp)
        + "\n**KL recovered, macro over families**\n\n" + md_table(hdr, km)
        + "\n**Agreement with the original model's verdict (pooled; macro with interval below)**\n\n"
        + md_table(hdr, ap) + "\n" + md_table(hdr, am)
    )  # fmt: skip

    # per-family details, training prompt and second prompt
    details = "## Recovery per family (components only, KL recovered)\n\n"
    details += (
        "Mean all-off KL = the denominator of KL recovered in that family (identical for all runs: "
        "the decomposed MLP outputs set to 0). Families with a small all-off KL have unstable ratios "
        "and dominate the macro average.\n\n"
    )
    for split, prompt in SETS:
        zz = z[FINALS[0].run_id]
        fam = families(zz[f"{split}_datasets"])
        off = zz[f"{split}_{prompt}_all_off_kl"]
        rows = []
        for g in sorted(set(fam)):
            row = [g, str(int((fam == g).sum())), f(float(off[fam == g].mean()))]
            row += [
                f(rec[o.run_id]["sets"][split][prompt]["states"]["components_ci"]["per_family"][g]["kl_recovered"])
                for o in DECOMPS
            ]  # fmt: skip
            rows.append(row)
        details += (
            f"**{split}/{prompt}**\n\n"
            + md_table(["family", "n", "mean all-off KL"] + [o.label for o in DECOMPS], rows)
            + "\n"
        )

    # ---- active set ----
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for o in FINALS:
        a = rec[o.run_id]["active_set"]
        ci_sum = rec[o.run_id]["sets"]["test"]["train"]["ci_sum_per_statement"]["mean"]
        rows.append([o.label, str(a["m"]), f(a["kl_recovered_all"]), f(a["target"]),
                     f(a["kl_recovered_active"]), f(ci_sum, 1), ", ".join(a["components"][:8])
                     + (" …" if a["m"] > 8 else "")])  # fmt: skip
        curve = sorted((int(k), v) for k, v in a["curve"].items())
        ax.plot(
            [c[0] for c in curve],
            [c[1] for c in curve],
            marker="o",
            ms=4,
            label=o.label,
            **style(o),
        )
    for o in UNTRAINED:
        a = rec[o.run_id]["active_set"]
        assert a is None, o.label
        all_rec = rec[o.run_id]["sets"]["test"]["train"]["states"]["components_ci"]["pooled"][
            "kl_recovered"
        ]
        ci_sum = rec[o.run_id]["sets"]["test"]["train"]["ci_sum_per_statement"]["mean"]
        rows.append(
            [
                o.label,
                "none",
                f"test: {f(all_rec)}",
                "—",
                "—",
                f(ci_sum, 1),
                "(all components recover ≤ 0 on val)",
            ]
        )
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("top m components by mean CI on val (others off, Δ off)")
    ax.set_ylabel("KL recovered (val, pooled)")
    ax.set_title("Functional active set: recovery of the top-m components")
    ax.legend(fontsize=8, ncol=2, handlelength=4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figs / "active_set_curve.png", dpi=150)
    plt.close(fig)
    md += (
        "\n## 1b. Functional active set and counts\n\n"
        "Active set S = smallest top-m set by mean CI on val whose CI-weighted components-only state "
        "recovers ≥ 95% of what all components recover (frozen plan; no CI threshold). Found on val by "
        "grid + bisection (first crossing). Sum of CIs = per statement, over all statement tokens and "
        "layers (counts without a cutoff), test, training prompt.\n\n"
        + md_table(
            [
                "run",
                "m = |S|",
                "KL rec. all (val)",
                "target (95%)",
                "KL rec. S (val)",
                "mean Σ CI / statement",
                "S (first 8, by mean CI)",
            ],
            rows,
        )  # fmt: skip
        + "\n![active-set curve](figs/active_set_curve.png)\n"
    )

    # ---- active vs inactive ----
    rows = []
    for o in FINALS:
        st = rec[o.run_id]["sets"]["test"]["train"]["states"]
        fw = st["full_without_inactive"]["macro"]
        rows.append([o.label] + [f(st[s]["pooled"]["kl_recovered"]) for s in
                                 ("components_ci", "active_ci", "mean_mask")]
                    + [f(st["full_without_inactive"]["pooled"]["agreement"]), f(fw["sep_share"]),
                       fs(fw["shift"]), f(fw["abs_shift"]), f(fw["flip_1_to_0"]), f(fw["flip_0_to_1"])])  # fmt: skip
    for o in UNTRAINED:
        st = rec[o.run_id]["sets"]["test"]["train"]["states"]
        rows.append([o.label, f(st["components_ci"]["pooled"]["kl_recovered"]), "—",
                     f(st["mean_mask"]["pooled"]["kl_recovered"]), "—", "—", "—", "—", "—", "—"])  # fmt: skip
    md += (
        "\n## 1c. Active vs inactive components (test, training prompt)\n\n"
        "Components only (Δ off): all components at mask = CI; only S at mask = CI; mean mask "
        "CI + (1 − CI)/2 (the training's expected mask; separates 'inactive components matter' from "
        "'they matter through the random masks'). Then the full model (Δ kept) with every component "
        "outside S removed: agreement (pooled), separation share, shift, |shift| and flips (macro).\n\n"
        + md_table(
            [
                "run",
                "all, CI",
                "S only, CI",
                "mean mask",
                "full − inactive: agreement",
                "sep. share",
                "shift",
                "|shift|",
                "1→0",
                "0→1",
            ],
            rows,
        )  # fmt: skip
    )
    rows = []
    for o in DECOMPS:
        for lay, v in rec[o.run_id]["sets"]["test"]["train"]["layers"].items():
            rows.append([o.label, lay] + [f(v.get(k)) for k in
                                          ("A/y", "I/y", "cos_AI", "C/y", "D/y", "inner_CD")])  # fmt: skip
    details += (
        "## Write norms per layer (test, training prompt)\n\n"
        "Mean norms relative to mean |y| (y = original MLP output) at the statement tokens. A = "
        "CI-weighted writes of S, I = CI-weighted writes of the other components, C = all component "
        "writes at mask 1, D = y − C (Δ's write); inner_CD = ⟨C, D⟩ / |y|² (cancellation diagnostic; "
        "large negative = C and D point in opposite directions and cancel).\n\n"
        + md_table(["run", "layer", "|A|/|y|", "|I|/|y|", "cos(A, I)", "|C|/|y|", "|D|/|y|", "⟨C,D⟩/|y|²"], rows)
    )  # fmt: skip

    # ---- T0 (lives in termination, computed here) ----
    rows = []
    for o in DECOMPS:
        row = [o.label]
        for split, prompt in SETS:
            r = rec[o.run_id]["sets"][split][prompt]["states"]["delta_only"]
            row.append(f"{f(r['pooled']['agreement'])} / {f(r['macro']['sep_share'])} / "
                       f"{fs(r['macro']['shift'])} ({f(r['macro']['abs_shift'])}) / "
                       f"{f(r['macro']['flip_1_to_0'])} / {f(r['macro']['flip_0_to_1'])}")  # fmt: skip
        rows.append(row)
    dp, dm = state_rows("delta_only", "kl_recovered", True)
    t0 = (
        "## 2a. T0: all components removed (Δ only)\n\n"
        "Full model with every component at mask 0 (Δ kept). KL recovered as in section 1 (same "
        "denominator), so it compares directly with the components-only recovery there: ideally "
        "components ≈ 1 and Δ only ≈ 0; the untrained decompositions (Δ = everything) are the "
        "ceiling. 95% bootstrap intervals as in section 1.\n\n"
        "**KL recovered by Δ only, pooled over statements**\n\n" + md_table(hdr, dp)
        + "\n**KL recovered by Δ only, macro over families**\n\n" + md_table(hdr, dm)
        + "\n**Verdict measures.** Each cell: agreement with the original verdict (pooled) / "
        "separation share (macro; 1 = unchanged true-false gap of the verdict score) / shift "
        "(|shift|) (macro) / share of '1' verdicts flipped to '0' / of '0' flipped to '1' (macro).\n\n"
        + md_table(hdr, rows)
    )  # fmt: skip
    return md, details, t0


def valley_section(phase_d: Path, figs: Path) -> tuple[str, str]:
    edges = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 1.0]
    labels = ["=0", "(0,1e-4)", "[1e-4,1e-3)", "[1e-3,.01)", "[.01,.1)", "[.1,.5)", "[.5,.9)",
              "[.9,.99)", "[.99,1)", "=1"]  # fmt: skip

    def shares(v: np.ndarray) -> list[float]:
        n = v.size
        inner = v[(v > 0) & (v < 1)]
        counts = (
            [int((v <= 0).sum())] + list(np.histogram(inner, bins=edges)[0]) + [int((v >= 1).sum())]
        )
        assert sum(counts) == n, (sum(counts), n)
        return [c / n for c in counts]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    rows, rows_layer = [], []
    for ax, o in zip(axes.flat, DECOMPS, strict=True):
        d = np.load(phase_d / "recovery" / f"ci_val_{o.run_id}.npz")
        allv = np.concatenate([d[f"L{lay}"].astype(np.float32).ravel() for lay in LAYERS])
        rows.append([o.label] + [f(s, 4) for s in shares(allv)])
        for lay in LAYERS:
            rows_layer.append(
                [o.label, f"L{lay}"]
                + [f(s, 4) for s in shares(d[f"L{lay}"].astype(np.float32).ravel())]
            )
        inner = allv[(allv > 0) & (allv < 1)]
        ax.hist(
            inner, bins=50, range=(0, 1), color=COLOR[o.method], edgecolor="white", linewidth=0.5
        )
        ax.set_yscale("log")
        ax.set_title(
            f"{o.label}: =0 {np.mean(allv <= 0):.3f}, =1 {np.mean(allv >= 1):.4f}", fontsize=9
        )
        ax.set_xlabel("CI in (0, 1)")
    axes[0, 0].set_ylabel("count (log)")
    axes[1, 0].set_ylabel("count (log)")
    fig.suptitle("CI values at every statement token × component × layer, val (valley check)")
    fig.tight_layout()
    fig.savefig(figs / "ci_histograms.png", dpi=150)
    plt.close(fig)
    # mean CI per component; linear y chosen by Julian over log (log stretched the tail, compressed the top)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for o in DECOMPS:
        m = np.load(phase_d / "recovery" / f"ci_val_{o.run_id}.npz")["mean_ci"]
        ax.plot(np.arange(1, len(m) + 1), np.sort(m)[::-1], label=o.label, **style(o))
    ax.set_xscale("log")
    ax.set_xlabel("component rank by mean CI (val)")
    ax.set_ylabel("mean CI over statement tokens")
    ax.set_title("Mean CI per component, sorted")
    ax.legend(fontsize=8, ncol=2, handlelength=4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figs / "mean_ci_sorted.png", dpi=150)
    plt.close(fig)
    md = (
        "\n## 1d. Valley check: distribution of CI values (val)\n\n"
        "Share of all (statement token × component × layer) CI values in each bin; 9,895 val "
        "statement tokens × 672 components. A valley = a minimum between a mode near 0 and a mode near "
        "1, in both seeds. Histograms of the values strictly inside (0, 1) below (log counts).\n\n"
        + md_table(["run"] + labels, rows)
        + "\n![CI histograms](figs/ci_histograms.png)\n\n![mean CI](figs/mean_ci_sorted.png)\n"
    )
    details = "\n## CI value bins per layer (val)\n\n" + md_table(
        ["run", "layer"] + labels, rows_layer
    )
    return md, details


def termination_section(phase_d: Path, figs: Path) -> tuple[str, str]:
    T = {o.stem: load_with_shift(phase_d, "termination", o) for o in ALL}
    md = ""
    # ---- T1 ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rows = []
    for o in ALL:
        t = T[o.stem]
        drops = sorted((1 - v["sep_share"] for v in t["t1_val"].values()), reverse=True)
        ax.plot(np.arange(1, min(50, len(drops)) + 1), drops[:50], label=o.label, **style(o))
        top = [(nm, 1 - t["t1_val"][nm]["sep_share"], t["t1_val"][nm]) for nm in t["ranked"][:5]]
        rows.append([o.label, str(len(t["candidates"])),
                     "active set" if t["candidates_are_active_set"] else "all components",
                     "; ".join(f"{nm} {d:.3f} ({v['flip_1_to_0']:.2f}/{v['flip_0_to_1']:.2f})"
                               for nm, d, v in top)])  # fmt: skip
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xlabel("rank (val)")
    ax.set_ylabel("drop of the verdict separation (1 − sep. share)")
    ax.set_title("T1: single-component removal, largest 50 drops (val, training prompt)")
    ax.legend(fontsize=8, ncol=2, handlelength=4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figs / "t1_single_removal.png", dpi=150)
    plt.close(fig)
    md += (
        "\n## 2b. T1: single removals (val, training prompt; the ranking for T2 and manipulation)\n\n"
        "Each candidate set to mask 0 in the full model (Δ and all others kept), pooled over val. "
        "Candidates: the functional active set for trained runs; all 672 components for the untrained "
        "decompositions and the data-aligned basis (they have no active set). Top 5: name, separation "
        "drop (1 − separation share), (1→0 / 0→1 flip shares); no shift (the val single removals "
        "were stored as summaries only).\n\n"
        + md_table(["basis", "candidates", "kind", "top 5 by separation drop"], rows)
        + "\n![T1](figs/t1_single_removal.png)\n"
    )

    # ---- T2 ----
    def cond(o: Obj, split: str, prompt: str, name: str, on_ctrl: bool) -> dict[str, Any] | None:
        r = T[o.stem]["sets"][split][prompt]
        if on_ctrl and r["main_on_control_rows"]:
            return r["main_on_control_rows"].get(
                name, r["conditions"].get(name) if name in T[o.stem]["controls"] else None
            )
        return r["conditions"].get(name)

    def rand_stats(
        o: Obj, split: str, prompt: str, k: int, metric: str
    ) -> tuple[float, float, float] | None:
        r = T[o.stem]["sets"][split][prompt]["conditions"]
        vals = [r[f"T2 rand{k} #{i}"]["macro"][metric] for i in range(5) if f"T2 rand{k} #{i}" in r]
        return (float(np.mean(vals)), float(min(vals)), float(max(vals))) if vals else None

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
    for row_i, (split, prompt, on_ctrl) in enumerate(
        (("test", "train", False), ("ood", "train", True))
    ):
        for col, meth in enumerate(("plain", "recon", "ortho")):
            ax = axes[row_i, col]
            for o in [x for x in FINALS if x.method == meth] + UNTRAINED + [DATA_ALIGNED]:
                ks = T[o.stem]["ks"]
                pts = [(k, cond(o, split, prompt, f"T2 top{k}", on_ctrl)) for k in ks]
                pts = [(k, c["macro"]["sep_share"]) for k, c in pts if c is not None]
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker="o",
                    ms=3,
                    label=o.label,
                    **style(o),
                )
                if o.trained:
                    rs = [(k, rand_stats(o, split, prompt, k, "sep_share")) for k in ks]
                    rs = [(k, v) for k, v in rs if v is not None]
                    ax.errorbar([k for k, _ in rs], [v[0] for _, v in rs],
                                yerr=[[v[0] - v[1] for _, v in rs], [v[2] - v[0] for _, v in rs]],
                                fmt="x", color=COLOR[o.method], alpha=0.6, capsize=2,
                                label=f"{o.label}: random k of S (5 draws)")  # fmt: skip
            ax.set_xscale("log")
            ax.axhline(1, color=GRAY, lw=1)
            ax.axhline(0, color=GRAY, lw=1)
            ax.set_title(f"{meth}: {split}/{prompt}" + (" (≤50 per dataset)" if on_ctrl else ""))
            ax.set_xlabel("k components removed (top-k by val T1 drop)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6, handlelength=4)
        axes[row_i, 0].set_ylabel("separation share (macro)")
    fig.suptitle(
        "T2: top-k removal from the full model vs random k of the active set, untrained, data-aligned"
    )
    fig.tight_layout()
    fig.savefig(figs / "t2_topk.png", dpi=150)
    plt.close(fig)

    # main tables: a few k plus every trained run's active-set size and "all"; full grid in details
    k_main = {1, 5, 10, 20} | {len(T[o.stem]["candidates"]) for o in FINALS}

    def k_rows(split: str, prompt: str, on_ctrl: bool, short: bool) -> list[list[str]]:
        rows = []
        for o in ALL:
            t = T[o.stem]
            for k in t["ks"]:
                if short and k not in k_main and k != t["ks"][-1]:
                    continue
                c = cond(o, split, prompt, f"T2 top{k}", on_ctrl)
                if c is None:
                    continue
                m = c["macro"]
                rnd = rand_stats(o, split, prompt, k, "sep_share") if o.trained else None
                rows.append([o.label, str(k), f(m["sep_share"]), fs(m["shift"]), f(m["abs_shift"]),
                             f(c["pooled"]["agreement"]),
                             f(m["flip_1_to_0"]), f(m["flip_0_to_1"]), f(m["digit_mass"]),
                             f"{f(rnd[0])} [{f(rnd[1])}, {f(rnd[2])}]" if rnd else "—"])  # fmt: skip
        return rows

    hdr = ["basis", "k", "sep. share", "shift", "|shift|", "agreement", "1→0", "0→1", "digit mass",
           "random k of S: sep. share mean [min, max]"]  # fmt: skip
    md += (
        "\n## 2c. T2: top-k removal\n\n"
        "The k candidates with the largest val T1 drop removed together from the full model. k grid "
        "1, 2, 5, 10, 20, 50, 100, 200 up to all candidates; the data-aligned basis also at the "
        "trained runs' active-set sizes. Controls: 5 random size-k subsets of the run's active set; "
        "the untrained decompositions and the data-aligned basis ranked by the same rule. Separation "
        "share, shift, |shift|, flips, digit mass macro; agreement pooled. OOD = the 13 ground-truth files; the OOD "
        "table is on the common control subset (≤ 50 per dataset, 950 statements) so main and random "
        "conditions use the same statements. Shown: k = 1, 5, 10, 20, the trained runs' active-set "
        "sizes and all candidates; the full grid, the second prompt and full OOD in details.md.\n\n"
        "![T2](figs/t2_topk.png)\n\n"
        "**test / training prompt**\n\n" + md_table(hdr, k_rows("test", "train", False, True))
        + "\n**OOD (≤ 50 per dataset) / training prompt**\n\n" + md_table(hdr, k_rows("ood", "train", True, True))
    )  # fmt: skip
    details = (
        "\n## T2: full k grid, second prompt, full OOD\n\n"
        "**test / training prompt**\n\n" + md_table(hdr, k_rows("test", "train", False, False))
        + "\n**OOD (≤ 50 per dataset) / training prompt**\n\n" + md_table(hdr, k_rows("ood", "train", True, False))
        + "\n**test / second prompt**\n\n" + md_table(hdr, k_rows("test", "second", False, False))
        + "\n**OOD (≤ 50 per dataset) / second prompt**\n\n" + md_table(hdr, k_rows("ood", "second", True, False))
        + "\n**OOD full (13 files, 4,292 statements) / training prompt** (random column empty: the "
        "random controls ran on the subset only)\n\n"
        + md_table(hdr, [r[:-1] + ["—"] for r in k_rows("ood", "train", False, False)])
    )  # fmt: skip

    # ---- T1 on test, active set ----
    rows = []
    for o in FINALS:
        r = T[o.stem]["sets"]["test"]["train"]["conditions"]
        for nm in T[o.stem]["ranked"]:
            v = r[f"T1 {nm}"]["macro"]
            rows.append([o.label, nm, f(1 - T[o.stem]["t1_val"][nm]["sep_share"]), f(1 - v["sep_share"]),
                         fs(v["shift"]), f(v["flip_1_to_0"]), f(v["flip_0_to_1"])])  # fmt: skip
    details += (
        "\n## T1 per active component: val drop (ranking) vs test (macro)\n\n"
        "No val shift: the val single removals were stored as summaries only.\n\n"
        + md_table(
            [
                "run",
                "component",
                "val drop (pooled)",
                "test drop (macro)",
                "test shift",
                "test 1→0",
                "test 0→1",
            ],
            rows,
        )  # fmt: skip
    )

    # ---- T3 = A8 ----
    rows = []
    for o in FINALS:
        r = T[o.stem]["sets"]["test"]["train"]["conditions"]
        for lay in LAYERS:
            row = [o.label, f"L{lay}"]
            for span in ("all", "active"):
                if f"A8 L{lay} span{span}" not in r:
                    row += ["—"] * 3
                    continue
                main = r[f"A8 L{lay} span{span}"]["macro"]
                pca = r[f"A8 L{lay} pca{span}"]["macro"]
                rnd = [r[f"A8 L{lay} rand{span} #{i}"]["macro"] for i in range(3)]
                row += [f"{f(main['sep_share'])} {fs(main['shift'], 2)} ({f(main['flip_1_to_0'], 2)}/{f(main['flip_0_to_1'], 2)})",
                        f"{f(pca['sep_share'])} {fs(pca['shift'], 2)} ({f(pca['flip_1_to_0'], 2)}/{f(pca['flip_0_to_1'], 2)})",
                        f"{f(float(np.mean([x['sep_share'] for x in rnd])))} "
                        f"{fs(float(np.mean([x['shift'] for x in rnd])), 2)} "
                        f"(max 0→1 {f(max(x['flip_0_to_1'] for x in rnd), 2)})"]  # fmt: skip
            rows.append(row)
    # figure: separation drop of the span vs the same-dimension controls, per method and span kind;
    # random draws shown only as a band (min-max over both seeds' draws) to keep it readable
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for col, meth in enumerate(("plain", "recon", "ortho")):
        for row_i, span in enumerate(("all", "active")):
            ax = axes[row_i, col]
            lo, hi = np.full(len(LAYERS), np.inf), np.full(len(LAYERS), -np.inf)
            for o in [x for x in FINALS if x.method == meth]:
                r = T[o.stem]["sets"]["test"]["train"]["conditions"]
                drop: dict[str, list[float]] = {"span": [], "pca": []}
                for i, lay in enumerate(LAYERS):
                    if f"A8 L{lay} span{span}" not in r:
                        drop["span"].append(np.nan)
                        drop["pca"].append(np.nan)
                        continue
                    drop["span"].append(1 - r[f"A8 L{lay} span{span}"]["macro"]["sep_share"])
                    drop["pca"].append(1 - r[f"A8 L{lay} pca{span}"]["macro"]["sep_share"])
                    rnd = [
                        1 - r[f"A8 L{lay} rand{span} #{d}"]["macro"]["sep_share"] for d in range(3)
                    ]
                    lo[i], hi[i] = min(lo[i], *rnd), max(hi[i], *rnd)
                ls = "--" if o.seed == 2 else "-"
                ax.plot(
                    LAYERS,
                    drop["span"],
                    marker="o",
                    ms=5,
                    label=f"{o.label}: span {span}",
                    **style(o),
                )
                ax.plot(LAYERS, drop["pca"], marker="s", ms=4, color=GRAY, linestyle=ls, lw=1.5,
                        label=f"{o.label}: top-PCA of y (same dim)")  # fmt: skip
            ok = np.isfinite(lo)
            ax.fill_between(np.array(LAYERS)[ok], lo[ok], hi[ok], color=GRAY, alpha=0.25,
                            label="random subspaces (min–max)")  # fmt: skip
            ax.axhline(0, color=GRAY, lw=1)
            ax.set_title(
                f"{meth}: span {span}" + (" (dim 96)" if span == "all" else " (dim = |S∩L|)")
            )
            ax.set_xlabel("layer (projection at this layer only)")
            ax.set_xticks(LAYERS)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, handlelength=4)
    for ax in axes[:, 0]:
        ax.set_ylabel("drop of the verdict separation (1 − sep. share)")
    fig.suptitle("T3 (A8): removing the components' write span from the original MLP output, vs controls "
                 "(test, training prompt)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(figs / "t3_a8.png", dpi=150)
    plt.close(fig)
    md += (
        "\n## 2d. T3 (A8): projecting the components' write span out of the original MLP output\n\n"
        "![T3](figs/t3_a8.png)\n\n"
        "One layer at a time, everything else original; test, training prompt. Span 'all' = all 96 "
        "write vectors of the layer; 'active' = the write vectors of S in that layer (dimension = their "
        "number; '—' where S has none). Controls of the same dimension: top principal directions of y "
        "(pca) and 3 random subspaces. Cells: separation share, shift (macro) and (1→0 / 0→1) flips; "
        "random: means over the 3 draws.\n\n"
        + md_table(
            [
                "run",
                "layer",
                "span all",
                "pca (dim 96)",
                "random (dim 96)",
                "span active",
                "pca (dim |S∩L|)",
                "random (dim |S∩L|)",
            ],
            rows,
        )  # fmt: skip
    )
    return md, details


# K2 pairs by the OOD family's form: the affirmative val form transfers to plain affirmative OOD files
# and to conjunctions/disjunctions, shown apart because "A and B" need not keep a simple statement's sign
K2_GROUPS = (
    ("negated", "negated → negated", "#6250d6"),
    ("affirmative", "affirmative → affirmative", "#2a78d6"),
    ("composed", "affirmative → conj/disj", "#eb6834"),
)


def k2_pairs(k2: dict[str, Any], target: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(val AUC, OOD AUC) of every (selective component, OOD family) pair of one basis (coef), by
    K2_GROUPS key; one target ("label" or "verdict")."""
    out: dict[str, tuple[list[float], list[float]]] = {g: ([], []) for g, _, _ in K2_GROUPS}
    for v in k2.values():
        for d in v["detail"]:
            if d["target"] != target:
                continue
            fm = form_of(d["family"])
            g = "composed" if fm in ("conjunction", "disjunction") else fm
            out[g][0].extend(v["val_auc"])
            out[g][1].extend(d["auc"])
    return {g: (np.array(x), np.array(y)) for g, (x, y) in out.items()}


def transfers(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The K2 rule (paper_correlation.py): same side of 0.5 as on val, at least TRANSFER away."""
    return (np.sign(y - 0.5) == np.sign(x - 0.5)) & (np.abs(y - 0.5) >= TRANSFER)


def k2_figures(C: dict[str, Any], figs: Path) -> None:
    """Scatter of val vs OOD AUC per basis (one figure per target) and a dot plot of the shares."""
    for target in ("label", "verdict"):
        fig, axes = plt.subplots(3, 3, figsize=(13, 12), sharex=True, sharey=True)
        for ax, o in zip(axes.flat, ALL, strict=True):
            k2 = C[o.stem]["K2_coef"]
            pairs = k2_pairs(k2, target)
            for g, name, color in K2_GROUPS:
                x, y = pairs[g]
                if len(x):
                    ax.scatter(x, y, s=10, alpha=0.5, color=color, edgecolors="none",
                               label=f"{name} ({len(x)} pairs)")  # fmt: skip
            ax.fill_between([0.5, 1], 0.5 + TRANSFER, 1, color=GRAY, alpha=0.15, lw=0)
            ax.fill_between([0, 0.5], 0, 0.5 - TRANSFER, color=GRAY, alpha=0.15, lw=0)
            for v in (0.5 - SELECT, 0.5 + SELECT):
                ax.axvline(v, color=GRAY, lw=0.8, ls=":")
            ax.axhline(0.5, color=GRAY, lw=1)
            ax.axvline(0.5, color=GRAY, lw=1)
            n_sel = {form: len(v["selective"]) for form, v in k2.items()}
            ax.set_title(f"{o.label} (selective: neg {n_sel['negated']}, aff {n_sel['affirmative']})",
                         fontsize=10)  # fmt: skip
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7, loc="upper left")
        for ax in axes[-1]:
            ax.set_xlabel("val macro AUC within the form")
        for ax in axes[:, 0]:
            ax.set_ylabel(f"AUC on the OOD family ({target})")
        fig.suptitle(
            f"K2: selectivity on val vs on each OOD family of the matching form, target = {target}\n"
            "(coef; selective on val = outside the dotted lines; shaded = counts as transfer)"
        )  # fmt: skip
        fig.tight_layout()
        fig.savefig(figs / f"k2_scatter_{target}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    ys = np.arange(len(ALL))[::-1]
    for ax, (g, name, _) in zip(axes, K2_GROUPS, strict=True):
        for yi, o in zip(ys, ALL, strict=True):
            k2 = C[o.stem]["K2_coef"]
            n_sel = len(k2["negated" if g == "negated" else "affirmative"]["selective"])
            for target, marker, dy in (("label", "o", 0.12), ("verdict", "s", -0.12)):
                x, y = k2_pairs(k2, target)[g]
                if len(x):
                    ax.scatter(transfers(x, y).mean(), yi + dy, marker=marker, s=40, color=COLOR[o.method],
                               edgecolors="white" if o.seed != 2 else "black", linewidths=0.8)  # fmt: skip
            ax.text(1.04, yi, f"n = {n_sel}", va="center", fontsize=8)
        ax.set_title(name)
        ax.set_xlim(-0.02, 1.02)  # room for dots at exactly 0 or 1
        ax.set_xlabel("share of pairs that transfer")
        ax.grid(alpha=0.25, axis="x")
    axes[0].set_yticks(ys, [o.label for o in ALL])
    # one legend below the panels (inside a panel it was too small to notice and covered data);
    # colour = basis, read from the row labels
    keys = [
        plt.Line2D([], [], ls="", marker="o", ms=8, color=GRAY, label="target: label (13 ground-truth files)"),
        plt.Line2D([], [], ls="", marker="s", ms=8, color=GRAY, label="target: verdict (all 24 files)"),
        plt.Line2D([], [], ls="", marker="o", ms=8, color=GRAY, markeredgecolor="white", label="seed 1 (no outline)"),
        plt.Line2D([], [], ls="", marker="o", ms=8, color=GRAY, markeredgecolor="black", label="seed 2 (black outline)"),
    ]  # fmt: skip
    fig.legend(handles=keys, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle("K2: transfer shares by OOD form (coef; n = selective components on val)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(figs / "k2_shares.png", dpi=150)
    plt.close(fig)


def check_k2_split(C: dict[str, Any]) -> None:
    """The split shares recombine to the stored K2 shares (same rule, same pairs)."""
    for o in ALL:
        k2 = C[o.stem]["K2_coef"]
        for target in ("label", "verdict"):
            pairs = k2_pairs(k2, target)
            for form, groups in (
                ("negated", ("negated",)),
                ("affirmative", ("affirmative", "composed")),
            ):
                ok = np.concatenate([transfers(*pairs[g]) for g in groups])
                stored = k2[form][f"share_{target}"]
                assert len(ok) == k2[form]["n_pairs"][target], (o.label, form, target)
                assert (np.isnan(stored) and not len(ok)) or abs(ok.mean() - stored) < 1e-12, (
                    o.label, form, target, ok.mean(), stored)  # fmt: skip


def correlation_section(phase_d: Path, figs: Path) -> tuple[str, str]:
    C = {o.stem: load_json(phase_d / "out" / f"correlation_{o.stem}.json") for o in ALL}
    targets = (("label", "test"), ("label", "ood"), ("verdict", "test"), ("verdict", "ood"))
    hdr = ["basis", "features"] + [f"{t} {s}" for t, s in targets]

    def cells(d: dict[str, Any]) -> list[str]:
        return [f"{f(d[t][s]['macro_accuracy'])} ({f(d[t][s]['auc'])})" for t, s in targets]

    rows = []
    resid = C[FINALS[0].stem]["K0"]
    for r in ("last", "mean"):
        rows.append(["(original model)", f"resid_{r}"] + cells(resid[f"resid_{r}"]))
    for o in ALL:
        for fs, d in C[o.stem]["K0"].items():
            if fs.startswith("resid"):
                continue
            rows.append([o.label, fs] + cells(d))
    md = (
        "\n## 3a. K0: probes over all features (logistic regression, fit on train)\n\n"
        "Cells: macro accuracy over families (pooled AUC). Features: coef = CI_c (v_c · x) per "
        "component (data-aligned: v_c · x), ci = CI_c, resid = residual stream after layer 17; at the "
        "statement's last token (last) or the mean over its tokens (mean). Label = ground truth "
        "(OOD: 13 ground-truth files), verdict = the model's own verdict (OOD: all 24 files).\n\n"
        + md_table(hdr, rows)
    )
    # ---- K1 sparse probes ----
    rows = []
    # 2 x 2: rows = target (label, verdict), columns = set (test, OOD), the order of `targets`
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for o in ALL:
        for kind in ("coef", "ci"):
            key = f"K1_{kind}"
            if key not in C[o.stem]:
                continue
            sp = C[o.stem][key]["sparse_probes"]
            for k, d in sp.items():
                rows.append([o.label, kind, k, ", ".join(d["components"][:5])] + cells(d))
            if kind == "coef":
                for ax, (t, s) in zip(axes.flat, targets, strict=True):
                    ks = sorted(int(k) for k in sp)
                    ax.plot(ks, [sp[str(k)][t][s]["macro_accuracy"] for k in ks], marker="o", ms=4,
                            label=o.label, **style(o))  # fmt: skip
    for ax, (t, s) in zip(axes.flat, targets, strict=True):
        ax.set_title(f"{t} / {s}")
        ax.set_xlabel("k components (top by train selectivity)")
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("probe macro accuracy (coef features)")
    # one legend for all panels, outside, so it covers no data
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="center right", fontsize=8, handlelength=4
    )
    fig.suptitle("K1: probes on the k most selective single components")
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(figs / "k1_sparse_probes.png", dpi=150)
    plt.close(fig)
    md += (
        "\n## 3b. K1: sparsity (probes on the top-k single components by train selectivity)\n\n"
        "Components ranked by |macro AUC − 0.5| on train (readout last/mean chosen per component on "
        "train); logistic regression on the top k, fit on train. Cells as in K0. The coef features "
        "are comparable across bases (the figure); ci only exists for decompositions.\n\n"
        "![K1](figs/k1_sparse_probes.png)\n\n"
        + md_table(
            ["basis", "kind", "k", "components (first 5)"] + [f"{t} {s}" for t, s in targets], rows
        )
    )
    # single-component AUC summaries from the npz tables
    rows = []
    for o in ALL:
        z = np.load(phase_d / "out" / f"correlation_{o.stem}.npz")
        for kind in ("coef", "ci"):
            if f"K1_{kind}_test_label_auc" not in z.files:
                continue
            row = [o.label, kind]
            for s, t in (
                ("test", "label"),
                ("test", "verdict"),
                ("ood", "label"),
                ("ood", "verdict"),
            ):
                tab = z[
                    f"K1_{kind}_{s}_{t}_auc"
                ]  # (families, components), nan for one-class families
                sel = np.abs(np.nanmean(tab, axis=0) - 0.5)
                sel = sel[~np.isnan(sel)]
                row += [f(float(sel.max())), f(float(np.mean(sel >= 0.2)), 3)]
            rows.append(row)
    md += (
        "\nSingle components: macro (over families) |AUC − 0.5| of each component on its own; the best "
        "component and the share of components at ≥ 0.2 (the K2 selection bar). OOD label: all 24 files "
        "here (the per-family tables include the 11 files where the model is often wrong).\n\n"
        + md_table(
            [
                "basis",
                "kind",
                "test label: best",
                "share ≥ 0.2",
                "test verdict: best",
                "share ≥ 0.2",
                "OOD label: best",
                "share ≥ 0.2",
                "OOD verdict: best",
                "share ≥ 0.2",
            ],
            rows,
        )  # fmt: skip
    )
    # ---- K2, K3 ----
    check_k2_split(C)
    k2_figures(C, figs)
    rows2, rows3 = [], []
    for o in ALL:
        for kind in ("coef", "ci"):
            if f"K2_{kind}" not in C[o.stem]:
                continue
            for form, v in C[o.stem][f"K2_{kind}"].items():
                rows2.append([o.label, kind, form, str(len(v["selective"])),
                              f"{f(v['share_label'])} ({v['n_pairs']['label']})",
                              f"{f(v['share_verdict'])} ({v['n_pairs']['verdict']})"])  # fmt: skip
            for form, v in C[o.stem][f"K3_{kind}"].items():
                oa = v.get("oriented_auc")
                rows3.append([o.label, kind, form, str(v["n"]),
                              str(len(oa)) if oa else "0",
                              f(v["share_follow_label"]) if v["share_follow_label"] is not None else "—",
                              f(float(np.median(oa))) if oa else "—"])  # fmt: skip
    md += (
        "\n## 3c. K2: transfer of single-component selectivity to OOD\n\n"
        "Selective on val within a sentence form: |macro AUC − 0.5| ≥ 0.2. Transfer: same sign and "
        "|AUC − 0.5| ≥ 0.1 on an OOD family of the matching form (negated → negated; affirmative → "
        "affirmative, conjunction, disjunction). Share of (component, OOD family) pairs that transfer "
        "(number of pairs). Label: 13 ground-truth files; verdict: all 24. Figures (coef only): the "
        "shares split by OOD form (affirmative → plain affirmative vs conjunctions/disjunctions), and "
        "every pair's val AUC against its OOD AUC, per basis (one point per component and OOD family, so "
        "a component appears once per family; shaded = counts as transfer).\n\n"
        "![K2 shares](figs/k2_shares.png)\n\n![K2 scatter, label](figs/k2_scatter_label.png)\n\n"
        "![K2 scatter, verdict](figs/k2_scatter_verdict.png)\n\n"
        + md_table(
            [
                "basis",
                "kind",
                "val form",
                "n selective",
                "share transfer: label (pairs)",
                "verdict (pairs)",
            ],
            rows2,
        )  # fmt: skip
        + "\n## 3d. K3: truth vs verdict\n\n"
        "Test + OOD statements of the matching forms where the model's verdict contradicts the label. "
        "Per K2-selective component, the AUC against the label, oriented by its val sign: > 0.5 = "
        "follows the label (truth), < 0.5 = follows the verdict. Share of components that follow the "
        "label, and the median oriented AUC.\n\n"
        + md_table(
            [
                "basis",
                "kind",
                "form",
                "n statements",
                "n components",
                "share follow label",
                "median oriented AUC",
            ],
            rows3,
        )  # fmt: skip
    )
    return md, ""


GROUP_EDITS = ("reverse top10 group", "reverse selective group", "amplify top10 group",
               "amplify selective group")  # fmt: skip


def manipulation_group_figures(M: dict[str, Any], figs: Path) -> None:
    """4a as figures: per group edit, the flip shares and the separation share and shift per basis
    (test and OOD, training prompt), and the paired-twin test (test). Second prompt left out to keep them readable (in the table)."""
    ys = np.arange(len(ALL))[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True, sharex=True)
    for ax, name in zip(axes.flat, GROUP_EDITS, strict=True):
        for yi, o in zip(ys, ALL, strict=True):
            m = M[o.stem]
            if name not in m["sets"]["test"]["train"]["conditions"]:
                ax.text(0.01, yi, "(no such group)", va="center", fontsize=8, color=GRAY)
                continue
            n = len(m["top"]) if "top10" in name else len(m["selective_group"])
            for split, dy, filled in (("test", 0.15, True), ("ood", -0.15, False)):
                r = m["sets"][split]["train"]["conditions"][name]["macro"]
                for key, marker in (("flip_1_to_0", "v"), ("flip_0_to_1", "^")):
                    ax.scatter(r[key], yi + dy, marker=marker, s=55, linewidths=1.5,
                               color=COLOR[o.method] if filled else "none",
                               edgecolors=COLOR[o.method])  # fmt: skip
            ax.text(1.04, yi, f"n = {n}", va="center", fontsize=8)
        ax.set_title(name)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(alpha=0.25, axis="x")
    for ax in axes[-1]:
        ax.set_xlabel("share of the model's own verdicts flipped (macro over families)")
    for ax in axes[:, 0]:
        ax.set_yticks(ys, [o.label for o in ALL])
    keys = [
        plt.Line2D([], [], ls="", marker="v", ms=8, color=GRAY, label="1 → 0 (true verdicts flipped to false)"),
        plt.Line2D([], [], ls="", marker="^", ms=8, color=GRAY, label="0 → 1 (false verdicts flipped to true)"),
        plt.Line2D([], [], ls="", marker="o", ms=8, color=GRAY, label="filled = test"),
        plt.Line2D([], [], ls="", marker="o", ms=8, color="none", markeredgecolor=GRAY, label="hollow = OOD (13 files)"),
    ]  # fmt: skip
    fig.legend(handles=keys, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle(
        "4a: group edits at every statement token, training prompt (n = components in the group)"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figs / "m_groups.png", dpi=150)
    plt.close(fig)

    # separation share and shift, both in units of the original true-false gap, so one x axis
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True, sharex=True)
    for ax, name in zip(axes.flat, GROUP_EDITS, strict=True):
        for yi, o in zip(ys, ALL, strict=True):
            m = M[o.stem]
            if name not in m["sets"]["test"]["train"]["conditions"]:
                ax.text(0.01, yi, "(no such group)", va="center", fontsize=8, color=GRAY)
                continue
            for split, dy, filled in (("test", 0.15, True), ("ood", -0.15, False)):
                r = m["sets"][split]["train"]["conditions"][name]["macro"]
                for x, marker, size in ((r["sep_share"], "o", 55), (r["shift"], "D", 45)):
                    ax.scatter(x, yi + dy, marker=marker, s=size, linewidths=1.5,
                               color=COLOR[o.method] if filled else "none",
                               edgecolors=COLOR[o.method])  # fmt: skip
                # '|' is a line marker: no face, so test and OOD are told apart by the row offset
                ax.scatter(r["abs_shift"], yi + dy, marker="|", s=90, linewidths=1.5,
                           color=COLOR[o.method])  # fmt: skip
        ax.axvline(0, color=GRAY, lw=1)
        ax.axvline(1, color=GRAY, lw=1, ls=":")
        ax.set_title(name)
        ax.grid(alpha=0.25, axis="x")
    for ax in axes[-1]:
        ax.set_xlabel(
            "units of the original true−false gap of the verdict score (macro over families)"
        )
    for ax in axes[:, 0]:
        ax.set_yticks(ys, [o.label for o in ALL])
    keys = [
        plt.Line2D([], [], ls="", marker="o", ms=8, color=GRAY, label="separation share (1 = unchanged)"),
        plt.Line2D([], [], ls="", marker="D", ms=7, color=GRAY, label="shift (> 0 = toward '1')"),
        plt.Line2D([], [], ls="", marker="|", ms=10, mew=1.5, color=GRAY, label="|shift| (macro of |·|)"),
        plt.Line2D([], [], ls="", marker="o", ms=8, color=GRAY, label="filled = test"),
        plt.Line2D([], [], ls="", marker="o", ms=8, color="none", markeredgecolor=GRAY, label="hollow = OOD (13 files)"),
    ]  # fmt: skip
    fig.legend(handles=keys, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("4a: group edits, separation share and shift (training prompt)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figs / "m_share_shift.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = {"reverse top10 group": "o", "reverse selective group": "s",
               "amplify top10 group": "D", "amplify selective group": "P"}  # fmt: skip
    originals = set()
    for yi, o in zip(ys, ALL, strict=True):
        for name, marker in markers.items():
            c = M[o.stem]["sets"]["test"]["train"]["conditions"].get(name)
            if c is None:
                continue
            tt = c["twin_test"]
            originals.add(round(tt["share_follows_twin_original"], 12))
            ax.scatter(tt["share_follows_twin"], yi, marker=marker, s=50, color=COLOR[o.method],
                       alpha=0.85)  # fmt: skip
    assert len(originals) == 1, originals  # the original model is the same for every basis
    ax.axvline(originals.pop(), color=GRAY, ls="--", lw=1.2)
    # gray legend markers: colour = basis (row labels), shape = edit
    keys = [
        plt.Line2D([], [], ls="", marker=mk, ms=7, color=GRAY, label=nm)
        for nm, mk in markers.items()
    ]
    keys.append(plt.Line2D([], [], color=GRAY, ls="--", lw=1.2, label="original model"))
    ax.set_yticks(ys, [o.label for o in ALL])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(
        "share of negated test statements whose edited verdict = the twin's original verdict"
    )
    ax.set_title("4a: paired-twin test ('the not is ignored'), test, training prompt")
    ax.grid(alpha=0.25, axis="x")
    ax.legend(handles=keys, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(figs / "m_twin.png", dpi=150)
    plt.close(fig)


def manipulation_section(phase_d: Path, out: Path) -> tuple[str, str]:
    M = {o.stem: load_with_shift(phase_d, "manipulation", o) for o in ALL}

    def get(o: Obj, split: str, prompt: str, name: str) -> dict[str, Any]:
        return M[o.stem]["sets"][split][prompt]["conditions"][name]

    def flips(r: dict[str, Any]) -> str:
        m = r["macro"]
        return f"{f(m['flip_1_to_0'])} / {f(m['flip_0_to_1'])}"

    rows = []
    for o in ALL:
        for op in ("reverse", "amplify"):
            for grp in ("top10 group", "selective group"):
                name = f"{op} {grp}"
                if name not in M[o.stem]["sets"]["test"]["train"]["conditions"]:
                    continue
                r = get(o, "test", "train", name)
                n = (
                    len(M[o.stem]["top"])
                    if grp.startswith("top")
                    else len(M[o.stem]["selective_group"])
                )
                rows.append([o.label, name, str(n), flips(r), f(r["macro"]["sep_share"]),
                             fs(r["macro"]["shift"]), f(r["macro"]["abs_shift"]), f(r["macro"]["digit_mass"]),
                             f(r["twin_test"]["share_follows_twin"]) + f" (orig. {f(r['twin_test']['share_follows_twin_original'])})",
                             flips(get(o, "test", "second", name)), flips(get(o, "ood", "train", name)),
                             flips(get(o, "ood", "second", name))])  # fmt: skip
    manipulation_group_figures(M, out / "figs")
    md = (
        "\n## 4a. Manipulation, fixed core: groups\n\n"
        "Full model, writes of the chosen components multiplied by −1 (reverse) or 2 (amplify) at every "
        "statement token. top10 = the 10 largest val T1 drops; selective = the K2 val-selective "
        "components (coef readout, both forms). Flips '1→0 / 0→1' = share of the model's own '1' "
        "verdicts that go to '0' / of '0' verdicts that go to '1' (macro). Separation share, shift and "
        "|shift| on test/train (macro). Twin test (test, negated "
        "statements with an affirmative twin): share whose edited verdict equals the twin's original "
        "verdict (original model in brackets; 'the not is ignored' predicts a rise). OOD = 13 "
        "ground-truth files, full.\n\n"
        "Figures (training prompt; test filled, OOD hollow; the second prompt is in the table only): "
        "separation share and shift of each group edit, then its flip shares, then the twin test.\n\n"
        "![manipulation groups: share and shift](figs/m_share_shift.png)\n\n"
        "![manipulation groups: flips](figs/m_groups.png)\n\n![twin test](figs/m_twin.png)\n\n"
        + md_table(
            [
                "basis",
                "condition",
                "n comps",
                "test/train flips",
                "sep. share",
                "shift",
                "|shift|",
                "digit mass",
                "twin follows",
                "test/second flips",
                "ood/train flips",
                "ood/second flips",
            ],
            rows,
        )  # fmt: skip
    )
    # singles: strongest of the top 10 vs random active components
    rows, detail_rows = [], []
    for o in ALL:
        for op in ("reverse", "amplify"):
            singles = [n for n in M[o.stem]["sets"]["test"]["train"]["conditions"]
                       if n.startswith(f"{op} L")]  # fmt: skip
            randoms = [n for n in M[o.stem]["sets"]["test"]["train"]["conditions"]
                       if n.startswith(f"{op} random")]  # fmt: skip

            def best(names: list[str], key: str, o: Obj = o) -> str:
                if not names:
                    return "—"
                vals = [(get(o, "test", "train", n)["macro"][key], n.split()[-1]) for n in names]
                v, nm = max(vals)
                return f"{f(v)} ({nm})"

            rows.append([o.label, op, best(singles, "flip_1_to_0"), best(singles, "flip_0_to_1"),
                         best(randoms, "flip_1_to_0"), best(randoms, "flip_0_to_1"),
                         f(float(np.mean([get(o, "test", "train", n)["macro"]["flip_0_to_1"] for n in randoms])))
                         if randoms else "—"])  # fmt: skip
            for n in singles + randoms:
                r = get(o, "test", "train", n)
                detail_rows.append([o.label, n, flips(r), f(r["macro"]["sep_share"]), fs(r["macro"]["shift"]),
                                    f(r["macro"]["digit_mass"]),
                                    f(r["twin_test"]["share_follows_twin"]), flips(get(o, "test", "second", n))])  # fmt: skip
    md += (
        "\n## 4b. Manipulation, fixed core: single components (test, training prompt)\n\n"
        "Largest flip share over the 10 top-ranked single components (which one) vs over 10 random "
        "components of the active set (trained runs only; the untrained decompositions and the "
        "data-aligned basis are themselves the effect-selected controls). Every single condition in "
        "details.md.\n\n"
        + md_table(
            [
                "basis",
                "op",
                "top-10 singles: max 1→0",
                "max 0→1",
                "random singles: max 1→0",
                "max 0→1",
                "random mean 0→1",
            ],
            rows,
        )  # fmt: skip
    )
    details = "\n## Manipulation: every single-component condition (test)\n\n" + md_table(
        ["basis", "condition", "train flips 1→0 / 0→1", "sep. share", "shift", "digit mass", "twin follows",
         "second-prompt flips"], detail_rows)  # fmt: skip

    # ---- wide val table (exploratory) ----
    md += (
        "\n## 4c. Exploratory (val only): every active component reversed alone\n\n"
        "**Exploratory, val only**: anything here becomes a finding only after a test/OOD check. Full "
        "per-family tables (flips and shift) in `wide_val_reverse_<run>.csv`. Top 5 per run by the "
        "larger flip share (pooled; share and shift pooled too).\n\n"
    )
    rows = []
    for o in FINALS:
        wide = M[o.stem]["wide_val_reverse"]
        fams = sorted(next(iter(wide.values()))["per_family"])
        with open(out / f"wide_val_reverse_{o.label.replace(' ', '_')}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            fam_keys = ("flip_1_to_0", "flip_0_to_1", "shift")
            w.writerow(["component", "flip_1_to_0", "flip_0_to_1", "sep_share", "shift", "digit_mass"]
                       + [f"{g}:{k}" for g in fams for k in fam_keys])  # fmt: skip
            for nm, v in wide.items():
                p = v["pooled"]
                w.writerow([nm, p["flip_1_to_0"], p["flip_0_to_1"], p["sep_share"], p["shift"], p["digit_mass"]]
                           + [v["per_family"][g][k] for g in fams for k in fam_keys])  # fmt: skip
        ranked = sorted(
            wide.items(),
            key=lambda kv: -max(kv[1]["pooled"]["flip_1_to_0"], kv[1]["pooled"]["flip_0_to_1"]),
        )
        for nm, v in ranked[:5]:
            p = v["pooled"]
            fam_best = max([(v["per_family"][g]["flip_0_to_1"], g, "0→1") for g in fams]
                           + [(v["per_family"][g]["flip_1_to_0"], g, "1→0") for g in fams],
                           key=lambda x: -1 if np.isnan(x[0]) else x[0])  # fmt: skip
            rows.append([o.label, nm, f"{f(p['flip_1_to_0'])} / {f(p['flip_0_to_1'])}", f(p["sep_share"]),
                         fs(p["shift"]), f"{fam_best[1]} {fam_best[2]} {f(fam_best[0])}"])  # fmt: skip
    md += md_table(
        ["run", "component", "flips 1→0 / 0→1", "sep. share", "shift", "largest family flip"], rows
    )
    return md, details


def directions_figures(D: dict[str, Any], figs: Path) -> None:
    """Section 5 as figures: gap shares of A and D per layer, alignment (max |cos|) against the random
    expectation, and the verdict cost of removing the direction itself (from the data-aligned run, the
    one that ran --remove_directions; the directions are the same for every basis)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for row_i, dn in enumerate(("mm", "lr")):
        for col, (part, what) in enumerate((("share_A", "components' CI-weighted writes (A)"),
                                            ("share_D", "Δ's write (D = y − C)"))):  # fmt: skip
            ax = axes[row_i, col]
            for o in ALL:
                ax.plot(LAYERS, [D[o.stem]["layers"][f"L{lay}"][dn][part] for lay in LAYERS],
                        marker="o", ms=4, label=o.label, **style(o))  # fmt: skip
            ax.axhline(0, color=GRAY, lw=1)
            ax.axhline(1, color=GRAY, lw=1, ls=":")
            ax.set_title(f"{dn} direction: gap share of {what}")
            ax.set_xticks(LAYERS)
            ax.grid(alpha=0.25)
        axes[row_i, 0].set_ylabel("share of the MLP output's true−false gap along d")
    for ax in axes[-1]:
        ax.set_xlabel("layer")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="center right", fontsize=9, handlelength=4
    )
    fig.suptitle("5: how much of the truth-direction gap of the MLP output each part writes "
                 "(test, last statement token; A + I + D = 1)")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(figs / "d_gap_shares.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, dn in zip(axes, ("mm", "lr"), strict=True):
        for o in ALL:
            ax.plot(LAYERS, [D[o.stem]["layers"][f"L{lay}"][dn]["max_abs_cos"] for lay in LAYERS],
                    marker="o", ms=4, label=o.label, **style(o))  # fmt: skip
        ref = [np.mean([D[o.stem]["layers"][f"L{lay}"][dn]["random_max_abs_cos_per_C"] for o in ALL])
               for lay in LAYERS]  # fmt: skip
        ax.plot(
            LAYERS, ref, color="black", lw=1.5, ls=":", label="expected max of 96 random vectors"
        )
        ax.set_title(f"{dn} direction")
        ax.set_xlabel("layer")
        ax.set_xticks(LAYERS)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("max |cos(write vector, d)| over the layer's 96")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=9,
               handlelength=4, frameon=False)  # fmt: skip
    fig.suptitle("5: best alignment of any single write vector with the truth direction (test)")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(figs / "d_alignment.png", dpi=150)
    plt.close(fig)

    rem = D[DATA_ALIGNED.stem]["direction_removal"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [*LAYERS, LAYERS[-1] + 1.5]
    for dn, color in (("mm", "#6250d6"), ("lr", "#1baf7a")):
        names = [f"{dn} L{lay}" for lay in LAYERS] + [f"{dn} all layers"]
        drop = [1 - rem[nm]["macro"]["sep_share"] for nm in names]
        ax.plot(xs[:-1], drop[:-1], marker="o", color=color, lw=2, label=f"{dn}: one layer")
        ax.scatter(
            xs[-1], drop[-1], marker="*", s=180, color=color, label=f"{dn}: all 7 layers at once"
        )
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xticks(xs, [f"L{lay}" for lay in LAYERS] + ["all"])
    ax.set_ylabel("drop of the verdict separation (1 − sep. share)")
    ax.set_title("5: removing the truth direction (1 dim) from the original MLP output (test)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "d_removal.png", dpi=150)
    plt.close(fig)


def directions_section(phase_d: Path, figs: Path) -> tuple[str, str]:
    D = {o.stem: load_json(phase_d / "out" / f"directions_{o.stem}.json") for o in ALL}
    directions_figures(D, figs)
    md = (
        "\n## 5a. Truth directions vs the components' writes (test, statement's last token)\n\n"
        "Directions after each layer, fit on train: mass-mean (mm) and logistic regression (lr). Gap "
        "share = (mean true − mean false of the projection of X onto d) / (the same for the MLP output "
        "y), X = A (CI-weighted writes of all components; data-aligned: all writes), I = the rest of "
        "the components' writes (C − A), D = Δ's write (y − C); A + I + D = 1.\n\n"
        "![gap shares](figs/d_gap_shares.png)\n\n![alignment](figs/d_alignment.png)\n\n"
        "![direction removal](figs/d_removal.png)\n\n"
    )
    first = D[FINALS[0].stem]["layers"]
    md += (
        "cos(mm, lr) per layer: "
        + ", ".join(f"L{lay} {f(first[f'L{lay}']['cos_mm_lr'])}" for lay in LAYERS)
        + "\n\n"
    )
    for dn in ("mm", "lr"):
        for part in ("share_A", "share_D"):
            rows = [
                [o.label] + [f(D[o.stem]["layers"][f"L{lay}"][dn][part]) for lay in LAYERS]
                for o in ALL
            ]
            md += f"**{dn}: {part.replace('share_', 'gap share of ')}**\n\n" + md_table(
                ["basis"] + [f"L{lay}" for lay in LAYERS], rows) + "\n"  # fmt: skip
        rows = [[o.label] + [f"{f(D[o.stem]['layers'][f'L{lay}'][dn]['max_abs_cos'])} / "
                             f"{f(D[o.stem]['layers'][f'L{lay}'][dn]['random_max_abs_cos_per_C'])}"
                             for lay in LAYERS] for o in ALL]  # fmt: skip
        md += (f"**{dn}: max |cos(u_c, d)| over the layer's 96 write vectors / expected max of 96 random "
               "vectors**\n\n" + md_table(["basis"] + [f"L{lay}" for lay in LAYERS], rows) + "\n")  # fmt: skip
        rows = [[o.label] + [f(D[o.stem]["layers"][f"L{lay}"][dn].get("spearman_t1_drop_vs_abs_gapshare"))
                             for lay in LAYERS] for o in ALL]  # fmt: skip
        md += (f"**{dn}: Spearman correlation of T1 drop (val) and |gap share| over the layer's "
               "candidates** ('—' < 3 candidates; nan = all candidates have CI 0 at the last token)\n\n"
               + md_table(["basis"] + [f"L{lay}" for lay in LAYERS], rows) + "\n")  # fmt: skip
    rem = D[DATA_ALIGNED.stem]["direction_removal"]
    rows = [[nm, f(r["macro"]["sep_share"]), f(r["pooled"]["agreement"]), f(r["macro"]["flip_1_to_0"]),
             f(r["macro"]["flip_0_to_1"]), f(r["pooled"]["kl_recovered"])] for nm, r in rem.items()]  # fmt: skip
    md += (
        "**Removing the direction itself** from the original MLP output at the statement tokens "
        "(1 dimension per layer; 'all layers' = every layer at once). Test, training prompt. Compare "
        "with T2 (removing components). KL rec. here = 1 − KL(edit) / KL(MLPs off). No shift: the "
        "removal was stored as summaries only.\n\n"
        + md_table(["direction, layer", "sep. share", "agreement", "1→0", "0→1", "KL rec."], rows)
    )
    details = "\n## Top gap-share components per layer (mm)\n\n"
    rows = []
    for o in ALL:
        for lay in LAYERS:
            top = D[o.stem]["layers"][f"L{lay}"]["mm"]["top_gap_share"][:5]
            rows.append([o.label, f"L{lay}", "; ".join(f"{nm} {v:+.3f}" for nm, v in top)])
    details += md_table(["basis", "layer", "top 5 by |gap share|"], rows)
    return md, details


def main(phase_d: str, phase_c: str, out: str) -> None:
    pd, pc, od = Path(phase_d), Path(phase_c), Path(out)
    figs = od / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    checks = check_recomputation(pd)
    for c in checks:
        print("check:", c)
    rec_md, rec_det, t0 = recovery_section(pd, figs)
    val_md, val_det = valley_section(pd, figs)
    term_md, term_det = termination_section(pd, figs)
    corr_md, _ = correlation_section(pd, figs)
    man_md, man_det = manipulation_section(pd, od)
    dir_md, dir_det = directions_section(pd, figs)
    runs = "; ".join(f"{o.label} = `{o.stem}`" for o in ALL)
    head = (
        "# Paper analyses: tables (stage 1a)\n\n"
        "Generated by `spd/experiments/lm/honesty_targeted_decomposition/paper_tables.py` from the "
        f"phase D outputs in `{pd}` and the phase C finals readout in `{pc}`. No interpretation here; "
        "see the topic LOG `convos/julian/26-10-29_paper_writing_LOG.md`.\n\n"
        f"Bases: {runs}. Seeds are never pooled. KL recovered is shown pooled and macro "
        "(aggregation open); separation share and flips are macro over families, agreement pooled.\n\n"
        "Shift (added after the outputs were read; descriptive, not in the frozen plan) = signed move "
        "of the midpoint between the true and false class means of the verdict score s = log P('1') − "
        "log P('0'), in units of the original true−false gap; > 0 = toward '1'. With the separation "
        "share it fixes both class means; flips mix a closing gap with a shift. Macro over families; "
        "|shift| = macro of the per-family |shift| (the signed macro can cancel across families). "
        "Computed from the per-statement npz arrays where they exist (not for T1's val ranking and the "
        "direction removal).\n\n"
        "Consistency checks (stored summaries recomputed from per-statement arrays):\n"
        + "".join(f"- {c}\n" for c in checks)
        + "- shift: the stored separation share of every summary that got a shift (pooled and per "
        "family) recomputed from the same arrays: equal ("
        + ", ".join(f"{k} {v}" for k, v in SHIFT_CHECKED.items()) + " summaries)\n\n"
    )  # fmt: skip
    (od / "tables.md").write_text(
        head + side_effects_section(pc) + "\n" + rec_md + val_md + "\n# 2. Termination\n\n" + t0
        + term_md + "\n# 3. Correlation\n" + corr_md + "\n# 4. Manipulation\n" + man_md
        + "\n# 5. Directions\n" + dir_md
    )  # fmt: skip
    (od / "details.md").write_text(
        "# Paper analyses: detail tables\n\n" + rec_det + val_det + term_det + man_det + dir_det
    )
    print(f"wrote {od / 'tables.md'}, {od / 'details.md'}, figures in {figs}")


if __name__ == "__main__":
    fire.Fire(main)
