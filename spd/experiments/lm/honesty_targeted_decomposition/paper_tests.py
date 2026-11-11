"""Behavioural tests of the paper draft: one intervention on one part of the decomposed down_proj
outputs, read by chosen readouts (draft reports/26-11-04_tpd_truth_paper_draft.typ, section *The
tests*, Table 3; site versions = section 3.2.1). Topic: convos/julian/26-11-04_paper_draft_LOG.md,
entry *can the site-version tests (3.2.1) be run individually?* and the answers under Julian's
comments there. Splits and interventions: paper_interventions.py. Shared data code: paper_common.py;
prompts, digit and truth readouts: paper_side_effects.py (imported, not copied). Exploratory: not part
of the frozen phase D plan; the frozen scripts (paper_recovery/termination/...) are left as they are.

Every call is one cell group of the draft's tables: a target (the whole site, or a component set's
P or R), a list of interventions and a list of readouts, on one statement split.

Statements and twins. Twins = the true and the false statement of one subject in one family (exactly
one of each; sp_en_trans' false statements mostly use another Spanish word, so few twins there).
Twin interventions (twin_average, swap) run on the twins of equal token layout whose training-prompt
verdicts differ (draft, *Evaluation setup*), each twin once as x and once as x'. The other
interventions run on all statements of the split.

Readouts (--readouts, comma-separated):
  verdict           the training prompt, '1' vs '0'; with non-twin interventions also the next-token
                    KL at the statement positions per class (prefix / recall / object_end, as G2)
  second            the second 1/0 prompt (paper_common.SECOND_STIMULUS)
  yes_no:<key>      a Yes/No prompt of paper_side_effects.YES_NO_PROMPTS, ' Yes' vs ' No'
  digit:<key>       a digit question of paper_side_effects.DIGIT_QUESTIONS, digits 0-9
  recall:subject:<form>  "<statement>" + a question for the subject (RECALL_TEMPLATES[form], e.g.
                    "\\nThe city named in this statement is") -> the subject as written
  recall:object:<form>   the same for the object the statement names (for a false statement, the
                    false one)
  Recall of the true object (a knowledge question about the subject alone) was proposed and dropped by
  Julian ("We have enough tests already"; LOG entry above, 26-11-11).
  Recall is scored teacher-forced: the summed log-prob of the answer tokens and whether every answer
  token is the argmax (exact). Not with twin interventions (Julian: few readouts for the swaps).

Train statistics (mean, shift): on all train statements under each readout's prompt, because the
statement's last token differs between prompts ('.\\n' merges with a stimulus starting with '\\n',
'.' does not with ' Is'). Recall prompts start with '\\n' and use the training prompt's statistics.

Output in --out: tests_<stem>.json (summaries), tests_<stem>.npz (per-statement arrays: rows, labels,
families, the original's and each intervention's readout values, train statistics).

Usage:
  python paper_tests.py run --run <s-id> --target site --part P --split val
      --interventions zero,mean:scan3,mean:role --readouts verdict,second,digit:words --out <dir>
  python paper_tests.py run ... --interventions "" --readouts recall:subject:named,...   (originals only:
      the format check of new readouts)
  python paper_tests.py check_g2 --tests <tests npz> --g2 <G2 side_*.npz> --tol <float>
"""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import fire
import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer

from spd.configs import LMTaskConfig, LossPositions, StimulusLens
from spd.experiments.lm.honesty_targeted_decomposition.paper_common import (
    Batch,
    Prompt,
    Statements,
    bootstrap_intervals,
    decomposition_basis,
    load_statements,
    make_batches,
    maybe_cap,
    sorted_modules,
    verdict_prompts,
)
from spd.experiments.lm.honesty_targeted_decomposition.paper_interventions import (
    GROUPINGS,
    N_GROUPS,
    BatchContext,
    GroupAccumulator,
    GroupingName,
    Intervention,
    Part,
    SiteSplit,
    Split,
    TrainStats,
    component_split,
    describe_stats,
    edit_fn,
    group_ids,
    grouping_of,
    intervention_name,
    needs_twin,
    parse_intervention,
    part_of,
    site_hooks,
    to_numpy,
)
from spd.experiments.lm.honesty_targeted_decomposition.paper_side_effects import (
    DIGIT_QUESTIONS,
    DIGITS,
    POSITION_CLASSES,
    YES_NO_PROMPTS,
    digit_readout,
    load_run,
    object_token,
    position_classes,
    prompt_of,
    single_tokens,
)
from spd.experiments.lm.prepared_datasets import resolve_dataset_dir
from spd.experiments.lm.prompts_dataset import _tokenize_jointly, build_position_mask
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import bf16_autocast

N_BOOT = 1000

# ---- statement records, twins, recall answers ----

# subject and object as written in the statement, per family without the "neg_" prefix
STATEMENT_PATTERNS = {
    "cities": r"^The city of (.+) is (?:not )?in (.+)\.$",
    "element_symb": r"^(.+?) (?:has|does not have) the symbol (.+)\.$",
    "animal_class": r"^The (.+?) is (?:not )?an? (.+)\.$",
    "sp_en_trans": r"^The Spanish word '(.+)' (?:means|does not mean) '(.+)'\.$",
    "larger_than": r"^(.+) is larger than (.+)\.$",
    "smaller_than": r"^(.+) is smaller than (.+)\.$",
}
# (subject noun, object noun) per family; the answer is the span as written, with a leading space
RECALL_NOUNS: dict[str, tuple[str, str]] = {
    "cities": ("city", "country"),
    "element_symb": ("element", "symbol"),
    "animal_class": ("animal", "animal class"),
    "sp_en_trans": ("Spanish word", "English word"),
    "larger_than": ("first number", "second number"),
    "smaller_than": ("first number", "second number"),
}
# Two phrasings, compared in the val format check (26-11-11): in the CPU smoke test the "named ... is"
# form was often continued with a judgement (' not', ' incorrect', ' actually') instead of the name,
# so it partly reads truth; the colon form is meant to invite the bare name.
RECALL_TEMPLATES = {
    "named": "\nThe {noun} named in this statement is",
    "colon": "\nThe {noun} mentioned in the statement above:",
}
RecallKind = Literal["subject", "object"]


@dataclass(frozen=True)
class Records:
    """Per statement of a Statements set, in its order."""

    subject: list[str]  # as written in the statement
    obj: list[str]  # the object as written in the statement
    # the record's `subject` field, which keys twins: the city, element, animal or Spanish word, and
    # for the comparisons the number pair ("52-54", the same for both orders); not the parsed subject,
    # which for a comparison is only its first number
    twin_key: list[str]


def base_family(family: str) -> str:
    return family.removeprefix("neg_")


def parse_statement(text: str, family: str) -> tuple[str, str]:
    m = re.match(STATEMENT_PATTERNS[base_family(family)], text)
    assert m is not None, (text, family)
    return m.group(1), m.group(2)


def load_records(st: Statements, train_dirs: list[str]) -> Records:
    """Subject, object and twin key per statement."""
    assert st.split in ("train", "val", "test"), st.split
    fields = [
        json.loads(line)
        for d in train_dirs
        for line in (resolve_dataset_dir(d) / f"{st.split}.jsonl").read_text().splitlines()
    ]
    # the order of load_statements (probe_ci.load_split_with_labels); a capped set is a subset of it
    by_text = {r["text"]: r["subject"] for r in fields}
    assert len(by_text) == len(fields), "duplicate statement texts"
    parsed = [parse_statement(t, f) for t, f in zip(st.texts, st.families, strict=True)]
    return Records(
        subject=[s for s, _ in parsed],
        obj=[o for _, o in parsed],
        twin_key=[by_text[t] for t in st.texts],
    )


def twin_partner(st: Statements, rec: Records) -> np.ndarray:
    """Per statement the row of its twin (same family and twin key, other label), -1 if none. Only
    keys with exactly one true and one false statement in the set have twins."""
    by_key: dict[tuple[str, str], list[int]] = {}
    for i, (fam, s) in enumerate(zip(st.families, rec.twin_key, strict=True)):
        by_key.setdefault((str(fam), s), []).append(i)
    partner = np.full(len(st), -1)
    for rows in by_key.values():
        if len(rows) == 2 and st.labels[rows[0]] != st.labels[rows[1]]:
            partner[rows[0]], partner[rows[1]] = rows[1], rows[0]
    return partner


# ---- readouts ----


@dataclass(frozen=True)
class TokenReadout:
    """An answer read at the prompt's last token from a fixed token set."""

    name: str
    prompt: Prompt
    tokens: tuple[str, ...]
    kind: Literal["truth", "digit"]


@dataclass(frozen=True)
class RecallReadout:
    name: str
    kind: RecallKind
    template: str  # a RECALL_TEMPLATES key


Readout = TokenReadout | RecallReadout


def parse_readout(spec: str, lens: StimulusLens, train: Prompt, second: Prompt) -> Readout:
    match spec.split(":"):
        case ["verdict"]:
            return TokenReadout("verdict", train, ("1", "0"), "truth")
        case ["second"]:
            return TokenReadout("second", second, ("1", "0"), "truth")
        case ["yes_no", key] if key in YES_NO_PROMPTS:
            # ' Yes' / ' No': in G0 (26-10-31) every candidate put < 3e-5 on the bare 'Yes' / 'No'
            p = prompt_of(key, YES_NO_PROMPTS[key], lens)
            return TokenReadout(spec, p, (" Yes", " No"), "truth")
        case ["digit", key] if key in DIGIT_QUESTIONS:
            return TokenReadout(spec, prompt_of(key, DIGIT_QUESTIONS[key], lens), DIGITS, "digit")
        case ["recall", kind, form] if kind in ("subject", "object") and form in RECALL_TEMPLATES:
            return RecallReadout(spec, kind, form)
        case _:
            raise AssertionError(f"unknown readout {spec!r}")


class RecallBatch(NamedTuple):
    rows: np.ndarray  # positions in the readout's row list
    ids: Tensor  # (b, pos)
    dec: Tensor  # (b, pos) statement positions
    end: Tensor  # (b,) the question's last token; the answer starts at end + 1
    answer: Tensor  # (b, k) answer token ids, right-padded with -1
    obj: Tensor  # (b,) the object's first token


class ObjBatch(NamedTuple):
    """paper_common.Batch plus the object token per row (for the role grouping and the statement KL)."""

    rows: np.ndarray  # positions in the readout's row list
    ids: Tensor
    dec: Tensor
    end: Tensor
    obj: Tensor


AnyBatch = ObjBatch | RecallBatch


def recall_question_answer(
    kind: RecallKind, template: str, fam: str, rec: Records, i: int
) -> tuple[str, str]:
    subj_noun, obj_noun = RECALL_NOUNS[base_family(fam)]
    match kind:
        case "subject":
            return RECALL_TEMPLATES[template].format(noun=subj_noun), " " + rec.subject[i]
        case "object":
            return RECALL_TEMPLATES[template].format(noun=obj_noun), " " + rec.obj[i]


def make_recall_batches(
    texts: list[str],
    families: list[str],
    qa: list[tuple[str, str]],
    tok: PreTrainedTokenizer,
    loss_positions: LossPositions,
    batch_size: int,
    device: str,
) -> list[RecallBatch]:
    """statement + question + answer, tokenized jointly as the verdict prompts are
    (prompts_dataset._tokenize_jointly), right-padded; the answer differs per statement, hence one
    tokenization per statement."""
    rows_ids, n_stmt, ends, answers, objs = [], [], [], [], []
    for text, fam, (q, a) in zip(texts, families, qa, strict=True):
        (ids_q,), (n_q,) = _tokenize_jointly([text], q, tok)
        (ids_qa,), (n_qa,) = _tokenize_jointly([text], q + a, tok)
        assert n_q == n_qa and ids_qa[: len(ids_q)] == ids_q, (text, q, a)
        assert len(ids_qa) > len(ids_q), (text, q, a)
        obj, obj_ids = object_token(text, q + a, fam, tok)
        assert obj_ids == ids_qa, (text, q, a)
        rows_ids.append(ids_qa)
        n_stmt.append(n_q)
        ends.append(len(ids_q) - 1)
        answers.append(ids_qa[len(ids_q) :])
        objs.append(obj)
    batches = []
    for b in range(0, len(texts), batch_size):
        sl = slice(b, b + batch_size)
        width = max(len(x) for x in rows_ids[sl])
        k = max(len(x) for x in answers[sl])
        pad = tok.pad_token_id
        assert pad is not None
        ids = torch.tensor([x + [pad] * (width - len(x)) for x in rows_ids[sl]])
        ans = torch.tensor([x + [-1] * (k - len(x)) for x in answers[sl]])
        dec = build_position_mask(torch.tensor(n_stmt[sl]), width, loss_positions)
        assert dec is not None
        batches.append(
            RecallBatch(
                np.arange(b, b + len(ids)),
                ids.to(device),
                dec.to(device).bool(),
                torch.tensor(ends[sl]).to(device),
                ans.to(device),
                torch.tensor(objs[sl]).to(device),
            )
        )
    return batches


# per batch: (state logits, clean logits, batch) -> per-row arrays
Measure = Callable[[Tensor, Tensor, Any], dict[str, np.ndarray]]


def token_measure(token_ids: list[int]) -> Measure:
    tid = torch.tensor(token_ids)

    def measure(logits: Tensor, clean: Tensor, bt: AnyBatch) -> dict[str, np.ndarray]:
        rows = torch.arange(len(bt.end), device=bt.end.device)
        lp = torch.log_softmax(logits[rows, bt.end].float(), -1)
        lp0 = torch.log_softmax(clean[rows, bt.end].float(), -1)
        return {
            "lp": lp[:, tid.to(lp.device)].cpu().numpy(),
            "kl": (lp0.exp() * (lp0 - lp)).sum(-1).cpu().numpy(),
        }

    return measure


def statement_kl_measure(logits: Tensor, clean: Tensor, bt: ObjBatch) -> dict[str, np.ndarray]:
    """Mean next-token KL(clean || state) over each position class of the statement (G2's readout 1)."""
    lp = torch.log_softmax(logits.float(), -1)
    lp0 = torch.log_softmax(clean.float(), -1)
    kl = (lp0.exp() * (lp0 - lp)).sum(-1)  # (b, pos)
    cls = position_classes(bt.obj, bt.dec)
    out = {}
    for ic, c in enumerate(POSITION_CLASSES):
        sel = (cls == ic).float()
        assert bool((sel.sum(1) > 0).all()), f"a statement without {c} positions"
        out[f"stmt_{c}"] = ((kl * sel).sum(1) / sel.sum(1)).cpu().numpy()
    return out


def recall_measure(logits: Tensor, _clean: Tensor, bt: RecallBatch) -> dict[str, np.ndarray]:
    """Teacher-forced: summed log-prob of the answer tokens, and whether each is the argmax."""
    k = bt.answer.shape[1]
    pos = bt.end[:, None] + torch.arange(k, device=bt.end.device)[None, :]  # predicts answer[j]
    valid = bt.answer >= 0
    pos = torch.where(valid, pos, bt.end[:, None])
    lp = torch.log_softmax(logits.float(), -1)
    picked = lp.gather(1, pos[..., None].expand(-1, -1, lp.shape[-1]))  # (b, k, vocab)
    ans = bt.answer.clamp(min=0)
    tok_lp = picked.gather(2, ans[..., None]).squeeze(2)
    hit = picked.argmax(-1) == ans
    return {
        "logprob": (tok_lp * valid).sum(1).cpu().numpy(),
        "exact": (hit | ~valid).all(1).cpu().numpy(),
    }


# ---- the engine ----


def run_cells(
    model: ComponentModel,
    modules: list[str],
    split: Split,
    part: Part,
    ivs: list[Intervention],
    batches: list[tuple[AnyBatch, AnyBatch | None]],
    sign: np.ndarray,
    stats: TrainStats | None,
    measures: list[Measure],
    autocast: bool,
    n: int,
) -> dict[str, np.ndarray]:
    """For each batch (and its twin batch for twin interventions): one clean pass, which captures
    what the interventions need, then one edited pass per intervention. Returns per-row arrays
    '<state>__<measure key>' for state 'original' and every intervention. sign: (n,) +1 / -1 per row
    (the model's training-prompt verdict)."""
    groupings: list[GroupingName] = [
        g for g in GROUPINGS if any(grouping_of(iv) == g for iv in ivs)
    ]
    twin = any(needs_twin(iv) for iv in ivs)
    out: dict[str, np.ndarray] = {}

    def store(state: str, rows: np.ndarray, vals: dict[str, np.ndarray]) -> None:
        for k, v in vals.items():
            key = f"{state}__{k}"
            if key not in out:
                out[key] = np.zeros((n, *v.shape[1:]), v.dtype)
            out[key][rows] = v

    for bt, tw in batches:
        assert (tw is not None) == twin
        norm: dict[str, Tensor] = {}
        own: dict[str, Tensor] = {}

        def capture_own(
            m: str,
            x: Tensor,
            y: Tensor,
            norm: dict[str, Tensor] = norm,
            own: dict[str, Tensor] = own,
        ) -> None:
            norm[m] = y.norm(dim=-1)
            if twin:
                own[m] = part_of(split, part, m, x, y)

        with site_hooks(model, modules, capture_own), bf16_autocast(enabled=autocast):
            clean = model(bt.ids)
        twin_clean: dict[str, Tensor] | None = None
        if tw is not None:
            assert tw.ids.shape == bt.ids.shape and bool((tw.dec == bt.dec).all()), "twin layout"
            tc: dict[str, Tensor] = {}
            twin_clean = tc

            def capture_twin(m: str, x: Tensor, y: Tensor, tc: dict[str, Tensor] = tc) -> None:
                tc[m] = part_of(split, part, m, x, y)

            with site_hooks(model, modules, capture_twin), bf16_autocast(enabled=autocast):
                model(tw.ids)
        groups: dict[GroupingName, dict[str, Tensor]] = {}
        for gr in groupings:
            assert stats is not None
            groups[gr] = {
                m: group_ids(gr, bt.dec, bt.obj, norm[m], stats.median[m]) for m in modules
            }
        ctx = BatchContext(
            bt.dec,
            groups,
            torch.tensor(sign[bt.rows], device=bt.ids.device, dtype=torch.float32),
            own if twin else None,
            twin_clean,
        )
        for ms in measures:
            store("original", bt.rows, ms(clean, clean, bt))
        for iv in ivs:
            with (
                site_hooks(model, modules, edit_fn(split, part, iv, ctx, stats)),
                bf16_autocast(enabled=autocast),
            ):
                logits = model(bt.ids)
            for ms in measures:
                store(intervention_name(iv), bt.rows, ms(logits, clean, bt))
            del logits
        del clean
    return out


def with_obj(batches: list[Batch], texts: list[str], fams: list[str], stimulus: str,
             tok: PreTrainedTokenizer) -> list[ObjBatch]:  # fmt: skip
    """Attach each row's object token index (checked against the batch's own tokens)."""
    out = []
    for bt in batches:
        objs = []
        for r, row in enumerate(bt.rows):
            o, ids = object_token(texts[row], stimulus, fams[row], tok)
            assert bt.ids[r, : len(ids)].tolist() == ids, (texts[row], ids)
            objs.append(o)
        out.append(
            ObjBatch(bt.rows, bt.ids, bt.dec, bt.end, torch.tensor(objs, device=bt.ids.device))
        )
    return out


# ---- train statistics ----


def train_stats(
    model: ComponentModel,
    modules: list[str],
    split: Split,
    part: Part,
    batches: list[ObjBatch],
    labels: np.ndarray,
    has_twin: np.ndarray,
    autocast: bool,
) -> TrainStats:
    """Two clean passes over the train statements: the median output norm per site (defines
    'massive'), then the per-group sums of the edited part for both groupings."""
    norms: dict[str, list[Tensor]] = {m: [] for m in modules}
    for bt in batches:

        def cap_norm(m: str, _x: Tensor, y: Tensor, bt: ObjBatch = bt) -> None:
            norms[m].append(y.norm(dim=-1)[bt.dec].cpu())

        with site_hooks(model, modules, cap_norm), bf16_autocast(enabled=autocast):
            model(bt.ids)
    median = {m: float(torch.cat(v).median()) for m, v in norms.items()}
    d_out = {m: int(model.target_weight(m).shape[0]) for m in modules}
    device = str(batches[0].ids.device)
    acc: dict[GroupingName, GroupAccumulator] = {
        gr: GroupAccumulator(modules, N_GROUPS[gr], d_out, device) for gr in GROUPINGS
    }
    for bt in batches:
        lab = torch.tensor(labels[bt.rows], device=bt.ids.device)
        tw = torch.tensor(has_twin[bt.rows], device=bt.ids.device)

        def cap_sum(m: str, x: Tensor, y: Tensor, bt: ObjBatch = bt) -> None:
            q = part_of(split, part, m, x, y)
            nrm = y.norm(dim=-1)
            for gr in GROUPINGS:
                acc[gr].add(m, q, group_ids(gr, bt.dec, bt.obj, nrm, median[m]), lab, tw)  # noqa: B023

        with site_hooks(model, modules, cap_sum), bf16_autocast(enabled=autocast):
            model(bt.ids)
    return TrainStats(median, {gr: a.result() for gr, a in acc.items()})


# ---- summaries (pure, numpy) ----


def truth_summary(
    s: np.ndarray, s0: np.ndarray, labels: np.ndarray, mass: np.ndarray, kl: np.ndarray,
    s_twin0: np.ndarray | None,
) -> dict[str, float]:  # fmt: skip
    """s, s0: the state's and the original's score log P(true answer) - log P(false answer) per row.
    With s_twin0 (the original score of each row's twin, for twin interventions): IIA (the state's
    verdict = the twin's original verdict, over rows whose twin verdict differs under this prompt),
    the pooled swap fraction nu = sum (s - s0) sgn / sum |s_twin0 - s0| with sgn = sign(s_twin0 - s0),
    and the number of rows behind IIA."""
    gap0 = s0[labels].mean() - s0[~labels].mean()
    base = s0 > 0
    res = {
        "sep_share": float((s[labels].mean() - s[~labels].mean()) / gap0),
        "accuracy": float(((s > 0) == labels).mean()),
        "agreement": float(((s > 0) == base).mean()),
        "answer_mass": float(mass.mean()),
        "kl": float(kl.mean()),
        "flip_1_to_0": float((s[base] <= 0).mean()) if base.any() else float("nan"),
        "flip_0_to_1": float((s[~base] > 0).mean()) if (~base).any() else float("nan"),
    }
    if s_twin0 is not None:
        differ = (s_twin0 > 0) != base
        sgn = np.sign(s_twin0 - s0)
        res["iia"] = (
            float(((s > 0) == (s_twin0 > 0))[differ].mean()) if differ.any() else float("nan")
        )
        res["iia_n"] = float(differ.sum())
        res["nu"] = float(((s - s0) * sgn).sum() / np.abs(s_twin0 - s0).sum())
    return res


def summarize(
    readout: Readout,
    arr: dict[str, np.ndarray],
    states: list[str],
    labels: np.ndarray,
    families: np.ndarray,
    twin_rows: np.ndarray | None,
    seed: int,
) -> dict[str, Any]:
    """Per state: the readout's metrics, pooled, with 95% bootstrap intervals (resampled within
    family, paper_common.bootstrap_intervals). twin_rows: for twin interventions, per row the
    position of its twin in this row list."""
    res: dict[str, Any] = {}
    match readout:
        case TokenReadout(kind="truth"):
            lp0 = arr["original__lp"]
            for stt in states:
                lp, kl = arr[f"{stt}__lp"], arr[f"{stt}__kl"]

                def fn(
                    idx: np.ndarray, lp: np.ndarray = lp, kl: np.ndarray = kl
                ) -> dict[str, float]:
                    s, s0 = lp[idx, 0] - lp[idx, 1], lp0[idx, 0] - lp0[idx, 1]
                    st0 = (
                        None if twin_rows is None else (lp0[twin_rows, 0] - lp0[twin_rows, 1])[idx]
                    )
                    mass = np.exp(lp[idx]).sum(1)
                    return truth_summary(s, s0, labels[idx], mass, kl[idx], st0)

                res[stt] = {"pooled": fn(np.arange(len(labels))),
                            "interval": bootstrap_intervals(families, fn, N_BOOT, seed)}  # fmt: skip
        case TokenReadout(kind="digit"):
            lp0 = arr["original__lp"]
            for stt in states:
                d = digit_readout(arr[f"{stt}__lp"], lp0)
                res[stt] = {
                    "digit_kl": float(d["kl"].mean()),
                    "shift": float(d["shift"].mean()),
                    "abs_shift": float(np.abs(d["shift"]).mean()),
                    "digit_mass": float(d["mass"].mean()),
                    "answer_kl": float(arr[f"{stt}__kl"].mean()),
                }
        case RecallReadout():
            lp0 = arr["original__logprob"]
            for stt in states:
                res[stt] = {
                    "logprob": float(arr[f"{stt}__logprob"].mean()),
                    "logprob_change": float((arr[f"{stt}__logprob"] - lp0).mean()),
                    "exact": float(arr[f"{stt}__exact"].mean()),
                    "per_family_exact": {
                        f: float(arr[f"{stt}__exact"][families == f].mean())
                        for f in sorted(set(families))
                    },
                }
        case TokenReadout():
            raise AssertionError(readout.kind)
    if "original__stmt_prefix" in arr:
        res["statement_kl"] = {
            stt: {c: float(arr[f"{stt}__stmt_{c}"].mean()) for c in POSITION_CLASSES}
            for stt in states
        }
    return res


def print_summary(name: str, summ: dict[str, Any]) -> None:
    for stt, v in summ.items():
        if stt == "statement_kl":
            for s2, w in v.items():
                print(
                    f"  {name:22s} {s2:18s} statement KL "
                    + " ".join(f"{c} {x:.4f}" for c, x in w.items())
                )
            continue
        flat = (
            v["pooled"]
            if "pooled" in v
            else {k: x for k, x in v.items() if not isinstance(x, dict)}
        )
        print(f"  {name:22s} {stt:18s} " + " ".join(f"{k} {x:.4f}" for k, x in flat.items()))


def show_examples(tok: PreTrainedTokenizer, batches: Sequence[AnyBatch], fams: list[str],
                  logits_of: Callable[[AnyBatch], Tensor]) -> None:  # fmt: skip
    """Observe & Verify: one row per family, the exact token sequence with the statement tokens in
    [..], the answer tokens in <..>, and the original model's top tokens at the answer position."""
    shown: set[str] = set()
    for bt in batches:
        logits = None
        for r, row in enumerate(bt.rows):
            fam = fams[row]
            if fam in shown:
                continue
            shown.add(fam)
            if logits is None:
                logits = logits_of(bt)
            end = int(bt.end[r])
            n_ans = int((bt.answer[r] >= 0).sum()) if isinstance(bt, RecallBatch) else 0
            parts = []
            for t in range(end + 1 + n_ans):
                s = repr(tok.decode([int(bt.ids[r, t])]))
                parts.append(f"[{s}]" if bool(bt.dec[r, t]) else (f"<{s}>" if t > end else s))
            p = torch.softmax(logits[r, end].float(), -1)
            top = torch.topk(p, 5)
            tops = ", ".join(f"{tok.decode([int(i)])!r} {float(v):.2f}"
                             for v, i in zip(top.values, top.indices, strict=True))  # fmt: skip
            print(f"    {fam}: {' '.join(parts)}\n      top at answer: {tops}")


# ---- CLI ----


def run(
    run: str,
    target: Literal["site", "active"],
    part: Part,
    split: Literal["val", "test"],
    interventions: str | tuple[str, ...],
    readouts: str | tuple[str, ...],
    out: str,
    recovery: str | None = None,
    n_per_dataset: int | None = None,
    batch_size: int = 64,
    device: str = "cuda",
    seed: int = 0,
) -> None:
    """--run supplies the model, the data and the decomposed positions (for target site any final run
    gives the same result; the decomposition is not used). target active: P = the run's active set
    from paper_recovery.py's JSON (--recovery). interventions: comma-separated specs of
    paper_interventions.parse_intervention, or "" for the originals only (format checks)."""
    torch.set_grad_enabled(False)
    # fire passes "a,b" as a tuple, but as one string when the items contain ':' (checked 26-11-11)
    iv_specs = [
        s
        for s in (interventions.split(",") if isinstance(interventions, str) else interventions)
        if s
    ]
    ro_specs = [s for s in (readouts.split(",") if isinstance(readouts, str) else readouts) if s]
    assert ro_specs, "no readouts"
    ivs = [parse_intervention(s) for s in iv_specs]
    model, config, run_id, tok = load_run(run, device)
    lens, task = config.target_lens, config.task_config
    assert isinstance(lens, StimulusLens) and isinstance(task, LMTaskConfig)
    assert task.prepared_datasets is not None
    train_dirs = task.prepared_datasets
    autocast = config.autocast_bf16
    modules = sorted_modules(model)
    train_p, second_p = verdict_prompts(lens, tok, task.max_seq_len)
    ros = [parse_readout(s, lens, train_p, second_p) for s in ro_specs]
    assert lens.stimulus.startswith("\n"), "recall readouts reuse the training prompt's statistics"
    for ro in ros:
        if isinstance(ro, RecallReadout):
            assert not any(needs_twin(iv) for iv in ivs), "no recall under twin interventions"
    assert not (target == "site" and part == "R"), "the site has no R part (R = 0)"
    match target:
        case "site":
            assert recovery is None
            sp: Split = SiteSplit()
        case "active":
            assert recovery is not None
            rj = json.loads(Path(recovery).read_text())
            assert rj["run"] == run, (rj["run"], run)
            sp = component_split(
                "active", decomposition_basis(model, run_id), rj["active_set"]["components"]
            )
    print(f"{run_id}: target {sp.name} part {part}, interventions {[intervention_name(iv) for iv in ivs]}, "
          f"readouts {[ro.name for ro in ros]}")  # fmt: skip

    st = maybe_cap(load_statements(split, train_dirs), n_per_dataset, seed)
    rec = load_records(st, train_dirs)
    fams = [str(f) for f in st.families]

    def obj_batches(
        rows: np.ndarray, prompt: Prompt, s: Statements, f: list[str]
    ) -> list[ObjBatch]:
        texts = [s.texts[i] for i in rows]
        b = make_batches(texts, tok, task.loss_positions, prompt, batch_size, device)
        return with_obj(b, texts, [f[i] for i in rows], prompt.lens.stimulus, tok)

    # the model's own verdict under the training prompt: twin selection and the shift's sign
    all_rows = np.arange(len(st))
    base = run_cells(model, modules, sp, part, [], [(b, None) for b in obj_batches(all_rows, train_p, st, fams)],
                     np.ones(len(st)), None, [token_measure(single_tokens(tok, ("1", "0")))], autocast, len(st))  # fmt: skip
    base_score = base["original__lp"][:, 0] - base["original__lp"][:, 1]
    sign = np.where(base_score > 0, 1.0, -1.0)

    partner = twin_partner(st, rec)
    n_tok = np.array([len(_tokenize_jointly([t], lens.stimulus, tok)[0][0]) for t in st.texts])
    ok = partner >= 0
    ok &= np.where(ok, n_tok == n_tok[partner], False)
    ok &= np.where(ok, (base_score > 0) != (base_score[partner] > 0), False)
    twin_rows_all = np.flatnonzero(ok)
    assert np.array_equal(np.sort(partner[twin_rows_all]), twin_rows_all), (
        "twins must come in pairs"
    )
    print(f"{split}: {len(st)} statements, {int((partner >= 0).sum()) // 2} label twins, "
          f"{len(twin_rows_all) // 2} pairs of equal length and different verdicts")  # fmt: skip

    def logits_of(b: AnyBatch) -> Tensor:
        with bf16_autocast(enabled=autocast):
            return model(b.ids)

    stats_cache: dict[str, TrainStats] = {}

    def stats_for(prompt: Prompt) -> TrainStats:
        if prompt.name not in stats_cache:
            tr = load_statements("train", train_dirs)
            tr_rec = load_records(tr, train_dirs)
            tr_f = [str(f) for f in tr.families]
            has_twin = twin_partner(tr, tr_rec) >= 0
            stats_cache[prompt.name] = train_stats(
                model, modules, sp, part, obj_batches(np.arange(len(tr)), prompt, tr, tr_f),
                tr.labels, has_twin, autocast,
            )  # fmt: skip
            print(f"train statistics under prompt {prompt.name} ({len(tr)} statements, "
                  f"{int(has_twin.sum())} with a twin):")  # fmt: skip
            print("\n".join(describe_stats(stats_cache[prompt.name], modules)))
        return stats_cache[prompt.name]

    summary: dict[str, Any] = {
        "run": run, "run_id": run_id, "target": sp.name, "part": part, "split": split,
        "n": len(st), "n_twin_rows": len(twin_rows_all), "interventions": [intervention_name(iv) for iv in ivs],
        "components": list(sp.components) if not isinstance(sp, SiteSplit) else None, "readouts": {},
    }  # fmt: skip
    arrays: dict[str, np.ndarray] = {
        "labels": st.labels, "families": st.families, "base_score": base_score, "partner": partner,
    }  # fmt: skip
    need_stats = any(grouping_of(iv) is not None for iv in ivs)
    for ro in ros:
        res: dict[str, Any] = {}
        for twin_mode in (False, True):
            group = [iv for iv in ivs if needs_twin(iv) == twin_mode]
            # the all-statements pass with no intervention only computes originals: kept only for a
            # call without interventions (a format check)
            if not group and (twin_mode or ivs):
                continue
            rows = twin_rows_all if twin_mode else all_rows
            pos_of = np.full(len(st), -1)
            pos_of[rows] = np.arange(len(rows))
            match ro:
                case TokenReadout(prompt=prompt, tokens=tokens):
                    bts = obj_batches(rows, prompt, st, fams)
                    tws = (
                        obj_batches(partner[rows], prompt, st, fams)
                        if twin_mode
                        else [None] * len(bts)
                    )
                    measures = [token_measure(single_tokens(tok, tokens))]
                    if ro.name == "verdict" and not twin_mode:
                        measures.append(statement_kl_measure)
                    stats = stats_for(prompt) if need_stats else None
                case RecallReadout(kind=kind, template=form):
                    qa = [recall_question_answer(kind, form, fams[i], rec, int(i)) for i in rows]
                    bts = make_recall_batches([st.texts[i] for i in rows], [fams[i] for i in rows], qa,
                                              tok, task.loss_positions, batch_size, device)  # fmt: skip
                    tws = [None] * len(bts)
                    measures = [recall_measure]
                    stats = stats_for(train_p) if need_stats else None
            print(f"{ro.name} ({'twins' if twin_mode else 'all'}, {len(rows)} rows), examples:")
            show_examples(tok, bts, [fams[i] for i in rows], logits_of)
            arr = run_cells(model, modules, sp, part, group, list(zip(bts, tws, strict=True)), sign[rows],
                            stats, measures, autocast, len(rows))  # fmt: skip
            key = "twins" if twin_mode else "all"
            arrays.update({f"{ro.name}__{key}__{k}": v for k, v in arr.items()})
            arrays[f"{ro.name}__{key}__rows"] = rows
            states = ["original"] + [intervention_name(iv) for iv in group]
            twin_pos = pos_of[partner[rows]] if twin_mode else None
            assert twin_pos is None or bool((twin_pos >= 0).all()), "a twin outside the row list"
            res[key] = summarize(
                ro, arr, states, st.labels[rows], st.families[rows], twin_pos, seed
            )
            print_summary(ro.name, res[key])
        summary["readouts"][ro.name] = res
    for p_name, s in stats_cache.items():
        arrays.update({f"{p_name}__{k}": v for k, v in to_numpy(s).items()})

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "_".join(
        [sp.name, part, split, run_id, "+".join(iv_specs) or "original", "+".join(ro_specs)]
    )
    stem = re.sub(r"[^A-Za-z0-9_.+-]", "-", stem)
    (out_dir / f"tests_{stem}.json").write_text(json.dumps(summary, indent=1))
    np.savez(out_dir / f"tests_{stem}.npz", **arrays)  # pyright: ignore[reportArgumentType]
    print(f"wrote {out_dir / f'tests_{stem}'}.{{json,npz}}")


def check_g2(tests: str, g2: str, tol: float) -> None:
    """The new code against G2 (paper_side_effects.py g1, same split): its site zero removal must give
    G2's all-off state per statement, and its originals G2's originals. Compares every readout the two
    files share; asserts max |difference| < tol (log-probs, KLs)."""
    t, g = np.load(tests), np.load(g2)
    assert np.array_equal(t["labels"], g["labels"]), "different statement sets"
    pairs = []
    for ro, g_prefix in (
        [("verdict", "train")]
        + [(f"yes_no:{k}", f"yes_no_{k}") for k in YES_NO_PROMPTS]
        + [(f"digit:{k}", f"digit_{k}") for k in DIGIT_QUESTIONS]
    ):
        if f"{ro}__all__original__lp" not in t.files or f"{g_prefix}_original_lp" not in g.files:
            continue
        pairs.append((f"{ro}__all__original__lp", f"{g_prefix}_original_lp"))
        if f"{ro}__all__zero__lp" in t.files:
            pairs += [(f"{ro}__all__zero__lp", f"{g_prefix}_all_off_lp"),
                      (f"{ro}__all__zero__kl", f"{g_prefix}_all_off_kl")]  # fmt: skip
        if ro == "verdict" and "verdict__all__zero__stmt_prefix" in t.files:
            pairs += [
                (f"verdict__all__zero__stmt_{c}", f"train_all_off_stmt_{c}")
                for c in POSITION_CLASSES
            ]
    assert pairs, "nothing to compare"
    worst = 0.0
    for a, b in pairs:
        rows = t[a.split("__")[0] + "__all__rows"]
        d = float(np.abs(t[a] - g[b][rows]).max())
        worst = max(worst, d)
        print(f"{a:40s} vs {b:32s} max |diff| {d:.2e}")
    assert worst < tol, f"max |diff| {worst:.2e} >= tol {tol}"
    print(f"all {len(pairs)} arrays agree within {tol}")


if __name__ == "__main__":
    fire.Fire({"run": run, "check_g2": check_g2})
