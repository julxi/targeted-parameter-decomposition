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
  recall:subject    "<statement>" + RECALL_TEMPLATE for the subject, e.g.
                    '\\nThe city in the statement above is "' -> the subject
  recall:object     the same for the object the statement names (for a false statement, the false one)
  Recall of the true object (a knowledge question about the subject alone) was proposed and dropped by
  Julian ("We have enough tests already"; LOG entry above, 26-11-11).
  Recall is scored teacher-forced on every spelling of recall_variants (as written, first letter's case
  flipped, and "the ..." for the animal subject); per statement p_any = the summed probability of the
  spellings (disjoint strings, so a probability; asserted <= 1) and recalled = p_any > 0.5. Under twin
  interventions the twin pass runs the twin's statement followed by the row's own question and answer:
  only the twin's writes at the statement positions are used, and by causal attention they do not
  depend on what follows. So recall:object scores the row's own stated object, not the twin's.

Targets: site (P = the whole down_proj output, no R), active (P = the run's active set), all (P = all
components of every decomposed layer, R = Delta's write y - C). Model-anchored targets (26-11-15,
convos/julian/26-11-15_open_research_LOG.md): readspan, writespan, bothspan = the ORIGINAL matrix W
restricted to the active set's read span, write span, or both; randspan, pcaspan = W restricted to a
random output subspace / the top input principal directions of the same per-layer size
(paper_interventions.span_split); diffspan = W restricted to the top principal directions of the
site output's difference between train twins, same size. Their swaps are subspace interchange interventions on the unsplit
model, the reference a component set's own swap is read against.

Scaling dial (--dial k, targets active and all): P's writes p become (1 + k) p and R = y - p becomes
y - (1 + k) p. This is the seeded re-implementation of the draft's *Validating the tests*:
W_c -> (1 + k) W_c and Delta -> Delta - k sum_c W_c leave every layer's output unchanged, and every
test on P then sees (1 + k) times P's own edit (e.g. P-swap: y(x) + (1 + k) (p(x') - p(x))). Scaling
only the active set's U is exact for every test here, since the inactive components sit in R and R is
computed as y minus the edited part.

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
    make_basis,
    make_batches,
    maybe_cap,
    sorted_modules,
    verdict_prompts,
)
from spd.experiments.lm.honesty_targeted_decomposition.paper_interventions import (
    GROUPINGS,
    MASSIVE_FACTOR,
    N_GROUPS,
    SPAN_KINDS,
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
    span_split,
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
# Chosen by Julian (26-11-11) from a screen of seven wordings on 190 val statements
# (~/spd_out/26-11-04_paper_draft/site_versions/recall_screen.py; LOG entry *recall wording screen*):
# it had the highest probability of the correct span (subject 0.66, object 0.77). Forms ending in
# "... is" were often continued with a judgement (' not', ' incorrect') and so partly read truth. The
# answer follows the opening quote without a space; the quote does not merge into it (checked).
RECALL_TEMPLATE = '\nThe {noun} in the statement above is "'
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
        case ["recall", kind] if kind in ("subject", "object"):
            return RecallReadout(spec, kind)
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


def recall_variants(kind: RecallKind, fam: str, word: str) -> list[tuple[str, str]]:
    """(label, answer) per spelling that counts as recalling the word. Most misses of the chosen
    wording were spelling, not recall (screen, 26-11-11): a sentence-initial subject recalled in lower
    case ('m'olybdenum, 'f'ifty-three) or an animal with its article ('the' lobster). The spellings
    differ in their first tokens or after a shared 'the', so they are disjoint outcomes."""
    flipped = (word[0].lower() if word[0].isupper() else word[0].upper()) + word[1:]
    out = [("written", word)] + ([("case", flipped)] if flipped != word else [])
    if kind == "subject" and base_family(fam) == "animal_class":
        out += [(f"the_{lab}", "the " + w) for lab, w in out]
    return out


def recall_question(kind: RecallKind, fam: str) -> str:
    subj_noun, obj_noun = RECALL_NOUNS[base_family(fam)]
    return RECALL_TEMPLATE.format(noun=subj_noun if kind == "subject" else obj_noun)


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


def aggregate_variants(
    arr: dict[str, np.ndarray], owner: np.ndarray, labels: np.ndarray, states: list[str], n: int
) -> dict[str, np.ndarray]:
    """Per statement and state: p_any = the summed probability of its spellings, and p_v_<label> the
    probability of each spelling (0 where a statement has no such spelling)."""
    out: dict[str, np.ndarray] = {}
    for stt in states:
        p = np.exp(arr[f"{stt}__logprob"].astype(np.float64))
        out[f"{stt}__p_any"] = np.bincount(owner, weights=p, minlength=n)
        for lab in sorted(set(labels)):
            out[f"{stt}__p_v_{lab}"] = np.bincount(owner, weights=p * (labels == lab), minlength=n)
        # disjoint spellings: their probabilities cannot sum to more than 1 (bf16 noise allowed)
        assert out[f"{stt}__p_any"].max() < 1.01, (stt, out[f"{stt}__p_any"].max())
    return out


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
        own_y: dict[str, Tensor] = {}

        def capture_own(
            m: str,
            x: Tensor,
            y: Tensor,
            norm: dict[str, Tensor] = norm,
            own: dict[str, Tensor] = own,
            own_y: dict[str, Tensor] = own_y,
        ) -> None:
            norm[m] = y.norm(dim=-1)
            if twin:
                own[m] = part_of(split, part, m, x, y)
                own_y[m] = y

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
            own_y if twin else None,
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


def twin_difference_directions(
    model: ComponentModel,
    modules: list[str],
    pairs: list[tuple[Batch, Batch]],
    autocast: bool,
) -> dict[str, Tensor]:
    """Per module the principal directions (d_out, d_out; columns in ASCENDING eigenvalue order, like
    the data-aligned basis) of the site output's twin difference y(x') - y(x) at the statement tokens
    of the given twin batches (uncentred second moment). For the diffspan target."""
    d_out = {m: int(model.target_weight(m).shape[0]) for m in modules}
    dev = pairs[0][0].ids.device
    second = {m: torch.zeros(d_out[m], d_out[m], device=dev, dtype=torch.float64) for m in modules}
    for own, twin in pairs:
        assert twin.ids.shape == own.ids.shape and bool((twin.dec == own.dec).all()), "twin layout"
        ys: list[dict[str, Tensor]] = [{}, {}]
        for i, ids in enumerate((own.ids, twin.ids)):

            def cap(m: str, _x: Tensor, y: Tensor, d: dict[str, Tensor] = ys[i]) -> None:
                d[m] = y

            with site_hooks(model, modules, cap), bf16_autocast(enabled=autocast):
                model(ids)
        for m in modules:
            d = (ys[1][m] - ys[0][m])[own.dec].double()
            second[m] += d.T @ d
    # eigh on the CPU (d_out is small; cusolver can fail on a GPU shared with other jobs)
    return {m: torch.linalg.eigh(second[m].cpu())[1].float().contiguous().to(dev) for m in modules}


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
            p0 = arr["original__p_any"]
            variant_keys = sorted(
                k.split("__", 1)[1] for k in arr if k.startswith("original__p_v_")
            )
            for stt in states:
                p = arr[f"{stt}__p_any"]
                res[stt] = {
                    "p_any": float(p.mean()),
                    "p_any_change": float((p - p0).mean()),
                    "recalled": float((p > 0.5).mean()),
                    # mean probability per spelling: which spellings carry the recall
                    **{k: float(arr[f"{stt}__{k}"].mean()) for k in variant_keys},
                    "per_family_p_any": {
                        f: float(p[families == f].mean()) for f in sorted(set(families))
                    },
                    "per_family_recalled": {
                        f: float((p[families == f] > 0.5).mean()) for f in sorted(set(families))
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
    target: Literal[
        "site",
        "active",
        "all",
        "readspan",
        "writespan",
        "bothspan",
        "randspan",
        "pcaspan",
        "diffspan",
    ],  # fmt: skip
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
    dial: float = 0.0,
    span_seed: int = 0,
    span_base: Literal["active", "all"] = "active",
    span_k: int | None = None,
) -> None:
    """--run supplies the model, the data and the decomposed positions (for target site any final run
    gives the same result; the decomposition is not used). target active: P = the run's active set
    from paper_recovery.py's JSON (--recovery). target all: P = every component (no --recovery).
    interventions: comma-separated specs of paper_interventions.parse_intervention, or "" for the
    originals only (format checks). dial: the scaling dial k (module docstring), targets active and
    all; 0 = the decomposition as trained. targets readspan / writespan / bothspan / randspan /
    pcaspan: the model-anchored splits of paper_interventions.span_split with the per-layer sizes of
    the run's active set (--recovery), or with --span_base all of all components of every layer (no
    --recovery; the split is then named all-<kind>); span_seed draws randspan's subspace. span_k:
    for the kinds that use only the sizes of the base set (randspan, pcaspan, diffspan), k dimensions
    in every layer instead (no --recovery; named k<k>-<kind>)."""
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
    assert not (target == "site" and part == "R"), "the site has no R part (R = 0)"
    match target:
        case "site":
            assert recovery is None
            assert dial == 0, "the dial needs a decomposition (target active or all)"
            sp: Split = SiteSplit()
        case "active":
            assert recovery is not None
            rj = json.loads(Path(recovery).read_text())
            assert rj["run"] == run, (rj["run"], run)
            sp = component_split(
                "active" if dial == 0 else f"active-dial{dial:g}",
                decomposition_basis(model, run_id),
                rj["active_set"]["components"],
                dial,
            )
        case "readspan" | "writespan" | "bothspan" | "randspan" | "pcaspan" | "diffspan":
            assert target in SPAN_KINDS and dial == 0
            basis = decomposition_basis(model, run_id)
            base_name: str = span_base
            if span_k is not None:
                assert target in ("randspan", "pcaspan", "diffspan") and recovery is None
                assert span_base == "active", "span_k replaces the base set"
                base_name = f"k{span_k}"
                base_names = [
                    f"L{int(m.split('.')[2])}:{c}" for m in basis.U for c in range(span_k)
                ]
            elif recovery is not None:
                assert span_base == "active", "--recovery iff span_base active"
                rj = json.loads(Path(recovery).read_text())
                assert rj["run"] == run, (rj["run"], run)
                base_names = rj["active_set"]["components"]
            else:
                assert span_base == "all", "--recovery iff span_base active"
                base_names = [f"L{int(m.split('.')[2])}:{c}" for m in basis.U for c in range(basis.U[m].shape[0])]  # fmt: skip
            base_sp = component_split(base_name, basis, base_names, 0.0)
            pca_V = None
            if target == "pcaspan":
                pca_V = make_basis(
                    "data_aligned", model, run_id, train_dirs,
                    lambda texts: make_batches(texts, tok, task.loss_positions, train_p, batch_size, device),
                    autocast, seed,
                ).V  # fmt: skip
            diff_Q = None
            if target == "diffspan":
                # train twins of equal token layout (label twins; the true statement as x): the
                # directions are fitted on train and tested on the --split twins
                tr = load_statements("train", train_dirs)
                tr_partner = twin_partner(tr, load_records(tr, train_dirs))
                tr_n = np.array([len(_tokenize_jointly([t], lens.stimulus, tok)[0][0]) for t in tr.texts])  # fmt: skip
                tr_rows = np.flatnonzero((tr_partner >= 0) & tr.labels)
                tr_rows = tr_rows[tr_n[tr_rows] == tr_n[tr_partner[tr_rows]]]
                print(f"diffspan: {len(tr_rows)} train twin pairs of equal length")
                mk = lambda rows: make_batches([tr.texts[i] for i in rows], tok, task.loss_positions, train_p, batch_size, device)  # noqa: E731  # fmt: skip
                diff_Q = twin_difference_directions(
                    model, modules, list(zip(mk(tr_rows), mk(tr_partner[tr_rows]), strict=True)), autocast
                )  # fmt: skip
            sp = span_split(
                target,
                base_sp,
                {m: model.target_weight(m).float() for m in modules},
                pca_V,
                span_seed,
                diff_Q,
            )
        case "all":
            assert recovery is None
            basis = decomposition_basis(model, run_id)
            names = [
                f"L{int(m.split('.')[2])}:{c}" for m in basis.U for c in range(basis.U[m].shape[0])
            ]
            sp = component_split("all" if dial == 0 else f"all-dial{dial:g}", basis, names, dial)
            assert all(sp.U[m].shape == basis.U[m].shape for m in basis.U), "all = every component"
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
        "n": len(st), "n_twin_rows": len(twin_rows_all), "dial": dial,
        "interventions": [intervention_name(iv) for iv in ivs],
        "components": list(sp.components) if not isinstance(sp, SiteSplit) else None,
        # recorded because outputs before 26-11-14 used 10 and their file names don't show it
        "massive_factor": MASSIVE_FACTOR, "readouts": {},
    }  # fmt: skip
    arrays: dict[str, np.ndarray] = {
        "labels": st.labels, "families": st.families, "base_score": base_score, "partner": partner,
        "massive_factor": np.array(MASSIVE_FACTOR),
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
            owner: np.ndarray | None = None  # recall: flat (statement, spelling) row -> statement
            labels_v: np.ndarray | None = None
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
                case RecallReadout(kind=kind):
                    # one sequence per (statement, spelling); aggregated per statement after the run
                    flat = [(int(r), lab, a) for r in rows
                            for lab, a in recall_variants(kind, fams[r], (rec.subject if kind == "subject" else rec.obj)[r])]  # fmt: skip
                    owner = np.array([pos_of[r] for r, _, _ in flat])
                    labels_v = np.array([lab for _, lab, _ in flat])
                    qa = [(recall_question(kind, fams[r]), a) for r, _, a in flat]
                    bts = make_recall_batches([st.texts[r] for r, _, _ in flat], [fams[r] for r, _, _ in flat],
                                              qa, tok, task.loss_positions, batch_size, device)  # fmt: skip
                    # the twin's statement with the row's own question and answer (module docstring);
                    # run_cells asserts that both have the same layout
                    tws = (
                        make_recall_batches(
                            [st.texts[partner[r]] for r, _, _ in flat],
                            [fams[partner[r]] for r, _, _ in flat],
                            qa,
                            tok,
                            task.loss_positions,
                            batch_size,
                            device,
                        )  # fmt: skip
                        if twin_mode
                        else [None] * len(bts)
                    )
                    measures = [recall_measure]
                    stats = stats_for(train_p) if need_stats else None
            print(f"{ro.name} ({'twins' if twin_mode else 'all'}, {len(rows)} rows), examples:")
            if isinstance(ro, RecallReadout):
                assert owner is not None and labels_v is not None
                flat_rows = rows[owner]
                show_examples(tok, bts, [fams[i] for i in flat_rows], logits_of)
                arr_flat = run_cells(model, modules, sp, part, group, list(zip(bts, tws, strict=True)),
                                     sign[flat_rows], stats, measures, autocast, len(flat_rows))  # fmt: skip
                arrays.update({f"{ro.name}__flat__{k}": v for k, v in arr_flat.items()})
                arrays[f"{ro.name}__flat__owner"] = owner
                arrays[f"{ro.name}__flat__label"] = labels_v
                states_all = ["original"] + [intervention_name(iv) for iv in group]
                arr = aggregate_variants(arr_flat, owner, labels_v, states_all, len(rows))
            else:
                show_examples(tok, bts, [fams[i] for i in rows], logits_of)
                arr = run_cells(model, modules, sp, part, group, list(zip(bts, tws, strict=True)),
                                sign[rows], stats, measures, autocast, len(rows))  # fmt: skip
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
    files share. Asserts that the means agree within the relative tolerance tol, and prints the
    per-statement scatter: on another GPU type than G2's (H100) single statements differ by up to
    ~0.75 nats even without an edit (RTX 4090, 26-11-11), so a per-statement maximum cannot pass."""
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
        x, y = t[a], g[b][rows]
        rel = float(abs(x.mean() - y.mean()) / abs(y.mean()))
        worst = max(worst, rel)
        d = np.abs(x - y)
        print(f"{a:40s} vs {b:32s} mean {x.mean():9.4f} / {y.mean():9.4f} (rel. diff {rel:.1e}); "
              f"per statement mean |diff| {d.mean():.3f}, p99 {np.quantile(d, 0.99):.3f}")  # fmt: skip
    assert worst < tol, f"relative mean difference {worst:.2e} >= tol {tol}"
    print(f"all {len(pairs)} means agree within {tol} (relative)")


if __name__ == "__main__":
    fire.Fire({"run": run, "check_g2": check_g2})
