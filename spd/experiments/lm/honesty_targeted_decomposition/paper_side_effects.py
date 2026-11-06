"""Side-effect probes of the paper decompositions (paper-writing topic,
convos/julian/26-10-29_paper_writing_LOG.md, entry *side-effect test: revised design after Julian's
comments*, and the staged GPU list in its SUMMARY). Exploratory: designed after the phase D outputs
were read, so nothing here is part of the frozen analysis plan. Shared code: paper_common.py.

Question: how close is each method to an ideal truth decomposition, under which the components matter
only where truth is computed? Plain tPD's components write large vectors that delta cancels; every
edit that breaks the cancellation (delta alone, reversal) injects them, and they may disturb readouts
the training never looked at.

g0 (original model only, val subset): which prompts are usable at all.
  - Yes/No truth-question candidates (YES_NO_PROMPTS): mass on the Yes/No tokens, with and without a
    leading space, accuracy and AUC of log P(Yes) - log P(No), next to the 1/0 training prompt.
  - Digit questions about other properties (DIGIT_QUESTIONS): mass on the digits 0-9, how much the
    mean digit varies across statements, and its mean per label.
g1 (one basis, one split: G1 = the s1 decompositions on val; G2 = all bases on test, plus T2): the
states
     full           all components + delta (sanity: must reproduce the original)
     all_off        the decomposed MLP outputs set to 0 (reference)
     delta_only     every component removed, delta kept
     components_ci  mask = CI, delta off (recovery; only for the two truth prompts)
     reverse_top10  the TOP_K components of the T1 ranking (paper_termination.py, val) at mask -1,
                    everything else unchanged = manipulation's "reverse top10 group"
     T2 ...         (--t2) phase D's T2 removal sets (top-k and random size-k), under the Yes/No
                    prompts (the GPU list's G2: "for Yes/No also T2")
  and the readouts
  1. next-token KL(original || state) at the statement positions, training prompt, per position class:
     prefix (input and predicted token both before the object), recall (the token before the object;
     it predicts the object, e.g. "The city of Lu'an is in" -> the country), object_end (the object
     and everything after it, including the merged '.\\n'). Before the object, true and false
     statements of a family are the same text, so truth is not yet determined there.
  2. digit questions: KL of the renormalised 0-9 distribution vs the original, shift of the mean digit.
  3. Yes/No transfer: KL recovered of components_ci under the chosen Yes/No prompts vs the 1/0 prompt.
Every result is stored per statement in the npz (log-probs of the readout tokens per state, answer KL,
statement-position KLs), so all summaries can be recomputed offline.

Usage: python paper_side_effects.py g0 --run <any s-id, for the model> --out <dir> [--n_per_dataset 20]
       python paper_side_effects.py g1 --run <s-id or checkpoint> --basis decomposition|data_aligned
              --split val|test --termination <json> --out <dir> --t2 True|False
              --yes_no <comma-separated YES_NO_PROMPTS keys> --digit_questions <same for DIGIT_QUESTIONS>
Output: side_<run id>_<basis>_<split>.{json,npz}
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import fire
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.configs import Config, LMTaskConfig, SamplingType, StimulusLens
from spd.experiments.lm.honesty_targeted_decomposition.paper_common import (
    Basis,
    Batch,
    Prompt,
    Statements,
    auc_columns,
    bootstrap_intervals,
    edited_outputs,
    load_statements,
    make_basis,
    make_batches,
    maybe_cap,
    sorted_modules,
    verdict_prompts,
)
from spd.experiments.lm.honesty_targeted_decomposition.paper_termination import N_RANDOM
from spd.experiments.lm.honesty_targeted_decomposition.probe_ci import WANDB_PREFIX
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.general_utils import bf16_autocast

# Candidates for the Yes/No transfer prompt (g0 picks one: most Yes/No mass among those about as
# accurate as the 1/0 prompt). Prompts ending in ': ' put the answer after a space token, so the
# answer is 'Yes'; prompts ending in ':' make it ' Yes'. g0 measures both forms for every prompt.
YES_NO_PROMPTS = {
    "is_true_sp": "\nIs this statement true? Answer Yes or No: ",
    "is_true": "\nIs this statement true? Answer Yes or No:",
    "is_true_nl": "\nIs this statement true? Answer Yes or No.\nAnswer:",
    "true_or_false": "\nIs this statement true or false? Answer Yes if true, No if false.\nAnswer:",
    "correct": "\nIs the above statement correct? Yes or No?\nAnswer:",
    "question": "\nQuestion: Is the statement above true? (Yes/No)\nAnswer:",
    "same_line": " Is this true? Yes or No? Answer:",
    "same_line_sp": " Is this true? Yes or No? Answer: ",
}
YES_NO_TOKENS = ("Yes", "No", " Yes", " No")
# Digit questions about properties other than truth (Julian's humor line), in the training prompt's
# form, so the answer is a digit read after the final space as for the verdict.
DIGIT_QUESTIONS = {
    "humor": "\nThe amount of humor in this statement is ",
    "emotion": "\nThe amount of emotion in this statement is ",
    "surprise": "\nThe amount of surprise in this statement is ",
    "words": "\nThe number of words in this statement is ",
}
DIGITS = tuple(str(d) for d in range(10))
TOP_K = 10
N_BOOT = 1000
POSITION_CLASSES = ("prefix", "recall", "object_end")


def object_char(text: str, family: str) -> int:
    """Character index of the first character of the statement's object: the part in which true and
    false statements of a family differ (city -> country, comparison -> second number, ...)."""
    match family.removeprefix("neg_"):
        case "cities":
            anchor = r" in "
        case "animal_class":
            anchor = r" an? "
        case "element_symb":
            anchor = r" symbol "
        case "sp_en_trans":
            anchor = r" means? '"  # the object is the English word, without its quote
        case "larger_than" | "smaller_than":
            anchor = r" than "
        case _:
            raise AssertionError(family)
    # the last match: the object ends the statement, and a subject may contain the anchor text
    hits = list(re.finditer(anchor, text))
    assert hits and text.endswith("."), (text, family, anchor)
    return hits[-1].end()


def object_token(
    text: str, stimulus: str, family: str, tok: PreTrainedTokenizer
) -> tuple[int, list[int]]:
    """Index of the token holding the object's first character in the jointly tokenized
    statement + stimulus, and the token ids (the caller checks them against the batch, which
    paper_common.make_batches tokenizes the same way; no BOS)."""
    enc: Any = tok(text + stimulus, return_offsets_mapping=True, add_special_tokens=False)
    c = object_char(text, family)
    holding = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s <= c < e]
    assert len(holding) == 1, (text, c, holding)
    return holding[0], enc["input_ids"]


def position_classes(obj: Tensor, dec: Tensor) -> Tensor:
    """(b, pos) class per position: 0 prefix, 1 recall, 2 object_end, -1 not a statement position.
    obj: (b,) object token index per row; dec: (b, pos) statement positions. Position t's next-token
    distribution predicts token t + 1, so t = obj - 1 is the position that predicts the object."""
    t = torch.arange(dec.shape[1], device=dec.device)[None, :]
    cls = torch.full(dec.shape, -1, dtype=torch.long, device=dec.device)
    cls = torch.where(t < obj[:, None] - 1, 0, cls)
    cls = torch.where(t == obj[:, None] - 1, 1, cls)
    cls = torch.where(t >= obj[:, None], 2, cls)
    return torch.where(dec, cls, -1)


def single_tokens(tok: PreTrainedTokenizer, words: tuple[str, ...]) -> list[int]:
    ids = [tok.encode(w, add_special_tokens=False) for w in words]
    assert all(len(i) == 1 for i in ids), list(zip(words, ids, strict=True))
    return [i[0] for i in ids]


def load_run(run: str, device: str) -> tuple[ComponentModel, Config, str, PreTrainedTokenizer]:
    run_info = SPDRunInfo.from_path(run if Path(run).exists() else WANDB_PREFIX + run)
    # name by run id, never by the checkpoint file name (all final checkpoints share it; FEEDBACK.md)
    run_id = Path(run).parent.name if Path(run).exists() else run
    config = run_info.config
    assert isinstance(config.target_lens, StimulusLens) and config.target_lens.tokenize_jointly
    task = config.task_config
    assert isinstance(task, LMTaskConfig) and task.prepared_datasets is not None, task
    model = ComponentModel.from_run_info(run_info).to(device).eval()
    assert config.tokenizer_name is not None
    return model, config, run_id, AutoTokenizer.from_pretrained(config.tokenizer_name)


def prompt_of(name: str, stimulus: str, base: StimulusLens) -> Prompt:
    # max_seq_len 0: padded_length then fits the longest statement + this stimulus
    return Prompt(name, base.model_copy(update={"stimulus": stimulus}), 0)


# (CIs of the batch or None) -> MaskFn; CIs exist only for a decomposition with a CI network
StateEdit = Callable[[dict[str, Tensor] | None], Callable[..., tuple[Tensor, bool]]]


def state_edit(state: str) -> StateEdit:
    """The named states of the module docstring other than the scaled ones (scaled_edit)."""

    def make(ci: dict[str, Tensor] | None) -> Callable[..., tuple[Tensor, bool]]:
        def mask(m: str, _bt: Batch, acts: Tensor, _y: Tensor) -> tuple[Tensor, bool]:
            match state:
                case "full":
                    return torch.ones_like(acts), True
                case "all_off":
                    return torch.zeros_like(acts), False
                case "delta_only":
                    return torch.zeros_like(acts), True
                case "components_ci":
                    assert ci is not None
                    return ci[m], False
                case _:
                    raise AssertionError(state)

        return mask

    return make


def scaled_edit(scale: dict[str, Tensor]) -> StateEdit:
    """Full model (remainder kept) with each component's write multiplied by scale[m][c]: -1 for
    reverse_top10, 0 for a T2 removal, 1 = unchanged."""

    def make(_ci: dict[str, Tensor] | None) -> Callable[..., tuple[Tensor, bool]]:
        return lambda m, _bt, acts, _y: (scale[m].expand_as(acts), True)

    return make


def run_prompt(
    model: ComponentModel,
    basis: Basis | None,
    batches: list[Batch],
    states: dict[str, StateEdit],
    token_ids: list[int],
    autocast: bool,
    ci_sampling: SamplingType | None,
    statement_kl: dict[int, Tensor] | None,
) -> dict[str, np.ndarray]:
    """Per statement, for the original and every state: log-probs of token_ids at the answer position
    ('<state>_lp', (n, k)), and for the states KL(original || state) over the vocabulary there
    ('<state>_kl'). With statement_kl (batch index -> (b,) object token index), also the mean
    next-token KL over each position class at the statement positions ('<state>_stmt_<class>').
    States run batch by batch next to the original pass, so the original's full-vocabulary log-probs
    at every position never have to be kept for the whole set."""
    n = int(sum(len(bt.rows) for bt in batches))
    out: dict[str, np.ndarray] = {"original_lp": np.zeros((n, len(token_ids)), np.float32)}
    for s in states:
        out[f"{s}_lp"] = np.zeros((n, len(token_ids)), np.float32)
        out[f"{s}_kl"] = np.zeros(n, np.float32)
        if statement_kl is not None:
            for c in POSITION_CLASSES:
                out[f"{s}_stmt_{c}"] = np.zeros(n, np.float32)
    tid = torch.tensor(token_ids)
    for ib, bt in enumerate(batches):
        rows = torch.arange(len(bt.end), device=bt.end.device)
        with bf16_autocast(enabled=autocast):
            o = model(bt.ids, cache_type="input")
            ci = (
                {
                    m: v.float()
                    for m, v in model.calc_causal_importances(
                        pre_weight_acts=o.cache, detach_inputs=True, sampling=ci_sampling
                    ).lower_leaky.items()
                }
                if ci_sampling is not None
                else None
            )
        lp_o = torch.log_softmax(o.output.float(), -1)  # (b, pos, vocab)
        del o
        ans_o = lp_o[rows, bt.end]
        out["original_lp"][bt.rows] = ans_o[:, tid.to(ans_o.device)].cpu().numpy()
        cls = position_classes(statement_kl[ib], bt.dec) if statement_kl is not None else None
        for s, make in states.items():
            assert basis is not None
            with edited_outputs(model, basis, bt, make(ci)), bf16_autocast(enabled=autocast):
                logits = model(bt.ids)
            lp = torch.log_softmax(logits.float(), -1)
            del logits
            ans = lp[rows, bt.end]
            out[f"{s}_lp"][bt.rows] = ans[:, tid.to(ans.device)].cpu().numpy()
            out[f"{s}_kl"][bt.rows] = (ans_o.exp() * (ans_o - ans)).sum(-1).cpu().numpy()
            if cls is not None:
                kl_pos = (lp_o.exp() * (lp_o - lp)).sum(-1)  # (b, pos)
                for ic, c in enumerate(POSITION_CLASSES):
                    sel = (cls == ic).float()
                    assert bool((sel.sum(1) > 0).all()), f"a statement without {c} positions"
                    out[f"{s}_stmt_{c}"][bt.rows] = (
                        ((kl_pos * sel).sum(1) / sel.sum(1)).cpu().numpy()
                    )
            del lp
    return out


# ---- summaries (pure, numpy) ----


def ratio_summary(
    num: np.ndarray, den: np.ndarray, families: np.ndarray, seed: int
) -> dict[str, Any]:
    """mean(num) / mean(den), pooled over statements, with a 95% bootstrap interval (resampled within
    family) and per family. Used for 'state KL as a share of the all-off KL' (1 = as bad as all-off)."""

    def fn(idx: np.ndarray) -> dict[str, float]:
        return {"ratio": float(num[idx].mean() / den[idx].mean())}

    return {
        "num": float(num.mean()),
        "den": float(den.mean()),
        "ratio": fn(np.arange(len(num)))["ratio"],
        "interval": bootstrap_intervals(families, fn, N_BOOT, seed)["ratio"],
        "per_family": {
            f: fn(np.flatnonzero(families == f))["ratio"] for f in sorted(set(families))
        },
    }


def digit_readout(lp: np.ndarray, lp_orig: np.ndarray) -> dict[str, np.ndarray]:
    """Per statement: KL(original || state) of the renormalised 0-9 distribution, the mean digit and
    its shift, and the digit mass. lp: (n, 10) log-probs of the digits 0..9."""
    q_o = np.exp(lp_orig - np.logaddexp.reduce(lp_orig, axis=1, keepdims=True))
    q = np.exp(lp - np.logaddexp.reduce(lp, axis=1, keepdims=True))
    d = np.arange(10)
    return {
        "kl": (q_o * (np.log(q_o) - np.log(q))).sum(1),
        "mean_digit": q @ d,
        "shift": q @ d - q_o @ d,
        "mass": np.exp(lp).sum(1),
    }


def truth_readout(
    lp: np.ndarray, lp_orig: np.ndarray, kl: np.ndarray, kl_off: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """KL recovered, accuracy, agreement and separation share of a state under a two-token truth
    prompt; lp[:, 0] = the 'true' answer ('1' / 'Yes'), lp[:, 1] = the 'false' answer."""
    s, s0 = lp[:, 0] - lp[:, 1], lp_orig[:, 0] - lp_orig[:, 1]
    gap0 = s0[labels].mean() - s0[~labels].mean()
    return {
        "kl_recovered": float(1 - kl.mean() / kl_off.mean()),
        "accuracy": float(((s > 0) == labels).mean()),
        "agreement": float(((s > 0) == (s0 > 0)).mean()),
        "sep_share": float((s[labels].mean() - s[~labels].mean()) / gap0),
        "answer_mass": float(np.exp(lp).sum(1).mean()),
    }


# ---- g0 ----


def g0(
    run: str,
    out: str,
    n_per_dataset: int = 20,
    batch_size: int = 64,
    device: str = "cuda",
    seed: int = 0,
) -> None:
    """Original model only (no edits); --run only supplies the model, tokenizer and data config."""
    torch.set_grad_enabled(False)
    model, config, _run_id, tok = load_run(run, device)
    lens = config.target_lens
    task = config.task_config
    assert isinstance(lens, StimulusLens) and isinstance(task, LMTaskConfig)
    assert task.prepared_datasets is not None
    st = maybe_cap(load_statements("val", task.prepared_datasets), n_per_dataset, seed)
    print(f"g0 on {len(st)} val statements ({n_per_dataset} per dataset)")

    def original(prompt: Prompt, words: tuple[str, ...]) -> tuple[np.ndarray, list[Batch]]:
        batches = make_batches(st.texts, tok, task.loss_positions, prompt, batch_size, device)
        r = run_prompt(
            model, None, batches, {}, single_tokens(tok, words), config.autocast_bf16, None, None
        )
        return r["original_lp"], batches

    summary: dict[str, Any] = {"run": run, "n": len(st), "yes_no": {}, "digit_questions": {}}
    arrays: dict[str, np.ndarray] = {"labels": st.labels, "datasets": st.datasets}
    train = verdict_prompts(lens, tok, task.max_seq_len)[0]
    lp, _ = original(train, ("1", "0"))
    arrays["train_lp"] = lp
    summary["train_1_0"] = binary_summary(lp[:, 0], lp[:, 1], st)
    print(f"1/0 training prompt: {summary['train_1_0']}")

    for key, stim in YES_NO_PROMPTS.items():
        lp, batches = original(prompt_of(key, stim, lens), YES_NO_TOKENS)
        arrays[f"yes_no_{key}_lp"] = lp
        res = {
            "bare": binary_summary(lp[:, 0], lp[:, 1], st),
            "space": binary_summary(lp[:, 2], lp[:, 3], st),
        }
        summary["yes_no"][key] = res
        print(f"{key} {stim!r}:\n  'Yes'/'No' {res['bare']}\n  ' Yes'/' No' {res['space']}")
        print_top_tokens(model, tok, batches[0], config.autocast_bf16, st, 3)

    for key, stim in DIGIT_QUESTIONS.items():
        lp, batches = original(prompt_of(key, stim, lens), DIGITS)
        arrays[f"digit_{key}_lp"] = lp
        r = digit_readout(lp, lp)
        res = {
            "digit_mass": float(r["mass"].mean()),
            "digit_mass_min": float(r["mass"].min()),
            "mean_digit": float(r["mean_digit"].mean()),
            "mean_digit_std": float(r["mean_digit"].std()),
            "mean_digit_true": float(r["mean_digit"][st.labels].mean()),
            "mean_digit_false": float(r["mean_digit"][~st.labels].mean()),
            "argmax_digit_counts": np.bincount(lp.argmax(1), minlength=10).tolist(),
        }
        summary["digit_questions"][key] = res
        print(f"{key} {stim!r}: {res}")
        print_top_tokens(model, tok, batches[0], config.autocast_bf16, st, 3)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "g0.json").write_text(json.dumps(summary, indent=1))
    np.savez(out_dir / "g0.npz", **arrays)  # pyright: ignore[reportArgumentType]


def binary_summary(lp_a: np.ndarray, lp_b: np.ndarray, st: Statements) -> dict[str, float]:
    """Mass of the two answer tokens, accuracy and AUC of log P(a) - log P(b) against the label."""
    s = lp_a - lp_b
    return {
        "mass": float((np.exp(lp_a) + np.exp(lp_b)).mean()),
        "mass_min": float((np.exp(lp_a) + np.exp(lp_b)).min()),
        "accuracy": float(((s > 0) == st.labels).mean()),
        "auc": float(auc_columns(s[:, None], st.labels)[0]),
        "share_a": float((s > 0).mean()),
    }


def print_top_tokens(
    model: ComponentModel,
    tok: PreTrainedTokenizer,
    bt: Batch,
    autocast: bool,
    st: Statements,
    n: int,
) -> None:
    """Observe & Verify: the decoded prompt and the original model's top next tokens for n rows."""
    with bf16_autocast(enabled=autocast):
        logits = model(bt.ids)
    for r in range(n):
        end = int(bt.end[r])
        p = torch.softmax(logits[r, end].float(), -1)
        top = torch.topk(p, 6)
        shown = ", ".join(
            f"{tok.decode([int(i)])!r} {float(v):.2f}"
            for v, i in zip(top.values, top.indices, strict=True)
        )
        print(f"    [{tok.decode(bt.ids[r, : end + 1])!r}] ({st.labels[bt.rows[r]]}) -> {shown}")


# ---- g1 ----


def g1(
    run: str,
    basis: Literal["decomposition", "data_aligned"],
    split: Literal["val", "test"],
    termination: str,
    out: str,
    yes_no: str | tuple[str, ...],
    digit_questions: str | tuple[str, ...],
    t2: bool,
    n_per_dataset: int | None = None,
    batch_size: int = 64,
    device: str = "cuda",
    seed: int = 0,
) -> None:
    """One basis on one split (G1: decompositions on val without T2; G2: all bases on test with T2).
    termination: paper_termination.py's JSON of the same run and basis (T1 ranking, T2's k grid and
    candidates). yes_no, digit_questions: keys of YES_NO_PROMPTS and DIGIT_QUESTIONS (from g0).
    t2: also T2 under the Yes/No prompts, with phase D's exact removal sets: the top-k of the T1
    ranking for every k of its grid, and for an active set N_RANDOM random size-k subsets, redrawn
    with paper_termination.py's generator (default_rng(seed), these are its first draws). Two of the
    sets also run under the 1/0 prompt (top10, rand10 #0), to compare per statement with phase D's
    stored termination scores (checks the redraw and the GPU).
    The data-aligned basis has no CIs, so it has no components_ci state."""
    torch.set_grad_enabled(False)
    # fire passes "a,b" as a tuple and a single key as a str
    yes_no_keys = [yes_no] if isinstance(yes_no, str) else list(yes_no)
    digit_keys = [digit_questions] if isinstance(digit_questions, str) else list(digit_questions)
    assert set(yes_no_keys) <= set(YES_NO_PROMPTS), yes_no_keys
    assert set(digit_keys) <= set(DIGIT_QUESTIONS), digit_keys
    model, config, run_id, tok = load_run(run, device)
    lens = config.target_lens
    task = config.task_config
    assert isinstance(lens, StimulusLens) and isinstance(task, LMTaskConfig)
    assert task.prepared_datasets is not None
    autocast = config.autocast_bf16
    modules = sorted_modules(model)
    layer = {m: int(m.split(".")[2]) for m in modules}
    train = verdict_prompts(lens, tok, task.max_seq_len)[0]
    bas = make_basis(
        basis, model, run_id, task.prepared_datasets,
        lambda tx: make_batches(tx, tok, task.loss_positions, train, batch_size, device),
        autocast, seed,
    )  # fmt: skip
    has_ci = basis == "decomposition"
    ci_sampling = config.sampling if has_ci else None
    n_c = {m: bas.U[m].shape[0] for m in modules}
    by_name = {f"L{layer[m]}:{c}": (m, c) for m in modules for c in range(n_c[m])}

    def scale_of(names: list[str], factor: float) -> dict[str, Tensor]:
        scale = {m: torch.ones(n_c[m], device=device) for m in modules}
        for nm in names:
            m, c = by_name[nm]
            scale[m][c] = factor
        return scale

    term = json.loads(Path(termination).read_text())
    assert term["run"] == run and term["basis"] == bas.name, (term["run"], term["basis"], run)
    top = term["ranked"][:TOP_K]
    print(f"{run_id} {bas.name}: reverse_top10 = {top}")
    edits = {s: state_edit(s) for s in ("full", "all_off", "delta_only")}
    if has_ci:
        edits["components_ci"] = state_edit("components_ci")
    edits["reverse_top10"] = scaled_edit(scale_of(top, -1.0))

    t2_edits: dict[str, StateEdit] = {}
    if t2:
        rng = np.random.default_rng(seed)
        active = term["candidates"] if term["candidates_are_active_set"] else None
        for k in term["ks"]:
            t2_edits[f"T2 top{k}"] = scaled_edit(scale_of(term["ranked"][:k], 0.0))
            if active is not None and k < len(active):
                for r in range(N_RANDOM):
                    draw = [active[i] for i in rng.choice(len(active), size=k, replace=False)]
                    t2_edits[f"T2 rand{k} #{r}"] = scaled_edit(scale_of(draw, 0.0))
        print(f"T2: {len(t2_edits)} removal sets (ks {term['ks']})")
    t2_check = {n: e for n, e in t2_edits.items() if n in ("T2 top10", "T2 rand10 #0")}

    st = maybe_cap(load_statements(split, task.prepared_datasets), n_per_dataset, seed)

    def batches_of(prompt: Prompt) -> list[Batch]:
        return make_batches(st.texts, tok, task.loss_positions, prompt, batch_size, device)

    summary: dict[str, Any] = {
        "run": run, "run_id": run_id, "basis": bas.name, "split": split, "n": len(st),
        "reverse_top10": top, "yes_no": yes_no_keys, "digit_questions": digit_keys,
        "t2_conditions": list(t2_edits),
    }  # fmt: skip
    arrays: dict[str, np.ndarray] = {"labels": st.labels, "datasets": st.datasets}

    # 1. training prompt: verdict readout + next-token KL at the statement positions
    batches = batches_of(train)
    obj_ids = [
        object_token(t, lens.stimulus, f, tok) for t, f in zip(st.texts, st.families, strict=True)
    ]
    for bt in batches:
        for r, row in enumerate(bt.rows):
            ids = obj_ids[row][1]
            assert bt.ids[r, : len(ids)].tolist() == ids, (st.texts[row], ids)
    obj = np.array([o for o, _ in obj_ids])
    show_position_classes(tok, batches, obj, st)
    obj_t = {ib: torch.tensor(obj[bt.rows], device=device) for ib, bt in enumerate(batches)}
    train_edits = {**edits, **t2_check}
    r = run_prompt(
        model, bas, batches, train_edits, single_tokens(tok, ("1", "0")), autocast, ci_sampling,
        obj_t,
    )  # fmt: skip
    arrays.update({f"train_{k}": v for k, v in r.items()})
    arrays["object_token"] = obj
    assert np.abs(r["full_lp"] - r["original_lp"]).max() < 1.0, "full state does not reproduce"
    summary["statement_kl"] = {
        c: {
            s: ratio_summary(r[f"{s}_stmt_{c}"], r[f"all_off_stmt_{c}"], st.families, seed)
            for s in train_edits
            if s != "all_off"
        }
        for c in POSITION_CLASSES
    }
    for c in POSITION_CLASSES:
        for s, v in summary["statement_kl"][c].items():
            print(f"statement KL {c:10s} {s:14s} {v['num']:.4f} / all-off {v['den']:.4f} = "
                  f"{v['ratio']:.3f} [{v['interval'][0]:.3f}, {v['interval'][1]:.3f}]")  # fmt: skip
    summary["train_1_0"] = {
        s: truth_readout(r[f"{s}_lp"], r["original_lp"], r[f"{s}_kl"], r["all_off_kl"], st.labels)
        for s in train_edits
    }

    # 3. Yes/No transfer. Always ' Yes' / ' No': in g0 (26-10-31) every candidate put < 3e-5 of the
    # mass on the bare 'Yes' / 'No', also those ending in ': '.
    yn_edits = {s: e for s, e in edits.items() if s != "reverse_top10"} | t2_edits
    summary["yes_no_readout"], summary["yes_no_original"] = {}, {}
    for key in yes_no_keys:
        r = run_prompt(
            model, bas, batches_of(prompt_of(key, YES_NO_PROMPTS[key], lens)), yn_edits,
            single_tokens(tok, (" Yes", " No")), autocast, ci_sampling, None,
        )  # fmt: skip
        arrays.update({f"yes_no_{key}_{k}": v for k, v in r.items()})
        summary["yes_no_readout"][key] = {
            s: truth_readout(
                r[f"{s}_lp"], r["original_lp"], r[f"{s}_kl"], r["all_off_kl"], st.labels
            )
            for s in yn_edits
        }
        summary["yes_no_original"][key] = binary_summary(
            r["original_lp"][:, 0], r["original_lp"][:, 1], st
        )
        for s in [s for s in ("components_ci", "delta_only", "T2 top10") if s in yn_edits]:
            a, b = summary["train_1_0"].get(s), summary["yes_no_readout"][key][s]
            print(f"{s:14s} Yes/No ({key}) KL rec. {b['kl_recovered']:.3f} sep {b['sep_share']:.3f}"
                  + (f" | 1/0 KL rec. {a['kl_recovered']:.3f} sep {a['sep_share']:.3f}" if a else ""))  # fmt: skip

    # 2. digit questions
    dq_edits = {s: edits[s] for s in ("full", "all_off", "delta_only", "reverse_top10")}
    summary["digit"] = {}
    for key in digit_keys:
        r = run_prompt(
            model, bas, batches_of(prompt_of(key, DIGIT_QUESTIONS[key], lens)), dq_edits,
            single_tokens(tok, DIGITS), autocast, None, None,
        )  # fmt: skip
        arrays.update({f"digit_{key}_{k}": v for k, v in r.items()})
        res: dict[str, Any] = {}
        off = digit_readout(r["all_off_lp"], r["original_lp"])
        for s in dq_edits:
            d = digit_readout(r[f"{s}_lp"], r["original_lp"])
            res[s] = {
                "digit_kl": float(d["kl"].mean()),
                "digit_kl_vs_all_off": ratio_summary(d["kl"], off["kl"], st.families, seed),
                "shift": float(d["shift"].mean()),
                "abs_shift": float(np.abs(d["shift"]).mean()),
                "shift_true": float(d["shift"][st.labels].mean()),
                "shift_false": float(d["shift"][~st.labels].mean()),
                "digit_mass": float(d["mass"].mean()),
                "answer_kl": float(r[f"{s}_kl"].mean()),
            }
            v = res[s]
            print(f"digit {key:9s} {s:14s} KL {v['digit_kl']:.4f} (share of all-off "
                  f"{v['digit_kl_vs_all_off']['ratio']:.3f}) shift {v['shift']:+.3f} "
                  f"|shift| {v['abs_shift']:.3f} mass {v['digit_mass']:.3f}")  # fmt: skip
        summary["digit"][key] = res

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"side_{run_id}_{basis}_{split}"
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    np.savez(out_dir / f"{stem}.npz", **arrays)  # pyright: ignore[reportArgumentType]


def show_position_classes(
    tok: PreTrainedTokenizer, batches: list[Batch], obj: np.ndarray, st: Statements
) -> None:
    """Observe & Verify: one statement per family with its tokens tagged by position class."""
    shown: set[str] = set()
    tags = {0: "P", 1: "R", 2: "O"}
    for bt in batches:
        cls = position_classes(torch.tensor(obj[bt.rows], device=bt.dec.device), bt.dec)
        for r in range(len(bt.rows)):
            fam = str(st.families[bt.rows[r]])
            if fam in shown:
                continue
            shown.add(fam)
            parts = [
                f"{tok.decode([int(i)])!r}/{tags.get(int(c), '-')}"
                for i, c in zip(
                    bt.ids[r, : int(bt.end[r]) + 1], cls[r, : int(bt.end[r]) + 1], strict=True
                )
            ]
            print(f"  {fam}: " + " ".join(parts))


if __name__ == "__main__":
    fire.Fire({"g0": g0, "g1": g1})
