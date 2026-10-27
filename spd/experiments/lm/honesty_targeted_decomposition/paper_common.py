"""Shared code of the paper analyses (topic convos/julian/26-10-22_tpd_truth_paper_LOG.md; analysis plan
frozen 26-10-27 in the LOG entries *D3 plan, revision 2*, *D3 plan, revision 3* and *manipulation: A plus
full exploratory tables*): the statement sets (val, test, OOD), the two verdict prompts, batching,
summary metrics with bootstrap intervals, and edits of the decomposed down_proj outputs.

Edits are forward hooks on the plain target model, written for any linear basis of a module:
W = sum_c u_c v_c^T + R, with R = delta for a decomposition (trained or untrained) and R = W (1 - P_V)
for the data-aligned basis. An edit sets y' = d_on * (y - C) + sum_c mask_c (v_c . x) u_c at the
decomposed positions (C = all basis writes at mask 1, so y - C = R x), and leaves every other position
on the original weights. One code path for trained decompositions and controls, so a difference
between them cannot come from the intervention code.

Conventions of the frozen plan: CI-weighted states, no CI threshold as a definition of "active";
metrics per run (never pooled over seeds), pooled over statements and macro-averaged over families;
95% bootstrap intervals resampled within family.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
import yaml
from jaxtyping import Float
from torch import Tensor
from transformers import PreTrainedTokenizer

from spd.configs import LossPositions, StimulusLens
from spd.experiments.lm.honesty_targeted_decomposition.probe_ci import load_split_with_labels
from spd.experiments.lm.honesty_targeted_decomposition.verdict_lens_analysis import (
    SECOND_STIMULUS,
)
from spd.experiments.lm.prepared_datasets import resolve_dataset_dir
from spd.experiments.lm.prompts_dataset import load_prompts_dataset, trim_to_loss_positions
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import POSITION_MASK_KEY, STIMULUS_END_KEY

# OOD files (upstream Truth-is-Universal files, each a _true and a _false dataset) excluded from all
# analyses: verdict-prompt AUC of the original model < 0.7 (B2, ~/spd_out/26-10-22_tpd_truth_paper/
# b2_full/b2.json; bar decided by Julian, LOG entry *D1 answers*, point 2). 24 of 28 files stay.
EXCLUDED_OOD = ("inventors_disj", "neg_element_symb_de", "neg_facts_de", "neg_sp_en_trans_de")
N_OOD_FILES = 24
# OOD files where the model is mostly right (B2: verdict-prompt accuracy >= 0.8 and AUC >= 0.9), the
# subset for analyses against the ground-truth label (same LOG entry). Analyses against the model's
# own verdict use all 24.
GROUND_TRUTH_OOD = (
    "animal_class_conj", "animal_class_de", "cities_conj", "cities_de", "element_symb_conj",
    "element_symb_de", "facts", "inventors", "inventors_de", "neg_cities_de", "neg_inventors_de",
    "sp_en_trans_conj", "sp_en_trans_de",
)  # fmt: skip

StatementSplit = Literal["val", "test", "ood"]
Form = Literal["affirmative", "negated", "comparison", "conjunction", "disjunction"]


def family_of(dataset: str) -> str:
    assert dataset.endswith(("_true", "_false")), dataset
    return dataset.removesuffix("_true").removesuffix("_false")


def form_of(family: str) -> Form:
    """Sentence form of a tiu family; negation is checked first (neg_cities_de is negated)."""
    if family.startswith("neg_"):
        return "negated"
    if family in ("larger_than", "smaller_than"):
        return "comparison"
    if family.endswith("_conj"):
        return "conjunction"
    if family.endswith("_disj"):
        return "disjunction"
    return "affirmative"


@dataclass(frozen=True)
class Statements:
    """One statement set; all arrays have one entry per statement, in loading order."""

    split: StatementSplit
    texts: list[str]
    labels: np.ndarray  # bool, ground truth
    datasets: np.ndarray  # str, e.g. "neg_cities_de_false"
    families: np.ndarray  # str, e.g. "neg_cities_de"
    forms: np.ndarray  # str, Form
    german: np.ndarray  # bool

    def __len__(self) -> int:
        return len(self.texts)


def ood_dataset_dirs(train_dirs: list[str]) -> list[str]:
    """The OOD test-only datasets of the training data's version, minus EXCLUDED_OOD.

    Found by their manifest's `role: ood_test` (written by build_tiu.py) next to the training datasets.
    """
    versions = {Path(d).name for d in train_dirs}
    assert len(versions) == 1, f"training datasets of mixed versions: {versions}"
    version = versions.pop()
    data_root = resolve_dataset_dir(train_dirs[0]).parent.parent
    found = []
    for manifest in sorted(data_root.glob(f"*/{version}/manifest.yaml")):
        if yaml.safe_load(manifest.read_text())["role"] != "ood_test":
            continue
        if family_of(manifest.parent.parent.name) in EXCLUDED_OOD:
            continue
        found.append(str(manifest.parent))
    assert len(found) == 2 * N_OOD_FILES, (
        f"expected {2 * N_OOD_FILES} OOD datasets, found {len(found)}"
    )
    return found


def load_statements(split: StatementSplit, train_dirs: list[str]) -> Statements:
    if split == "ood":
        texts, labels, names = load_split_with_labels(ood_dataset_dirs(train_dirs), "test")
    else:
        texts, labels, names = load_split_with_labels(train_dirs, split)
    families = [family_of(nm) for nm in names]
    return Statements(
        split=split,
        texts=texts,
        labels=np.asarray(labels, dtype=bool),
        datasets=np.array(names),
        families=np.array(families),
        forms=np.array([form_of(f) for f in families]),
        german=np.array(["_de" in f for f in families]),
    )


@dataclass(frozen=True)
class Prompt:
    name: Literal["train", "second"]
    lens: StimulusLens
    max_seq_len: int


def verdict_prompts(lens: StimulusLens, tok: PreTrainedTokenizer, max_seq_len: int) -> list[Prompt]:
    """The training verdict prompt and the held-out second prompt (SECOND_STIMULUS).

    The run's max_seq_len fits its own prompt. The second prompt is longer, and its statement end does
    not merge with the stimulus ('.' + ' Is' vs '.' + '\\n' -> '.\\n'), hence the + 1 (as in
    verdict_lens_analysis.py).
    """
    extra = len(tok.encode(SECOND_STIMULUS, add_special_tokens=False)) - len(
        tok.encode(lens.stimulus, add_special_tokens=False)
    )
    second = lens.model_copy(update={"stimulus": SECOND_STIMULUS})
    return [Prompt("train", lens, max_seq_len), Prompt("second", second, max_seq_len + extra + 1)]


class Batch(NamedTuple):
    rows: np.ndarray  # statement indices
    ids: Tensor  # (b, pos)
    dec: Tensor  # (b, pos) bool, the run's decomposed positions (the statement tokens)
    end: Tensor  # (b,) position after which the verdict digit is read


def make_batches(
    texts: list[str],
    tok: PreTrainedTokenizer,
    loss_positions: LossPositions,
    prompt: Prompt,
    batch_size: int,
    device: str,
) -> list[Batch]:
    data = load_prompts_dataset(texts, tok, prompt.max_seq_len, loss_positions, prompt.lens)[:]
    first_end = int(data[STIMULUS_END_KEY][0])
    print(
        f"prompt {prompt.name} {prompt.lens.stimulus!r}: {len(texts)} statements; example tokens "
        f"{[tok.decode([i]) for i in data['input_ids'][0][: first_end + 1]]}, decomposed "
        f"{data[POSITION_MASK_KEY][0].nonzero().flatten().tolist()}"
    )
    batches = []
    for b in range(0, len(texts), batch_size):
        bt = trim_to_loss_positions({k: v[b : b + batch_size] for k, v in data.items()})
        batches.append(
            Batch(
                np.arange(b, b + len(bt["input_ids"])),
                bt["input_ids"].to(device),
                bt[POSITION_MASK_KEY].to(device).bool(),
                bt[STIMULUS_END_KEY].to(device),
            )
        )
    assert sum(len(bt.rows) for bt in batches) == len(texts)
    return batches


@dataclass(frozen=True)
class Basis:
    """A linear basis of each decomposed module: write of c = (v_c . x) u_c. fp32 on the device."""

    name: str
    U: dict[str, Float[Tensor, "C d_out"]]
    V: dict[str, Float[Tensor, "d_in C"]]


def decomposition_basis(model: ComponentModel, name: str) -> Basis:
    modules = sorted_modules(model)
    return Basis(
        name,
        {m: model.components[m].U.float() for m in modules},
        {m: model.components[m].V.float() for m in modules},
    )


def sorted_modules(model: ComponentModel) -> list[str]:
    modules = sorted(model.target_module_paths, key=lambda m: int(m.split(".")[2]))
    for m in modules:
        assert m.endswith(".mlp.down_proj"), m
        # the all-off state must give an output of exactly 0
        assert model.target_model.get_submodule(m).bias is None, m
    return modules


# (module, batch, x, y) -> the per-component mask (b, pos, C) and whether the remainder y - C stays
MaskFn = Callable[[str, Batch, Tensor, Tensor], tuple[Tensor, bool]]


@contextmanager
def edited_outputs(
    model: ComponentModel, basis: Basis, batch: Batch, mask_fn: MaskFn
) -> Iterator[None]:
    """Within the context, a plain forward of model.target_model computes at the decomposed positions
    y' = d_on (y - C) + sum_c mask_c (v_c . x) u_c (module docstring). Call with model(ids) and no
    mask_infos, which runs the target model without ComponentModel's own hooks."""
    handles = []
    for m in basis.U:

        def hook(
            _mod: torch.nn.Module, args: tuple[Tensor, ...], output: Tensor, m: str = m
        ) -> Tensor:
            x, y = args[0].float(), output.float()
            acts = x @ basis.V[m]  # (b, pos, C)
            mask, d_on = mask_fn(m, batch, acts, y)
            assert mask.shape == acts.shape, (mask.shape, acts.shape)
            new = (acts * mask) @ basis.U[m]
            if d_on:
                new = new + y - acts @ basis.U[m]
            return torch.where(batch.dec[..., None], new, y).to(output.dtype)

        handles.append(model.target_model.get_submodule(m).register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def verdict_logprobs(logits: Tensor, end: Tensor) -> Tensor:
    """Log-probabilities over the vocabulary at the answer position, fp32, (b, vocab)."""
    return torch.log_softmax(logits[torch.arange(len(end), device=end.device), end].float(), -1)


# ---- summary metrics (pure, numpy) ----


def kl_recovered(kl_state: np.ndarray, kl_off: np.ndarray) -> float:
    """1 - mean KL(state) / mean KL(all off): 1 = the original verdict, 0 = the MLPs off."""
    assert kl_off.mean() > 0
    return 1 - float(kl_state.mean()) / float(kl_off.mean())


def state_metrics(
    score: np.ndarray,
    kl: np.ndarray,
    base_score: np.ndarray,
    kl_off: np.ndarray,
    labels: np.ndarray,
    mass: np.ndarray,
) -> dict[str, float]:
    """Metrics of one state on one set of statements. Verdict score s = log P('1') - log P('0')."""
    pred, base = score > 0, base_score > 0
    sep0 = base_score[labels].mean() - base_score[~labels].mean()
    return {
        "kl_recovered": kl_recovered(kl, kl_off),
        "kl": float(kl.mean()),
        "accuracy": float((pred == labels).mean()),  # against the ground truth
        "agreement": float((pred == base).mean()),  # against the model's own verdict
        "sep_share": float((score[labels].mean() - score[~labels].mean()) / sep0),
        "digit_mass": float(mass.mean()),
    }


def macro(per_family: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = next(iter(per_family.values())).keys()
    return {k: float(np.mean([v[k] for v in per_family.values()])) for k in keys}


def family_metrics(
    families: np.ndarray, fn: Callable[[np.ndarray], dict[str, float]]
) -> dict[str, dict[str, float]]:
    """fn(statement indices) per family. Every family holds true and false statements."""
    return {f: fn(np.flatnonzero(families == f)) for f in sorted(set(families))}


def bootstrap_intervals(
    families: np.ndarray,
    fn: Callable[[np.ndarray], dict[str, float]],
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """95% intervals of every metric of fn(indices), resampling statements with replacement within
    each family (so every family keeps its size and the macro average stays defined)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(families == f) for f in sorted(set(families))]
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        for k, v in fn(idx).items():
            draws.setdefault(k, []).append(v)
    return {
        k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for k, v in draws.items()
    }


def summarize_state(
    score: np.ndarray,
    kl: np.ndarray,
    mass: np.ndarray,
    base_score: np.ndarray,
    kl_off: np.ndarray,
    labels: np.ndarray,
    families: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """state_metrics pooled over all statements, per family, macro-averaged over families, and the
    bootstrap intervals of the macro averages. All arrays per statement."""

    def fn(idx: np.ndarray) -> dict[str, float]:
        return state_metrics(
            score[idx], kl[idx], base_score[idx], kl_off[idx], labels[idx], mass[idx]
        )

    def fn_macro(idx: np.ndarray) -> dict[str, float]:
        fam = families[idx]
        return macro({f: fn(idx[fam == f]) for f in sorted(set(fam))})

    per_family = family_metrics(families, fn)
    return {
        "pooled": fn(np.arange(len(score))),
        "macro": macro(per_family),
        "macro_interval": bootstrap_intervals(families, fn_macro, n_boot, seed),
        "per_family": per_family,
    }


def smallest_reaching(curve: dict[int, float], target: float) -> int:
    """Smallest m whose value reaches target. The curve need not be monotone; asserts one m does."""
    hits = [m for m, v in sorted(curve.items()) if v >= target]
    assert hits, f"no m reaches {target:.4f}; best {max(curve.values()):.4f}"
    return hits[0]
