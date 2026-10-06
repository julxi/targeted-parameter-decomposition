"""Truth-writer analyses on finished decompositions: do tPD components write into the truth
direction, where in the model is truth written, and does the next-token output depend on truth?

Analyses A1-A7 and gate G of convos/julian/26-10-06_truth_writing_components_LOG.md. Everything
uses the last real token of each sample (the readout of the earlier truth probes).

Two steps:
- `extract` (GPU): one pass of the frozen model over all samples, storing per-sample last-token
  features, per-group sums of every sublayer's output, and the component weights; then the A7
  pass (forward + backward with component masks) on a per-dataset subsample.
- `analyze` (CPU): truth directions, A1-A7, G from the stored arrays; writes `results.json`,
  `report.txt`, `a5_matrices.npz`. Rerunnable without the GPU.

Data (`data` argument):
- `tiu`: the Truth-is-Universal datasets the first run was trained on. Splits: `fit` and `tune`
  = the run's train split, divided by subject (TIU_TUNE_FRAC of the subject strings go to tune,
  so a subject never sits in both); `test` = the run's test split (held-out subjects, same
  families). `tune` exists only to pick the logistic-regression C without touching test.
- `uth`: the UTH datasets as in probe_uth.py: `fit`/`tune` = train tasks' train/test splits,
  `test` = held-out task categories (cross-task).

Truth pairs: tiu samples pair by (family, subject); UTH samples by the text before the final
"\\nAnswer:" (e.g. copa, story_cloze: one context, two endings). Only keys with exactly one true
and one false sample count as a pair; pairs keep pair members in the same CV fold (A4) and give
the paired KL of gate G.

Stored per sample (features.npz, keys '<split>:<name>'):
- `resid_L<l>` (fp16, 3,584) for l in RESID_LAYERS: residual stream after decoder layer l;
- `<run>:ci` (480): lower_leaky CI values; `<run>:acts` (480): inner activations a_c = V_c . h of
  all components, h = the down_proj input, in the unmodified model (component index = 96 x
  position of the module in sorted order + c, i.e. layers 15..19);
- `lastlp` (VOCAB_K + 1): next-token log-probs at the last token for the VOCAB_K tokens most
  likely there on average (chosen on a fit subsample), plus the log of the remaining mass;
- `gsum:*`: per (dataset, label) group sums of the embedding, every attention and MLP output of
  layers 0..MAX_LAYER, and per run the Delta part `Delta_l h` of each decomposed layer. A2 needs
  only class means, and these sums give them for any direction chosen later; per-sample sublayer
  outputs would take ~10 GB for UTH.
- in a separate a7.npz (written after features.npz): `a7*` arrays, the A7 attributions and
  linearity check (see `a7_pass`).

Usage:
    python truth_writers.py extract <out_dir> <tiu|uth> <run_id> [<run_id> ...] [--cap N]
    python truth_writers.py analyze <out_dir> <n_workers>
    e.g. python truth_writers.py extract ~/spd_out/26-10-06_truth_writers/tiu tiu \\
             s-bd23f0d1 s-b6cce5de s-7fad0c14
Smoke tests: `--cap N` keeps at most N samples per (split, dataset); `--a7_cap M` lowers the A7
subsample to M per dataset (A7 is ~15x the forward cost and very slow on a CPU).
"""

import json
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import Tensor
from transformers import AutoTokenizer

from spd.experiments.lm.honesty_targeted_decomposition.probe_ci import WANDB_PREFIX, load_ci_fns
from spd.experiments.lm.honesty_targeted_decomposition.probe_uth import (
    CI_ACTIVE,
    LOGREG_CS,
    Targets,
    length_batches,
    load_uth,
    probe_feature,
)
from spd.experiments.lm.prepared_datasets import Split, load_prepared_split, resolve_dataset_dir
from spd.log import logger
from spd.models.component_model import (
    ComponentModel,
    OutputWithCache,
    SPDRunInfo,
    handle_deprecated_state_dict_keys_,
)
from spd.models.components import make_mask_infos
from spd.utils.general_utils import bf16_autocast

SPLITS = ("fit", "tune", "test")
RESID_LAYERS = (15, 16, 17, 18, 19, 23)
# Sublayer outputs are summed up to this layer: A2 attributes the truth separation along the
# layer-23 direction too.
MAX_LAYER = 23
TERMS = ["emb"] + [f"{kind}_{layer}" for layer in range(MAX_LAYER + 1) for kind in ("attn", "mlp")]
# Same memory budget as probe_uth.py (peak ~20.5 of 24 GB on an RTX 4090). The A7 pass keeps
# activations of layers 15-23 for the backward pass, hence the smaller batches.
BATCH_TOKENS = 2048
A7_BATCH_TOKENS = 1024
# A7 costs a forward and a backward pass per run, so it runs on a subsample of the fit and test
# splits: at most this many samples per dataset (random, seed 0).
A7_PER_DATASET = 200
A7_SPLITS = ("fit", "test")
# components per run and split whose first-order A7 estimate is checked by a real ablation
A7_N_CHECK = 5
VOCAB_K = 1000
VOCAB_PREPASS = 512
TIU_TUNE_FRAC = 0.2
# Per-dataset truth directions are fit only for datasets with at least this many fit samples per
# class; fewer give directions dominated by noise (split-half reliability is reported).
MIN_PER_CLASS = 20
N_NULL_DIRECTIONS = 200
TOP_WRITERS = 20
CV_FOLDS = 5


# ---------------------------------------------------------------------------------------------
# Data


@dataclass
class Samples:
    texts: list[str]
    labels: np.ndarray  # bool (n,)
    dataset: np.ndarray  # str (n,)
    category: np.ndarray  # str (n,)
    pair_key: np.ndarray  # str (n,); "" = no partner

    def subset(self, idx: np.ndarray) -> "Samples":
        return Samples(
            texts=[self.texts[i] for i in idx],
            labels=self.labels[idx],
            dataset=self.dataset[idx],
            category=self.category[idx],
            pair_key=self.pair_key[idx],
        )


def pair_keys(raw_keys: list[str], labels: np.ndarray) -> np.ndarray:
    """Keep a raw key only if exactly one true and one false sample carry it; else ""."""
    by_key: dict[str, list[bool]] = {}
    for key, label in zip(raw_keys, labels, strict=True):
        by_key.setdefault(key, []).append(bool(label))
    paired = {k for k, ls in by_key.items() if k and sorted(ls) == [False, True]}
    return np.array([k if k in paired else "" for k in raw_keys])


def load_tiu(dataset_dirs: list[str]) -> dict[str, Samples]:
    """tiu splits fit/tune/test (see module docstring); labels checked against the dir names."""
    file_splits: tuple[Split, Split] = ("train", "test")
    rows: dict[str, list[dict[str, Any]]] = {s: [] for s in file_splits}
    for dataset_dir in dataset_dirs:
        name = Path(dataset_dir).parent.name
        assert name.endswith(("_true", "_false")), name
        family = name.removesuffix("_true").removesuffix("_false")
        for split in file_splits:
            texts = load_prepared_split(dataset_dir, split)  # verifies the manifest hash
            path = resolve_dataset_dir(dataset_dir) / f"{split}.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert [r["text"] for r in records] == texts, f"{path}: texts differ"
            assert all(r["label"] is name.endswith("_true") for r in records), path
            rows[split] += [{**r, "family": family} for r in records]
    subjects = sorted({r["subject"] for r in rows["train"]})
    rng = np.random.default_rng(0)
    tune_subjects = set(rng.permutation(subjects)[: round(TIU_TUNE_FRAC * len(subjects))])
    parts = {
        "fit": [r for r in rows["train"] if r["subject"] not in tune_subjects],
        "tune": [r for r in rows["train"] if r["subject"] in tune_subjects],
        "test": rows["test"],
    }
    out = {}
    for split, recs in parts.items():
        labels = np.array([r["label"] for r in recs], dtype=bool)
        out[split] = Samples(
            texts=[r["text"] for r in recs],
            labels=labels,
            dataset=np.array([r["family"] for r in recs]),
            # category = domain: affirmative and negated versions of a family together
            category=np.array([r["family"].removeprefix("neg_") for r in recs]),
            pair_key=pair_keys([f"{r['family']}|{r['subject']}" for r in recs], labels),
        )
    return out


def load_uth_samples(version: str) -> dict[str, Samples]:
    out = {}
    for split, data in load_uth(version).items():
        t = data.targets
        raw = [
            f"{ds}|{text.rsplit(chr(10) + 'Answer:', 1)[0]}" if "\nAnswer:" in text else ""
            for ds, text in zip(t.dataset, data.texts, strict=True)
        ]
        out[split] = Samples(
            texts=data.texts,
            labels=t.labels,
            dataset=t.dataset,
            category=t.category,
            pair_key=pair_keys(raw, t.labels),
        )
    return out


def cap_per_dataset(samples: Samples, cap: int, seed: int) -> np.ndarray:
    """Sorted indices of at most `cap` random samples per dataset."""
    rng = np.random.default_rng(seed)
    keep = [
        rng.permutation(np.flatnonzero(samples.dataset == d))[:cap]
        for d in sorted(set(samples.dataset))
    ]
    return np.sort(np.concatenate(keep))


# ---------------------------------------------------------------------------------------------
# Extraction


class LastTokenCapture:
    """Forward hooks that keep, for selected modules, only each sample's last real token (fp32).

    Set `last_idx` (batch,) before every forward pass. Values keep their autograd graph, which
    the A7 pass needs."""

    def __init__(self, model: torch.nn.Module, paths: dict[str, str]):
        self.last_idx: Tensor | None = None
        self.values: dict[str, Tensor] = {}
        self.handles = [
            model.get_submodule(path).register_forward_hook(self._hook(key))
            for key, path in paths.items()
        ]

    def _hook(self, key: str):
        def hook(_module: torch.nn.Module, _inputs: object, out: object) -> None:
            # self_attn returns (attn_output, attn_weights); the other modules a tensor
            x = out[0] if isinstance(out, tuple) else out
            assert isinstance(x, Tensor) and self.last_idx is not None, key
            self.values[key] = x[torch.arange(x.shape[0], device=x.device), self.last_idx].float()

        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def sublayer_paths() -> dict[str, str]:
    paths = {"emb": "model.embed_tokens", "final": "model.norm"}
    for layer in range(MAX_LAYER + 1):
        paths[f"attn_{layer}"] = f"model.layers.{layer}.self_attn"
        paths[f"mlp_{layer}"] = f"model.layers.{layer}.mlp"
    for layer in RESID_LAYERS:
        paths[f"resid_{layer}"] = f"model.layers.{layer}"
    return paths


def padded_batch(
    token_ids: list[list[int]], rows: list[int], pad_id: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Right-padded ids (no attention mask, as in training: with causal attention padding cannot
    affect real positions) and the last real index of each row."""
    lengths = [len(token_ids[i]) for i in rows]
    width = max(lengths)
    ids = torch.tensor([token_ids[i] + [pad_id] * (width - len(token_ids[i])) for i in rows])
    return ids.to(device), torch.tensor(lengths, device=device) - 1


@dataclass
class RunWeights:
    V: dict[str, Tensor]  # module -> (d_in, C), fp32 CPU
    U: dict[str, Tensor]  # module -> (C, d_out)


def load_component_weights(run_info: SPDRunInfo, module_order: list[str]) -> RunWeights:
    weights = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=True)
    handle_deprecated_state_dict_keys_(weights)
    keys = {m: f"_components.{m.replace('.', '-')}" for m in module_order}
    return RunWeights(
        V={m: weights[f"{k}.V"].float() for m, k in keys.items()},
        U={m: weights[f"{k}.U"].float() for m, k in keys.items()},
    )


@torch.no_grad()
def choose_vocab(
    comp_model: ComponentModel, token_ids: list[list[int]], pad_id: int, capture: LastTokenCapture
) -> Tensor:
    """The VOCAB_K token ids with the highest mean next-token probability at the last token."""
    device = next(comp_model.parameters()).device
    lm_head = comp_model.target_model.get_submodule("lm_head")
    total: Tensor | None = None
    for rows in length_batches([len(t) for t in token_ids], BATCH_TOKENS):
        ids, capture.last_idx = padded_batch(token_ids, rows, pad_id, device)
        with bf16_autocast(enabled=True):
            comp_model(ids, logits_to_keep=1)
            logits = lm_head(capture.values["final"])
        probs = torch.softmax(logits.float(), -1).sum(0)
        total = probs if total is None else total + probs
    assert total is not None
    return torch.topk(total, VOCAB_K).indices.sort().values


@torch.no_grad()
def extract_split(
    comp_model: ComponentModel,
    ci_fns: dict[str, torch.nn.Module],
    run_weights: dict[str, RunWeights],
    samples: Samples,
    token_ids: list[list[int]],
    pad_id: int,
    vocab: Tensor,
    capture: LastTokenCapture,
) -> dict[str, np.ndarray]:
    """Per-sample features and per-group sums of one split (keys without the split prefix)."""
    device = next(comp_model.parameters()).device
    module_order = sorted(comp_model.target_module_paths)
    decomposed_layers = [int(m.split(".")[2]) for m in module_order]
    lm_head = comp_model.target_model.get_submodule("lm_head")
    lm_weight = lm_head.weight
    assert isinstance(lm_weight, Tensor)
    vocab_mask = torch.zeros(lm_weight.shape[0], dtype=torch.bool, device=device)
    vocab_mask[vocab] = True
    n = len(token_ids)
    feats: dict[str, np.ndarray] = {}

    def store(key: str, rows: list[int], values: Tensor, dtype: type) -> None:
        if key not in feats:
            feats[key] = np.zeros((n, values.shape[1]), dtype=dtype)
        feats[key][rows] = values.cpu().numpy().astype(dtype)

    groups = sorted({(d, bool(y)) for d, y in zip(samples.dataset, samples.labels, strict=True)})
    group_of = {g: i for i, g in enumerate(groups)}
    sample_group = torch.tensor(
        [group_of[(d, bool(y))] for d, y in zip(samples.dataset, samples.labels, strict=True)]
    )
    d_model = lm_weight.shape[1]
    gsum_terms = torch.zeros(len(groups), len(TERMS), d_model, dtype=torch.float64)
    gsum_delta = {
        r: torch.zeros(len(groups), len(module_order), d_model, dtype=torch.float64)
        for r in run_weights
    }
    weights_dev = {
        r: RunWeights(
            V={m: w.V[m].to(device) for m in module_order},
            U={m: w.U[m].to(device) for m in module_order},
        )
        for r, w in run_weights.items()
    }
    max_resid_err = 0.0
    batches = length_batches([len(t) for t in token_ids], BATCH_TOKENS)
    for n_done, rows in enumerate(batches):
        ids, capture.last_idx = padded_batch(token_ids, rows, pad_id, device)
        ar = torch.arange(len(rows), device=device)
        with bf16_autocast(enabled=True):  # as in training (autocast_bf16: true)
            out = comp_model(ids, cache_type="input", logits_to_keep=1)
            logits = lm_head(capture.values["final"])
        assert isinstance(out, OutputWithCache) and set(out.cache) == set(module_order)
        v = capture.values
        h_last = {m: out.cache[m][ar, capture.last_idx] for m in module_order}

        # The residual stream after layer l is exactly the embedding plus all attention and MLP
        # outputs up to l; A2 relies on this, so check it on every batch (bf16 rounding of the
        # residual adds is the only expected difference).
        terms = torch.stack([v[t] for t in TERMS], 1)  # (b, n_terms, d_model)
        for layer in RESID_LAYERS:
            summed = terms[:, : 1 + 2 * (layer + 1)].sum(1)
            err = float(
                (
                    (summed - v[f"resid_{layer}"]).norm(dim=-1) / v[f"resid_{layer}"].norm(dim=-1)
                ).max()
            )
            max_resid_err = max(max_resid_err, err)
            assert err < 0.05, f"residual decomposition off by {err:.3f} at layer {layer}"
        gsum_terms.index_add_(0, sample_group[rows], terms.double().cpu())

        for layer in RESID_LAYERS:
            store(f"resid_L{layer}", rows, v[f"resid_{layer}"], np.float16)
        for run_id, w in weights_dev.items():
            with bf16_autocast(enabled=True):
                pre = ci_fns[run_id](h_last)
            ci = torch.cat([comp_model.lower_leaky_fn(pre[m].float()) for m in module_order], -1)
            acts = {m: h_last[m].float() @ w.V[m] for m in module_order}
            # Delta_l h = W h - sum_c a_c U_c exactly (W = sum_c U_c V_c^T + Delta), with W h the
            # model's own down_proj output (= the MLP output in Qwen2)
            delta = torch.stack(
                [
                    v[f"mlp_{layer}"] - acts[m] @ w.U[m]
                    for m, layer in zip(module_order, decomposed_layers, strict=True)
                ],
                1,
            )
            gsum_delta[run_id].index_add_(0, sample_group[rows], delta.double().cpu())
            store(f"{run_id}:ci", rows, ci, np.float32)
            store(
                f"{run_id}:acts", rows, torch.cat([acts[m] for m in module_order], -1), np.float32
            )
        logp = torch.log_softmax(logits.float(), -1)
        rest = torch.logsumexp(logp.masked_fill(vocab_mask, float("-inf")), -1, keepdim=True)
        store("lastlp", rows, torch.cat([logp[:, vocab], rest], -1), np.float32)
        if n_done % 50 == 0:
            logger.info(f"batch {n_done}/{len(batches)} ({len(rows)} samples)")
    logger.info(f"max relative error of the residual decomposition: {max_resid_err:.4f}")
    for key, x in feats.items():
        assert np.isfinite(x).all(), f"non-finite values in {key}"
    feats["gsum:terms"] = gsum_terms.numpy()
    for run_id, s in gsum_delta.items():
        feats[f"gsum:{run_id}:delta"] = s.numpy()
    feats["gsum:dataset"] = np.array([g[0] for g in groups])
    feats["gsum:label"] = np.array([g[1] for g in groups])
    feats["gsum:count"] = np.bincount(sample_group.numpy(), minlength=len(groups))
    return feats


@contextmanager
def truncated_layers(comp_model: ComponentModel, n_layers: int) -> Iterator[None]:
    """Run only the first n_layers decoder layers. A7 needs the residual stream after layer
    MAX_LAYER only; without this, autograd would also keep the activations of the later layers,
    which depend on the masks but never enter the backward pass."""
    base = comp_model.target_model.get_submodule("model")
    full = base.layers
    assert isinstance(full, torch.nn.ModuleList) and len(full) > n_layers
    base.layers = torch.nn.ModuleList(list(full)[:n_layers])
    try:
        yield
    finally:
        base.layers = full


def grad_of(t: Tensor) -> Tensor:
    assert t.grad is not None, "mask received no gradient"
    return t.grad


@dataclass
class A7Result:
    attr: np.ndarray  # (n, n_components): mask gradient summed over all positions
    attr_first: np.ndarray  # (n, n_components): its part at position 0
    attr_last: np.ndarray  # (n, n_components): its part at the last real token
    attr_delta: np.ndarray  # (n, n_modules): Delta-mask gradient summed over all positions
    p: np.ndarray  # (n,)
    check: np.ndarray  # (A7_N_CHECK, 3): component index, predicted change -T_c, actual change


def a7_pass(
    comp_model: ComponentModel,
    weights: RunWeights,
    token_ids: list[list[int]],
    labels: np.ndarray,
    pad_id: int,
    w: Tensor,
    capture: LastTokenCapture,
) -> A7Result:
    """Attribution patching of p = (residual after layer MAX_LAYER at the last token) . w.

    The model runs through the components (mask 1) plus Delta (mask 1), which equals the original
    model up to rounding. Per sample, the gradient of p with respect to component c's mask, summed
    over positions, is the first-order change of p when c is switched off everywhere (with the
    opposite sign); it includes c's indirect effects through later layers and through attention
    from earlier positions. The first-position part is stored separately because the first token
    often carries very large activations, where a first-order estimate can be poor.

    Linearity check: for the A7_N_CHECK components with the largest |T_c| (T_c = true/false
    separation of the gradients), the separation is recomputed with that component really
    switched off at all positions; `check` holds the first-order prediction -T_c next to the
    actual change.
    """
    device = next(comp_model.parameters()).device
    module_order = sorted(comp_model.target_module_paths)
    n_c = [comp_model.components[m].C for m in module_order]
    comp_model.requires_grad_(False)
    for m in module_order:
        comp_model.components[m].V.data.copy_(weights.V[m])
        comp_model.components[m].U.data.copy_(weights.U[m])
    # bf16 like the frozen weights; autocast casts Delta to bf16 inside the einsum anyway
    deltas = {m: d.to(torch.bfloat16) for m, d in comp_model.calc_weight_deltas().items()}
    n = len(token_ids)
    attr = np.zeros((n, sum(n_c)), dtype=np.float32)
    attr_first = np.zeros_like(attr)
    attr_last = np.zeros_like(attr)
    attr_delta = np.zeros((n, len(module_order)), dtype=np.float32)
    p_all = np.zeros(n, dtype=np.float32)
    key = f"resid_{MAX_LAYER}"
    batches = length_batches([len(t) for t in token_ids], A7_BATCH_TOKENS)

    def masked_p(
        ids: Tensor, off: int | None, requires_grad: bool
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        """p with all masks 1, except global component index `off` set to 0 everywhere."""
        b, pos = ids.shape
        masks = {
            m: torch.ones(b, pos, c, device=device) for m, c in zip(module_order, n_c, strict=True)
        }
        if off is not None:
            i_mod = int(np.searchsorted(np.cumsum(n_c), off, side="right"))
            masks[module_order[i_mod]][..., off - sum(n_c[:i_mod])] = 0.0
        dmasks = {m: torch.ones(b, pos, device=device) for m in module_order}
        for t in [*masks.values(), *dmasks.values()]:
            t.requires_grad_(requires_grad)
        mask_infos = make_mask_infos(
            masks, weight_deltas_and_masks={m: (deltas[m], dmasks[m]) for m in module_order}
        )
        with bf16_autocast(enabled=True):
            comp_model(ids, mask_infos=mask_infos, logits_to_keep=1)
        return capture.values[key] @ w, masks, dmasks

    with truncated_layers(comp_model, MAX_LAYER + 1):
        for n_done, rows in enumerate(batches):
            ids, capture.last_idx = padded_batch(token_ids, rows, pad_id, device)
            last = capture.last_idx
            p, masks, dmasks = masked_p(ids, None, requires_grad=True)
            if n_done == 0:
                # Components + Delta must reproduce the original model; a forgotten Delta or a
                # wrong weight would show here as a large difference.
                with torch.no_grad(), bf16_autocast(enabled=True):
                    comp_model(ids, logits_to_keep=1)
                p_orig = capture.values[key] @ w
                rel = float((p.detach() - p_orig).abs().max() / p_orig.abs().mean())
                logger.info(f"A7 check: max |p_components - p_original| / mean |p| = {rel:.4f}")
                assert rel < 0.1, f"component forward differs from the original model ({rel:.3f})"
            p.sum().backward()  # samples are independent, so each mask row gets its own gradient
            grads = torch.cat([grad_of(masks[m]) for m in module_order], -1)  # (b, pos, n_comp)
            ar = torch.arange(len(rows), device=device)
            attr[rows] = grads.sum(1).cpu().numpy()
            attr_first[rows] = grads[:, 0].cpu().numpy()
            attr_last[rows] = grads[ar, last].cpu().numpy()
            attr_delta[rows] = (
                torch.stack([grad_of(dmasks[m]).sum(1) for m in module_order], -1).cpu().numpy()
            )
            p_all[rows] = p.detach().cpu().numpy()
            if n_done % 20 == 0:
                logger.info(f"A7 batch {n_done}/{len(batches)} ({len(rows)} samples)")

        t_c = mean_diff(attr.astype(np.float64), labels)
        s_base = float(mean_diff(p_all.astype(np.float64), labels))
        check = []
        for c in np.argsort(-np.abs(t_c))[:A7_N_CHECK]:
            p_off = np.zeros(n)
            for rows in batches:
                ids, capture.last_idx = padded_batch(token_ids, rows, pad_id, device)
                with torch.no_grad():
                    p_off[rows] = masked_p(ids, int(c), requires_grad=False)[0].cpu().numpy()
            check.append([float(c), -float(t_c[c]), float(mean_diff(p_off, labels)) - s_base])
            logger.info(
                f"A7 linearity: component {c}: predicted {check[-1][1]:+.3f}, actual {check[-1][2]:+.3f}"
            )
    return A7Result(attr, attr_first, attr_last, attr_delta, p_all, np.array(check))


def extract(
    out_dir: str, data: str, *run_ids: str, cap: int | None = None, a7_cap: int = A7_PER_DATASET
) -> None:
    """GPU step: features.npz, a7.npz, samples.jsonl, meta.json in out_dir (module docstring)."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    assert run_ids, "give the run ids (trained runs and the untrained baseline)"
    run_infos = {r: SPDRunInfo.from_path(WANDB_PREFIX + r) for r in run_ids}
    # One forward pass serves all runs, so everything that shapes the CI inputs must agree.
    ref = run_infos[run_ids[0]].config
    for run_id, info in run_infos.items():
        c = info.config
        assert c.pretrained_model_name == ref.pretrained_model_name, run_id
        assert c.pretrained_model_dtype == ref.pretrained_model_dtype, run_id
        assert c.module_info == ref.module_info and c.ci_config == ref.ci_config, run_id
        assert c.sigmoid_type == ref.sigmoid_type and c.sampling == "continuous", run_id
        assert c.tokenizer_name == ref.tokenizer_name and c.autocast_bf16, run_id
    assert ref.tokenizer_name is not None

    match data:
        case "tiu":
            dirs = ref.task_config.prepared_datasets  # pyright: ignore[reportAttributeAccessIssue]
            assert dirs and all("/tiu/" in d for d in dirs), f"first run is not a tiu run: {dirs}"
            samples = load_tiu(dirs)
        case "uth":
            samples = load_uth_samples("v1")
        case _:
            raise AssertionError(f"data must be tiu or uth, got {data!r}")
    if cap is not None:
        samples = {s: x.subset(cap_per_dataset(x, cap, seed=0)) for s, x in samples.items()}
        for x in samples.values():  # the cap can drop one member of a pair
            x.pair_key = pair_keys([str(k) for k in x.pair_key], x.labels)

    tokenizer = AutoTokenizer.from_pretrained(ref.tokenizer_name)
    pad_id = tokenizer.pad_token_id
    assert isinstance(pad_id, int)
    token_ids = {s: tokenizer(x.texts)["input_ids"] for s, x in samples.items()}
    with open(out / "samples.jsonl", "w") as f:
        for split, x in samples.items():
            for i, text in enumerate(x.texts):
                rec = {
                    "split": split,
                    "index": i,
                    "dataset": str(x.dataset[i]),
                    "label": bool(x.labels[i]),
                    "pair_key": str(x.pair_key[i]),
                    "n_tokens": len(token_ids[split][i]),
                    "last_tokens": [tokenizer.decode([t]) for t in token_ids[split][i][-4:]],
                    "text": text,
                }
                f.write(json.dumps(rec) + "\n")
    for split, x in samples.items():
        lens = [len(t) for t in token_ids[split]]
        assert min(lens) >= 2, f"{split}: a sample with fewer than 2 tokens"
        logger.info(
            f"{split}: {len(x.texts)} samples, {x.labels.mean():.3f} true, "
            f"{len(set(x.dataset))} datasets, {int((x.pair_key != '').sum()) // 2} pairs, "
            f"tokens median {int(np.median(lens))} max {max(lens)}"
        )

    comp_model = ComponentModel.from_run_info(run_infos[run_ids[0]])
    comp_model.eval()
    if torch.cuda.is_available():
        comp_model.to("cuda")
    device = next(comp_model.parameters()).device
    module_order = sorted(comp_model.target_module_paths)
    layers = [int(m.split(".")[2]) for m in module_order]
    assert layers == sorted(layers) and max(layers) <= MAX_LAYER, module_order
    ci_fns = load_ci_fns(run_infos, comp_model)
    for ci_fn in ci_fns.values():
        ci_fn.to(device)
    run_weights = {r: load_component_weights(i, module_order) for r, i in run_infos.items()}
    capture = LastTokenCapture(comp_model.target_model, sublayer_paths())

    rng = np.random.default_rng(0)
    pre_idx = rng.permutation(len(token_ids["fit"]))[:VOCAB_PREPASS]
    vocab = choose_vocab(comp_model, [token_ids["fit"][i] for i in pre_idx], pad_id, capture)
    arrays: dict[str, np.ndarray] = {"vocab": vocab.cpu().numpy()}
    for split, x in samples.items():
        logger.info(f"extracting {split}")
        feats = extract_split(
            comp_model, ci_fns, run_weights, x, token_ids[split], pad_id, vocab, capture
        )
        arrays.update({f"{split}:{k}": a for k, a in feats.items()})
        arrays[f"{split}:labels"] = x.labels
        arrays[f"{split}:dataset"] = x.dataset
        arrays[f"{split}:category"] = x.category
        arrays[f"{split}:pair_key"] = x.pair_key
    for run_id, w in run_weights.items():
        arrays[f"U:{run_id}"] = torch.cat([w.U[m] for m in module_order], 0).numpy()

    # Saved before A7, so a failure there doesn't cost the forward-pass features
    np.savez(out / "features.npz", **arrays)  # pyright: ignore[reportArgumentType]
    meta = {
        "data": data,
        "cap": cap,
        "runs": {
            r: {
                "checkpoint": i.checkpoint_path.name,
                "label": i.config.label,
                "target": i.config.task_config.prepared_datasets,  # pyright: ignore[reportAttributeAccessIssue]
            }
            for r, i in run_infos.items()
        },
        "module_order": module_order,
        "n_components_per_module": [comp_model.components[m].C for m in module_order],
        "resid_layers": list(RESID_LAYERS),
        "terms": TERMS,
        "vocab_tokens": [tokenizer.decode([t]) for t in vocab.tolist()],
        "n": {s: len(x.texts) for s, x in samples.items()},
        "a7_per_dataset": a7_cap,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"wrote {out / 'features.npz'}")

    a7_arrays: dict[str, np.ndarray] = {}
    # A7 direction: global mass-mean direction of the residual stream after MAX_LAYER on fit
    y = samples["fit"].labels
    resid = arrays[f"fit:resid_L{MAX_LAYER}"].astype(np.float64)
    w_np = resid[y].mean(0) - resid[~y].mean(0)
    w_np /= np.linalg.norm(w_np)
    a7_arrays["a7:w"] = w_np
    w = torch.tensor(w_np, dtype=torch.float32, device=device)
    for split in A7_SPLITS:
        idx = cap_per_dataset(samples[split], a7_cap, seed=1)
        a7_arrays[f"{split}:a7idx"] = idx
        for run_id, weights in run_weights.items():
            logger.info(f"A7 {split} {run_id}: {len(idx)} samples")
            r = a7_pass(
                comp_model,
                weights,
                [token_ids[split][i] for i in idx],
                samples[split].labels[idx],
                pad_id,
                w,
                capture,
            )
            a7_arrays[f"{split}:a7:{run_id}"] = r.attr
            a7_arrays[f"{split}:a7first:{run_id}"] = r.attr_first
            a7_arrays[f"{split}:a7last:{run_id}"] = r.attr_last
            a7_arrays[f"{split}:a7delta:{run_id}"] = r.attr_delta
            a7_arrays[f"{split}:a7p:{run_id}"] = r.p
            a7_arrays[f"{split}:a7check:{run_id}"] = r.check
    capture.remove()
    np.savez(out / "a7.npz", **a7_arrays)  # pyright: ignore[reportArgumentType]
    logger.info(f"wrote {out / 'a7.npz'}")


# ---------------------------------------------------------------------------------------------
# Analysis helpers


def unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    assert norm > 0
    return x / norm


def mean_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean over true rows minus mean over false rows (axis 0)."""
    assert y.any() and (~y).any(), "both classes needed"
    return x[y].mean(0) - x[~y].mean(0)


def group_mean_diff(
    gsum: np.ndarray, count: np.ndarray, label: np.ndarray, sel: np.ndarray
) -> np.ndarray:
    """Class-mean difference from per-group sums, pooling the selected groups' samples."""
    t, f = sel & label, sel & ~label
    assert t.any() and f.any(), "both classes needed"
    return gsum[t].sum(0) / count[t].sum() - gsum[f].sum(0) / count[f].sum()


def pooled_sd(proj: np.ndarray, y: np.ndarray) -> float:
    """Within-class standard deviation of a projection, pooled over the two classes."""
    var = (proj[y].var() * y.sum() + proj[~y].var() * (~y).sum()) / len(y)
    return float(np.sqrt(var))


def component_names(meta: dict[str, Any]) -> list[str]:
    names = []
    for m, n_c in zip(meta["module_order"], meta["n_components_per_module"], strict=True):
        names += [f"L{m.split('.')[2]}:{c}" for c in range(n_c)]
    return names


class Store:
    """Lazy access to an extract output directory: features.npz and a7.npz as one key space."""

    def __init__(self, out_dir: Path):
        self.files = [np.load(out_dir / "features.npz"), np.load(out_dir / "a7.npz")]
        keys = [set(f.files) for f in self.files]
        assert not keys[0] & keys[1], "a key in both features.npz and a7.npz"

    def __getitem__(self, key: str) -> np.ndarray:
        for f in self.files:
            if key in f.files:
                return f[key]
        raise KeyError(key)

    def f64(self, key: str) -> np.ndarray:
        return self[key].astype(np.float64)


def has_both_classes(store: Store, split: str, dataset: str) -> bool:
    y = store[f"{split}:labels"][store[f"{split}:dataset"] == dataset]
    return bool(y.any() and (~y).any())


def term_layer(term: str) -> int:
    """Layer of a TERMS entry; -1 for the embedding."""
    return -1 if term == "emb" else int(term.split("_")[1])


def home_split(store: Store, dataset: str) -> str:
    """The split a dataset is analysed on: fit if it has fit samples, else test."""
    for split in ("fit", "test"):
        if dataset in set(store[f"{split}:dataset"]):
            return split
    raise AssertionError(dataset)


# ---------------------------------------------------------------------------------------------
# Directions


def truth_directions(store: Store) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Unit truth directions in the residual stream (positive = true), all fit on the fit split.

    - mm_L<l>: global mass-mean direction after layer l;
    - lr_L19: logistic-regression direction (standardised features, C picked on tune), mapped
      back to the raw residual basis;
    - mm_L19|<dataset>: per-dataset mass-mean direction, for datasets with >= MIN_PER_CLASS fit
      samples per class.
    Info: split-half reliability of each per-dataset direction (cosine between the directions
    of two random halves of each class), and the chosen C.
    """
    y = store["fit:labels"]
    dirs = {f"mm_L{i}": unit(mean_diff(store.f64(f"fit:resid_L{i}"), y)) for i in RESID_LAYERS}
    x19 = store.f64("fit:resid_L19")
    x19_tune, y_tune = store.f64("tune:resid_L19"), store["tune:labels"]
    best: tuple[float, float, np.ndarray] | None = None
    for c in LOGREG_CS:
        scaler = StandardScaler().fit(x19)
        lr = LogisticRegression(C=c, max_iter=5000).fit(scaler.transform(x19), y)
        acc = float((lr.predict(scaler.transform(x19_tune)) == y_tune).mean())
        if best is None or acc > best[0]:
            best = (acc, c, lr.coef_[0] / scaler.scale_)
    assert best is not None
    dirs["lr_L19"] = unit(best[2])
    info: dict[str, Any] = {"lr_L19": {"C": best[1], "tune_acc": best[0]}, "reliability": {}}
    ds = store["fit:dataset"]
    rng = np.random.default_rng(0)
    for d in sorted(set(ds)):
        sel = ds == d
        if min((y & sel).sum(), (~y & sel).sum()) < MIN_PER_CLASS:
            continue
        dirs[f"mm_L19|{d}"] = unit(mean_diff(x19[sel], y[sel]))
        halves = []
        for _ in range(2):
            pick = np.zeros(len(y), dtype=bool)
            for cls in (True, False):
                idx = rng.permutation(np.flatnonzero(sel & (y == cls)))
                pick[idx[: len(idx) // 2]] = True
            halves.append(pick)
        a = unit(mean_diff(x19[halves[0]], y[halves[0]]))
        rest = sel & ~halves[0]
        b = unit(mean_diff(x19[rest], y[rest]))
        info["reliability"][d] = float(a @ b)
    return dirs, info


def null_directions(store: Store) -> np.ndarray:
    """Mass-mean directions after layer 19 for randomly permuted fit labels: directions with the
    residual stream's anisotropy but no truth content, the null for A1."""
    x = store.f64("fit:resid_L19")
    y = store["fit:labels"]
    rng = np.random.default_rng(0)
    return np.stack([unit(mean_diff(x, rng.permutation(y))) for _ in range(N_NULL_DIRECTIONS)])


# ---------------------------------------------------------------------------------------------
# A1 static alignment


def a1_alignment(
    store: Store, runs: list[str], dirs: dict[str, np.ndarray], null: np.ndarray, names: list[str]
) -> dict[str, Any]:
    """Cosine of each component's output direction U_c with each truth direction. Per run and
    direction: the largest |cos| over components, compared with the largest |cos| of the same U
    against each null direction (fraction of null directions reaching it = p-value)."""
    out: dict[str, Any] = {}
    for run in runs:
        u = store.f64(f"U:{run}")
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        null_max = np.abs(u @ null.T).max(0)  # (n_null,)
        res = {}
        for name, w in dirs.items():
            cos = u @ w
            top = np.argsort(-np.abs(cos))[:5]
            best = float(np.abs(cos).max())
            res[name] = {
                "max_abs_cos": best,
                "null_max_abs_cos_median": float(np.median(null_max)),
                "p_null": float((null_max >= best).mean()),
                "top": [(names[i], float(cos[i])) for i in top],
            }
        out[run] = res
    return out


# ---------------------------------------------------------------------------------------------
# A2 per-layer attribution


def a2_attribution(
    store: Store,
    runs: list[str],
    w: np.ndarray,
    layer: int,
    split: str,
    sel_dataset: str | None,
    names: list[str],
    meta: dict[str, Any],
    with_top: bool,
) -> dict[str, Any]:
    """Exact split of the true/false separation along w after `layer` into embedding, attention
    and MLP terms of layers 0..layer; for the decomposed layers, per run, the MLP term further
    into components and Delta. All values are raw separations (mean_true - mean_false of the
    projection); `sd` = pooled within-class sd of the projection, to read them in sd units."""
    n_terms = 1 + 2 * (layer + 1)
    label, count, gds = (
        store[f"{split}:gsum:label"],
        store[f"{split}:gsum:count"],
        store[f"{split}:gsum:dataset"],
    )
    gsel = np.ones(len(label), dtype=bool) if sel_dataset is None else gds == sel_dataset
    diff = group_mean_diff(store.f64(f"{split}:gsum:terms"), count, label, gsel)  # (n_terms_all, d)
    s_terms = diff[:n_terms] @ w
    y = store[f"{split}:labels"]
    ssel = (
        np.ones(len(y), dtype=bool)
        if sel_dataset is None
        else store[f"{split}:dataset"] == sel_dataset
    )
    proj = store.f64(f"{split}:resid_L{layer}")[ssel] @ w
    s_resid = float(mean_diff(proj, y[ssel]))
    s_sum = float(s_terms.sum())
    sd = pooled_sd(proj, y[ssel])
    # fp16 storage of the residual and bf16 residual adds are the only differences expected; the
    # sd term keeps the check meaningful when the separation itself is near zero (cross-task)
    assert abs(s_sum - s_resid) <= 0.05 * abs(s_resid) + 0.02 * sd, (
        split,
        sel_dataset,
        layer,
        s_sum,
        s_resid,
        sd,
    )
    res: dict[str, Any] = {
        "S_total": s_resid,
        "S_sum_of_terms": s_sum,
        "sd": sd,
        "terms": {t: float(s) for t, s in zip(TERMS[:n_terms], s_terms, strict=True)},
        "runs": {},
    }
    n_c = meta["n_components_per_module"][0]
    for run in runs:
        u = store.f64(f"U:{run}")
        acts_diff = mean_diff(store.f64(f"{split}:{run}:acts")[ssel], y[ssel])
        s_c = acts_diff * (u @ w)  # (n_components,)
        delta_diff = group_mean_diff(store.f64(f"{split}:gsum:{run}:delta"), count, label, gsel)
        per_layer = {}
        for i, m in enumerate(meta["module_order"]):
            mod_layer = int(m.split(".")[2])
            if mod_layer > layer:
                continue
            comp = s_c[i * n_c : (i + 1) * n_c]
            s_delta = float(delta_diff[i] @ w)
            s_mlp = res["terms"][f"mlp_{mod_layer}"]
            # exact identity W h = sum_c a_c U_c + Delta h (Delta h was stored as the remainder)
            assert abs(comp.sum() + s_delta - s_mlp) <= 1e-3 * (abs(s_mlp) + sd), (run, mod_layer)
            entry: dict[str, Any] = {"components": float(comp.sum()), "delta": s_delta}
            if with_top:
                top = np.argsort(-np.abs(comp))[:10]
                entry["top"] = [(names[i * n_c + j], float(comp[j])) for j in top]
            per_layer[mod_layer] = entry
        res["runs"][run] = {"layers": per_layer}
    return res


# ---------------------------------------------------------------------------------------------
# A3 / G1 probes and A4 within-task CV


def feature_set(store: Store, split: str, name: str) -> np.ndarray:
    """Feature matrix of a named set: 'resid_L19', 'lastlp', '<run>:ci', '<run>:acts',
    '<run>:actsci' (a_c x CI_c). No 'write along w' set: a_c x CI_c x (U_c . w) is a_c x CI_c
    times a constant per feature, which standardised logistic regression cannot tell apart."""
    match name.rsplit(":", 1):
        case [run, "actsci"]:
            return store[f"{split}:{run}:acts"] * store[f"{split}:{run}:ci"]
        case _:
            return store[f"{split}:{name}"].astype(np.float32)


def _probe_worker(out_dir: str, name: str, n_threads: int) -> tuple[str, dict[str, Any]]:
    with threadpool_limits(limits=n_threads):
        store = Store(Path(out_dir))
        feats = {s: feature_set(store, s, name) for s in SPLITS}
        targets = {
            s: Targets(
                labels=store[f"{s}:labels"],
                dataset=store[f"{s}:dataset"],
                category=store[f"{s}:category"],
            )
            for s in SPLITS
        }
        result = probe_feature(feats, targets)
    lr = result["logreg"]
    logger.info(
        f"A3 {name}: logreg (C={lr['selected_C']}) tune {lr['tune']['acc_pooled']:.3f}, "
        f"test {lr['test']['acc_pooled']:.3f}"
    )
    return name, result


def _cv_worker(out_dir: str, name: str, c: float, n_threads: int) -> tuple[str, dict[str, float]]:
    """Within-dataset cross-validated accuracy on each test-split dataset, pair members kept in
    the same fold. C is the one picked on tune in A3, never chosen on the test data."""
    with threadpool_limits(limits=n_threads):
        store = Store(Path(out_dir))
        x = feature_set(store, "test", name)
        y, ds, pk = store["test:labels"], store["test:dataset"], store["test:pair_key"]
        out = {}
        for d in sorted(set(ds)):
            sel = np.flatnonzero(ds == d)
            if min(y[sel].sum(), (~y[sel]).sum()) < 2 * CV_FOLDS:
                continue
            groups = np.array([k if k else f"single{i}" for i, k in zip(sel, pk[sel], strict=True)])
            folds = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=0)
            correct = 0
            for tr, te in folds.split(x[sel], y[sel], groups):
                probe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
                probe.fit(x[sel][tr], y[sel][tr])
                correct += int((probe.predict(x[sel][te]) == y[sel][te]).sum())
            out[d] = correct / len(sel)
    return name, out


# ---------------------------------------------------------------------------------------------
# A5 writers per dataset


def a5_writers(
    store: Store, runs: list[str], dirs: dict[str, np.ndarray], names: list[str]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Per dataset (on its home split) and component: write separation S_c = (mean_true a_c -
    mean_false a_c) (U_c . w) in the unmodified model, its CI-gated version, and activity (share
    of samples with CI > CI_ACTIVE). w = global layer-19 mass-mean direction, and each dataset's
    own direction where it has one. Datasets are compared through the correlation of their S_c
    vectors over components and the overlap of their top writers, against a split-half noise
    floor (two random halves of one dataset)."""
    w = dirs["mm_L19"]
    datasets = sorted(set(store["fit:dataset"]) | set(store["test:dataset"]))
    rng = np.random.default_rng(0)
    results: dict[str, Any] = {}
    matrices: dict[str, np.ndarray] = {"datasets": np.array(datasets)}
    for run in runs:
        u = store.f64(f"U:{run}")
        uw = u @ w
        s_global = mean_diff(store.f64(f"fit:{run}:acts"), store["fit:labels"]) * uw
        top = np.argsort(-np.abs(s_global))[:TOP_WRITERS]
        s_mat, s_own, gated, activity, total, halves = [], [], [], [], [], []
        for d in datasets:
            split = home_split(store, d)
            sel = store[f"{split}:dataset"] == d
            y = store[f"{split}:labels"][sel]
            acts = store.f64(f"{split}:{run}:acts")[sel]
            ci = store.f64(f"{split}:{run}:ci")[sel]
            s_mat.append(mean_diff(acts, y) * uw)
            own_key = f"mm_L19|{d}"  # absent for datasets with too few fit samples per class
            s_own.append(
                mean_diff(acts, y) * (u @ dirs[own_key])
                if own_key in dirs
                else np.full(len(uw), np.nan)
            )
            gated.append(mean_diff(acts * ci, y) * uw)
            activity.append((ci > CI_ACTIVE).mean(0))
            total.append(float(mean_diff(store.f64(f"{split}:resid_L19")[sel], y) @ w))
            # split-half noise floor; NaN (left out of the means) when a class is too small to halve
            if min(y.sum(), (~y).sum()) < 4:
                halves.append(np.nan)
                continue
            pick = np.zeros(len(y), dtype=bool)
            for cls in (True, False):
                idx = rng.permutation(np.flatnonzero(y == cls))
                pick[idx[: len(idx) // 2]] = True
            halves.append(
                np.corrcoef(
                    mean_diff(acts[pick], y[pick]) * uw, mean_diff(acts[~pick], y[~pick]) * uw
                )[0, 1]
            )
        s_mat_a, act_a = np.array(s_mat), np.array(activity)
        corr = np.corrcoef(s_mat_a)
        off = corr[~np.eye(len(datasets), dtype=bool)]
        top_sets = [set(np.argsort(-np.abs(row))[:TOP_WRITERS]) for row in s_mat_a]
        jacc = [
            len(top_sets[i] & top_sets[j]) / len(top_sets[i] | top_sets[j])
            for i in range(len(datasets))
            for j in range(i + 1, len(datasets))
        ]
        results[run] = {
            "top_writers_fit": [
                {
                    "component": names[c],
                    "S_c_fit": float(s_global[c]),
                    "activity": {d: float(act_a[k, c]) for k, d in enumerate(datasets)},
                    "S_c_over_total": {
                        d: float(s_mat_a[k, c] / total[k]) for k, d in enumerate(datasets)
                    },
                    "gated_over_total": {
                        d: float(gated[k][c] / total[k]) for k, d in enumerate(datasets)
                    },
                }
                for c in top
            ],
            "S_total_L19": dict(zip(datasets, total, strict=True)),
            "between_dataset_corr_mean": float(np.nanmean(off)),
            "split_half_corr_mean": float(np.nanmean(halves)),
            "split_half_corr": dict(zip(datasets, map(float, halves), strict=True)),
            "top_writer_jaccard_mean": float(np.mean(jacc)) if jacc else None,
        }
        matrices[f"{run}:S"] = s_mat_a
        matrices[f"{run}:S_own"] = np.array(s_own)
        matrices[f"{run}:gated"] = np.array(gated)
        matrices[f"{run}:activity"] = act_a
        matrices[f"{run}:corr"] = corr
    return results, matrices


# ---------------------------------------------------------------------------------------------
# A6 inversions


def a6_inversions(
    store: Store, runs: list[str], names: list[str], samples_path: Path
) -> dict[str, Any]:
    """For every test dataset where the CI probe (C = 1.0, fit split) scores more than 2 standard
    errors below chance: which components drive the inverted readout, whether a strongly
    regularised or a binarised probe still inverts, and the most confidently wrong samples."""
    texts: dict[int, str] = {}
    for line in samples_path.read_text().splitlines():
        rec = json.loads(line)
        if rec["split"] == "test":
            texts[rec["index"]] = rec["text"]
    y_fit, y_te, ds = store["fit:labels"], store["test:labels"], store["test:dataset"]
    out: dict[str, Any] = {}
    for run in runs:
        x_fit, x_te = store.f64(f"fit:{run}:ci"), store.f64(f"test:{run}:ci")
        probes = {}
        for variant in ("ci", "binary"):
            for c in LOGREG_CS:
                xf = x_fit if variant == "ci" else (x_fit > CI_ACTIVE).astype(np.float64)
                probes[(variant, c)] = make_pipeline(
                    StandardScaler(), LogisticRegression(C=c, max_iter=5000)
                ).fit(xf, y_fit)
        main = probes[("ci", 1.0)]
        # the same fit as `main`, unpacked: standardised coefficients and inputs
        scaler = StandardScaler().fit(x_fit)
        z_fit, z_te = np.asarray(scaler.transform(x_fit)), np.asarray(scaler.transform(x_te))
        beta = LogisticRegression(C=1.0, max_iter=5000).fit(z_fit, y_fit).coef_[0]
        delta_fit = mean_diff(z_fit, y_fit)
        res = {}
        for d in sorted(set(ds)):
            sel = ds == d
            prob = main.predict_proba(x_te[sel])[:, 1]
            acc = float(((prob > 0.5) == y_te[sel]).mean())
            se = 0.5 / np.sqrt(sel.sum())
            if acc >= 0.5 - 2 * se:
                continue
            delta_d = mean_diff(z_te[sel], y_te[sel])
            kappa = beta * delta_d
            worst = np.argsort(kappa)[:10]
            variable = (np.abs(delta_fit) > 0) & (np.abs(delta_d) > 0)
            idx = np.flatnonzero(sel)
            wrong = np.flatnonzero((prob > 0.5) != y_te[sel])
            conf_wrong = wrong[np.argsort(-np.abs(prob[wrong] - 0.5))][:20]
            variants = {}
            for (variant, c), p in probes.items():
                xv = x_te[sel] if variant == "ci" else (x_te[sel] > CI_ACTIVE).astype(np.float64)
                variants[f"{variant} C={c}"] = float((p.predict(xv) == y_te[sel]).mean())
            res[d] = {
                "n": int(sel.sum()),
                "acc_C1": acc,
                "mean_logit_separation": float(kappa.sum()),
                "corr_delta_fit_vs_dataset": float(
                    np.corrcoef(delta_fit[variable], delta_d[variable])[0, 1]
                ),
                "most_negative_contributors": [
                    {
                        "component": names[c],
                        "contribution": float(kappa[c]),
                        "beta": float(beta[c]),
                        "delta_fit": float(delta_fit[c]),
                        "delta_dataset": float(delta_d[c]),
                    }
                    for c in worst
                ],
                "probe_variants": variants,
                "confidently_wrong": [
                    {
                        "label": bool(y_te[idx[i]]),
                        "p_true": float(prob[i]),
                        "text_end": texts[int(idx[i])][-400:],
                    }
                    for i in conf_wrong
                ],
            }
        out[run] = res
    return out


# ---------------------------------------------------------------------------------------------
# A7 total effects


def a7_effects(store: Store, runs: list[str], names: list[str]) -> dict[str, Any]:
    """Per component: total-effect separation T_c = mean_true(g_c) - mean_false(g_c) of the A7
    gradients (first-order: switching c off changes the separation along the layer-23 direction
    by -T_c), next to the direct write D_c = (mean_true a_c - mean_false a_c) (U_c . w) on the
    same samples. T_c - D_c is the indirect part through later layers."""
    w = store["a7:w"]
    out: dict[str, Any] = {}
    for run in runs:
        u = store.f64(f"U:{run}")
        res: dict[str, Any] = {}
        t_fit: np.ndarray | None = None
        for split in A7_SPLITS:
            idx = store[f"{split}:a7idx"]
            y = store[f"{split}:labels"][idx]
            g = store.f64(f"{split}:a7:{run}")
            p = store.f64(f"{split}:a7p:{run}")
            p_orig = store.f64(f"{split}:resid_L{MAX_LAYER}")[idx] @ w
            corr = float(np.corrcoef(p, p_orig)[0, 1])
            assert corr > 0.99, (
                f"A7 {run} {split}: p differs from the original model (corr {corr:.4f})"
            )
            t_c = mean_diff(g, y)
            d_c = mean_diff(store.f64(f"{split}:{run}:acts")[idx], y) * (u @ w)
            t_delta = mean_diff(store.f64(f"{split}:a7delta:{run}"), y)
            t_first = mean_diff(store.f64(f"{split}:a7first:{run}"), y)
            t_last = mean_diff(store.f64(f"{split}:a7last:{run}"), y)
            if split == "fit":
                t_fit = t_c
            assert t_fit is not None
            top = np.argsort(-np.abs(t_fit))[:TOP_WRITERS]
            ds = store[f"{split}:dataset"][idx]
            per_ds = {}
            for d in sorted(set(ds)):
                sel = ds == d
                if y[sel].all() or (~y[sel]).all():
                    continue
                per_ds[d] = {
                    "S_total": float(mean_diff(p_orig[sel], y[sel])),
                    "top_T": [float(v) for v in mean_diff(g[sel], y[sel])[top]],
                }
            res[split] = {
                "n": len(idx),
                "p_corr_with_original": corr,
                "S_total": float(mean_diff(p_orig, y)),
                "sum_T_components": float(t_c.sum()),
                "T_delta_per_layer": [float(v) for v in t_delta],
                "top_by_fit_T": [
                    {
                        "component": names[c],
                        "T": float(t_c[c]),
                        "T_first_position": float(t_first[c]),
                        "T_last_position": float(t_last[c]),
                        "D": float(d_c[c]),
                    }
                    for c in top
                ],
                # first-order prediction (-T_c) vs the change measured by really ablating c
                "linearity_check": [
                    {"component": names[int(c)], "predicted": float(pr), "actual": float(ac)}
                    for c, pr, ac in store[f"{split}:a7check:{run}"]
                ],
                "per_dataset": per_ds,
            }
        out[run] = res
    return out


# ---------------------------------------------------------------------------------------------
# G gate: does the next-token output at the last token depend on truth?


def kl_rows(lp_a: np.ndarray, lp_b: np.ndarray) -> np.ndarray:
    """KL(a || b) per row over the VOCAB_K + 1 buckets (a lower bound on the full-vocabulary KL,
    since merging outcomes can only shrink a KL)."""
    return (np.exp(lp_a) * (lp_a - lp_b)).sum(1)


def g_gate(store: Store) -> dict[str, Any]:
    """Paired KL between the true and the false version of an item at the last token, against
    the KL between two different items of the same dataset and label (content variation)."""
    rng = np.random.default_rng(0)
    out: dict[str, Any] = {}
    for split in SPLITS:
        lp = store.f64(f"{split}:lastlp")
        coverage = float(np.exp(lp[:, :-1]).sum(1).mean())
        y, ds, pk = store[f"{split}:labels"], store[f"{split}:dataset"], store[f"{split}:pair_key"]
        members: dict[str, list[int]] = {}
        for i, k in enumerate(pk):
            if k:
                members.setdefault(str(k), []).append(i)
        assert all(len(m) == 2 for m in members.values()), "a pair key with other than 2 samples"
        per_ds = {}
        for d in sorted(set(ds)):
            keys = sorted({str(k) for k in pk[ds == d] if k})
            if not keys:
                continue
            t_idx = np.array([next(i for i in members[k] if y[i]) for k in keys])
            f_idx = np.array([next(i for i in members[k] if not y[i]) for k in keys])
            paired = kl_rows(lp[t_idx], lp[f_idx])
            other = rng.permutation(t_idx)
            ok = other != t_idx
            baseline = kl_rows(lp[t_idx[ok]], lp[other[ok]])
            per_ds[d] = {
                "n_pairs": len(keys),
                "paired_kl_mean": float(paired.mean()),
                "paired_kl_median": float(np.median(paired)),
                "baseline_kl_mean": float(baseline.mean()) if ok.any() else None,
            }
        out[split] = {"vocab_coverage_mean": coverage, "per_dataset": per_ds}
    return out


# ---------------------------------------------------------------------------------------------
# Report


def fmt(x: float) -> str:
    return f"{x:+.3f}" if abs(x) < 100 else f"{x:+.1f}"


def write_report(results: dict[str, Any], meta: dict[str, Any]) -> str:
    """The human-readable digest; the full numbers are in results.json."""
    lines: list[str] = []
    add = lines.append
    runs = list(meta["runs"])
    add(f"Truth-writer analyses, data = {meta['data']}, runs = {runs}, n = {meta['n']}")
    add(
        "Separations S = mean(true) - mean(false) of a projection on a unit truth direction; 'sd' is"
    )
    add("the pooled within-class sd of that projection.")
    add("")
    add("== Directions (fit split)")
    add(f"lr_L19: {results['directions']['lr_L19']}")
    rel = results["directions"]["reliability"]
    add(
        "per-dataset direction split-half reliability (cos): "
        + ", ".join(f"{d} {v:.2f}" for d, v in rel.items())
    )
    add("")
    add(
        "== A1 static alignment: max |cos(U_c, w)| over components; p_null = share of null directions"
    )
    add("   (permuted-label mass-mean) with at least that max. Global directions only here.")
    for run, res in results["A1"].items():
        for name, r in res.items():
            if "|" in name:
                continue
            add(
                f"{run} {name}: max {r['max_abs_cos']:.3f} (null median {r['null_max_abs_cos_median']:.3f}, p {r['p_null']:.3f}); top {r['top'][:3]}"
            )
    per_ds_p = {
        run: sum(1 for n, r in res.items() if "|" in n and r["p_null"] < 0.05)
        for run, res in results["A1"].items()
    }
    n_ds = sum(1 for n in next(iter(results["A1"].values())) if "|" in n)
    add(f"per-dataset directions with p_null < 0.05 (of {n_ds}): {per_ds_p}")
    add("")
    for dname in ("mm_L19", "mm_L23"):
        add(f"== A2 attribution along {dname} (fraction of S_total per term)")
        a2 = results["A2"][dname]
        layer = int(dname.removeprefix("mm_L"))
        add(
            "split: S_total (sd units) | "
            + " | ".join(
                f"{s}: {a2[s]['S_total']:.2f} ({a2[s]['S_total'] / a2[s]['sd']:.2f} sd)"
                for s in SPLITS
            )
        )
        add(f"{'term':>8} " + " ".join(f"{s:>8}" for s in SPLITS))
        for t in TERMS[: 1 + 2 * (layer + 1)]:
            add(
                f"{t:>8} "
                + " ".join(f"{a2[s]['terms'][t] / a2[s]['S_total']:>+8.3f}" for s in SPLITS)
            )
        for run in runs:
            add(f"  {run}: decomposed layers, components vs Delta (fraction of S_total)")
            for lyr, e in a2["fit"]["runs"][run]["layers"].items():
                cells = " ".join(
                    f"{s}: comp {a2[s]['runs'][run]['layers'][lyr]['components'] / a2[s]['S_total']:+.3f} delta {a2[s]['runs'][run]['layers'][lyr]['delta'] / a2[s]['S_total']:+.3f}"
                    for s in SPLITS
                )
                add(f"    L{lyr}: {cells}; top fit {e['top'][:3]}")
        add("")
    add(
        "== A2 per dataset, own direction (tune split for fit datasets): MLP 15-19 vs attention 15-19,"
    )
    add("   and per run components vs Delta in 15-19 (fractions of the dataset's S_total)")
    for d, r in results["A2_own"].items():
        mlp = sum(r["terms"][f"mlp_{lyr}"] for lyr in range(15, 20)) / r["S_total"]
        attn = sum(r["terms"][f"attn_{lyr}"] for lyr in range(15, 20)) / r["S_total"]
        early = sum(v for t, v in r["terms"].items() if term_layer(t) < 15) / r["S_total"]
        comps = " ".join(
            f"{run}: comp {sum(e['components'] for e in r['runs'][run]['layers'].values()) / r['S_total']:+.2f} delta {sum(e['delta'] for e in r['runs'][run]['layers'].values()) / r['S_total']:+.2f}"
            for run in runs
        )
        add(
            f"  {d}: S {r['S_total']:.2f} ({r['S_total'] / r['sd']:.2f} sd); layers<15 {early:+.2f}; attn15-19 {attn:+.2f}; mlp15-19 {mlp:+.2f}; {comps}"
        )
    add("")
    add(
        "== A3 / G1 probes (logistic regression, C picked on tune): tune acc, test acc (pooled), test mean over datasets"
    )
    for name, r in results["A3"].items():
        lr = r["logreg"]
        add(
            f"  {name:>22}: C {lr['selected_C']:>6}  tune {lr['tune']['acc_pooled']:.3f}  test {lr['test']['acc_pooled']:.3f}  test-mean {lr['test']['acc_mean_over_datasets']:.3f}"
        )
    add("")
    add("== A4 within-dataset 5-fold CV accuracy on test-split datasets")
    a4 = results["A4"]
    test_ds = sorted({d for r in a4.values() for d in r})
    add(f"{'feature set':>22} " + " ".join(f"{d[:11]:>11}" for d in test_ds))
    for name, r in a4.items():
        add(
            f"{name:>22} " + " ".join(f"{r[d]:>11.3f}" if d in r else f"{'-':>11}" for d in test_ds)
        )
    add("")
    add("== A5 writers (global mm_L19): dataset comparison of per-component write separations")
    for run, r in results["A5"].items():
        add(
            f"  {run}: between-dataset corr mean {r['between_dataset_corr_mean']:.3f}, split-half corr mean {r['split_half_corr_mean']:.3f}, top-{TOP_WRITERS} Jaccard mean {r['top_writer_jaccard_mean']:.3f}"
        )
        for wr in r["top_writers_fit"][:5]:
            acts = wr["activity"]
            add(
                f"    {wr['component']}: S_c fit {fmt(wr['S_c_fit'])}; activity min/median/max over datasets {min(acts.values()):.2f}/{np.median(list(acts.values())):.2f}/{max(acts.values()):.2f}"
            )
    add("")
    add("== A6 inverted test datasets (CI probe C = 1.0 more than 2 SE below chance)")
    for run, r in results["A6"].items():
        for d, e in r.items():
            add(
                f"  {run} {d}: acc {e['acc_C1']:.3f}, mean logit separation {e['mean_logit_separation']:+.2f}, corr(delta fit, delta {d}) {e['corr_delta_fit_vs_dataset']:+.2f}"
            )
            add(f"    variants: {e['probe_variants']}")
            add(
                f"    most negative contributors: {[(c['component'], round(c['contribution'], 3)) for c in e['most_negative_contributors'][:5]]}"
            )
    add("")
    add(
        "== A7 total (T, first order; pos0/last = its parts at position 0 and at the last token) vs"
    )
    add("   direct (D, last token) effect on the separation along mm_L23")
    for run, r in results["A7"].items():
        for split in A7_SPLITS:
            e = r[split]
            add(
                f"  {run} {split}: n {e['n']}, p corr {e['p_corr_with_original']:.4f}, S_total {e['S_total']:.2f}, sum T components {e['sum_T_components']:.2f}, T delta per layer {[round(v, 2) for v in e['T_delta_per_layer']]}"
            )
            add(
                "    top by fit T: "
                + ", ".join(
                    f"{c['component']} T {c['T']:+.2f} (pos0 {c['T_first_position']:+.2f}, "
                    f"last {c['T_last_position']:+.2f}) D {c['D']:+.2f}"
                    for c in e["top_by_fit_T"][:8]
                )
            )
            add(
                "    linearity (predicted vs actual change on ablation): "
                + ", ".join(
                    f"{c['component']} {c['predicted']:+.2f}/{c['actual']:+.2f}"
                    for c in e["linearity_check"]
                )
            )
    add("")
    add(
        "== G gate: paired KL(true || false) at the last token vs KL between two different true items"
    )
    for split, e in results["G"].items():
        add(f"  {split}: vocab coverage {e['vocab_coverage_mean']:.3f}")
        for d, v in e["per_dataset"].items():
            add(
                f"    {d}: {v['n_pairs']} pairs, paired KL mean {v['paired_kl_mean']:.4f} (median {v['paired_kl_median']:.4f}), baseline {v['baseline_kl_mean']}"
            )
    return "\n".join(lines) + "\n"


def to_jsonable(x: Any) -> Any:
    match x:
        case dict():
            return {str(k): to_jsonable(v) for k, v in x.items()}
        case np.ndarray():
            raise AssertionError("arrays belong in a5_matrices.npz, not results.json")
        case list() | tuple():
            return [to_jsonable(v) for v in x]
        case np.floating() | np.integer():
            return x.item()
        case _:
            return x


def analyze(out_dir: str, n_workers: int) -> None:
    """CPU step: results.json, report.txt, a5_matrices.npz in out_dir."""
    out = Path(out_dir).expanduser()
    meta = json.loads((out / "meta.json").read_text())
    store = Store(out)
    runs = list(meta["runs"])
    names = component_names(meta)
    results: dict[str, Any] = {}

    dirs, dir_info = truth_directions(store)
    results["directions"] = dir_info
    logger.info(f"{len(dirs)} truth directions")
    results["A1"] = a1_alignment(store, runs, dirs, null_directions(store), names)

    results["A2"] = {
        dname: {
            split: a2_attribution(
                store, runs, dirs[dname], layer, split, None, names, meta, with_top=True
            )
            for split in SPLITS
        }
        for dname, layer in (("mm_L19", 19), ("mm_L23", 23), ("lr_L19", 19))
    }
    results["A2_own"] = {
        d.split("|")[1]: a2_attribution(
            store, runs, w, 19, "tune", d.split("|")[1], names, meta, with_top=False
        )
        for d, w in dirs.items()
        if "|" in d and has_both_classes(store, "tune", d.split("|")[1])
    }
    logger.info("A1, A2 done")

    sets = ["resid_L19", "lastlp"] + [
        f"{r}:{kind}" for r in runs for kind in ("ci", "acts", "actsci")
    ]
    n_threads = max(1, (os.cpu_count() or 1) // n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        a3 = dict(
            f.result() for f in [pool.submit(_probe_worker, str(out), s, n_threads) for s in sets]
        )
        results["A3"] = a3
        futures = [
            pool.submit(_cv_worker, str(out), s, float(a3[s]["logreg"]["selected_C"]), n_threads)
            for s in sets
        ]
        results["A4"] = dict(f.result() for f in futures)
    logger.info("A3, A4 done")

    results["A5"], matrices = a5_writers(store, runs, dirs, names)
    matrices["components"] = np.array(names)
    np.savez(out / "a5_matrices.npz", **matrices)  # pyright: ignore[reportArgumentType]
    results["A6"] = a6_inversions(store, runs, names, out / "samples.jsonl")
    results["A7"] = a7_effects(store, runs, names)
    results["G"] = g_gate(store)
    (out / "results.json").write_text(
        json.dumps(to_jsonable({"meta": meta, "results": results}), indent=1)
    )
    report = write_report(results, meta)
    (out / "report.txt").write_text(report)
    logger.info(f"wrote {out / 'report.txt'}")


if __name__ == "__main__":
    fire.Fire({"extract": extract, "analyze": analyze})
