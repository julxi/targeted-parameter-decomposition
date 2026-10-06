"""Cross-task truth probes on the UTH datasets: residual stream, CI values of SPD runs, log-prob.

Protocol (Liu et al. 2024, "cross-task"): probes are fit on the train-task datasets' `train` splits,
tuned on their held-out `test` splits, and scored on the test-task datasets (the held-out task
categories), which neither the probes nor any decomposition ever trained on. Dataset roles come
from the `role` field of each `data/uth/<dataset>/<version>/manifest.yaml` (see build_uth.py).

Two steps. `run` extracts features (GPU) and then fits the probes; `probe` fits the probes again
from a saved `features.npz` (CPU only), e.g. after changing the probe settings. Probes are always
fit on the float16 values stored in `features.npz`, so `probe` reproduces `run`'s results exactly.

Reads: data/uth/*/<version>, SPD runs on WandB (final_config.yaml + latest checkpoint each).
Writes into `out_dir`: `features.npz` (all probe inputs and targets, float16 features),
`meta.json`, `results.json` (accuracies), `examples.txt` (tokenisation check).

Per sample two readouts: the value at the last real token (the paper's readout) and the mean over
real tokens. Feature sets:
- `resid_L<i>`: residual stream after decoder layer i, for each i in RESID_LAYERS (coarse sweep);
- `<run_id>`: the 480 `lower_leaky` CI values of that run (e.g. the untrained baseline s-7fad0c14);
- `logprob`: mean next-token log-probability of the sample (one scalar, positions 1..n-1).

Probes per feature set: mass-mean (no hyperparameter) and logistic regression for each C in
LOGREG_CS; the reported logistic-regression score uses the C with the best accuracy on the
train-task held-out split, never chosen on test tasks. Accuracies are given pooled, as the
unweighted mean over datasets (the paper's average), per dataset and per category.

Sequences have very different lengths (test tasks up to ~3,000 tokens, uncapped), so forward passes
run on length-sorted batches of at most BATCH_TOKENS padded tokens. Right padding with no attention
mask, as in training; with causal attention padding cannot affect real positions. One forward pass
of the frozen model serves all runs: every CI function reads the unmasked `down_proj` inputs.

Feature sets are probed in parallel worker processes (`n_workers`), each with an equal share of
the CPU threads; one sklearn fit uses few cores, and a sequential loop over 15 feature sets took
~4.5 min per 3,584-dim set on a 26-10-05 VM.

`sparsity` compares CI runs at matched sparsity (last-token readout), as the stage 1 success
criterion requires: a trained decomposition has ~10 active components per sample, an untrained one
hundreds, so their plain probe scores are not comparable. Per sample only the k largest CI values
are kept (the rest set to 0), for each k in TOPK_VALUES, and probed like any feature set. Also
reported: the number of active components (CI > CI_ACTIVE) per sample, and a probe on the binary
on/off pattern. Same method as the tiu analysis of 26-10-05 (~/spd_out/26-10-05_no_truth_baseline/
followup.py), but cross-task and with C chosen on the tune split.

Usage:
    python probe_uth.py run <out_dir> <data_version> <n_workers> <run_id> [<run_id> ...]
    python probe_uth.py probe <out_dir> <n_workers>
    python probe_uth.py sparsity <out_dir> <n_workers> <run_id> [<run_id> ...]
    e.g. python probe_uth.py run ~/spd_out/26-10-05_uth_experiments/stage0 v1 15 s-7fad0c14
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
import yaml
from jaxtyping import Float, Int
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import Tensor
from transformers import AutoTokenizer

from spd.experiments.lm.honesty_targeted_decomposition.probe_ci import (
    WANDB_PREFIX,
    load_ci_fns,
    mass_mean_acc,
    readouts,
)
from spd.experiments.lm.prepared_datasets import Split, load_prepared_split
from spd.log import logger
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.settings import REPO_ROOT
from spd.utils.general_utils import bf16_autocast

RESID_LAYERS = (7, 11, 15, 19, 23, 27)
# 2,048 padded tokens per batch and log-softmax in chunks of LOGPROB_CHUNK positions keep peak memory
# within a 24 GB GPU (RTX 4090): the bf16 model takes ~15 GB, full fp32 logits of 4,096 positions
# would take 2.5 GB per copy (vocabulary ~152k).
BATCH_TOKENS = 2048
LOGPROB_CHUNK = 512
# Extended down to 1e-4 after the first stage 0 run (26-10-05) picked C = 0.01, then the grid's
# strongest regularisation, for every residual layer, with accuracy still rising towards it.
LOGREG_CS = (1e-4, 1e-3, 1e-2, 0.1, 1.0)
TOPK_VALUES = (1, 2, 5, 10, 20, 50)
# A component counts as active above this CI (the runs' ci_alive_threshold, and the tiu analysis's).
CI_ACTIVE = 0.01
# fit: train-task train splits; tune: train-task held-out splits; test: test-task datasets
SPLITS = ("fit", "tune", "test")


@dataclass
class Targets:
    labels: np.ndarray  # bool (n,)
    dataset: np.ndarray  # str (n,)
    category: np.ndarray  # str (n,)


@dataclass
class ProbeData:
    texts: list[str]
    targets: Targets


def load_uth(version: str) -> dict[str, ProbeData]:
    """The three probe splits, assembled from the manifests' roles."""
    manifests = sorted((REPO_ROOT / "data" / "uth").glob(f"*/{version}/manifest.yaml"))
    assert manifests, f"no data/uth/*/{version}"
    parts: dict[str, list[tuple[list[str], list[dict[str, Any]]]]] = {s: [] for s in SPLITS}
    for path in manifests:
        manifest = yaml.safe_load(path.read_text())
        dataset_dir = str(path.parent.relative_to(REPO_ROOT))
        wanted: dict[str, Split]
        match manifest["role"]:
            case "train_task":
                wanted = {"fit": "train", "tune": "test"}
            case "test_task":
                wanted = {"test": "test"}
            case role:
                raise AssertionError(f"{path}: unknown role {role!r}")
        for probe_split, file_split in wanted.items():
            texts = load_prepared_split(dataset_dir, file_split)  # verifies the manifest hash
            records = [
                json.loads(line)
                for line in (path.parent / f"{file_split}.jsonl").read_text().splitlines()
            ]
            assert [r["text"] for r in records] == texts, f"{dataset_dir}: texts differ"
            parts[probe_split].append((texts, records))
    data = {}
    for split, chunks in parts.items():
        records = [r for _, recs in chunks for r in recs]
        data[split] = ProbeData(
            texts=[t for texts, _ in chunks for t in texts],
            targets=Targets(
                labels=np.array([r["label"] for r in records], dtype=bool),
                dataset=np.array([r["dataset"] for r in records]),
                category=np.array([r["category"] for r in records]),
            ),
        )
        assert data[split].texts, f"empty probe split {split}"
    fit_cats = set(data["fit"].targets.category)
    test_cats = set(data["test"].targets.category)
    assert not fit_cats & test_cats, f"categories in both fit and test: {fit_cats & test_cats}"
    return data


def length_batches(lengths: list[int], max_tokens: int) -> list[list[int]]:
    """Index batches over length-sorted samples, each padded size (n * longest) <= max_tokens
    (a single sample longer than max_tokens gets a batch of its own)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    for i in order:
        # sorted ascending, so lengths[i] is the longest in the batch if appended
        if current and (len(current) + 1) * lengths[i] > max_tokens:
            batches.append(current)
            current = []
        current.append(i)
    batches.append(current)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    return batches


@torch.no_grad()
def extract_features(
    comp_model: ComponentModel,
    ci_fns: dict[str, torch.nn.Module],
    token_ids: list[list[int]],
    pad_token_id: int,
) -> dict[str, np.ndarray]:
    """Feature arrays keyed '<set>/<readout>' plus 'logprob', rows in the order of token_ids."""
    device = next(comp_model.parameters()).device
    lengths = [len(ids) for ids in token_ids]
    assert min(lengths) >= 2, "logprob needs at least two tokens"
    resid: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook(_module: torch.nn.Module, _inputs: object, out: object) -> None:
            x = out[0] if isinstance(out, tuple) else out
            assert isinstance(x, torch.Tensor), type(x)
            resid[layer] = x

        return hook

    hooks = [
        comp_model.target_model.get_submodule(f"model.layers.{i}").register_forward_hook(
            make_hook(i)
        )
        for i in RESID_LAYERS
    ]
    module_order = sorted(comp_model.target_module_paths)
    feats: dict[str, np.ndarray] = {}

    def store(key: str, rows: list[int], values: torch.Tensor) -> None:
        if key not in feats:
            feats[key] = np.zeros((len(token_ids), values.shape[1]), dtype=np.float32)
        feats[key][rows] = values.float().cpu().numpy()

    batches = length_batches(lengths, BATCH_TOKENS)
    for n_done, rows in enumerate(batches):
        batch_len = torch.tensor([lengths[i] for i in rows])
        width = int(batch_len.max())
        ids = torch.tensor(
            [token_ids[i] + [pad_token_id] * (width - lengths[i]) for i in rows], device=device
        )
        with bf16_autocast(enabled=True):  # as in training (autocast_bf16: true)
            out = comp_model(ids, cache_type="input")
        assert isinstance(out, OutputWithCache)
        assert set(resid) == set(RESID_LAYERS) and set(out.cache) == set(module_order)
        sets = {f"resid_L{i}": resid.pop(i).float() for i in RESID_LAYERS}
        for run_id, ci_fn in ci_fns.items():
            with bf16_autocast(enabled=True):
                pre = ci_fn(out.cache)
            sets[run_id] = torch.cat(
                [comp_model.lower_leaky_fn(pre[m].float()) for m in module_order], -1
            )
        lengths_dev = batch_len.to(device)
        for name, x in sets.items():
            assert x.shape[:2] == ids.shape, (name, x.shape, ids.shape)
            last, mean = readouts(x, lengths_dev)
            store(f"{name}/last", rows, last)
            store(f"{name}/mean", rows, mean)
        logp = next_token_logprobs(out.output, ids)
        real = torch.arange(1, width, device=device).unsqueeze(0) < lengths_dev.unsqueeze(1)
        store("logprob", rows, ((logp * real).sum(1) / (lengths_dev - 1)).unsqueeze(1))
        if n_done % 50 == 0:
            logger.info(f"batch {n_done}/{len(batches)} (width {width}, {len(rows)} samples)")
    for h in hooks:
        h.remove()
    return feats


def next_token_logprobs(
    logits: Float[Tensor, "batch pos vocab"], ids: Int[Tensor, "batch pos"]
) -> Float[Tensor, "batch pos_minus_1"]:
    """log p(ids[:, t+1] | prefix) for t = 0..pos-2, computed in chunks of LOGPROB_CHUNK positions.
    Logits go to fp32 before log_softmax: bf16 log-probs are too coarse (see probe_ci.py)."""
    batch, pos, vocab = logits.shape
    flat_logits = logits[:, :-1].reshape(-1, vocab)
    flat_targets = ids[:, 1:].reshape(-1, 1)
    chunks = [
        torch.log_softmax(flat_logits[i : i + LOGPROB_CHUNK].float(), -1)
        .gather(-1, flat_targets[i : i + LOGPROB_CHUNK])
        .squeeze(-1)
        for i in range(0, len(flat_targets), LOGPROB_CHUNK)
    ]
    return torch.cat(chunks).view(batch, pos - 1)


def scores(pred: np.ndarray, data: Targets) -> dict[str, Any]:
    correct = pred == data.labels
    per_dataset = {d: float(correct[data.dataset == d].mean()) for d in sorted(set(data.dataset))}
    per_category = {
        c: float(correct[data.category == c].mean()) for c in sorted(set(data.category))
    }
    return {
        "acc_pooled": float(correct.mean()),
        "acc_mean_over_datasets": float(np.mean(list(per_dataset.values()))),
        "per_dataset": per_dataset,
        "per_category": per_category,
    }


def probe_feature(feats: dict[str, np.ndarray], targets: dict[str, Targets]) -> dict[str, Any]:
    """Mass-mean and logistic-regression scores of one feature set ({split: (n, d) array}) on the
    tune and test splits."""
    x_fit, y_fit = feats["fit"], targets["fit"].labels
    result: dict[str, Any] = {}
    if x_fit.shape[1] > 1:
        mm = {}
        for split in ("tune", "test"):
            _, pred = mass_mean_acc(x_fit, y_fit, feats[split], targets[split].labels)
            mm[split] = scores(pred, targets[split])
        result["mass_mean"] = mm
    by_c = {}
    for c in LOGREG_CS:
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        probe.fit(x_fit, y_fit)
        by_c[str(c)] = {
            split: scores(probe.predict(feats[split]), targets[split]) for split in ("tune", "test")
        }
    best_c = max(by_c, key=lambda c: by_c[c]["tune"]["acc_pooled"])
    result["logreg"] = {"selected_C": best_c, **by_c[best_c], "by_C": by_c}
    return result


def load_targets(npz: Any) -> dict[str, Targets]:
    return {
        s: Targets(
            labels=npz[f"{s}:labels"], dataset=npz[f"{s}:dataset"], category=npz[f"{s}:category"]
        )
        for s in SPLITS
    }


def _probe_worker(features_path: str, key: str, n_threads: int) -> tuple[str, dict[str, Any]]:
    """One feature set, in its own process: reads only that set's arrays from the npz."""
    with threadpool_limits(limits=n_threads), np.load(features_path) as npz:
        feats = {s: npz[f"{s}:{key}"].astype(np.float32) for s in SPLITS}
        result = probe_feature(feats, load_targets(npz))
    lr = result["logreg"]
    logger.info(
        f"{key}: logreg (C={lr['selected_C']}) tune {lr['tune']['acc_pooled']:.3f}, "
        f"test mean-over-datasets {lr['test']['acc_mean_over_datasets']:.3f}"
    )
    return key, result


def probe(out_dir: str, n_workers: int) -> None:
    """Fit all probes from `out_dir/features.npz`; writes `out_dir/results.json`."""
    out = Path(out_dir).expanduser()
    features_path = out / "features.npz"
    with np.load(features_path) as npz:
        keys = [n.removeprefix("fit:") for n in npz.files if n.startswith("fit:")]
    keys = [k for k in keys if k not in ("labels", "dataset", "category")]
    assert keys, f"no features in {features_path}"
    n_threads = max(1, (os.cpu_count() or 1) // n_workers)
    logger.info(f"probing {len(keys)} feature sets, {n_workers} workers x {n_threads} threads")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_probe_worker, str(features_path), k, n_threads) for k in keys]
        results = dict(f.result() for f in futures)
    meta = json.loads((out / "meta.json").read_text())
    (out / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))


def topk(x: np.ndarray, k: int) -> np.ndarray:
    """Per row keep the k largest values, zero the rest."""
    out = np.zeros_like(x)
    idx = np.argsort(-x, axis=1)[:, :k]
    np.put_along_axis(out, idx, np.take_along_axis(x, idx, 1), 1)
    return out


def _sparsity_worker(
    features_path: str, run_id: str, variant: str, n_threads: int
) -> tuple[str, str, dict[str, Any]]:
    """One (run, variant) probe; variant is 'top<k>' or 'binary', on the last-token CI values."""
    with threadpool_limits(limits=n_threads), np.load(features_path) as npz:
        raw = {s: npz[f"{s}:{run_id}/last"].astype(np.float32) for s in SPLITS}
        if variant == "binary":
            feats = {s: (x > CI_ACTIVE).astype(np.float32) for s, x in raw.items()}
        else:
            k = int(variant.removeprefix("top"))
            feats = {s: topk(x, k) for s, x in raw.items()}
        result = probe_feature(feats, load_targets(npz))
    lr = result["logreg"]
    logger.info(
        f"{run_id} {variant}: logreg (C={lr['selected_C']}) tune {lr['tune']['acc_pooled']:.3f}, "
        f"test mean-over-datasets {lr['test']['acc_mean_over_datasets']:.3f}"
    )
    return run_id, variant, result


def sparsity(out_dir: str, n_workers: int, *run_ids: str) -> None:
    """Matched-sparsity comparison of CI runs from `out_dir/features.npz`; writes sparsity.json."""
    out = Path(out_dir).expanduser()
    features_path = out / "features.npz"
    assert run_ids, "give the CI run ids to compare (e.g. trained and untrained)"
    results: dict[str, dict[str, Any]] = {}
    with np.load(features_path) as npz:
        for run_id in run_ids:
            n_active = {}
            for split in SPLITS:
                n = (npz[f"{split}:{run_id}/last"] > CI_ACTIVE).sum(1)
                n_active[split] = {
                    "mean": float(n.mean()),
                    "p10": float(np.percentile(n, 10)),
                    "p50": float(np.percentile(n, 50)),
                    "p90": float(np.percentile(n, 90)),
                    "frac_zero": float((n == 0).mean()),
                }
            results[run_id] = {"n_active_last": n_active, "probes": {}}
            logger.info(f"{run_id}: active components per sample (last token) {n_active}")
    variants = ["binary"] + [f"top{k}" for k in TOPK_VALUES]
    jobs = [(r, v) for r in run_ids for v in variants]
    n_threads = max(1, (os.cpu_count() or 1) // n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_sparsity_worker, str(features_path), r, v, n_threads) for r, v in jobs
        ]
        for f in futures:
            run_id, variant, result = f.result()
            results[run_id]["probes"][variant] = result
    meta = {"topk_values": list(TOPK_VALUES), "ci_active": CI_ACTIVE, "readout": "last"}
    (out / "sparsity.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))


def run(out_dir: str, data_version: str, n_workers: int, *run_ids: str) -> None:
    """Extract features of all samples (GPU if available), then fit the probes."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    assert run_ids, "give at least one run id (e.g. the untrained baseline)"
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

    data = load_uth(data_version)
    tokenizer = AutoTokenizer.from_pretrained(ref.tokenizer_name)
    assert tokenizer.pad_token_id is not None
    token_ids = {s: tokenizer(d.texts)["input_ids"] for s, d in data.items()}
    for split, d in data.items():
        lens = [len(t) for t in token_ids[split]]
        logger.info(
            f"{split}: {len(d.texts)} samples, {d.targets.labels.mean():.3f} true, "
            f"{len(set(d.targets.dataset))} datasets, "
            f"tokens median {int(np.median(lens))} max {max(lens)}"
        )

    with open(out / "examples.txt", "w") as f:
        for split, d in data.items():
            for ds in sorted(set(d.targets.dataset)):
                i = int(np.flatnonzero(d.targets.dataset == ds)[0])
                ids = token_ids[split][i]
                f.write(f"[{split}] {ds} label={d.targets.labels[i]} len={len(ids)}\n")
                f.write(f"  text: {d.texts[i]!r}\n  last 8 tokens: ")
                f.write(f"{[tokenizer.decode([t]) for t in ids[-8:]]}\n")

    comp_model = ComponentModel.from_run_info(run_infos[run_ids[0]])
    comp_model.eval()
    if torch.cuda.is_available():
        comp_model.to("cuda")
    ci_fns = load_ci_fns(run_infos, comp_model)
    for ci_fn in ci_fns.values():
        ci_fn.to(next(comp_model.parameters()).device)

    feats = {
        s: extract_features(comp_model, ci_fns, token_ids[s], tokenizer.pad_token_id)
        for s in SPLITS
    }
    np.savez(
        out / "features.npz",
        **{f"{s}:{k}": v.astype(np.float16) for s, f in feats.items() for k, v in f.items()},
        **{
            f"{s}:{field}": getattr(d.targets, field)
            for s, d in data.items()
            for field in ("labels", "dataset", "category")
        },
    )
    meta = {
        "data_version": data_version,
        "runs": {
            r: {"checkpoint": i.checkpoint_path.name, "label": i.config.label}
            for r, i in run_infos.items()
        },
        "resid_layers": list(RESID_LAYERS),
        "logreg_Cs": list(LOGREG_CS),
        "n": {s: len(d.texts) for s, d in data.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    del feats
    probe(out_dir, n_workers)


if __name__ == "__main__":
    fire.Fire({"run": run, "probe": probe, "sparsity": sparsity})
