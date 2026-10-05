"""Truth probes on the CI values of finished tiu decompositions, with baselines.

Reads: SPD runs on WandB (final_config.yaml + latest checkpoint each) and the tiu prepared
datasets those runs were trained on. Writes into `out_dir`: `features.npz` (all probe inputs),
`results.json` (test accuracies) and `examples.txt` (tokenisation check).

Per statement two readouts: the value at the last real token, and the mean over real tokens
(right-padding excluded). Each readout of each feature set is probed with logistic regression and
a mass-mean probe, fit on the train split and scored on the held-out test split.

Feature sets:
- CI of each run: the 480 `lower_leaky` CI values (the variant the recon losses mask with).
- Baseline "resid_L19": residual stream after layer 19 (the last decomposed layer), raw.
- Baseline "logprob": mean next-token log-probability of the statement under the frozen model
  (one scalar; positions 1..n-1, since the first token has no prefix). Logistic regression only.
The untrained baseline is just another run id (a 0-step run of config_truth_untrained.yaml).
Probe data = the tiu datasets the FIRST run id was trained on; later runs may have a different
target (e.g. the code-lines control, config_code_control.yaml) and are probed on that same tiu data.

All CI functions read the unmasked model's `down_proj` inputs, so one forward pass serves every
run: the target model is loaded once, in the runs' `pretrained_model_dtype`, under bf16 autocast
as in training, and each run contributes only its CI function.

Usage: python probe_ci.py <out_dir> <tiu run_id> [<run_id> ...]
"""

import copy
import json
from pathlib import Path

import fire
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from spd.experiments.lm.prepared_datasets import load_prepared_split, resolve_dataset_dir
from spd.log import logger
from spd.models.component_model import (
    ComponentModel,
    OutputWithCache,
    SPDRunInfo,
    handle_deprecated_state_dict_keys_,
)
from spd.utils.general_utils import bf16_autocast

WANDB_PREFIX = "wandb:bitt-j-personal/spd/"
RESID_LAYER = 19
BATCH_SIZE = 16


def load_split_with_labels(
    dataset_dirs: list[str], split: str
) -> tuple[list[str], np.ndarray, list[str]]:
    """Texts, truth labels and dataset names of one split, in the training loader's order.

    `load_prepared_split` verifies the manifest hash; labels are read from the same jsonl and the
    texts are asserted identical, so labels cannot drift from the hashed data.
    """
    texts, labels, names = [], [], []
    for dataset_dir in dataset_dirs:
        hashed_texts = load_prepared_split(dataset_dir, split)  # pyright: ignore[reportArgumentType]
        path = resolve_dataset_dir(dataset_dir) / f"{split}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["text"] for r in records] == hashed_texts, (
            f"{path}: texts differ from manifest load"
        )
        name = Path(dataset_dir).parent.name
        expected = name.endswith("_true")
        assert name.endswith(("_true", "_false")), name
        assert all(r["label"] is expected for r in records), (
            f"{path}: label disagrees with dir name"
        )
        texts += hashed_texts
        labels += [expected] * len(records)
        names += [name] * len(records)
    return texts, np.array(labels), names


def tokenize(
    texts: list[str], tokenizer_name: str, max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded input ids (n, max_seq_len) and lengths, exactly as the training loader builds
    them (`load_prompts_dataset`: no BOS, pad with pad_token_id, no attention mask)."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    assert tokenizer.pad_token_id is not None
    encoded = tokenizer(texts)["input_ids"]
    lengths = torch.tensor([len(ids) for ids in encoded])
    assert int(lengths.max()) <= max_seq_len and int(lengths.min()) >= 2
    ids = torch.tensor(
        [ids + [tokenizer.pad_token_id] * (max_seq_len - len(ids)) for ids in encoded]
    )
    return ids, lengths


def load_ci_fns(
    run_infos: dict[str, SPDRunInfo], comp_model: ComponentModel
) -> dict[str, torch.nn.Module]:
    """One CI function per run, each a copy of comp_model.ci_fn with that run's weights."""
    ci_fns = {}
    for run_id, run_info in run_infos.items():
        weights = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=True)
        handle_deprecated_state_dict_keys_(weights)
        ci_weights = {
            k.removeprefix("ci_fn."): v for k, v in weights.items() if k.startswith("ci_fn.")
        }
        other = [k for k in weights if not k.startswith(("ci_fn.", "_components."))]
        assert not other, f"{run_id}: unexpected checkpoint keys {other}"
        ci_fn = copy.deepcopy(comp_model.ci_fn)
        ci_fn.load_state_dict(ci_weights, strict=True)
        ci_fn.eval()
        ci_fns[run_id] = ci_fn
        logger.info(f"{run_id}: CI function from {run_info.checkpoint_path.name}")
    return ci_fns


def readouts(x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(batch, pos, d) -> last-real-token values (batch, d) and mean over real tokens (batch, d)."""
    batch, pos, _ = x.shape
    real = (
        (torch.arange(pos, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))
        .unsqueeze(-1)
        .to(x.dtype)
    )
    last = x[torch.arange(batch, device=x.device), lengths - 1]
    mean = (x * real).sum(1) / lengths.unsqueeze(1).to(x.dtype)
    return last, mean


@torch.no_grad()
def extract_features(
    comp_model: ComponentModel,
    ci_fns: dict[str, torch.nn.Module],
    ids: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Feature arrays keyed '<set>/<readout>' (CI sets keyed by run id), plus 'logprob'."""
    resid: list[torch.Tensor] = []

    def save_resid(
        _module: torch.nn.Module, _inputs: object, out: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> None:
        resid.append(out[0] if isinstance(out, tuple) else out)

    layer = comp_model.target_model.get_submodule(f"model.layers.{RESID_LAYER}")
    hook = layer.register_forward_hook(save_resid)
    module_order = sorted(comp_model.target_module_paths)
    feats: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(ids), BATCH_SIZE):
        batch_ids, batch_len = ids[start : start + BATCH_SIZE], lengths[start : start + BATCH_SIZE]
        # bf16 autocast on fp32 weights, as in training (autocast_bf16: true)
        with bf16_autocast(enabled=True):
            out = comp_model(batch_ids, cache_type="input")
        assert isinstance(out, OutputWithCache)
        logits, cache = out.output.float(), out.cache
        assert set(cache) == set(module_order)
        sets = {f"resid_L{RESID_LAYER}": resid.pop().float()}
        assert not resid
        for run_id, ci_fn in ci_fns.items():
            with bf16_autocast(enabled=True):
                pre = ci_fn(cache)
            sets[run_id] = torch.cat(
                [comp_model.lower_leaky_fn(pre[m].float()) for m in module_order], -1
            )
        for name, x in sets.items():
            last, mean = readouts(x, batch_len)
            feats.setdefault(f"{name}/last", []).append(last)
            feats.setdefault(f"{name}/mean", []).append(mean)
        logp = (
            torch.log_softmax(logits[:, :-1], -1)
            .gather(-1, batch_ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )
        target_real = torch.arange(1, ids.shape[1]).unsqueeze(0) < batch_len.unsqueeze(1)
        feats.setdefault("logprob", []).append(
            ((logp * target_real).sum(1) / (batch_len - 1)).unsqueeze(1)
        )
        if start % (BATCH_SIZE * 20) == 0:
            logger.info(f"forward {start}/{len(ids)}")
    hook.remove()
    return {k: torch.cat(v).numpy() for k, v in feats.items()}


def logreg_acc(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray
) -> tuple[float, np.ndarray]:
    # Default L2 strength (C=1) on standardised features; not tuned.
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    probe.fit(x_tr, y_tr)
    pred = probe.predict(x_te)
    return float((pred == y_te).mean()), pred


def mass_mean_acc(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray
) -> tuple[float, np.ndarray]:
    """Direction = mean(true) - mean(false) on train; threshold = midpoint of the projected class means."""
    direction = x_tr[y_tr].mean(0) - x_tr[~y_tr].mean(0)
    proj_tr = x_tr @ direction
    threshold = (proj_tr[y_tr].mean() + proj_tr[~y_tr].mean()) / 2
    pred = x_te @ direction > threshold
    return float((pred == y_te).mean()), pred


def main(out_dir: str, *run_ids: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert run_ids, "give at least one run id"
    run_infos = {r: SPDRunInfo.from_path(WANDB_PREFIX + r) for r in run_ids}

    # All runs must share everything that determines the CI function's inputs and architecture.
    ref = run_infos[run_ids[0]].config
    for run_id, info in run_infos.items():
        c = info.config
        assert c.pretrained_model_name == ref.pretrained_model_name, run_id
        # One forward pass serves all runs, so they must share the frozen model's dtype too.
        assert c.pretrained_model_dtype == ref.pretrained_model_dtype, run_id
        assert c.module_info == ref.module_info and c.ci_config == ref.ci_config, run_id
        assert c.sigmoid_type == ref.sigmoid_type and c.sampling == "continuous", run_id
        assert c.task_config.max_seq_len == ref.task_config.max_seq_len, run_id  # pyright: ignore[reportAttributeAccessIssue]
        assert c.autocast_bf16, run_id
    dataset_dirs = ref.task_config.prepared_datasets  # pyright: ignore[reportAttributeAccessIssue]
    assert dataset_dirs is not None
    max_seq_len = ref.task_config.max_seq_len  # pyright: ignore[reportAttributeAccessIssue]
    assert ref.tokenizer_name is not None

    splits = {}
    for split in ("train", "test"):
        texts, labels, names = load_split_with_labels(dataset_dirs, split)
        ids, lengths = tokenize(texts, ref.tokenizer_name, max_seq_len)
        splits[split] = (texts, labels, names, ids, lengths)
        logger.info(f"{split}: {len(texts)} statements, {labels.mean():.3f} true")

    tokenizer = AutoTokenizer.from_pretrained(ref.tokenizer_name)
    with open(out / "examples.txt", "w") as f:
        texts, labels, names, ids, lengths = splits["test"]
        for i in range(0, len(texts), len(texts) // 25):
            toks = [tokenizer.decode([t]) for t in ids[i].tolist()]
            f.write(
                f"{names[i]} label={labels[i]} len={int(lengths[i])} last={toks[int(lengths[i]) - 1]!r}\n"
            )
            f.write(f"  text: {texts[i]!r}\n  tokens: {toks}\n")

    comp_model = ComponentModel.from_run_info(run_infos[run_ids[0]])
    comp_model.eval()
    ci_fns = load_ci_fns(run_infos, comp_model)

    feats = {split: extract_features(comp_model, ci_fns, s[3], s[4]) for split, s in splits.items()}
    arrays = {
        **{f"{split}:{k}": v for split, f in feats.items() for k, v in f.items()},
        **{f"{split}:labels": s[1] for split, s in splits.items()},
        **{f"{split}:dataset": np.array(s[2]) for split, s in splits.items()},
    }
    np.savez(out / "features.npz", **arrays)  # pyright: ignore[reportArgumentType]

    y_tr, y_te = splits["train"][1], splits["test"][1]
    test_names = np.array(splits["test"][2])
    results: dict[str, dict[str, float | dict[str, float]]] = {}
    for key in feats["test"]:
        x_tr, x_te = feats["train"][key], feats["test"][key]
        probes = (
            {"logreg": logreg_acc}
            if key == "logprob"
            else {"logreg": logreg_acc, "mass_mean": mass_mean_acc}
        )
        for probe_name, probe_fn in probes.items():
            acc, pred = probe_fn(x_tr, y_tr, x_te, y_te)
            per_dataset = {
                n: float((pred[test_names == n] == y_te[test_names == n]).mean())
                for n in sorted(set(test_names))
            }
            results[f"{key}/{probe_name}"] = {"test_acc": acc, "per_dataset": per_dataset}
            logger.info(f"{key}/{probe_name}: test acc {acc:.3f}")
    meta = {
        r: {
            "checkpoint": i.checkpoint_path.name,
            "label": i.config.label,
            "target": i.config.task_config.prepared_datasets,  # pyright: ignore[reportAttributeAccessIssue]
        }
        for r, i in run_infos.items()
    }
    (out / "results.json").write_text(json.dumps({"runs": meta, "results": results}, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
