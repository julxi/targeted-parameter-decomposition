"""Analysis V of the bolder experiments: the model's true/false verdict after a stimulus when the
statement's last token runs on parts of a decomposition.

Topic: convos/julian/26-10-10_bolder_experiments_LOG.md. Each tiu test statement gets a stimulus
appended (tokenized separately, so the statement's tokens are unchanged). At the statement's last
token ('.' or "'."; the decomposed position of the runs config_truth_{last_token,residual_lens,
stimulus_lens}.yaml), and with --decompose_stimulus also at every stimulus token (as in
config_truth_stimulus_span.yaml), the decomposed `down_proj` outputs are replaced; every other
position runs on the original weights. States:
  original    no replacement
  components  components with rounded CI (CI > ROUNDING_THRESHOLD, per position) only, Delta
              off: what the decomposition itself accounts for
  delta_only  all component masks 0, Delta on
  all_off     component masks 0, Delta off (the MLP outputs zero there)
  full        all component masks 1, Delta on: the original weights computed through the
              decomposition. Equal to `original` up to bf16 rounding (on CPU, tiu, arm A: verdict
              scores within 0.25 of the original, separation ~11), which is why single-component
              ablations are compared with this state, not with `original`.
  minus_<c>   `full` with component c set to 0, for every component active (rounded CI, at any
              decomposed position) on at least MIN_ACTIVE_SHARE of the statements. Global component index c = 96 x (position
              of the module in sorted module order, i.e. layers 15..19) + index within the
              module, as in truth_writers.py.
Verdict score = log P(true-like) - log P(false-like) at the last stimulus token, with the token sets
below (single-token spellings of true/correct/yes/right and false/incorrect/wrong/no, with and
without a leading space; chosen in the R1 stimulus screen, where " True"/" False" carried most of
the mass after " Is this true or false? Answer:"). Per state: accuracy (score > 0 means "true"),
AUC, separation (mean score of true minus false statements), share predicted true, mean
KL(original || state) at the stimulus end; per family accuracy for the main states.

The decomposed positions are set by --decompose_stimulus, not by the run's config, so runs
decomposed elsewhere (e.g. the untrained baseline, configured for all tokens) are evaluated at the
same positions as the others.

Usage:
    python verdict_accuracy.py <out_dir> "<stimulus>" <run> [<run> ...] [--cap N] [--batch_size B]
        [--max_singles K] [--decompose_stimulus]
    <run>: a WandB run id of bitt-j-personal/spd (e.g. s-7fad0c14) or a local checkpoint path.
    --cap N keeps the first N test statements per dataset and --max_singles K ablates only the K
    most often active components (both for smoke tests).
Writes <out_dir>/verdict_<run name>[_span].json (_span with --decompose_stimulus) and prints a
table per run.
"""

import json
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor
from transformers import AutoTokenizer

from spd.experiments.lm.honesty_targeted_decomposition.probe_ci import (
    WANDB_PREFIX,
    load_split_with_labels,
)
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import make_mask_infos, route_only_selected_positions
from spd.utils.general_utils import bf16_autocast

ROUNDING_THRESHOLD = 0.01  # as TargetReconLoss's rounding_threshold in the run configs
MIN_ACTIVE_SHARE = 0.01
TRUE_WORDS = ["true", "True", "TRUE", "correct", "Correct", "yes", "Yes", "YES", "right", "Right"]
FALSE_WORDS = [
    "false", "False", "FALSE", "incorrect", "Incorrect", "wrong", "Wrong", "no", "No", "NO",
]  # fmt: skip
MAIN_STATES = ["original", "components", "delta_only", "all_off", "full"]


def single_token_ids(tokenizer: Any, words: list[str]) -> list[int]:
    """Ids of the words' single-token spellings, with and without a leading space."""
    ids = []
    for w in words:
        for variant in (w, " " + w):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
    return sorted(set(ids))


def run_path(run: str) -> str:
    return run if Path(run).exists() else WANDB_PREFIX + run


class Batches:
    """Right-padded statement + stimulus batches, with the decomposed positions (last statement
    token, plus every stimulus token if `decompose_stimulus`) and the stimulus end of every row."""

    def __init__(
        self,
        tokenizer: Any,
        texts: list[str],
        stimulus: str,
        decompose_stimulus: bool,
        batch_size: int,
        device: str,
    ) -> None:
        stim_ids = tokenizer.encode(stimulus, add_special_tokens=False)
        assert stim_ids, f"empty stimulus {stimulus!r}"
        statement_ids = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
        self.seqs = [s + stim_ids for s in statement_ids]
        self.n_statement = [len(s) for s in statement_ids]
        self.decompose_stimulus = decompose_stimulus
        self.pad_id = tokenizer.pad_token_id
        assert isinstance(self.pad_id, int)
        self.batch_size = batch_size
        self.device = device
        logger.info(
            f"stimulus {stimulus!r} = {[tokenizer.decode([i]) for i in stim_ids]}; e.g. "
            f"{tokenizer.decode(self.seqs[0])!r}, decomposed tokens "
            f"{[tokenizer.decode([t]) for t in self.seqs[0][self.n_statement[0] - 1 :]] if decompose_stimulus else [tokenizer.decode([statement_ids[0][-1]])]}"
        )

    def __iter__(self):
        for start in range(0, len(self.seqs), self.batch_size):
            rows = range(start, min(start + self.batch_size, len(self.seqs)))
            width = max(len(self.seqs[r]) for r in rows)
            ids = torch.tensor(
                [self.seqs[r] + [self.pad_id] * (width - len(self.seqs[r])) for r in rows]
            )
            first = torch.tensor([self.n_statement[r] - 1 for r in rows])
            end = torch.tensor([len(self.seqs[r]) - 1 for r in rows])
            last = end if self.decompose_stimulus else first
            pos = torch.arange(width)[None, :]
            decomposed = (pos >= first[:, None]) & (pos <= last[:, None])
            yield (
                list(rows),
                ids.to(self.device),
                decomposed.to(self.device),
                end.to(self.device),
            )


def evaluate_run(
    run: str,
    texts: list[str],
    labels: np.ndarray,
    families: np.ndarray,
    stimulus: str,
    decompose_stimulus: bool,
    batch_size: int,
    max_singles: int | None,
    device: str,
) -> dict[str, Any]:
    run_info = SPDRunInfo.from_path(run_path(run))
    config = run_info.config
    model = ComponentModel.from_run_info(run_info).to(device).eval()
    assert config.tokenizer_name is not None
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    t_ids = torch.tensor(single_token_ids(tokenizer, TRUE_WORDS), device=device)
    f_ids = torch.tensor(single_token_ids(tokenizer, FALSE_WORDS), device=device)
    assert not set(t_ids.tolist()) & set(f_ids.tolist())
    modules = sorted(model.target_module_paths)
    n_c = [model.module_to_c[m] for m in modules]
    offsets = np.concatenate([[0], np.cumsum(n_c)])
    deltas = model.calc_weight_deltas()
    batches = Batches(tokenizer, texts, stimulus, decompose_stimulus, batch_size, device)
    n = len(texts)

    def forward_lp(ids: Tensor, end: Tensor, mask_infos: dict[str, Any] | None) -> Tensor:
        """Next-token log-probs at each row's stimulus end."""
        with torch.no_grad(), bf16_autocast(enabled=config.autocast_bf16):
            logits = model(ids, mask_infos=mask_infos)
        return torch.log_softmax(logits[torch.arange(len(end), device=device), end].float(), -1)

    def masked(
        ids: Tensor, decomposed: Tensor, masks: dict[str, Tensor], delta_on: bool
    ) -> dict[str, Any]:
        delta_masks = {m: torch.full(ids.shape, float(delta_on), device=device) for m in modules}
        infos = make_mask_infos(
            masks, weight_deltas_and_masks={m: (deltas[m], delta_masks[m]) for m in modules}
        )
        return route_only_selected_positions(infos, decomposed)

    # Pass 1: original verdicts and the rounded CIs at the decomposed positions.
    # (n, vocab) fp32 on the CPU: ~1 GB for the 1,794 tiu test statements and Qwen's vocabulary.
    lp_orig_parts: list[Tensor] = []
    # per batch (in iteration order): rounded CIs (b, pos, C) per module, for the components state
    rounded_batches: list[dict[str, Tensor]] = []
    rounded = np.zeros((n, int(offsets[-1])), dtype=bool)  # active at any decomposed position
    n_active_per_position: list[float] = []
    for rows, ids, decomposed, end in batches:
        with torch.no_grad(), bf16_autocast(enabled=config.autocast_bf16):
            out = model(ids, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=out.cache, detach_inputs=True, sampling=config.sampling
            )
        rr = torch.arange(len(rows), device=device)
        lp_orig_parts.append(torch.log_softmax(out.output[rr, end].float(), -1).cpu())
        on = {m: ci.lower_leaky[m] > ROUNDING_THRESHOLD for m in modules}
        rounded_batches.append({m: v.cpu() for m, v in on.items()})
        at_d = torch.cat([on[m] & decomposed[..., None] for m in modules], dim=-1)  # (b, pos, C)
        rounded[rows] = at_d.any(dim=1).cpu().numpy()
        n_active_per_position += at_d.sum(-1)[decomposed].float().cpu().tolist()
    lp_orig = torch.cat(lp_orig_parts)  # batches run through the rows in order
    assert lp_orig.shape[0] == n
    active_share = rounded.mean(0)
    singles = [int(c) for c in np.flatnonzero(active_share >= MIN_ACTIVE_SHARE)]
    n_frequent = len(singles)
    if max_singles is not None:
        singles = sorted(singles, key=lambda c: -active_share[c])[:max_singles]
    logger.info(
        f"{run}: mean active components per decomposed position "
        f"{np.mean(n_active_per_position):.1f}; {n_frequent} components active on >= "
        f"{MIN_ACTIVE_SHARE:.0%} of statements, {len(singles)} of them ablated singly"
    )

    def state_masks(state: str, ids: Tensor, i_batch: int) -> dict[str, Tensor]:
        b, pos = ids.shape
        match state:
            case "components":
                masks = {m: v.float().to(device) for m, v in rounded_batches[i_batch].items()}
                assert all(v.shape == (b, pos, c) for v, c in zip(masks.values(), n_c, strict=True))
                return masks
            case "delta_only" | "all_off":
                return {
                    m: torch.zeros(b, pos, c, device=device)
                    for m, c in zip(modules, n_c, strict=True)
                }
            case "full":
                return {
                    m: torch.ones(b, pos, c, device=device)
                    for m, c in zip(modules, n_c, strict=True)
                }
            case _:
                assert state.startswith("minus_"), state
                c = int(state.removeprefix("minus_"))
                masks = {
                    m: torch.ones(b, pos, k, device=device)
                    for m, k in zip(modules, n_c, strict=True)
                }
                i = int(np.searchsorted(offsets, c, side="right")) - 1
                masks[modules[i]][..., c - offsets[i]] = 0.0
                return masks

    states = [*MAIN_STATES[1:], *[f"minus_{c}" for c in singles]]
    verdict: dict[str, np.ndarray] = {}
    kl_end: dict[str, np.ndarray] = {}

    def score(lp: Tensor) -> np.ndarray:
        lp = lp.to(device)
        return (torch.logsumexp(lp[:, t_ids], -1) - torch.logsumexp(lp[:, f_ids], -1)).cpu().numpy()

    verdict["original"] = score(lp_orig)
    kl_end["original"] = np.zeros(n)
    for state in states:
        s_out, kl_out = np.zeros(n), np.zeros(n)
        delta_on = state in ("delta_only", "full") or state.startswith("minus_")
        for i_batch, (rows, ids, decomposed, end) in enumerate(batches):
            infos = masked(ids, decomposed, state_masks(state, ids, i_batch), delta_on)
            lp = forward_lp(ids, end, infos)
            s_out[rows] = score(lp)
            p = lp_orig[rows].to(device)
            kl_out[rows] = F.kl_div(lp, p, log_target=True, reduction="none").sum(-1).cpu().numpy()
        verdict[state], kl_end[state] = s_out, kl_out
        if not state.startswith("minus_") or len(verdict) % 20 == 0:
            logger.info(f"{run}: {state} done")

    # `full` reproduces the original weights; a large deviation would mean a wrong Delta or a
    # wrong component weight, not rounding.
    max_dev = float(np.abs(verdict["full"] - verdict["original"]).max())
    logger.info(f"{run}: full (components + Delta) vs original, max |score diff| {max_dev:.4f}")
    assert max_dev < 0.1 * float(np.abs(verdict["original"]).mean()), (
        f"components + Delta do not reproduce the original model (max score diff {max_dev:.3f})"
    )

    def summary(s: np.ndarray, kl: np.ndarray) -> dict[str, float]:
        pred = s > 0
        return {
            "accuracy": float((pred == labels).mean()),
            "auc": float(roc_auc_score(labels, s)),
            "separation": float(s[labels].mean() - s[~labels].mean()),
            "share_predicted_true": float(pred.mean()),
            "kl_to_original": float(kl.mean()),
        }

    result: dict[str, Any] = {
        "run": run,
        "stimulus": stimulus,
        "n_statements": n,
        "components_plus_delta_max_score_dev": max_dev,
        "decompose_stimulus": decompose_stimulus,
        "mean_active_components": float(np.mean(n_active_per_position)),
        "n_components_active_min_share": n_frequent,
        "states": {st: summary(verdict[st], kl_end[st]) for st in ["original", *states]},
        "active_share": {str(c): float(active_share[c]) for c in singles},
        "per_family_accuracy": {
            st: {
                fam: float(((verdict[st][families == fam] > 0) == labels[families == fam]).mean())
                for fam in sorted(set(families))
            }
            for st in MAIN_STATES
        },
    }
    return result


def report(result: dict[str, Any]) -> str:
    lines = [
        f"=== {result['run']}  stimulus {result['stimulus']!r}  n={result['n_statements']}  "
        f"decompose_stimulus {result['decompose_stimulus']}  "
        f"mean active components per decomposed position {result['mean_active_components']:.1f}  "
        f"components+Delta max score dev {result['components_plus_delta_max_score_dev']:.4f}",
        f"{'state':12s} {'acc':>6s} {'AUC':>6s} {'sep':>7s} {'%true':>6s} {'KL':>7s}",
    ]
    states = result["states"]
    full_sep = states["full"]["separation"]
    for st in MAIN_STATES:
        v = states[st]
        lines.append(
            f"{st:12s} {v['accuracy']:6.3f} {v['auc']:6.3f} {v['separation']:7.2f} "
            f"{v['share_predicted_true']:6.3f} {v['kl_to_original']:7.4f}"
        )
    singles = sorted(
        (st for st in states if st.startswith("minus_")),
        key=lambda st: -abs(states[st]["separation"] - full_sep),
    )
    lines.append(f"single-component ablations: {len(singles)}; largest separation changes vs full:")
    for st in singles[:10]:
        v = states[st]
        lines.append(
            f"  {st:12s} sep change {v['separation'] - full_sep:+7.2f}  acc {v['accuracy']:.3f}  "
            f"KL to original {v['kl_to_original']:.4f}  "
            f"active on {result['active_share'][st.removeprefix('minus_')]:.0%}"
        )
    return "\n".join(lines)


def main(
    out_dir: str,
    stimulus: str,
    *runs: str,
    cap: int | None = None,
    batch_size: int = 64,
    max_singles: int | None = None,
    decompose_stimulus: bool = False,
) -> None:
    assert runs, "give at least one run"
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    first = SPDRunInfo.from_path(run_path(runs[0])).config
    dirs = first.task_config.prepared_datasets  # pyright: ignore[reportAttributeAccessIssue]
    assert dirs and all("/tiu/" in d for d in dirs), f"not a tiu run: {dirs}"
    texts, labels, names = load_split_with_labels(dirs, "test")
    labels = np.asarray(labels, dtype=bool)
    families = np.array([nm.removesuffix("_true").removesuffix("_false") for nm in names])
    if cap is not None:
        keep = np.concatenate(
            [np.flatnonzero(np.array(names) == nm)[:cap] for nm in sorted(set(names))]
        )
        texts = [texts[i] for i in keep]
        labels, families = labels[keep], families[keep]
    logger.info(f"{len(texts)} test statements, {labels.mean():.3f} true")
    for run in runs:
        result = evaluate_run(
            run,
            texts,
            labels,
            families,
            stimulus,
            decompose_stimulus,
            batch_size,
            max_singles,
            device,
        )
        name = Path(run).parent.name if Path(run).exists() else run
        suffix = "_span" if decompose_stimulus else ""
        (out / f"verdict_{name}{suffix}.json").write_text(json.dumps(result, indent=1))
        print(report(result), flush=True)


if __name__ == "__main__":
    fire.Fire(main)
