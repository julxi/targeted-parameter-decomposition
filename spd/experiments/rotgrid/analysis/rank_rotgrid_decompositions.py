"""Rank the RotGrid decompositions in `bitt-j-personal/spd` by how well they identify mechanisms.

Every run in that project decomposes the same RotGrid target model, so the runs are directly
comparable. Training already optimises reconstruction against sparsity, so the logged losses cannot
settle which decomposition found *mechanisms* — a run can sit on a fine KL/L0 trade-off with
components that are not mechanisms at all. This script therefore scores four axes, three of which
measure properties the loss does not directly optimise:

1. causal faithfulness   — does ablating a component matter exactly where its CI says it should?
2. reproducibility       — do independent seeds converge on the same components?
3. ground-truth alignment— is a component's causal role described by a simple predicate over the
                           48-state automaton that RotGrid actually is?
4. parsimony             — how many components, at what CI-L0.

The task's exact next-token distribution is uniform over the legal tokens, so every KL here has a
true zero floor, and every position carries an exact world-state label obtained by replaying
`game.NEXT_STATE` over the tokens.

Usage:
    python spd/experiments/rotgrid/analysis/rank_rotgrid_decompositions.py                 # all runs, reuse cache
    python spd/experiments/rotgrid/analysis/rank_rotgrid_decompositions.py --only=s-3a8b8eae
    python spd/experiments/rotgrid/analysis/rank_rotgrid_decompositions.py --refresh       # recompute everything
"""

import json
import warnings
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from pathlib import Path

import fire
import numpy as np
import torch
import wandb
from jaxtyping import Bool, Float, Int
from scipy.optimize import linear_sum_assignment
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from torch import Tensor

from spd.configs import ImportanceMinimalityLossConfig, LayerwiseCiConfig
from spd.experiments.rotgrid import game
from spd.experiments.rotgrid.dataset import RotGridDataset
from spd.experiments.rotgrid.model import RotGridTransformer
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import make_mask_infos
from spd.settings import SPD_OUT_DIR

OUT_DIR = SPD_OUT_DIR / "rotgrid_analysis"
WANDB_PROJECT = "bitt-j-personal/spd"
CACHE_VERSION = 1

STAT_SEQS = 128  # sequences behind the CI statistics, features, matching and decoders
EVAL_SEQS = 64  # sequences behind the per-component ablation sweep (a prefix of the STAT batch)
TAU = 0.1  # ci_alive_threshold; asserted against every run's config
EPS_LEAK = 0.01  # below this CI the decomposition is claiming the component is simply off
DELTA_KL = 1e-3  # nats; above this an ablation measurably changed the output
CLEAN_LIFT = 0.5  # held-out F1 lift above which a component counts as cleanly describable
N_NULL_DRAWS = 5


@dataclass(frozen=True)
class RotGridRun:
    """One decomposition, keyed by the hyperparameters that vary across the sweep."""

    run_id: str
    name: str
    hidden: int
    C: int
    impmin: float
    steps: int
    seed: int

    @property
    def config_key(self) -> tuple[int, int, float, int]:
        return (self.hidden, self.C, self.impmin, self.steps)

    @property
    def config_label(self) -> str:
        return f"mlp{self.hidden}_C{self.C}_impmin{self.impmin:g}_{self.steps // 1000}k"

    @property
    def label(self) -> str:
        return f"{self.config_label}_seed{self.seed}"

    @cached_property
    def run_info(self) -> SPDRunInfo:
        return SPDRunInfo.from_path(f"wandb:{WANDB_PROJECT}/runs/{self.run_id}")

    @property
    def cache_dir(self) -> Path:
        path = OUT_DIR / "runs" / self.run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_model(self) -> ComponentModel:
        """Load the decomposition and check the sweep metadata against the checkpoint's own config."""
        config = self.run_info.config
        assert isinstance(config.ci_config, LayerwiseCiConfig), (
            f"{self.run_id}: expected layerwise CI"
        )
        (hidden,) = config.ci_config.hidden_dims
        (C,) = {info.C for info in config.module_info}
        (impmin,) = [
            lc.coeff
            for lc in config.loss_metric_configs
            if isinstance(lc, ImportanceMinimalityLossConfig)
        ]
        assert (hidden, C, impmin, config.steps, config.seed) == (
            self.hidden,
            self.C,
            self.impmin,
            self.steps,
            self.seed,
        ), f"{self.run_id}: wandb metadata disagrees with final_config.yaml"
        assert config.ci_alive_threshold == TAU, f"{self.run_id}: unexpected alive threshold"
        assert config.sampling == "continuous", f"{self.run_id}: CI would not be deterministic"
        return ComponentModel.from_run_info(self.run_info).eval()


def discover_runs(refresh: bool) -> list[RotGridRun]:
    """List the finished rotgrid decompositions, caching the wandb query so reruns stay offline."""
    path = OUT_DIR / "runs.json"
    if path.exists() and not refresh:
        rows = json.loads(path.read_text())
    else:
        rows = []
        for run in wandb.Api().runs(WANDB_PROJECT):
            if not run.name.startswith("rotgrid") or run.state != "finished":
                continue
            (hidden,) = run.config["ci_config"]["hidden_dims"]
            (C,) = {info["C"] for info in run.config["module_info"]}
            rows.append(
                {
                    "run_id": run.id,
                    "name": run.name,
                    "hidden": hidden,
                    "C": C,
                    "impmin": run.config["loss.ImpMin.coeff"],
                    "steps": run.config["steps"],
                    "seed": run.config["seed"],
                }
            )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1))
    runs = [RotGridRun(**row) for row in rows]
    return sorted(runs, key=lambda r: (r.hidden, r.C, -r.impmin, r.steps, r.seed))


# --------------------------------------------------------------------------------------------
# Shared evaluation data: one batch of rollouts, its exact world-state labels, and the reference
# statistics of the target model on it. Every run is scored on these same tokens.
# --------------------------------------------------------------------------------------------


@dataclass
class EvalData:
    tokens: Int[Tensor, "stat seq"]
    legal: Bool[Tensor, "stat seq n_tokens"]
    state: Int[Tensor, "stat seq"]
    features: dict[str, Int[Tensor, "stat seq"]]
    fit: Bool[Tensor, " n_positions"]  # parity split over sequences, flattened


def _steps_since_new(tokens: Int[Tensor, "batch seq"]) -> Int[Tensor, "batch seq"]:
    batch, seq_len = tokens.shape
    since = torch.empty(batch, seq_len, dtype=torch.long)
    counter = torch.zeros(batch, dtype=torch.long)
    for position in range(seq_len):
        counter = torch.where(tokens[:, position] == game.NEW, 0, counter + 1)
        since[:, position] = counter
    return since


def build_eval_data(seq_len: int) -> EvalData:
    torch.manual_seed(0)
    tokens, legal = RotGridDataset(seq_len, "cpu").generate_batch(STAT_SEQS)
    batch, seq_len = tokens.shape

    state = torch.empty(batch, seq_len, dtype=torch.long)
    state[:, 0] = game.INITIAL_STATE
    for position in range(1, seq_len):
        state[:, position] = game.NEXT_STATE[state[:, position - 1], tokens[:, position]]
    assert (state != game.ILLEGAL_TRANSITION).all(), "the replay hit an illegal transition"
    assert (game.LEGAL_MASK[state] == legal).all(), "replayed states disagree with generate_batch"

    before = torch.cat([state[:, :1], state[:, :-1]], dim=1)
    cell, orientation = state // game.N_ORIENTATIONS, state % game.N_ORIENTATIONS
    cell_before = before // game.N_ORIENTATIONS
    orientation_before = before % game.N_ORIENTATIONS

    packed = (legal.long() * (2 ** torch.arange(game.N_TOKENS))).sum(-1)
    _, legalset = packed.unique(return_inverse=True)

    # 0 = not a goal, 1 = A (turns anticlockwise), 2 = B (clockwise), 3 = X (no turn)
    goal_kind = torch.zeros(game.N_CELLS, dtype=torch.long)
    for goal_cell, delta in game.GOALS.items():
        goal_kind[game.cell_index(goal_cell)] = {1: 1, -1: 2, 0: 3}[delta]

    since_new = _steps_since_new(tokens)
    features = {
        "tok": tokens,
        "prev_tok": torch.cat([tokens[:, :1], tokens[:, :-1]], dim=1),
        "state_after": state,
        "state_before": before,
        "cell_after": cell,
        "cell_before": cell_before,
        "orient_after": orientation,
        "orient_before": orientation_before,
        "row_after": cell // game.N_COLS,
        "col_after": cell % game.N_COLS,
        "legalset_after": legalset,
        "n_legal_after": legal.sum(-1) - 1,
        "goal_after": goal_kind[cell],
        "rotation_delta": (orientation - orientation_before) % game.N_ORIENTATIONS,
        "run_pos_bucket": torch.bucketize(since_new, torch.tensor([1, 2, 3, 4, 5, 8, 16])),
        "seq_pos_bucket": torch.arange(seq_len).expand(batch, seq_len) // (seq_len // 8),
    }
    for name, values in features.items():
        assert values.shape == (batch, seq_len), f"feature {name} has shape {values.shape}"

    fit = (torch.arange(batch) % 2 == 0)[:, None].expand(batch, seq_len).flatten()
    return EvalData(tokens=tokens, legal=legal, state=state, features=features, fit=fit)


@dataclass
class TargetStats:
    """The target model's own predictions on the shared batch, plus the scales that calibrate KL."""

    log_p: Float[Tensor, "stat seq n_tokens"]
    p: Float[Tensor, "stat seq n_tokens"]
    resid: Float[Tensor, "stat seq d_model"]  # pre-unembed residual: the decodability ceiling
    kl_const: float  # KL from the target to the best constant predictor
    kl_to_optimal: float


def build_target_stats(cm: ComponentModel, data: EvalData) -> TargetStats:
    assert isinstance(cm.target_model, RotGridTransformer)
    resid: list[Tensor] = []
    handle = cm.target_model.unembed.register_forward_pre_hook(
        lambda _module, args: resid.append(args[0].detach())
    )
    with torch.no_grad():
        logits = cm(data.tokens)
    handle.remove()
    assert len(resid) == 1

    log_p = torch.log_softmax(logits, dim=-1)
    p = log_p.exp()
    q = data.legal.float() / data.legal.sum(-1, keepdim=True)

    constant = p.reshape(-1, game.N_TOKENS).mean(0)
    kl_const = float((p * (log_p - constant.log())).sum(-1).mean())
    kl_to_optimal = float((q * (q.clamp(min=1e-30).log() - log_p)).sum(-1).mean())
    assert kl_to_optimal < 1e-4, f"target model is not near-optimal: {kl_to_optimal}"
    assert 0.5 < kl_const < 0.8, f"unexpected constant-predictor scale: {kl_const}"
    return TargetStats(
        log_p=log_p, p=p, resid=resid[0], kl_const=kl_const, kl_to_optimal=kl_to_optimal
    )


def kl_from_target(
    target: TargetStats, logits: Float[Tensor, "batch seq n_tokens"]
) -> Float[Tensor, "batch seq"]:
    """KL(target model || variant), per position, matching the convention in the validation scripts."""
    batch = logits.shape[0]
    log_p, p = target.log_p[:batch], target.p[:batch]
    return (p * (log_p - torch.log_softmax(logits, dim=-1))).sum(-1)


# --------------------------------------------------------------------------------------------
# Per-run stages. Each writes one cache file and is skipped when that file is already present.
# --------------------------------------------------------------------------------------------


def _ones_masks(cm: ComponentModel, batch: int, seq_len: int) -> dict[str, Tensor]:
    return {name: torch.ones(batch, seq_len, c) for name, c in cm.module_to_c.items()}


def _delta_masks(
    weight_deltas: dict[str, Tensor], batch: int, seq_len: int, on: bool
) -> dict[str, tuple[Tensor, Tensor]]:
    fill = torch.full((batch, seq_len), 1.0 if on else 0.0)
    return {name: (delta, fill) for name, delta in weight_deltas.items()}


def _delta_masks_off_in(
    weight_deltas: dict[str, Tensor], batch: int, seq_len: int, off: set[str]
) -> dict[str, tuple[Tensor, Tensor]]:
    on, off_fill = torch.ones(batch, seq_len), torch.zeros(batch, seq_len)
    return {name: (delta, off_fill if name in off else on) for name, delta in weight_deltas.items()}


def _save(path: Path, payload: dict[str, object]) -> None:
    torch.save({"version": CACHE_VERSION, **payload}, path)


def _load(path: Path) -> dict[str, object]:
    payload = torch.load(path, weights_only=False)
    assert payload["version"] == CACHE_VERSION, f"stale cache at {path}; rerun with --refresh"
    return payload


def stage_ci(run: RotGridRun, cm: ComponentModel, data: EvalData) -> dict[str, Tensor]:
    """Causal importances on the shared batch. Deterministic, since sampling is continuous."""
    path = run.cache_dir / "ci.pt"
    if path.exists():
        return _load(path)["ci"]  # pyright: ignore[reportReturnType]
    with torch.no_grad():
        cache = cm(data.tokens, cache_type="input").cache
        ci = cm.calc_causal_importances(cache, sampling=run.run_info.config.sampling).lower_leaky
    stored = {name: values.half() for name, values in ci.items()}
    _save(path, {"ci": stored})
    return stored


def stage_circuit_kl(
    run: RotGridRun, cm: ComponentModel, data: EvalData, target: TargetStats, ci: dict[str, Tensor]
) -> dict[str, float]:
    """Whole-decomposition KLs under the masking regimes that define a circuit."""
    path = run.cache_dir / "circuit_kl.pt"
    if path.exists():
        return _load(path)["kls"]  # pyright: ignore[reportReturnType]

    kl_const = target.kl_const
    batch, seq_len = data.tokens.shape
    weight_deltas = cm.calc_weight_deltas()
    delta_on = _delta_masks(weight_deltas, batch, seq_len, on=True)
    delta_off = _delta_masks(weight_deltas, batch, seq_len, on=False)
    ones = _ones_masks(cm, batch, seq_len)
    soft = {name: values.float() for name, values in ci.items()}
    hard = {name: (values.float() > TAU).float() for name, values in ci.items()}
    zeros = {name: torch.zeros_like(values) for name, values in ones.items()}

    regimes = {
        "kl_full": (ones, delta_on),
        "kl_soft_ci": (soft, delta_on),
        "kl_mask_delta": (hard, delta_on),
        "kl_circuit": (hard, delta_off),
        "kl_zero": (zeros, delta_off),
    }
    kls: dict[str, float] = {}
    with torch.no_grad():
        for name, (masks, deltas) in regimes.items():
            logits = cm(
                data.tokens, mask_infos=make_mask_infos(masks, weight_deltas_and_masks=deltas)
            )
            kls[name] = float(kl_from_target(target, logits).mean())

    assert abs(kls["kl_full"]) < 1e-6, (
        f"{run.label}: components+delta do not reconstruct the target ({kls['kl_full']:.2e})"
    )
    # Not "zero is the worst regime": a near-collapsed run whose circuit is a single component can
    # be marginally worse than silence, because one rank-1 map firing alone pushes the output
    # further from the target than emitting nothing does. What must hold is that deleting all three
    # decomposed matrices breaks the model worse than predicting a constant.
    assert kls["kl_zero"] > kl_const, (
        f"{run.label}: zeroing every decomposed matrix left the model at KL {kls['kl_zero']:.3f}, "
        f"no worse than the constant predictor at {kl_const:.3f}"
    )
    _save(path, {"kls": kls})
    return kls


def stage_delta_kl(
    run: RotGridRun, cm: ComponentModel, data: EvalData, target: TargetStats, ci: dict[str, Tensor]
) -> dict[str, dict[str, float]]:
    """KL with delta removed from every block and from one block at a time, under two component masks.

    Returns {mask: {delta_off_in: kl}} where mask is "all_on" (every component at 1, so the weights are
    exactly the component sum) or "hard_ci" (the run's own ci > TAU mask), and delta_off_in is "none",
    "all", or a module name. Comparing the two masks separates delta carrying weight the components do
    not cover from CI switching off components that are still needed.
    """
    path = run.cache_dir / "delta_kl.pt"
    if path.exists():
        return _load(path)["kls"]  # pyright: ignore[reportReturnType]

    batch, seq_len = data.tokens.shape
    weight_deltas = cm.calc_weight_deltas()
    component_masks = {
        "all_on": _ones_masks(cm, batch, seq_len),
        "hard_ci": {name: (values.float() > TAU).float() for name, values in ci.items()},
    }
    off_sets = {"none": set(), "all": set(weight_deltas)} | {name: {name} for name in weight_deltas}
    kls: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for mask_name, masks in component_masks.items():
            kls[mask_name] = {}
            for off_name, off in off_sets.items():
                deltas = _delta_masks_off_in(weight_deltas, batch, seq_len, off)
                logits = cm(
                    data.tokens, mask_infos=make_mask_infos(masks, weight_deltas_and_masks=deltas)
                )
                kls[mask_name][off_name] = float(kl_from_target(target, logits).mean())

    assert abs(kls["all_on"]["none"]) < 1e-6, (
        f"{run.label}: components+delta do not reconstruct the target"
    )
    _save(path, {"kls": kls})
    return kls


def stage_ablation(
    run: RotGridRun, cm: ComponentModel, data: EvalData, target: TargetStats
) -> dict[str, Tensor]:
    """Per-position KL from ablating each component on its own, against the exact-target baseline.

    The baseline is all components on with the delta present, which reproduces the target model
    exactly, so each number is damage done to the target rather than to some other reconstruction.
    All C components are ablated, not just the alive ones: the dead ones are cheap and their damage
    is the check that the alive threshold means anything.
    """
    path = run.cache_dir / "ablation_kl.pt"
    if path.exists():
        return _load(path)["kl"]  # pyright: ignore[reportReturnType]

    tokens = data.tokens[:EVAL_SEQS]
    batch, seq_len = tokens.shape
    weight_deltas = cm.calc_weight_deltas()
    delta_on = _delta_masks(weight_deltas, batch, seq_len, on=True)
    masks = _ones_masks(cm, batch, seq_len)

    out: dict[str, Tensor] = {}
    with torch.no_grad():
        for layer, n_components in cm.module_to_c.items():
            damage = torch.empty(n_components, batch, seq_len)
            for component in range(n_components):
                masks[layer][:, :, component] = 0.0
                logits = cm(
                    tokens, mask_infos=make_mask_infos(masks, weight_deltas_and_masks=delta_on)
                )
                damage[component] = kl_from_target(target, logits)
                masks[layer][:, :, component] = 1.0
            out[layer] = damage.half()
    _save(path, {"kl": out})
    return out


# --------------------------------------------------------------------------------------------
# Axis 1: causal faithfulness. Does ablating a component matter where its CI says it should?
# --------------------------------------------------------------------------------------------


@dataclass
class ComponentStats:
    """Everything known about one subcomponent of one run."""

    layer: str
    index: int
    alive: bool
    max_ci: float
    active_frac: float
    weight: float  # share of the run's total ablation damage
    mean_kl: float
    ci_precision: float  # share of this component's damage landing where ci > TAU
    ci_auc: float  # damage-weighted rank of CI: 0.5 = CI says nothing about where it acts
    participation: float  # effective fraction of positions it acts on
    effect_frac: float
    states_on: int
    f1_act: float
    f1_dmg: float
    f1_ceiling: float
    best_feature: str
    best_predicate: str


def _damage_weighted_auc(ci: Tensor, damage: Tensor) -> float:
    """Mann-Whitney AUC of CI as a ranker of where ablation damage falls."""
    total = float(damage.sum())
    if total <= 0:
        return 0.5
    n = ci.numel()
    order = ci.argsort()
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float64)
    # average ranks within ties, so a constant CI scores exactly 0.5
    unique, inverse = ci.unique(return_inverse=True)
    summed = torch.zeros(unique.numel(), dtype=torch.float64).index_add_(0, inverse, ranks)
    counts = torch.zeros(unique.numel(), dtype=torch.float64).index_add_(
        0, inverse, torch.ones(n, dtype=torch.float64)
    )
    ranks = (summed / counts)[inverse]
    mean_rank = float((damage.double() * ranks).sum() / total)
    return (mean_rank - (n + 1) / 2) / n + 0.5


def _f1(truth: Tensor, predicted: Tensor) -> float:
    hits = float((truth & predicted).sum())
    denominator = float(truth.sum()) + float(predicted.sum())
    return 2 * hits / denominator if denominator > 0 else 0.0


def _f1_lift(truth: Tensor, predicted: Tensor) -> float:
    """How far a predicate beats "always true", which already scores 2p/(1+p) for base rate p.

    Without this, a component that is simply active everywhere scores a high F1 from a vacuous
    predicate over every value of some feature, and looks like a cleanly identified mechanism.
    """
    base_rate = float(truth.float().mean())
    trivial = 2 * base_rate / (1 + base_rate) if base_rate > 0 else 0.0
    if trivial >= 1.0:
        return 0.0
    return max(0.0, (_f1(truth, predicted) - trivial) / (1 - trivial))


def _best_value_set(truth: Tensor, feature: Tensor, n_values: int) -> Tensor:
    """The value set maximising F1 of `feature in S` as a predictor of `truth`.

    Ordering values by precision and sweeping prefixes is exact, not greedy-approximate: over unions
    of cells of a partition, adding a cell raises F1 exactly when its precision exceeds half the
    current F1, so precision order makes the sequence single-peaked.
    """
    positives = torch.bincount(feature[truth], minlength=n_values).float()
    totals = torch.bincount(feature, minlength=n_values).float()
    precision = positives / totals.clamp(min=1.0)
    order = precision.argsort(descending=True)
    true_positives = positives[order].cumsum(0)
    predicted = totals[order].cumsum(0)
    f1 = 2 * true_positives / (predicted + float(truth.sum())).clamp(min=1e-9)
    return order[: int(f1.argmax()) + 1]


def _fit_predicate(
    truth: Tensor, features: dict[str, Tensor], fit: Tensor, names: list[str]
) -> tuple[float, str, str]:
    """Pick the best single-feature predicate on the fit half, score it on the held-out half."""
    if truth.sum() == 0 or truth.all():
        return 0.0, "none", "degenerate"
    best = (0.0, "none", "degenerate")
    for name in names:
        feature = features[name]
        n_values = int(feature.max()) + 1
        chosen = _best_value_set(truth[fit], feature[fit], n_values)
        member = torch.zeros(n_values, dtype=torch.bool)
        member[chosen] = True
        if int(member.sum()) == n_values:
            continue  # selecting every value is the vacuous predicate, not an explanation
        score = _f1_lift(truth[~fit], member[feature[~fit]])
        if score > best[0]:
            values = ",".join(str(int(v)) for v in sorted(chosen.tolist()))
            best = (score, name, f"{name} in {{{values}}}")
    return best


def component_stats(
    run: RotGridRun,
    cm: ComponentModel,
    data: EvalData,
    ci: dict[str, Tensor],
    ablation: dict[str, Tensor],
) -> list[ComponentStats]:
    path = run.cache_dir / "components.pt"
    if path.exists():
        return _load(path)["stats"]  # pyright: ignore[reportReturnType]

    n_eval = EVAL_SEQS * data.tokens.shape[1]
    total_damage = sum(float(v.float().sum()) for v in ablation.values())
    flat_state = data.state[:EVAL_SEQS].flatten()
    eval_fit = data.fit.reshape(STAT_SEQS, -1)[:EVAL_SEQS].flatten()
    eval_features = {name: values[:EVAL_SEQS].flatten() for name, values in data.features.items()}
    simple = [n for n in eval_features if n not in ("state_after", "state_before")]
    ceiling_features = {
        "transition": eval_features["state_before"] * game.N_TOKENS + eval_features["tok"]
    }

    stats: list[ComponentStats] = []
    for layer in cm.module_to_c:
        ci_layer = ci[layer][:EVAL_SEQS].float()
        damage_layer = ablation[layer].float()
        for index in range(cm.module_to_c[layer]):
            ci_flat = ci_layer[:, :, index].flatten()
            damage = damage_layer[index].flatten()
            damage_sum = float(damage.sum())
            active = ci_flat > TAU
            hurt = damage > DELTA_KL

            state_on = torch.zeros(game.N_STATES, dtype=torch.bool)
            for state in range(game.N_STATES):
                in_state = flat_state == state
                state_on[state] = bool(active[in_state].float().mean() > 0.5)

            f1_act, feature_act, predicate_act = _fit_predicate(
                active, eval_features, eval_fit, simple
            )
            f1_dmg, feature_dmg, predicate_dmg = _fit_predicate(
                hurt, eval_features, eval_fit, simple
            )
            f1_ceiling, _, _ = _fit_predicate(hurt, ceiling_features, eval_fit, ["transition"])
            stats.append(
                ComponentStats(
                    layer=layer,
                    index=index,
                    alive=bool(ci[layer][:, :, index].float().max() > TAU),
                    max_ci=float(ci[layer][:, :, index].float().max()),
                    active_frac=float(active.float().mean()),
                    weight=damage_sum / total_damage if total_damage > 0 else 0.0,
                    mean_kl=damage_sum / n_eval,
                    ci_precision=float(damage[active].sum()) / damage_sum
                    if damage_sum > 0
                    else 0.0,
                    ci_auc=_damage_weighted_auc(ci_flat, damage),
                    participation=damage_sum**2 / (n_eval * float((damage**2).sum()))
                    if damage_sum > 0
                    else 0.0,
                    effect_frac=float(hurt.float().mean()),
                    states_on=int(state_on.sum()),
                    f1_act=f1_act,
                    f1_dmg=f1_dmg,
                    f1_ceiling=f1_ceiling,
                    best_feature=feature_dmg if f1_dmg >= f1_act else feature_act,
                    best_predicate=predicate_dmg if f1_dmg >= f1_act else predicate_act,
                )
            )
    _save(path, {"stats": stats})
    return stats


# --------------------------------------------------------------------------------------------
# Axis 3: what the components track. How much of the world state does the active set pin down,
# and is the code compositional or one component per state?
# --------------------------------------------------------------------------------------------


def _entropy(labels: Tensor) -> float:
    counts = torch.bincount(labels).float()
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * probabilities.log()).sum())


def _decodability(design: np.ndarray, labels: Tensor, fit: Tensor, entropy: float) -> float:
    """1 - heldout_crossentropy/entropy for a multinomial decoder of `labels` from `design`."""
    from sklearn.metrics import log_loss

    if design.shape[1] == 0:
        return float("nan")
    y = labels.numpy()
    fit_np = fit.numpy()
    if len(np.unique(y[fit_np])) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(max_iter=1000).fit(design[fit_np], y[fit_np])
    held = ~fit_np
    assert set(np.unique(y[held])) <= set(model.classes_), "held-out half has an unseen class"
    cross_entropy = log_loss(y[held], model.predict_proba(design[held]), labels=model.classes_)
    return 1.0 - cross_entropy / entropy


def alive_code(cm: ComponentModel, ci: dict[str, Tensor], binarise: bool) -> np.ndarray:
    """The alive components' activations as a design matrix: (n_positions, n_alive)."""
    columns = []
    n_positions = 0
    for layer in cm.module_to_c:
        values = ci[layer].float()
        n_positions = values.shape[0] * values.shape[1]
        alive = values.amax(dim=(0, 1)) > TAU
        # spell the row count out: a collapsed run has zero alive columns and `-1` is ambiguous there
        selected = values[:, :, alive].reshape(n_positions, int(alive.sum()))
        columns.append((selected > TAU).float() if binarise else selected)
    return torch.cat(columns, dim=1).numpy() if columns else np.zeros((n_positions, 0))


def summarise_run(
    run: RotGridRun,
    cm: ComponentModel,
    data: EvalData,
    target: TargetStats,
    ci: dict[str, Tensor],
    kls: dict[str, float],
    ablation: dict[str, Tensor],
    stats: list[ComponentStats],
    ceilings: dict[str, float],
) -> dict[str, object]:
    path = run.cache_dir / "summary.pt"
    if path.exists():
        return _load(path)["row"]  # pyright: ignore[reportReturnType]

    layers = list(cm.module_to_c)
    row: dict[str, object] = {
        "run_id": run.run_id,
        "label": run.label,
        "config": run.config_label,
        "hidden": run.hidden,
        "C": run.C,
        "impmin": run.impmin,
        "steps": run.steps,
        "seed": run.seed,
    }

    # --- axis 1: causal faithfulness -------------------------------------------------------
    row.update({name: value for name, value in kls.items()})
    row["R_mask"] = 1.0 - kls["kl_mask_delta"] / target.kl_const
    row["R_circuit"] = 1.0 - kls["kl_circuit"] / target.kl_const
    row["binarisation_cost"] = kls["kl_mask_delta"] - kls["kl_soft_ci"]

    alive_stats = [s for s in stats if s.alive]
    dead_stats = [s for s in stats if not s.alive]
    total_damage = sum(float(v.float().sum()) for v in ablation.values())
    precise = leaked = 0.0
    for layer in layers:
        ci_eval = ci[layer][:EVAL_SEQS].float().permute(2, 0, 1)
        damage = ablation[layer].float()
        precise += float(damage[ci_eval > TAU].sum())
        leaked += float(damage[ci_eval < EPS_LEAK].sum())
    row["ci_precision"] = precise / total_damage if total_damage > 0 else float("nan")
    row["leak"] = leaked / total_damage if total_damage > 0 else float("nan")

    row["ci_auc"] = sum(s.weight * s.ci_auc for s in stats)
    row["mean_participation"] = (
        sum(s.weight * s.participation for s in stats) if total_damage > 0 else float("nan")
    )
    row["max_dead_kl"] = max((s.mean_kl for s in dead_stats), default=0.0)
    hurt_counts = torch.stack([(ablation[layer] > DELTA_KL).sum(0) for layer in layers]).sum(0)
    row["components_per_position"] = float(hurt_counts.float().mean())

    # --- axis 3: ground-truth alignment ----------------------------------------------------
    collapsed = not alive_stats
    row["EXPL"] = float("nan") if collapsed else sum(s.weight * s.f1_dmg for s in stats)
    row["EXPL_act"] = float("nan") if collapsed else sum(s.weight * s.f1_act for s in stats)
    row["EXPL_ceiling"] = float("nan") if collapsed else sum(s.weight * s.f1_ceiling for s in stats)
    row["n_clean"] = sum(1 for s in alive_stats if s.f1_dmg > CLEAN_LIFT)
    row["mean_states_on"] = (
        float(np.mean([s.states_on for s in alive_stats])) if alive_stats else float("nan")
    )

    code = alive_code(cm, ci, binarise=True)
    n_alive = code.shape[1]
    state_entropy = _entropy(data.state.flatten())
    legalset_entropy = _entropy(data.features["legalset_after"].flatten())
    row["state_info"] = _decodability(code, data.state.flatten(), data.fit, state_entropy)
    row["legal_info"] = _decodability(
        code, data.features["legalset_after"].flatten(), data.fit, legalset_entropy
    )
    row["state_info_ci"] = _decodability(
        alive_code(cm, ci, binarise=False), data.state.flatten(), data.fit, state_entropy
    )
    row["state_info_frac_of_resid"] = (
        row["state_info"] / ceilings["state_info_resid"] if n_alive else float("nan")
    )

    if n_alive:
        on_by_state = np.zeros((game.N_STATES, n_alive), dtype=bool)
        flat_state = data.state.flatten().numpy()
        for state in range(game.N_STATES):
            on_by_state[state] = code[flat_state == state].mean(0) > 0.5
        n_codes = len(np.unique(on_by_state, axis=0))
        row["code_efficiency"] = float(np.log2(n_codes) / n_alive)
        row["n_distinct_codes"] = n_codes
        row["components_per_state"] = float(on_by_state.sum(1).mean())
    else:
        row.update(
            {
                "code_efficiency": float("nan"),
                "n_distinct_codes": 0,
                "components_per_state": float("nan"),
            }
        )

    # --- axis 4: parsimony -----------------------------------------------------------------
    row["n_alive"] = len(alive_stats)
    row["n_dead"] = len(dead_stats)
    for position, layer in enumerate(layers):
        values = ci[layer].float()
        row[f"n_alive_b{position}"] = int((values.amax(dim=(0, 1)) > TAU).sum())
        row[f"ci_l0_b{position}"] = float((values > TAU).float().sum(-1).mean())
    row["ci_l0_total"] = sum(
        float((ci[layer].float() > TAU).float().sum(-1).mean()) for layer in layers
    )
    row["ci_l1_total"] = sum(float(ci[layer].float().sum(-1).mean()) for layer in layers)
    _save(path, {"row": row})
    return row


# --------------------------------------------------------------------------------------------
# Axis 2: cross-seed reproducibility. Weight-space matching, and functional matching on the CI
# values themselves -- every run sees the same tokens, so the CI vectors are directly comparable.
# --------------------------------------------------------------------------------------------


def _mechanisms(cm: ComponentModel, layer: str) -> Tensor:
    """Each component as its flattened rank-1 map V_c U_c, unit-normalised: (C, d_in * d_out)."""
    components = cm.components[layer]
    with torch.no_grad():
        outer = torch.einsum("ic,co->cio", components.V, components.U).flatten(start_dim=1)
    return torch.nn.functional.normalize(outer, dim=1)


def _ci_directions(ci_layer: Tensor) -> Tensor:
    """Centred, unit-normalised CI vectors over positions: (C, n_positions). Dead ones become 0."""
    values = ci_layer.float().reshape(-1, ci_layer.shape[-1]).T
    centred = values - values.mean(dim=1, keepdim=True)
    norm = centred.norm(dim=1, keepdim=True)
    return torch.where(norm > 1e-8, centred / norm.clamp(min=1e-8), torch.zeros_like(centred))


def _matched_score(similarity: Tensor, weight_a: Tensor, weight_b: Tensor) -> float:
    """Hungarian-match, then score by the damage the two runs put on each matched pair.

    Matching over all C components rather than only the alive ones is what makes runs with
    different alive counts comparable: an important component with no counterpart is forced onto a
    dead one and contributes ~0, which is the right penalty.
    """
    rows, columns = linear_sum_assignment(similarity.numpy(), maximize=True)
    pair_weight = 0.5 * (weight_a[rows] + weight_b[columns])
    return float((pair_weight * similarity[rows, columns].clamp(min=0)).sum())


@dataclass
class MatchInputs:
    """Per-layer inputs to cross-run matching, so the pairwise pass never reloads a model."""

    mechanisms: dict[str, Float[Tensor, "C d_in_times_d_out"]]
    shape: dict[str, tuple[int, int]]  # (d_in, d_out)
    weight: dict[str, Float[Tensor, " C"]]  # share of the run's total ablation damage


def stage_match_inputs(
    run: RotGridRun, cm: ComponentModel, ablation: dict[str, Tensor]
) -> MatchInputs:
    path = run.cache_dir / "match_inputs.pt"
    if path.exists():
        return MatchInputs(**_load(path)["inputs"])
    total = sum(float(v.float().sum()) for v in ablation.values())
    inputs = MatchInputs(
        mechanisms={layer: _mechanisms(cm, layer) for layer in cm.module_to_c},
        shape={
            layer: (cm.components[layer].V.shape[0], cm.components[layer].U.shape[1])
            for layer in cm.module_to_c
        },
        weight={
            layer: ablation[layer].float().sum(dim=(1, 2)) / total
            if total > 0
            else torch.zeros(cm.module_to_c[layer])
            for layer in cm.module_to_c
        },
    )
    _save(path, {"inputs": vars(inputs)})
    return inputs


def _pair_key(run_id_a: str, run_id_b: str) -> tuple[str, str]:
    return (run_id_a, run_id_b) if run_id_a < run_id_b else (run_id_b, run_id_a)


def match_pair(
    ci_a: dict[str, Tensor],
    ci_b: dict[str, Tensor],
    inputs_a: MatchInputs,
    inputs_b: MatchInputs,
    generator: torch.Generator,
) -> dict[str, float]:
    """Score how much of two decompositions matches, in weight space and in function."""
    scores = {
        "repro_weight": 0.0,
        "repro_weight_null": 0.0,
        "repro_func": 0.0,
        "repro_func_null": 0.0,
    }
    for layer in ci_a:
        weight_a, weight_b = inputs_a.weight[layer], inputs_b.weight[layer]
        mech_a, mech_b = inputs_a.mechanisms[layer], inputs_b.mechanisms[layer]
        scores["repro_weight"] += _matched_score(mech_a @ mech_b.T, weight_a, weight_b)

        directions_a = _ci_directions(ci_a[layer])
        directions_b = _ci_directions(ci_b[layer])
        scores["repro_func"] += _matched_score(directions_a @ directions_b.T, weight_a, weight_b)

        n_components, n_positions = directions_b.shape
        d_in, d_out = inputs_b.shape[layer]
        assert mech_b.shape == (n_components, d_in * d_out)
        weight_null = func_null = 0.0
        for _ in range(N_NULL_DRAWS):
            random_v = torch.randn(d_in, n_components, generator=generator)
            random_u = torch.randn(n_components, d_out, generator=generator)
            random_mech = torch.nn.functional.normalize(
                torch.einsum("ic,co->cio", random_v, random_u).flatten(start_dim=1), dim=1
            )
            weight_null += _matched_score(mech_a @ random_mech.T, weight_a, weight_b)
            shuffled = directions_b[:, torch.randperm(n_positions, generator=generator)]
            func_null += _matched_score(directions_a @ shuffled.T, weight_a, weight_b)
        scores["repro_weight_null"] += weight_null / N_NULL_DRAWS
        scores["repro_func_null"] += func_null / N_NULL_DRAWS
    return scores


# --------------------------------------------------------------------------------------------
# Aggregation, tables and the reasoned pick.
# --------------------------------------------------------------------------------------------

GATES = {
    "G1_faithful": ("R_mask", 0.95, "the run's own hard CI mask barely damages the output"),
    "G2_calibrated": ("ci_precision", 0.90, "ablation damage lands where CI says it should"),
    "G3_reproducible": ("repro_func_excess", 0.30, "seeds converge on the same components"),
    "G4_alive": ("n_alive", 2, "there is more than one component to talk about"),
}
RANK_AXES = [
    ("R_mask", True),
    ("ci_precision", True),
    ("repro_func_excess", True),
    ("EXPL", True),
    ("ci_l0_total", False),
]


def aggregate_configs(
    rows: list[dict[str, object]],
    pair_scores: dict[tuple[str, str], dict[str, float]],
    runs: dict[str, RotGridRun],
) -> list[dict[str, object]]:
    import pandas as pd

    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.columns if frame[c].dtype.kind in "fiu"]
    grouped = frame.groupby("config")[numeric].agg(["mean", "std"])

    configs: list[dict[str, object]] = []
    for config_label, means in grouped.iterrows():
        members = [r for r in runs.values() if r.config_label == config_label]
        row: dict[str, object] = {"config": config_label, "n_seeds": len(members)}
        row.update({name: float(means[(name, "mean")]) for name in numeric})
        row.update(
            {f"{name}_std": float(means[(name, "std")]) for name in ("R_mask", "EXPL", "n_alive")}
        )

        within = [
            pair_scores[key]
            for a, b in combinations(sorted(m.run_id for m in members), 2)
            if (key := (a, b)) in pair_scores
        ]
        for field in ("repro_weight", "repro_weight_null", "repro_func", "repro_func_null"):
            row[field] = float(np.mean([p[field] for p in within])) if within else float("nan")
        row["repro_func_excess"] = row["repro_func"] - row["repro_func_null"]  # pyright: ignore[reportOperatorIssue]
        row["repro_weight_excess"] = row["repro_weight"] - row["repro_weight_null"]  # pyright: ignore[reportOperatorIssue]
        configs.append(row)
    return configs


def add_cross_hidden(
    configs: list[dict[str, object]],
    pair_scores: dict[tuple[str, str], dict[str, float]],
    runs: dict[str, RotGridRun],
) -> None:
    """Match mlp32 against mlp64 at the same impmin/steps/seed.

    The CI function's width is not part of the mechanism, so agreement across it is a genuine, if
    weaker, reproducibility probe -- and it is the only one available for the single-seed configs.
    """
    for row in configs:
        members = [r for r in runs.values() if r.config_label == row["config"]]
        scores = []
        for run in members:
            partners = [
                other
                for other in runs.values()
                if other.hidden != run.hidden
                and (other.impmin, other.steps, other.seed) == (run.impmin, run.steps, run.seed)
            ]
            for other in partners:
                key = _pair_key(run.run_id, other.run_id)
                if key in pair_scores:
                    scores.append(
                        pair_scores[key]["repro_func"] - pair_scores[key]["repro_func_null"]
                    )
        row["repro_cross_hidden"] = float(np.mean(scores)) if scores else float("nan")


def gate_report(configs: list[dict[str, object]]) -> list[dict[str, object]]:
    for row in configs:
        failed = [
            name
            for name, (field, threshold, _) in GATES.items()
            if not (isinstance(row.get(field), (int, float)) and row[field] >= threshold)  # pyright: ignore[reportOperatorIssue]
        ]
        row["gates_failed"] = ",".join(failed) if failed else "-"
    ranks = {}
    for field, higher_is_better in RANK_AXES:
        values = [(row["config"], row.get(field, float("nan"))) for row in configs]
        order = sorted(
            values,
            key=lambda kv: (np.isnan(kv[1]), -kv[1] if higher_is_better else kv[1]),  # pyright: ignore[reportArgumentType]
        )
        for position, (config_label, _) in enumerate(order):
            ranks.setdefault(config_label, []).append(position + 1)
    for row in configs:
        row["mean_rank"] = float(np.mean(ranks[row["config"]]))
    return sorted(configs, key=lambda r: r["mean_rank"])  # pyright: ignore[reportArgumentType]


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, float_format="%.6g")
    logger.info(f"wrote {path} ({len(rows)} rows)")


def build_ceilings(data: EvalData, target: TargetStats) -> dict[str, float]:
    """How much world state the target model itself exposes, and how much a token alone gives."""
    path = OUT_DIR / "ceilings.pt"
    if path.exists():
        return _load(path)["ceilings"]  # pyright: ignore[reportReturnType]
    state = data.state.flatten()
    entropy = _entropy(state)
    resid = target.resid.reshape(-1, target.resid.shape[-1]).numpy()
    one_hot = np.eye(game.N_TOKENS)[data.tokens.flatten().numpy()]
    ceilings = {
        "state_entropy": entropy,
        "state_info_resid": _decodability(resid, state, data.fit, entropy),
        "state_info_tok": _decodability(one_hot, state, data.fit, entropy),
    }
    _save(path, {"ceilings": ceilings})
    return ceilings


def print_report(
    configs: list[dict[str, object]], target: TargetStats, ceilings: dict[str, float]
) -> None:
    def fmt(value: object, spec: str = "7.3f") -> str:
        if isinstance(value, float) and np.isnan(value):
            return f"{'-':>{spec.split('.')[0]}}"
        return f"{value:{spec}}" if isinstance(value, (int, float)) else str(value)

    print()
    print("=" * 118)
    print("ROTGRID DECOMPOSITIONS, RANKED")
    print("=" * 118)
    print(
        f"Calibration: KL to the best constant predictor = {target.kl_const:.3f} nats. "
        f"R = 1 - KL/{target.kl_const:.3f}, so R=1 is perfect, R=0 is a constant predictor, R<0 is"
    )
    print(
        f"worse than saying nothing. Target model KL to ground truth = {target.kl_to_optimal:.1e}. "
        f"The model's own residual exposes {ceilings['state_info_resid']:.2f} of the world state;"
    )
    print(f"the current token alone gives {ceilings['state_info_tok']:.2f}.")
    print()
    header = (
        f"{'config':<28}{'n':>2} {'R_mask':>7} {'R_circ':>7} {'ci_prec':>8} {'leak':>6} "
        f"{'AUC':>5} {'repro':>6} {'xhid':>6} {'EXPL':>6} {'clean':>6} {'state':>6} "
        f"{'codeff':>7} {'alive':>6} {'L0':>6} {'rank':>5}"
    )
    print(header)
    print("-" * len(header))
    for row in configs:
        print(
            f"{row['config']:<28}{row['n_seeds']:>2} "
            f"{fmt(row['R_mask'])} {fmt(row['R_circuit'])} {fmt(row['ci_precision'], '8.3f')} "
            f"{fmt(row['leak'], '6.3f')} {fmt(row['ci_auc'], '5.2f')} "
            f"{fmt(row['repro_func_excess'], '6.2f')} {fmt(row['repro_cross_hidden'], '6.2f')} "
            f"{fmt(row['EXPL'], '6.2f')} {fmt(row['n_clean'], '6.0f')} "
            f"{fmt(row['state_info'], '6.2f')} {fmt(row['code_efficiency'], '7.3f')} "
            f"{fmt(row['n_alive'], '6.1f')} {fmt(row['ci_l0_total'], '6.2f')} "
            f"{fmt(row['mean_rank'], '5.1f')}"
        )
    print()
    print(
        "Gates (a decomposition failing these has not identified a mechanism, whatever else it scores):"
    )
    for name, (field, threshold, why) in GATES.items():
        print(f"  {name:<16} {field} >= {threshold:<6} -- {why}")
    print()
    survivors = [row for row in configs if row["gates_failed"] == "-"]
    if survivors:
        best = max(survivors, key=lambda r: r["EXPL"])  # pyright: ignore[reportArgumentType]
        print(f"PICK: {best['config']} -- the only gate-passing config with the highest EXPL.")
    else:
        print("No config passes every gate. Falling back to mean rank across the four axes,")
        print("among the configs that are at least non-degenerate (G4_alive).")
        configs = [r for r in configs if "G4_alive" not in str(r["gates_failed"])] or configs
        for row in configs[:3]:
            print(
                f"  {row['config']:<28} mean_rank {row['mean_rank']:.1f}  fails {row['gates_failed']}"
            )
        print(f"PICK (by mean rank): {configs[0]['config']}")
    print("=" * 118)


def main(refresh: bool = False, only: str | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if refresh:
        import shutil

        for path in (OUT_DIR / "runs", OUT_DIR / "eval_data.pt", OUT_DIR / "ceilings.pt"):
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(
                missing_ok=True
            )

    runs = discover_runs(refresh)
    if only is not None:
        runs = [r for r in runs if r.run_id == only]
        assert runs, f"no run matches {only}"
    logger.info(f"{len(runs)} decompositions to score")

    first = runs[0].load_model()
    assert isinstance(first.target_model, RotGridTransformer)
    seq_len = first.target_model.config.seq_len
    data = build_eval_data(seq_len)
    target = build_target_stats(first, data)
    ceilings = build_ceilings(data, target)
    logger.info(
        f"kl_const={target.kl_const:.4f}  kl_to_optimal={target.kl_to_optimal:.2e}  "
        f"state_info_resid={ceilings['state_info_resid']:.3f}"
    )

    rows: list[dict[str, object]] = []
    ci_cache: dict[str, dict[str, Tensor]] = {}
    match_cache: dict[str, MatchInputs] = {}
    for position, run in enumerate(runs, start=1):
        logger.info(f"[{position}/{len(runs)}] {run.label} ({run.run_id})")
        cm = first if run is runs[0] else run.load_model()
        assert (
            run.run_info.config.pretrained_model_path
            == runs[0].run_info.config.pretrained_model_path
        )
        ci = stage_ci(run, cm, data)
        kls = stage_circuit_kl(run, cm, data, target, ci)
        stage_delta_kl(run, cm, data, target, ci)
        ablation = stage_ablation(run, cm, data, target)
        match_cache[run.run_id] = stage_match_inputs(run, cm, ablation)
        stats = component_stats(run, cm, data, ci, ablation)
        rows.append(summarise_run(run, cm, data, target, ci, kls, ablation, stats, ceilings))
        ci_cache[run.run_id] = ci

    by_id = {run.run_id: run for run in runs}
    generator = torch.Generator().manual_seed(0)
    pair_scores: dict[tuple[str, str], dict[str, float]] = {}
    for run_a, run_b in combinations(runs, 2):
        same_config = run_a.config_key == run_b.config_key
        cross_hidden = (run_a.impmin, run_a.steps, run_a.seed) == (
            run_b.impmin,
            run_b.steps,
            run_b.seed,
        ) and run_a.hidden != run_b.hidden
        if not (same_config or cross_hidden):
            continue
        pair_scores[_pair_key(run_a.run_id, run_b.run_id)] = match_pair(
            ci_cache[run_a.run_id],
            ci_cache[run_b.run_id],
            match_cache[run_a.run_id],
            match_cache[run_b.run_id],
            generator,
        )
    logger.info(f"matched {len(pair_scores)} run pairs")

    configs = aggregate_configs(rows, pair_scores, by_id)
    add_cross_hidden(configs, pair_scores, by_id)
    configs = gate_report(configs)

    write_tsv(OUT_DIR / "runs.tsv", rows)
    write_tsv(OUT_DIR / "configs.tsv", configs)
    all_components = []
    for run in runs:
        for stat in _load(run.cache_dir / "components.pt")["stats"]:  # pyright: ignore[reportGeneralTypeIssues]
            all_components.append({"run_id": run.run_id, "label": run.label, **vars(stat)})
    write_tsv(OUT_DIR / "components.tsv", all_components)
    print_report(configs, target, ceilings)


if __name__ == "__main__":
    fire.Fire(main)
