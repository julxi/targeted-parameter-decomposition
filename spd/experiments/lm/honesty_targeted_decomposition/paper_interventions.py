"""Splits of the decomposed down_proj outputs and interventions on one part of them: the tensor side
of the behavioural tests (paper draft reports/26-11-04_tpd_truth_paper_draft.typ, section *The
tests* and Table 3; site versions in section 3.2.1; design in convos/julian/26-11-04_paper_draft_LOG.md,
entry *can the site-version tests (3.2.1) be run individually?* and the answers under Julian's
comments there). The driver is paper_tests.py.

A split divides each site's output y (one decomposed down_proj, at the statement tokens) into a
part p and the rest r = y - p:
  SiteSplit       p = y: the whole site (the "site version" of every test; P = W)
  ComponentSplit  p = sum_{c in S} (v_c . x) u_c for a component set S of a basis, at mask 1 (not
                  CI-weighted), so p + r = y exactly; with part "R" the edited part is y - p.
Site and component tests run through the same hooks, so a difference between them cannot come
from the intervention code.

An intervention replaces the edited part q at the statement tokens of every site at once:
  y' = y - q + q', with
  Zero          q' = 0
  Mean          q' = the train mean of q in the position's group (GroupingName)
  TwinAverage   q' = (q(x) + q(x')) / 2, both from clean forward passes (fixed values, like a mean)
  Swap          q' = q(x'), from the clean forward pass on the twin x' (an interchange intervention:
                every site gets the value it has on x', whatever the edits upstream do)
  SwapClamped   y' = q(x') + [y(x) - q(x)]: the edited part from the twin AND the other part held at
                its clean value on x, at every site. Under Swap the other part is recomputed, so at
                later layers it follows what the swapped part wrote earlier (total effect); here it
                cannot (the direct effect of the part's own writes). Added 26-11-15 to check that the
                R-swap's effect is not just the recomputed P following R
                (convos/julian/26-11-15_open_research_LOG.md).
  Shift         q' = q + sign * alpha * d_g, d_g = train mean of q over false minus true statements
                that have a twin, in the position's group; sign = +1 where the model calls the
                statement true, -1 where it calls it false (draft: "added for statements the model
                calls true, subtracted for those it calls false"); q is the current run's value.

Position groups (GroupingName), both from the train statistics and applied per site:
  scan3  first statement token / last statement token / every token between: the grouping of the
         26-10-16 causal layer scan (~/spd_out/26-10-10_bolder_experiments/layer_scan/layer_scan.py),
         whose mean removal the draft cites (Table 2), so "mean:scan3" is that intervention.
  role   massive / first / pre-object / object / end. "massive" = positions whose clean site output
         norm exceeds MASSIVE_FACTOR x the train median at that site, checked first. Why: the layer
         scan's norm check (mean_norms.out) found such tokens in 40% of first tokens and 7.6% of
         middle tokens at layer 11, so a scan3 mean mixes them with ordinary tokens. Julian asked to
         compare the two groupings on val before one becomes the default (LOG, same entry).
Default for the paper (Julian, 26-11-14, after the val comparison; LOG entry "role for mean and shift,
massive at 5x"): role for both the mean removal and the shift. scan3 stays selectable so the layer
scan's intervention can still be reproduced.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from spd.experiments.lm.honesty_targeted_decomposition.paper_common import Basis
from spd.models.component_model import ComponentModel

Part = Literal["P", "R"]
GroupingName = Literal["scan3", "role"]
GROUPINGS: tuple[GroupingName, ...] = ("scan3", "role")
N_GROUPS: dict[GroupingName, int] = {"scan3": 3, "role": 5}
ROLE_GROUPS = ("massive", "first", "pre_object", "object", "end")
SCAN3_GROUPS = ("first", "middle", "end")
# a position is "massive" at a site if its clean output norm exceeds this multiple of the site's train
# median norm. 5 (Julian, 26-11-14), lowered from the layer scan's 10: at 10 the one massive token per
# statement counted only at layer 11 (14x the median), though it is still 8x at layer 12.
MASSIVE_FACTOR = 5.0


# ---- splits ----


@dataclass(frozen=True)
class SiteSplit:
    name: Literal["site"] = "site"


@dataclass(frozen=True)
class ComponentSplit:
    """A component set S of a basis, per module: U (k, d_out), V (d_in, k); k may be 0."""

    name: str  # e.g. "active"
    U: dict[str, Float[Tensor, "k d_out"]]
    V: dict[str, Float[Tensor, "d_in k"]]
    components: tuple[str, ...]  # "L13:19" names, for the record


Split = SiteSplit | ComponentSplit


def component_split(name: str, basis: Basis, components: list[str], dial: float) -> ComponentSplit:
    """The split whose P is the named components ("L<layer>:<index>") of the basis, their writes
    scaled by 1 + dial (the scaling dial, paper_tests.py docstring; 0 = the decomposition as is)."""
    assert dial >= 0, dial
    layer_of = {m: int(m.split(".")[2]) for m in basis.U}
    chosen: dict[str, list[int]] = {m: [] for m in basis.U}
    for nm in components:
        lay, c = nm.removeprefix("L").split(":")
        (m,) = [m for m in basis.U if layer_of[m] == int(lay)]
        assert int(c) < basis.U[m].shape[0] and int(c) not in chosen[m], nm
        chosen[m].append(int(c))
    idx = {
        m: torch.tensor(cs, dtype=torch.long, device=basis.U[m].device) for m, cs in chosen.items()
    }
    return ComponentSplit(
        name,
        {m: (1 + dial) * basis.U[m][idx[m]] for m in basis.U},
        {m: basis.V[m][:, idx[m]] for m in basis.U},
        tuple(components),
    )


SpanKind = Literal["readspan", "writespan", "bothspan", "randspan", "pcaspan", "diffspan"]
SPAN_KINDS: tuple[SpanKind, ...] = (
    "readspan",
    "writespan",
    "bothspan",
    "randspan",
    "pcaspan",
    "diffspan",
)


def _orth(a: Float[Tensor, "d k"]) -> Float[Tensor, "d k"]:
    """Orthonormal basis of the column span of a: (d, k') with k' <= k, dropping directions whose
    singular value is below 1e-6 of the largest (dead components of an "all components" set have
    near-zero vectors). k may be 0."""
    if a.shape[1] == 0:
        return a
    # on the CPU: cusolver failed to create its handle when the GPU was shared with other jobs
    # (26-11-15), and these matrices are small
    u, sv, _ = torch.linalg.svd(a.cpu(), full_matrices=False)
    return u[:, sv > 1e-6 * sv.max()].contiguous().to(a.device)


def span_split(
    kind: SpanKind,
    base: ComponentSplit,
    W: dict[str, Float[Tensor, "d_out d_in"]],
    pca_V: dict[str, Float[Tensor, "d_in C"]] | None,
    seed: int,
    diff_Q: dict[str, Float[Tensor, "d_out K"]] | None = None,
) -> ComponentSplit:
    """Model-anchored splits with the dimensions of a component set S (base): P is the ORIGINAL
    weight matrix W restricted to a subspace, so every edit of P is an edit of what the model itself
    writes (a swap of P is a subspace interchange intervention on the unsplit model). Per module, with
    k = |S| in that module, Q_V an orthonormal basis of span{v_c : c in S} (reads) and Q_U of
    span{u_c : c in S} (writes):
      readspan   P = W Q_V Q_V^T      what W writes in response to the input directions S reads
      writespan  P = Q_U Q_U^T W      the part of W's output in the directions S writes to
      bothspan   P = Q_U Q_U^T W Q_V Q_V^T   W compressed to S's read and write spans
      randspan   P = Q Q^T W, Q a random k-dim subspace of the output space (CPU generator, seed)
      pcaspan    P = W Q_k Q_k^T, Q_k the top-k principal directions of the module input (pca_V: the
                 data-aligned basis, columns in ascending eigenvalue order)
      diffspan   P = Q_k Q_k^T W, Q_k the top-k principal directions of the site output's difference
                 between TRAIN twins (diff_Q, ascending order): the simplest truth-specific linear
                 baseline of the same size, fitted without any decomposition
    Why: a component set whose own swap moves the verdict while these do not gets its effect from
    gains W does not have along its own directions (a re-implementation); randspan and pcaspan say
    what a subspace of the same size does without any training. Returned as a ComponentSplit
    (p = (x @ V) @ U), so the hooks are the ones every other test uses."""
    U: dict[str, Tensor] = {}
    V: dict[str, Tensor] = {}
    for m in base.U:
        w = W[m]
        k = base.U[m].shape[0]
        assert base.V[m].shape == (w.shape[1], k) and base.U[m].shape == (k, w.shape[0])
        match kind:
            case "readspan":
                qv = _orth(base.V[m])
                U[m], V[m] = (w @ qv).T.contiguous(), qv
            case "writespan":
                qu = _orth(base.U[m].T)
                U[m], V[m] = qu.T.contiguous(), w.T @ qu
            case "bothspan":
                qv, qu = _orth(base.V[m]), _orth(base.U[m].T)
                U[m], V[m] = (qu @ (qu.T @ (w @ qv))).T.contiguous(), qv
            case "randspan":
                # drawn on the CPU so the subspace is the same on every GPU (as T3's random subspaces)
                g = torch.Generator().manual_seed(seed * 1000 + int(m.split(".")[2]))
                qu = _orth(torch.randn(w.shape[0], k, generator=g).to(w.device))
                U[m], V[m] = qu.T.contiguous(), w.T @ qu
            case "pcaspan":
                assert pca_V is not None and k <= pca_V[m].shape[1], (m, k)
                qv = pca_V[m][:, pca_V[m].shape[1] - k :]
                U[m], V[m] = (w @ qv).T.contiguous(), qv
            case "diffspan":
                assert diff_Q is not None and k <= diff_Q[m].shape[1], (m, k)
                qu = diff_Q[m][:, diff_Q[m].shape[1] - k :]
                U[m], V[m] = qu.T.contiguous(), w.T @ qu
        kk = U[m].shape[0]  # < k only if the set's vectors are linearly dependent
        assert kk <= k and U[m].shape == (kk, w.shape[0]) and V[m].shape == (w.shape[1], kk), m
    suffix = f"{kind}{seed}" if kind == "randspan" else kind
    return ComponentSplit(f"{base.name}-{suffix}", U, V, base.components)


def part_of(split: Split, part: Part, m: str, x: Tensor, y: Tensor) -> Tensor:
    """The edited part q of the output y = W x of module m (fp32, (b, pos, d_out))."""
    match split:
        case SiteSplit():
            assert part == "P", "the site has no R part (R = 0)"
            return y
        case ComponentSplit(U=U, V=V):
            p = (x @ V[m]) @ U[m]
            return p if part == "P" else y - p


# ---- interventions ----


@dataclass(frozen=True)
class Zero:
    pass


@dataclass(frozen=True)
class Mean:
    grouping: GroupingName


@dataclass(frozen=True)
class TwinAverage:
    pass


@dataclass(frozen=True)
class Swap:
    pass


@dataclass(frozen=True)
class SwapClamped:
    pass


@dataclass(frozen=True)
class Shift:
    grouping: GroupingName
    alpha: float


Intervention = Zero | Mean | TwinAverage | Swap | SwapClamped | Shift


def parse_intervention(spec: str) -> Intervention:
    """'zero', 'mean:<grouping>', 'twin_average', 'swap', 'swap_clamped',
    'shift:<grouping>:<alpha>'."""
    match spec.split(":"):
        case ["zero"]:
            return Zero()
        case ["mean", g] if g in GROUPINGS:
            return Mean(g)
        case ["twin_average"]:
            return TwinAverage()
        case ["swap"]:
            return Swap()
        case ["swap_clamped"]:
            return SwapClamped()
        case ["shift", g, a] if g in GROUPINGS:
            return Shift(g, float(a))
        case _:
            raise AssertionError(f"unknown intervention {spec!r}")


def intervention_name(iv: Intervention) -> str:
    match iv:
        case Zero():
            return "zero"
        case Mean(grouping=g):
            return f"mean:{g}"
        case TwinAverage():
            return "twin_average"
        case Swap():
            return "swap"
        case SwapClamped():
            return "swap_clamped"
        case Shift(grouping=g, alpha=a):
            return f"shift:{g}:{a:g}"


def needs_twin(iv: Intervention) -> bool:
    return isinstance(iv, TwinAverage | Swap | SwapClamped)


def grouping_of(iv: Intervention) -> GroupingName | None:
    match iv:
        case Mean(grouping=g) | Shift(grouping=g):
            return g
        case _:
            return None


# ---- position groups ----


def group_ids(
    grouping: GroupingName,
    dec: Tensor,
    obj: Tensor,
    norm: Tensor,
    median: float,
) -> Tensor:
    """(b, pos) group index per position, -1 outside the statement tokens.
    dec: (b, pos) statement positions; obj: (b,) index of the object's first token; norm: (b, pos)
    clean output norm at this site; median: the site's train median norm over statement positions."""
    t = torch.arange(dec.shape[1], device=dec.device)[None, :]
    first = dec.float().argmax(1)[:, None]  # statement tokens are contiguous from the first one
    last = dec.shape[1] - 1 - dec.flip(1).float().argmax(1)[:, None]
    assert bool((dec.sum(1) == (last - first + 1).squeeze(1)).all()), "non-contiguous statement"
    g = torch.full(dec.shape, -1, dtype=torch.long, device=dec.device)
    match grouping:
        case "scan3":
            g = torch.where(t > first, 1, g)
            g = torch.where(t == first, 0, g)
            g = torch.where(t == last, 2, g)
        case "role":
            g = torch.where(t > first, 2, g)
            g = torch.where(t >= obj[:, None], 3, g)
            g = torch.where(t == first, 1, g)
            g = torch.where(t == last, 4, g)
            g = torch.where(norm > MASSIVE_FACTOR * median, 0, g)
    return torch.where(dec, g, -1)


# ---- train statistics ----


@dataclass(frozen=True)
class GroupStats:
    """Per module: the train mean of the edited part per group, and the false-minus-true difference of
    the means over statements that have a twin; nan rows for groups without positions."""

    mean: dict[str, Float[Tensor, "G d_out"]]
    diff: dict[str, Float[Tensor, "G d_out"]]
    count: dict[str, Tensor]  # (G,) positions per group, all statements
    count_true: dict[str, Tensor]  # (G,) positions per group, true statements with a twin
    count_false: dict[str, Tensor]


@dataclass(frozen=True)
class TrainStats:
    median: dict[str, float]  # per module: median clean output norm over train statement positions
    groups: dict[GroupingName, GroupStats]


class GroupAccumulator:
    """Running per-group sums of the edited part over the train statements, for one grouping."""

    def __init__(self, modules: list[str], n_groups: int, d_out: dict[str, int], device: str):
        z = lambda m: torch.zeros(n_groups, d_out[m], device=device, dtype=torch.float64)  # noqa: E731
        zc = lambda: torch.zeros(n_groups, device=device, dtype=torch.float64)  # noqa: E731
        self.n_groups = n_groups
        self.sum = {m: z(m) for m in modules}
        self.sum_true = {m: z(m) for m in modules}
        self.sum_false = {m: z(m) for m in modules}
        self.cnt = {m: zc() for m in modules}
        self.cnt_true = {m: zc() for m in modules}
        self.cnt_false = {m: zc() for m in modules}

    def add(self, m: str, q: Tensor, g: Tensor, label: Tensor, has_twin: Tensor) -> None:
        """q (b, pos, d), g (b, pos) group ids (-1 = skip), label / has_twin (b,) bool."""
        onehot = torch.nn.functional.one_hot(g.clamp(min=0), self.n_groups).double()
        onehot = onehot * (g >= 0)[..., None]  # (b, pos, G)
        s = torch.einsum("bpg,bpd->bgd", onehot, q.double())
        c = onehot.sum(1)  # (b, G)
        self.sum[m] += s.sum(0)
        self.cnt[m] += c.sum(0)
        t, f = (label & has_twin), (~label & has_twin)
        self.sum_true[m] += s[t].sum(0)
        self.cnt_true[m] += c[t].sum(0)
        self.sum_false[m] += s[f].sum(0)
        self.cnt_false[m] += c[f].sum(0)

    def result(self) -> GroupStats:
        def div(s: Tensor, c: Tensor) -> Tensor:
            return (s / c[:, None]).float()  # 0/0 = nan: a group without positions

        mean = {m: div(self.sum[m], self.cnt[m]) for m in self.sum}
        diff = {
            m: div(self.sum_false[m], self.cnt_false[m]) - div(self.sum_true[m], self.cnt_true[m])
            for m in self.sum
        }
        return GroupStats(mean, diff, self.cnt, self.cnt_true, self.cnt_false)


# ---- hooks ----

# (module, x, y) -> the new output (b, pos, d_out) fp32, or None to leave it unchanged; x, y fp32
SiteFn = Callable[[str, Tensor, Tensor], Tensor | None]


@contextmanager
def site_hooks(model: ComponentModel, modules: list[str], fn: SiteFn) -> Iterator[None]:
    """Within the context, a plain forward of model.target_model (model(ids), no mask_infos) calls fn
    at every listed module; used both to capture values (fn returns None) and to edit them."""
    handles = []
    for m in modules:

        def hook(
            _mod: torch.nn.Module, args: tuple[Tensor, ...], output: Tensor, m: str = m
        ) -> Tensor | None:
            new = fn(m, args[0].float(), output.float())
            return None if new is None else new.to(output.dtype)

        handles.append(model.target_model.get_submodule(m).register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@dataclass(frozen=True)
class BatchContext:
    """What an intervention needs about one batch, from the clean passes (all on the device)."""

    dec: Tensor  # (b, pos) statement positions
    groups: dict[GroupingName, dict[str, Tensor]]  # grouping -> module -> (b, pos) group ids
    sign: Tensor  # (b,) +1 where the model calls the statement true (training prompt), else -1
    own_clean: dict[str, Tensor] | None  # module -> clean q(x), for TwinAverage and SwapClamped
    own_y: dict[str, Tensor] | None  # module -> clean y(x), for SwapClamped
    twin_clean: dict[str, Tensor] | None  # module -> clean q(x'), for TwinAverage and Swap


def edit_fn(
    split: Split, part: Part, iv: Intervention, ctx: BatchContext, stats: TrainStats | None
) -> SiteFn:
    """The SiteFn of one intervention on one batch: y' = y - q + q' at the statement positions."""

    def by_group(table: dict[str, Tensor], m: str, g: Tensor) -> Tensor:
        vals = table[m][g.clamp(min=0)]  # (b, pos, d)
        used = vals[g >= 0]
        assert bool(torch.isfinite(used).all()), (
            f"{m}: a position falls in a group without train data"
        )
        return vals

    def fn(m: str, x: Tensor, y: Tensor) -> Tensor:
        q = part_of(split, part, m, x, y)
        match iv:
            case Zero():
                target = torch.zeros_like(q)
            case Mean(grouping=gr):
                assert stats is not None
                target = by_group(stats.groups[gr].mean, m, ctx.groups[gr][m])
            case TwinAverage():
                assert ctx.own_clean is not None and ctx.twin_clean is not None
                target = (ctx.own_clean[m] + ctx.twin_clean[m]) / 2
            case Swap():
                assert ctx.twin_clean is not None
                target = ctx.twin_clean[m]
            case SwapClamped():
                assert ctx.twin_clean is not None and ctx.own_clean is not None
                assert ctx.own_y is not None
                # y' = q(x') + [y(x) - q(x)]: target is chosen so that y - q + target gives exactly that
                target = ctx.twin_clean[m] + (ctx.own_y[m] - ctx.own_clean[m]) - (y - q)
            case Shift(grouping=gr, alpha=a):
                assert stats is not None
                d = by_group(stats.groups[gr].diff, m, ctx.groups[gr][m])
                target = q + a * ctx.sign[:, None, None] * d
        assert target.shape == q.shape, (target.shape, q.shape)
        return torch.where(ctx.dec[..., None], y - q + target, y)

    return fn


def describe_stats(stats: TrainStats, modules: list[str]) -> list[str]:
    """Observe & Verify: per module and grouping, positions per group and the norms of the means and
    differences, so empty or degenerate groups are visible before anything is applied."""
    lines = []
    for m in modules:
        lines.append(f"{m}: median clean norm {stats.median[m]:.2f}")
        for gr, gs in stats.groups.items():
            names = SCAN3_GROUPS if gr == "scan3" else ROLE_GROUPS
            cells = []
            for i, nm in enumerate(names):
                cells.append(
                    f"{nm} n={int(gs.count[m][i])} (t {int(gs.count_true[m][i])}/f "
                    f"{int(gs.count_false[m][i])}) |mean| {float(gs.mean[m][i].norm()):.2f} "
                    f"|diff| {float(gs.diff[m][i].norm()):.3f}"
                )
            lines.append(f"  {gr}: " + "; ".join(cells))
    return lines


def to_numpy(stats: TrainStats) -> dict[str, np.ndarray]:
    """The statistics as flat arrays for the npz (so a run's means and directions can be inspected)."""
    out: dict[str, np.ndarray] = {}
    for gr, gs in stats.groups.items():
        for m in gs.mean:
            out[f"stats__{gr}__{m}__mean"] = gs.mean[m].cpu().numpy()
            out[f"stats__{gr}__{m}__diff"] = gs.diff[m].cpu().numpy()
            out[f"stats__{gr}__{m}__count"] = gs.count[m].cpu().numpy()
    for m, v in stats.median.items():
        out[f"stats__median__{m}"] = np.array(v)
    return out
