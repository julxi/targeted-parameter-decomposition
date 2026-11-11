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
# median norm (the layer scan's norm check used the same factor)
MASSIVE_FACTOR = 10.0


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


def component_split(name: str, basis: Basis, components: list[str]) -> ComponentSplit:
    """The split whose P is the named components ("L<layer>:<index>") of the basis."""
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
        {m: basis.U[m][idx[m]] for m in basis.U},
        {m: basis.V[m][:, idx[m]] for m in basis.U},
        tuple(components),
    )


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
class Shift:
    grouping: GroupingName
    alpha: float


Intervention = Zero | Mean | TwinAverage | Swap | Shift


def parse_intervention(spec: str) -> Intervention:
    """'zero', 'mean:<grouping>', 'twin_average', 'swap', 'shift:<grouping>:<alpha>'."""
    match spec.split(":"):
        case ["zero"]:
            return Zero()
        case ["mean", g] if g in GROUPINGS:
            return Mean(g)
        case ["twin_average"]:
            return TwinAverage()
        case ["swap"]:
            return Swap()
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
        case Shift(grouping=g, alpha=a):
            return f"shift:{g}:{a:g}"


def needs_twin(iv: Intervention) -> bool:
    return isinstance(iv, TwinAverage | Swap)


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
    own_clean: dict[str, Tensor] | None  # module -> clean q(x), for TwinAverage
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
