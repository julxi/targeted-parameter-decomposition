"""Inspection of several decompositions at once on the twins of a statement split: how each
decomposition's component writes relate to what the ORIGINAL model writes, and to each other.
Topic: convos/julian/26-11-15_open_research_LOG.md (entry *plan*). Reads no intervention outputs and
edits nothing: one clean forward pass per twin, then linear algebra on the captured down_proj inputs x
and outputs y. Companion of the behavioural tests in paper_tests.py (same twins, same component sets).

Twins: as in paper_tests.run (true/false statement of one subject, equal token layout, different
training-prompt verdicts), each twin once as x with its partner as x'.

Fields. Per decomposed module, a "field" is a function of (x, y) with values in the output space:
  site                 y
  <run>:active:p       the active set's write at mask 1, p = sum_c (v_c . x) u_c
  <run>:active:wspan   Q_U Q_U^T y        the model's own output inside the active set's write span
  <run>:active:rspan   W Q_V Q_V^T x      the model's own response to the input directions it reads
  <run>:all:p          the write of all components of the layer (C in the draft's notation)
(Q_U, Q_V: orthonormal bases of the active set's write / read vectors in that module; the anchored
fields are the ones paper_interventions.span_split edits.) All fields are linear, so the twin
difference of a field is the field of (x' - x, y' - y).

Output (npz), Gram matrices of the fields, summed over positions, per module, twin family and
position class (pre = statement tokens before the first token in which the twins differ, obj = from
that token up to but excluding the statement's last token, end = the statement's last token):
  G_raw[m, fam, cls, i, j]  sum over positions of <f_i, f_j> / |y|^2          (own pass)
  G_dp[m, fam, cls, i, j]   sum of <df_i, df_j>, df = f(x') - f(x)             (pooled)
  G_dn[m, fam, cls, i, j]   sum of <df_i, df_j> / |dy|^2                        (per-position ratio)
  count[fam, cls]           positions behind each sum (G_raw / count = mean of per-position ratios)
Everything of interest is a ratio of entries, e.g. with p = a run's active write and y = site:
  mean |p|^2/|y|^2 = G_raw[p,p]/count; cancellation <p, r>/|y|^2 = (G_raw[p,y] - G_raw[p,p])/count;
  kappa (cancellation between twins) = (G[p,p] + G[r,r]) / G[y,y] with G[r,r] = G[y,y] - 2 G[p,y] + G[p,p];
  share of the P difference present in the model's difference = G_dp[p,y] / G_dp[p,p];
  agreement of two runs = G_dp[p_1,p_2] / sqrt(G_dp[p_1,p_1] G_dp[p_2,p_2]).
Weight-space numbers without data, per module and run (active set and all components):
  frob[m, split] = (<P, W>_F, |P|_F^2, |W|_F^2, |Q_U^T W Q_V|_F^2 (active only, else nan))
  overlap_write / overlap_read[m, i, j] = |Q_i^T Q_j|_F^2 between the active sets of runs i, j (the
  diagonal is the set's size k).

Usage:
  python paper_inspect.py run --runs plain1=s-7152548c,ortho1=s-0c2fa0d7 --recovery_dir <dir with
      recovery_<s-id>.json> --split test --out <dir>
"""

import json
from pathlib import Path
from typing import Literal

import fire
import numpy as np
import torch
from torch import Tensor

from spd.configs import LMTaskConfig, StimulusLens
from spd.experiments.lm.honesty_targeted_decomposition.paper_common import (
    Basis,
    decomposition_basis,
    load_statements,
    make_batches,
    maybe_cap,
    sorted_modules,
    verdict_prompts,
)
from spd.experiments.lm.honesty_targeted_decomposition.paper_interventions import (
    _orth,
    component_split,
    site_hooks,
)
from spd.experiments.lm.honesty_targeted_decomposition.paper_side_effects import (
    load_run,
    single_tokens,
)
from spd.experiments.lm.honesty_targeted_decomposition.paper_tests import (
    load_records,
    twin_partner,
)
from spd.experiments.lm.honesty_targeted_decomposition.probe_ci import WANDB_PREFIX
from spd.experiments.lm.prompts_dataset import _tokenize_jointly
from spd.models.component_model import SPDRunInfo
from spd.utils.general_utils import bf16_autocast

CLASSES = ("pre", "obj", "end")


def run_id_of(run: str) -> str:
    """A run is a WandB run id or the path of a local checkpoint file (as paper_side_effects.load_run);
    recovery files are named by the run id, which for a path is the checkpoint's directory name."""
    return Path(run).parent.name if Path(run).exists() else run


def checkpoint_basis(run: str, modules: list[str], device: str) -> Basis:
    """U, V of a run straight from its checkpoint (no second copy of the 7B model)."""
    info = SPDRunInfo.from_path(run if Path(run).exists() else WANDB_PREFIX + run)
    sd = torch.load(info.checkpoint_path, map_location="cpu", weights_only=True)
    key = lambda m, w: f"_components.{m.replace('.', '-')}.{w}"  # noqa: E731
    return Basis(
        run,
        {m: sd[key(m, "U")].float().to(device) for m in modules},
        {m: sd[key(m, "V")].float().to(device) for m in modules},
    )


def run(
    runs: str | tuple[str, ...],
    recovery_dir: str,
    split: Literal["val", "test"],
    out: str,
    n_per_dataset: int | None = None,
    batch_size: int = 32,
    device: str = "cuda",
    seed: int = 0,
) -> None:
    torch.set_grad_enabled(False)
    pairs = [s.split("=") for s in (runs.split(",") if isinstance(runs, str) else runs)]
    names, ids = [p[0] for p in pairs], [p[1] for p in pairs]
    assert len(set(names)) == len(names) and all(len(p) == 2 for p in pairs), pairs
    model, config, run_id, tok = load_run(ids[0], device)
    lens, task = config.target_lens, config.task_config
    assert isinstance(lens, StimulusLens) and isinstance(task, LMTaskConfig)
    assert task.prepared_datasets is not None
    autocast = config.autocast_bf16
    modules = sorted_modules(model)
    train_p = verdict_prompts(lens, tok, task.max_seq_len)[0]
    W = {m: model.target_weight(m).float() for m in modules}

    # ---- the fields ----
    first = decomposition_basis(model, run_id)
    field_names = ["site"]
    # per field and module: ("y", A) -> y @ A ; ("x", V, U) -> (x @ V) @ U
    y_maps: dict[str, dict[str, Tensor]] = {}
    x_maps: dict[str, dict[str, tuple[Tensor, Tensor]]] = {}
    Qu: dict[str, dict[str, Tensor]] = {}
    Qv: dict[str, dict[str, Tensor]] = {}
    frob = np.full((len(modules), 2 * len(names), 4), np.nan)
    split_names = []
    for r, (name, rid) in enumerate(zip(names, ids, strict=True)):
        basis = checkpoint_basis(rid, modules, device)
        if r == 0:  # the key format and dtype of the checkpoint route, against the loaded model
            assert all(torch.equal(basis.U[m], first.U[m]) for m in modules), "checkpoint != model"
            assert all(torch.equal(basis.V[m], first.V[m]) for m in modules), "checkpoint != model"
        rj = json.loads((Path(recovery_dir) / f"recovery_{run_id_of(rid)}.json").read_text())
        assert rj["run"] == rid, (rj["run"], rid)
        act = component_split("active", basis, rj["active_set"]["components"], 0.0)
        print(f"{name} ({rid}): active set {len(act.components)}, per module "
              f"{[act.U[m].shape[0] for m in modules]}")  # fmt: skip
        Qu[name] = {m: _orth(act.U[m].T) for m in modules}
        Qv[name] = {m: _orth(act.V[m]) for m in modules}
        x_maps[f"{name}:active:p"] = {m: (act.V[m], act.U[m]) for m in modules}
        y_maps[f"{name}:active:wspan"] = {m: Qu[name][m] @ Qu[name][m].T for m in modules}
        x_maps[f"{name}:active:rspan"] = {
            m: (Qv[name][m], (W[m] @ Qv[name][m]).T.contiguous()) for m in modules
        }
        x_maps[f"{name}:all:p"] = {m: (basis.V[m], basis.U[m]) for m in modules}
        field_names += [f"{name}:active:p", f"{name}:active:wspan", f"{name}:active:rspan",
                        f"{name}:all:p"]  # fmt: skip
        split_names += [f"{name}:active", f"{name}:all"]
        for mi, m in enumerate(modules):
            for j, (U, V) in enumerate(((act.U[m], act.V[m]), (basis.U[m], basis.V[m]))):
                pw = float(((W[m] @ V) * U.T).sum())  # <P, W>_F = sum_c u_c^T W v_c
                pp = float(((U @ U.T) * (V.T @ V)).sum())
                comp = (
                    float((Qu[name][m].T @ W[m] @ Qv[name][m]).pow(2).sum()) if j == 0 else np.nan
                )
                frob[mi, 2 * r + j] = (pw, pp, float(W[m].pow(2).sum()), comp)
    F = len(field_names)
    ov_w = np.array([[[float((Qu[a][m].T @ Qu[b][m]).pow(2).sum()) for b in names] for a in names]
                     for m in modules])  # fmt: skip
    ov_r = np.array([[[float((Qv[a][m].T @ Qv[b][m]).pow(2).sum()) for b in names] for a in names]
                     for m in modules])  # fmt: skip

    def fields(m: str, x: Tensor, y: Tensor) -> Tensor:
        """(F, b, pos, d_out) in the order of field_names."""
        outs = []
        for f in field_names:
            if f == "site":
                outs.append(y)
            elif f in y_maps:
                outs.append(y @ y_maps[f][m])
            else:
                V, U = x_maps[f][m]
                outs.append((x @ V) @ U)
        return torch.stack(outs)

    # ---- twins (as paper_tests.run) ----
    st = maybe_cap(load_statements(split, task.prepared_datasets), n_per_dataset, seed)
    rec = load_records(st, task.prepared_datasets)
    tid = torch.tensor(single_tokens(tok, ("1", "0")), device=device)
    score = np.zeros(len(st))
    for bt in make_batches(list(st.texts), tok, task.loss_positions, train_p, 64, device):
        with bf16_autocast(enabled=autocast):
            logits = model(bt.ids)
        lp = torch.log_softmax(logits[torch.arange(len(bt.end), device=device), bt.end].float(), -1)
        score[bt.rows] = (lp[:, tid[0]] - lp[:, tid[1]]).cpu().numpy()
    partner = twin_partner(st, rec)
    n_tok = np.array([len(_tokenize_jointly([t], lens.stimulus, tok)[0][0]) for t in st.texts])
    ok = partner >= 0
    ok &= np.where(ok, n_tok == n_tok[partner], False)
    ok &= np.where(ok, (score > 0) != (score[partner] > 0), False)
    rows = np.flatnonzero(ok)
    fam_names = sorted({str(st.families[i]) for i in rows})
    fam_idx = np.array([fam_names.index(str(st.families[i])) for i in rows])
    print(f"{split}: {len(st)} statements, {len(rows)} twin rows in families "
          f"{ {f: int((fam_idx == i).sum()) for i, f in enumerate(fam_names)} }")  # fmt: skip
    own_b = make_batches([st.texts[i] for i in rows], tok, task.loss_positions, train_p, batch_size, device)  # fmt: skip
    twin_b = make_batches([st.texts[partner[i]] for i in rows], tok, task.loss_positions, train_p, batch_size, device)  # fmt: skip

    M, NF, NC = len(modules), len(fam_names), len(CLASSES)
    G_raw = torch.zeros(M, NF, NC, F, F, dtype=torch.float64, device=device)
    G_dp, G_dn = torch.zeros_like(G_raw), torch.zeros_like(G_raw)
    count = torch.zeros(NF, NC, dtype=torch.float64, device=device)
    shown = False
    for bt, tw in zip(own_b, twin_b, strict=True):
        assert tw.ids.shape == bt.ids.shape and bool((tw.dec == bt.dec).all()), "twin layout"
        cap: dict[str, dict[str, tuple[Tensor, Tensor]]] = {"own": {}, "twin": {}}
        for which, ids_ in (("own", bt.ids), ("twin", tw.ids)):

            def capture(
                m: str,
                x: Tensor,
                y: Tensor,
                which: str = which,
                cap: dict[str, dict[str, tuple[Tensor, Tensor]]] = cap,
            ) -> None:
                cap[which][m] = (x, y)

            with site_hooks(model, modules, capture), bf16_autocast(enabled=autocast):
                model(ids_)
        t = torch.arange(bt.dec.shape[1], device=device)[None, :]
        differs = (bt.ids != tw.ids) & bt.dec
        assert bool(differs.any(1).all()), "twins with identical statement tokens"
        first_diff = differs.float().argmax(1)[:, None]
        last = bt.dec.shape[1] - 1 - bt.dec.flip(1).float().argmax(1)[:, None]
        cls = torch.stack([
            bt.dec & (t < first_diff),
            bt.dec & (t >= first_diff) & (t < last),
            bt.dec & (t == last),
        ]).double()  # (3, b, pos)  # fmt: skip
        assert bool((cls.sum(0) == bt.dec).all())
        fi = torch.tensor(fam_idx[bt.rows], device=device)
        count.index_add_(0, fi, cls.sum(2).T)
        if not shown:  # Observe & Verify: the position classes of one twin, token by token
            shown = True
            toks = [tok.decode([int(i)]) for i in bt.ids[0]]
            toks2 = [tok.decode([int(i)]) for i in tw.ids[0]]
            lab = ["pre" if cls[0, 0, p] else "obj" if cls[1, 0, p] else "end" if cls[2, 0, p] else "-"
                   for p in range(len(toks))]  # fmt: skip
            print("position classes, first twin:", list(zip(toks, toks2, lab, strict=True)))
        for mi, m in enumerate(modules):
            x, y = cap["own"][m]
            x2, y2 = cap["twin"][m]
            f_own = fields(m, x, y)
            df = fields(m, x2 - x, y2 - y)
            dy2 = (y2 - y).pow(2).sum(-1)
            # twins are identical before the first differing token (causal attention)
            pre_max = float((dy2 * cls[0]).max())
            assert pre_max < 1e-6 * float(dy2.max()), (m, pre_max)

            def gram(f: Tensor, weight: Tensor) -> Tensor:
                """(b, F, F): sum over positions of weight * <f_i, f_j>; weight (b, pos)."""
                a = (f * weight.float().sqrt()[None, :, :, None]).permute(1, 0, 2, 3).flatten(2)
                return torch.bmm(a, a.transpose(1, 2)).double()

            inv_y = 1 / y.pow(2).sum(-1).clamp(min=1e-12)
            inv_dy = 1 / dy2.clamp(min=1e-12)
            for c in range(NC):
                G_raw[mi, :, c].index_add_(0, fi, gram(f_own, cls[c] * inv_y))
                if c > 0:
                    G_dp[mi, :, c].index_add_(0, fi, gram(df, cls[c]))
                    G_dn[mi, :, c].index_add_(0, fi, gram(df, cls[c] * inv_dy))
            del f_own, df

    # ---- self-checks ----
    cnt = count[None].expand(M, NF, NC)
    assert torch.allclose(G_raw[..., 0, 0], cnt, rtol=1e-3), "site/site raw Gram != position count"
    assert torch.allclose(G_dn[:, :, 1:, 0, 0], cnt[:, :, 1:], rtol=1e-3), "site/site delta Gram"
    for name in names:
        p, w = field_names.index(f"{name}:active:p"), field_names.index(f"{name}:active:wspan")
        # the P difference lies in the write span, so <dp, Q Q^T dy> = <dp, dy>
        a, b = G_dp[..., p, w].sum(), G_dp[..., p, 0].sum()
        assert abs(float(a - b)) <= 2e-2 * max(abs(float(b)), 1e-9), (name, float(a), float(b))

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"inspect_{split}_{'+'.join(names)}"
    np.savez(
        out_dir / f"{stem}.npz",
        G_raw=G_raw.cpu().numpy(), G_dp=G_dp.cpu().numpy(), G_dn=G_dn.cpu().numpy(),
        count=count.cpu().numpy(), fields=np.array(field_names), modules=np.array(modules),
        families=np.array(fam_names), classes=np.array(CLASSES), runs=np.array(names),
        run_ids=np.array(ids), frob=frob, frob_splits=np.array(split_names),
        overlap_write=ov_w, overlap_read=ov_r, n_twin_rows=np.array(len(rows)),
    )  # fmt: skip
    print(f"wrote {out_dir / stem}.npz ({F} fields, {len(rows)} twin rows)")
    # a first look: per run, pooled over modules/families, obj + end positions
    tot = G_dp[:, :, 1:].sum((0, 1, 2))
    for name in names:
        p = field_names.index(f"{name}:active:p")
        ws, rs = (
            field_names.index(f"{name}:active:wspan"),
            field_names.index(f"{name}:active:rspan"),
        )
        yy, pp, py = float(tot[0, 0]), float(tot[p, p]), float(tot[p, 0])
        print(f"  {name:8s} |dp|^2/|dy|^2 {pp / yy:.3f}  kappa {(2 * pp + yy - 2 * py) / yy:.3f}  "
              f"present <dp,dy>/|dp|^2 {py / pp:.3f}  wspan share {float(tot[ws, ws]) / yy:.3f}  "
              f"rspan share {float(tot[rs, rs]) / yy:.3f}  cos(dp, rspan) "
              f"{float(tot[p, rs]) / (pp * float(tot[rs, rs])) ** 0.5:.3f}")  # fmt: skip


if __name__ == "__main__":
    fire.Fire({"run": run})
