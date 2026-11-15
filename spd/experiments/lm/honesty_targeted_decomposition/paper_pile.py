"""Non-target side of the behavioural tests: what removing a part P of the decomposed down_proj
outputs does to the model's next-token predictions on the Pile (the runs' non-target data), for the
active set and for all components of several decompositions at once, next to removing the whole
sites. Topic: convos/julian/26-11-15_open_research_LOG.md. Draft cell: "P removed, other qualities,
on the Pile" (Table 3), which so far existed only for all components off, on validation, without a
site reference (SE-Pile).

Why it measures targeting: the Pile windows contain no truth judgement, so whatever a part P does
there is computation the truth verdict was not the reason for. The reference is the same removal of
the whole sites: share = KL(P removed) / KL(sites removed), 0 = P is silent off-target, 1 = P is the
whole matrix. Removal is at every position of the window (the runs' non-target loss positions), at
mask 0 for P with everything else kept (y' = y - p), like the removals of paper_tests.py.

Data: the first n_batches batches of the runs' own non-target eval loader (Pile validation windows of
nontarget_task_config.max_seq_len tokens), so the windows are the ones the training evaluation used.

Output: pile_<names>.json with, per split, the mean KL over positions (nats), its share of the site
removal's KL, and the mean of |p|^2/|y|^2 over positions and layers; plus per-window KL arrays (npz).

Usage:
  python paper_pile.py run --runs plain1=s-7152548c,ortho1=s-0c2fa0d7 --recovery_dir <dir> --out <dir>
"""

import json
from pathlib import Path

import fire
import numpy as np
import torch
from torch import Tensor

from spd.configs import LMTaskConfig
from spd.experiments.lm.honesty_targeted_decomposition.paper_common import sorted_modules
from spd.experiments.lm.honesty_targeted_decomposition.paper_inspect import (
    checkpoint_basis,
    run_id_of,
)
from spd.experiments.lm.honesty_targeted_decomposition.paper_interventions import (
    ComponentSplit,
    SiteSplit,
    Split,
    component_split,
    part_of,
    site_hooks,
)
from spd.experiments.lm.honesty_targeted_decomposition.paper_side_effects import load_run
from spd.experiments.lm.lm_decomposition import _create_lm_loaders
from spd.utils.general_utils import bf16_autocast, extract_batch_data


def run(
    runs: str | tuple[str, ...],
    recovery_dir: str,
    out: str,
    n_batches: int = 16,
    device: str = "cuda",
) -> None:
    torch.set_grad_enabled(False)
    pairs = [s.split("=") for s in (runs.split(",") if isinstance(runs, str) else runs)]
    names, ids = [p[0] for p in pairs], [p[1] for p in pairs]
    model, config, _run_id, tok = load_run(ids[0], device)
    nt = config.nontarget_task_config
    assert isinstance(nt, LMTaskConfig) and config.nontarget_eval_batch_size is not None
    autocast = config.autocast_bf16
    modules = sorted_modules(model)

    splits: dict[str, Split] = {"site": SiteSplit()}
    for name, rid in zip(names, ids, strict=True):
        basis = checkpoint_basis(rid, modules, device)
        rj = json.loads((Path(recovery_dir) / f"recovery_{run_id_of(rid)}.json").read_text())
        assert rj["run"] == rid, (rj["run"], rid)
        splits[f"{name}:active"] = component_split(
            "active", basis, rj["active_set"]["components"], 0.0
        )
        all_names = [
            f"L{int(m.split('.')[2])}:{c}" for m in basis.U for c in range(basis.U[m].shape[0])
        ]
        splits[f"{name}:all"] = component_split("all", basis, all_names, 0.0)

    _train, eval_loader = _create_lm_loaders(
        task_config=nt,
        tokenizer_name=config.tokenizer_name,
        train_batch_size=config.nontarget_eval_batch_size,
        eval_batch_size=config.nontarget_eval_batch_size,
        seed=config.seed,
        dist_state=None,
        stimulus=None,
    )
    kl: dict[str, list[np.ndarray]] = {s: [] for s in splits}
    energy: dict[str, list[float]] = {s: [] for s in splits}
    it = iter(eval_loader)
    for b in range(n_batches):
        ids_b = extract_batch_data(next(it)).to(device)
        if b == 0:  # Observe & Verify: the first window as text
            print(f"Pile batch shape {tuple(ids_b.shape)}; first window: {tok.decode(ids_b[0])!r}")
        with bf16_autocast(enabled=autocast):
            lp0 = torch.log_softmax(model(ids_b).float(), -1)
        for sname, sp in splits.items():
            ratios: list[float] = []

            def remove(
                m: str, x: Tensor, y: Tensor, sp: Split = sp, ratios: list[float] = ratios
            ) -> Tensor:
                p = part_of(sp, "P", m, x, y)
                ratios.append(float((p.pow(2).sum(-1) / y.pow(2).sum(-1).clamp(min=1e-12)).mean()))
                return y - p

            with site_hooks(model, modules, remove), bf16_autocast(enabled=autocast):
                lp = torch.log_softmax(model(ids_b).float(), -1)
            kl[sname].append((lp0.exp() * (lp0 - lp)).sum(-1).mean(1).cpu().numpy())  # per window
            assert len(ratios) == len(modules)
            energy[sname].append(float(np.mean(ratios)))
            del lp
        del lp0
    kl_site = float(np.concatenate(kl["site"]).mean())
    summary: dict[str, dict[str, float]] = {}
    for sname, sp in splits.items():
        k = float(np.concatenate(kl[sname]).mean())
        summary[sname] = {
            "kl": k,
            "share_of_site": k / kl_site,
            # edited-pass ratio: later layers see the removal upstream (as in every removal test)
            "energy_share": float(np.mean(energy[sname])),
            "n_components": float(len(sp.components)) if isinstance(sp, ComponentSplit) else -1.0,
        }
        print(f"  {sname:16s} KL {k:.4f}  share of site {k / kl_site:.3f}  |p|^2/|y|^2 "
              f"{summary[sname]['energy_share']:.3f}")  # fmt: skip
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pile_{'+'.join(names)}"
    n_windows = len(np.concatenate(kl["site"]))
    (out_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "runs": dict(zip(names, ids, strict=True)),
                "n_windows": n_windows,
                "window_len": nt.max_seq_len,
                "splits": summary,
            },
            indent=1,
        )  # fmt: skip
    )
    np.savez(out_dir / f"{stem}.npz", **{f"kl__{s}": np.concatenate(v) for s, v in kl.items()})  # pyright: ignore[reportArgumentType]
    print(f"wrote {out_dir / stem}.{{json,npz}} ({n_windows} windows)")


if __name__ == "__main__":
    fire.Fire({"run": run})
