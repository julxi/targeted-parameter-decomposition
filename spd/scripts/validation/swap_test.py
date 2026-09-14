"""Swap pairs of components' U vectors (rescaled by their mean activations) and test the result.

Builds a modified decomposed model where, for each requested swap, the output directions of a pair
of components in the same matrix are swapped. Each swap is rescaled by the ratio of its components'
mean inner activations (V^T x) so the post-swap output magnitudes match what the other component
used to produce. The swapped model is then evaluated on either the LM target prompts (default) or,
with `--nontarget`, `--n-batches` batches of the nontarget dataset; for each (prompt, pos) the
script stores the orig/swapped predictions and the probability of both task target tokens.

Usage:
    python -m spd.scripts.validation.swap_test <model_path> <alive_components_tsv> \
        9:attn.v_proj:52/35 [9:mlp.down_proj:47/48 ...] \
        --target-a=" np" --target-b=" pd" \
        [--nontarget] [--n-batches=1] [--prompts=PATH] [--batch-size=N] [--output=PATH]
"""

import csv
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.scripts.validation.common import (
    build_lm_loader,
    build_module_lookup,
    escape_tsv_value,
    is_prompt_task,
    iterate_input_ids,
    load_spd_run,
    resolve_task_config,
)
from spd.spd_types import ModelPath


@dataclass
class SwapSpec:
    module_name: str
    a_component: int
    b_component: int
    a_mean: float
    b_mean: float


def _parse_swap(raw: str) -> tuple[int, str, int, int]:
    """Parse a swap of the form '<layer>:<matrix>:<a>/<b>'."""
    parts = raw.split(":")
    assert len(parts) == 3, f"Swap {raw!r} must be '<layer>:<matrix>:<a>/<b>'"
    layer_s, matrix, pair = parts
    a_s, sep, b_s = pair.partition("/")
    assert sep == "/" and a_s and b_s, f"Swap {raw!r} must have the form '<layer>:<matrix>:<a>/<b>'"
    return int(layer_s), matrix, int(a_s), int(b_s)


def _read_mean_activations(
    alive_components_path: Path, needed: set[tuple[int, str, int]]
) -> dict[tuple[int, str, int], float]:
    """Look up mean_activation for every (layer, matrix, component) triple in one TSV scan."""
    found: dict[tuple[int, str, int], float] = {}
    with alive_components_path.open() as f:
        for record in csv.DictReader(f, delimiter="\t"):
            key = (int(record["layer"]), record["matrix"], int(record["component"]))
            if key in needed:
                found[key] = float(record["mean_activation"])
                if len(found) == len(needed):
                    break
    missing = needed - found.keys()
    assert not missing, f"Missing rows for {sorted(missing)} in {alive_components_path}"
    return found


def _single_token_id(tokenizer: PreTrainedTokenizer, text: str, label: str) -> int:
    encoded: Any = tokenizer(text, add_special_tokens=False)
    ids: list[int] = encoded["input_ids"]
    assert len(ids) == 1, f"{label} {text!r} must tokenize to exactly one token, got {ids}"
    return ids[0]


@contextmanager
def _swapped_u_vectors(spd_model: ComponentModel, specs: list[SwapSpec]) -> Generator[None]:
    """Swap U[A] and U[B] for each spec, rescaled by the activation ratio.

    All original U rows are cloned before any write, so overlapping specs are rejected rather than
    silently chained through half-modified state. Restores the original rows on exit so the model's
    state is unchanged once the `with` block is done.
    """
    touched: set[tuple[str, int]] = set()
    for spec in specs:
        for c in (spec.a_component, spec.b_component):
            key = (spec.module_name, c)
            assert key not in touched, (
                f"Overlapping swap on {spec.module_name}[{c}] — each (module, component) may "
                f"appear in at most one swap"
            )
            touched.add(key)

    saved: list[tuple[Tensor, SwapSpec, Tensor, Tensor]] = []
    for spec in specs:
        u_param = spd_model.components[spec.module_name].U
        assert spec.a_component < u_param.shape[0] and spec.b_component < u_param.shape[0], (
            f"Component indices out of range for {spec.module_name} (C={u_param.shape[0]})"
        )
        saved.append(
            (
                u_param,
                spec,
                u_param.data[spec.a_component].clone(),
                u_param.data[spec.b_component].clone(),
            )
        )

    for u_param, spec, u_a_old, u_b_old in saved:
        u_param.data[spec.a_component] = (spec.b_mean / spec.a_mean) * u_b_old
        u_param.data[spec.b_component] = (spec.a_mean / spec.b_mean) * u_a_old
    try:
        yield
    finally:
        for u_param, spec, u_a_old, u_b_old in saved:
            u_param.data[spec.a_component] = u_a_old
            u_param.data[spec.b_component] = u_b_old


def _build_full_mask_infos(
    spd_model: ComponentModel,
    weight_deltas: dict[str, Tensor],
    batch_shape: tuple[int, ...],
    device: torch.device,
) -> dict[str, ComponentsMaskInfo]:
    """All-ones component masks + delta enabled everywhere, using the pre-swap weight deltas."""
    delta_mask = torch.ones(batch_shape, device=device)
    component_masks = {
        name: torch.ones((*batch_shape, C), device=device)
        for name, C in spd_model.module_to_c.items()
    }
    deltas_and_masks = {name: (weight_deltas[name], delta_mask) for name in weight_deltas}
    return make_mask_infos(component_masks, weight_deltas_and_masks=deltas_and_masks)


def _probs_from_logits(
    logits: Tensor, target_a_id: int, target_b_id: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (log_probs, pred, pred_prob, p_target_a, p_target_b)."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    pred_prob, pred = probs.max(dim=-1)
    return log_probs, pred, pred_prob, probs[..., target_a_id], probs[..., target_b_id]


_CORE_FIELDS = [
    "prompt",
    "pos",
    "token",
    "token_str",
    "orig_pred",
    "orig_pred_str",
    "orig_prob",
    "swapped_pred",
    "swapped_pred_str",
    "swapped_prob",
    "p_target_a_orig",
    "p_target_a_swapped",
    "p_target_b_orig",
    "p_target_b_swapped",
    "kl",
]


def _make_decoder(tokenizer: PreTrainedTokenizer) -> Callable[[int], str]:
    """Memoised, TSV-safe single-token decoder."""
    cache: dict[int, str] = {}

    def decode(tid: int) -> str:
        s = cache.get(tid)
        if s is None:
            s = escape_tsv_value(tokenizer.decode([tid]))
            cache[tid] = s
        return s

    return decode


def _write_rows(
    writer: "csv.DictWriter[str]",
    batch: Tensor,
    orig_logits: Tensor,
    swapped_logits: Tensor,
    target_a_id: int,
    target_b_id: int,
    decode: Callable[[int], str],
    prompt_offset: int,
    batch_idx: int | None,
) -> None:
    """Compute per-(prompt, pos) rows from the two logit tensors and write them out."""
    orig_log_probs, orig_pred, orig_prob, p_a_orig, p_b_orig = _probs_from_logits(
        orig_logits, target_a_id, target_b_id
    )
    swapped_log_probs, swapped_pred, swapped_prob, p_a_swapped, p_b_swapped = _probs_from_logits(
        swapped_logits, target_a_id, target_b_id
    )
    kl = (orig_log_probs.exp() * (orig_log_probs - swapped_log_probs)).sum(dim=-1)

    batch_size, seq_len = batch.shape
    batch_cpu = batch.cpu().tolist()
    orig_pred_cpu = orig_pred.cpu().tolist()
    orig_prob_cpu = orig_prob.cpu().tolist()
    swapped_pred_cpu = swapped_pred.cpu().tolist()
    swapped_prob_cpu = swapped_prob.cpu().tolist()
    p_a_orig_cpu = p_a_orig.cpu().tolist()
    p_b_orig_cpu = p_b_orig.cpu().tolist()
    p_a_swapped_cpu = p_a_swapped.cpu().tolist()
    p_b_swapped_cpu = p_b_swapped.cpu().tolist()
    kl_cpu = kl.cpu().tolist()
    extra = {"batch_idx": batch_idx} if batch_idx is not None else {}

    for b in range(batch_size):
        prompt_idx = prompt_offset + b
        for t in range(seq_len):
            token_id = batch_cpu[b][t]
            orig_id = orig_pred_cpu[b][t]
            swapped_id = swapped_pred_cpu[b][t]
            writer.writerow(
                {
                    **extra,
                    "prompt": prompt_idx,
                    "pos": t,
                    "token": token_id,
                    "token_str": decode(token_id),
                    "orig_pred": orig_id,
                    "orig_pred_str": decode(orig_id),
                    "orig_prob": orig_prob_cpu[b][t],
                    "swapped_pred": swapped_id,
                    "swapped_pred_str": decode(swapped_id),
                    "swapped_prob": swapped_prob_cpu[b][t],
                    "p_target_a_orig": p_a_orig_cpu[b][t],
                    "p_target_a_swapped": p_a_swapped_cpu[b][t],
                    "p_target_b_orig": p_b_orig_cpu[b][t],
                    "p_target_b_swapped": p_b_swapped_cpu[b][t],
                    "kl": kl_cpu[b][t],
                }
            )


def _run_pass(
    spd_model: ComponentModel,
    iterator: Iterator[Tensor],
    n_batches: int,
    weight_deltas: dict[str, Tensor],
    specs: list[SwapSpec],
    target_a_id: int,
    target_b_id: int,
    decode: Callable[[int], str],
    out_path: Path,
    device: torch.device,
    include_batch_idx: bool,
) -> None:
    """Run `n_batches` through orig + swapped, streaming rows to `out_path`.

    `mask_infos` is built from the first batch's shape and reused (DataLoader uses `drop_last=True`
    and `StaticBatchLoader` yields the same cached batch, so shapes are stable across the pass).
    """
    fieldnames = (["batch_idx"] if include_batch_idx else []) + _CORE_FIELDS
    first_shape: tuple[int, ...] | None = None
    mask_infos: dict[str, ComponentsMaskInfo] | None = None
    batches_done = 0

    with out_path.open("w", newline="") as f, torch.no_grad():
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for batch_idx, batch in zip(
            tqdm(range(n_batches), desc=out_path.name), iterator, strict=False
        ):
            batches_done = batch_idx + 1
            assert batch.ndim == 2, f"Expected (batch, seq), got {tuple(batch.shape)}"
            shape = tuple(batch.shape)
            if first_shape is None:
                first_shape = shape
                mask_infos = _build_full_mask_infos(spd_model, weight_deltas, shape, device)
            else:
                assert shape == first_shape, (
                    f"Batch shape changed mid-pass: got {shape}, expected {first_shape}"
                )
            assert mask_infos is not None

            orig_logits = spd_model(batch)
            assert isinstance(orig_logits, Tensor)

            with _swapped_u_vectors(spd_model, specs):
                swapped_logits = spd_model(batch, mask_infos=mask_infos)
                assert isinstance(swapped_logits, Tensor)

            _write_rows(
                writer=writer,
                batch=batch,
                orig_logits=orig_logits,
                swapped_logits=swapped_logits,
                target_a_id=target_a_id,
                target_b_id=target_b_id,
                decode=decode,
                prompt_offset=batch_idx * first_shape[0],
                batch_idx=batch_idx if include_batch_idx else None,
            )
    if batches_done < n_batches:
        logger.warning(
            f"Loader exhausted after {batches_done}/{n_batches} batches (dataset too small?)"
        )


def _spec_slug(spec: SwapSpec) -> str:
    return f"{spec.module_name.replace('.', '_')}_c{spec.a_component}_c{spec.b_component}"


def swap_test(
    model_path: ModelPath,
    alive_components_path: str,
    *swaps: str,
    target_a: str,
    target_b: str,
    nontarget: bool = False,
    n_batches: int = 1,
    prompts: str | None = None,
    split: str | None = None,
    batch_size: int | None = None,
    output: str | None = None,
) -> Path:
    """Run the swap test on target prompts, or on nontarget batches if `--nontarget` is set."""
    assert len(swaps) > 0, "At least one swap argument is required"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spd_model, config, run_dir = load_spd_run(model_path)
    spd_model = spd_model.to(device)

    module_lookup = build_module_lookup(spd_model.target_module_paths)

    parsed = [_parse_swap(s) for s in swaps]
    for layer, matrix, a, b in parsed:
        assert a != b, f"Swap {layer}:{matrix}:{a}/{b} has identical components"
        assert (layer, matrix) in module_lookup, (
            f"No decomposed module matches (layer={layer}, matrix={matrix!r}). "
            f"Available: {sorted(module_lookup.keys())}"
        )

    needed = {(layer, matrix, c) for layer, matrix, a, b in parsed for c in (a, b)}
    means = _read_mean_activations(Path(alive_components_path).expanduser(), needed)

    specs: list[SwapSpec] = []
    for layer, matrix, a, b in parsed:
        a_mean = means[(layer, matrix, a)]
        b_mean = means[(layer, matrix, b)]
        assert a_mean != 0 and b_mean != 0, (
            f"Mean activations must be non-zero for {layer}:{matrix}:{a}/{b}; "
            f"got a_mean={a_mean}, b_mean={b_mean}"
        )
        specs.append(
            SwapSpec(
                module_name=module_lookup[(layer, matrix)],
                a_component=a,
                b_component=b,
                a_mean=a_mean,
                b_mean=b_mean,
            )
        )

    assert config.tokenizer_name is not None, "config.tokenizer_name is required"
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    target_a_id = _single_token_id(tokenizer, target_a, "target_a")
    target_b_id = _single_token_id(tokenizer, target_b, "target_b")
    decode = _make_decoder(tokenizer)

    for spec in specs:
        logger.info(
            f"Swap on {spec.module_name}: "
            f"A={spec.a_component} (mean={spec.a_mean:.4g}); "
            f"B={spec.b_component} (mean={spec.b_mean:.4g})"
        )
    logger.info(f"Target probs tracked: A={target_a!r}={target_a_id}, B={target_b!r}={target_b_id}")

    task_config = resolve_task_config(
        config,
        use_nontarget=nontarget,
        prompts_override=None if nontarget else prompts,
        split_override=split,
    )
    if not nontarget:
        assert is_prompt_task(task_config), (
            "swap_test target mode requires a prompt-based task (config.task_config.prompts_file)"
        )
    iterator = iterate_input_ids(
        build_lm_loader(task_config, config, batch_size_override=batch_size), device
    )
    n_to_run = 1 if not nontarget else n_batches

    slug = "__".join(_spec_slug(s) for s in specs)
    default_name = f"swap_test_{slug}_nontarget.tsv" if nontarget else f"swap_test_{slug}.tsv"
    out_path = Path(output).expanduser() if output else run_dir / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weight_deltas = spd_model.calc_weight_deltas()

    _run_pass(
        spd_model=spd_model,
        iterator=iterator,
        n_batches=n_to_run,
        weight_deltas=weight_deltas,
        specs=specs,
        target_a_id=target_a_id,
        target_b_id=target_b_id,
        decode=decode,
        out_path=out_path,
        device=device,
        include_batch_idx=nontarget,
    )
    return out_path


if __name__ == "__main__":
    fire.Fire(swap_test)
