"""Ablate a fixed set of components and measure per-language effects on model output.

Takes either a components file (one `<layer>:<matrix>:<component>` per line; `#` comments
and blank lines ignored) OR a comma-separated spec string (`1:mlp.down_proj:248,3:mlp.c_fc:98`)
directly on the CLI. Ablates every listed component simultaneously (their masks set to 0, every
other component and the delta kept fully enabled), and runs sequences from several programming
languages through both the original and the ablated model.

Writes two TSVs:
- per-position KL/predictions (one row per `(lang, prompt, pos)`).
- per-sequence mean KL and next-token cross-entropy of both the original and the ablated model
  (one row per `(lang, prompt)`); useful for comparing the ablated model's CE against the
  original training-loss curve to estimate how much training compute the ablation undoes.

With `--invert`, the listed components are the only ones kept on: every other component
*and* the delta are disabled. This isolates a model that can only run the listed components
(e.g. only the alive components from a CSS decomposition).

Data sources:
- `css` — `Antovigo/pile-css-chunks`, the pre-tokenised training data of the CSS decomposition.
- `css_bigcode` — CSS from `bigcode/the-stack-smol` with `/* ... */` comments stripped to match
  how `pile-css-chunks` was built (useful as an independent CSS sample).
- other code languages — `bigcode/the-stack-smol` (one subset per language).
- `english` — `wikitext-103-raw-v1`.

HTML files have `<style>...</style>` blocks stripped so the 'html' sample measures HTML without
embedded CSS.

Usage:
    # File-based
    python -m spd.scripts.validation.multilang_ablation <model_path> <components_txt> [...]

    # Inline (single or comma-separated)
    python -m spd.scripts.validation.multilang_ablation <model_path> 1:mlp.down_proj:248 [...]
    python -m spd.scripts.validation.multilang_ablation <model_path> \\
        "1:mlp.down_proj:248,3:mlp.c_fc:98" [...]

Optional flags:
    [--tokens-per-lang=10000] [--languages=css,html,javascript,python,c,rust,english]
    [--batch-size=N] [--seq-len=N] [--output=PATH] [--output-sequences=PATH] [--invert]
"""

import csv
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import fire
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.configs import LMTaskConfig
from spd.log import logger
from spd.models.components import make_mask_infos
from spd.scripts.validation.common import (
    build_module_lookup,
    escape_tsv_value,
    load_spd_run,
)
from spd.spd_types import ModelPath

_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

_PREPROCESSORS: dict[str, Callable[[str], str]] = {
    "html": lambda s: _STYLE_RE.sub("", s),
    "css_bigcode": lambda s: _CSS_COMMENT_RE.sub("", s),
}

_BIGCODE_SUBSETS: dict[str, str] = {
    "css_bigcode": "data/css",
    "html": "data/html",
    "javascript": "data/javascript",
    "python": "data/python",
    "c": "data/c",
    "rust": "data/rust",
}

_DEFAULT_LANGUAGES: tuple[str, ...] = (
    "css",
    "css_bigcode",
    "html",
    "javascript",
    "python",
    "c",
    "rust",
    "english",
)

FIELDS = [
    "lang",
    "prompt",
    "pos",
    "token_str",
    "orig_pred_str",
    "orig_prob",
    "ablated_pred_str",
    "ablated_prob",
    "kl",
]

SEQ_FIELDS = ["lang", "prompt", "n_positions", "mean_kl", "orig_ce", "ablated_ce"]


def _parse_component_line(raw: str) -> tuple[int, str, int]:
    """Parse a '<layer>:<matrix>:<component>' line."""
    parts = raw.split(":")
    assert len(parts) == 3, f"Component line {raw!r} must be '<layer>:<matrix>:<component>'"
    return int(parts[0]), parts[1], int(parts[2])


def _resolve_components(
    arg: str, module_lookup: dict[tuple[int, str], str]
) -> list[tuple[str, int]]:
    """Return [(module_name, component_idx), ...] from a file path or comma-separated spec.

    If `arg` names an existing file, each line is a `<layer>:<matrix>:<component>` spec (blank
    lines and `#` comments are ignored). Otherwise `arg` is treated as one or more specs
    joined by `,`.
    """
    path = Path(arg).expanduser()
    if path.is_file():
        lines = [ln.strip() for ln in path.read_text().splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith("#")]
    else:
        lines = [s.strip() for s in arg.split(",") if s.strip()]

    out: list[tuple[str, int]] = []
    for line in lines:
        layer, matrix, component = _parse_component_line(line)
        key = (layer, matrix)
        assert key in module_lookup, (
            f"No decomposed module matches layer={layer}, matrix={matrix}. "
            f"Available: {sorted(module_lookup.keys())}"
        )
        out.append((module_lookup[key], component))
    assert out, f"No components resolved from {arg!r}"
    return out


def _iter_tokens(lang: str, tokenizer: PreTrainedTokenizer) -> Iterator[int]:
    """Stream token IDs from a language's dataset with no inter-file separator.

    `css` reads the pre-tokenised `Antovigo/pile-css-chunks` used during CSS decomposition
    training — chunks are 512 token IDs each and we just flatten them. All other languages
    tokenise text on-the-fly with `add_special_tokens=False` and concatenate docs back-to-back,
    matching how `extract_css_from_pile.py` builds the CSS training chunks.
    """
    if lang == "css":
        ds: Any = load_dataset("Antovigo/pile-css-chunks", split="train", streaming=True)
        for row in ds:
            yield from row["input_ids"]
        return

    if lang == "english":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        content_field = "text"
    else:
        assert lang in _BIGCODE_SUBSETS, (
            f"Unknown language {lang!r}; expected one of {sorted(_BIGCODE_SUBSETS)} "
            f"or 'css' or 'english'"
        )
        ds = load_dataset(
            "bigcode/the-stack-smol",
            data_dir=_BIGCODE_SUBSETS[lang],
            split="train",
            streaming=True,
        )
        content_field = "content"

    preprocess = _PREPROCESSORS.get(lang, lambda s: s)
    for row in ds:
        text = preprocess(row[content_field])
        if not text:
            continue
        encoded: Any = tokenizer(text, add_special_tokens=False)
        ids: list[int] = encoded["input_ids"]
        if ids:
            yield from ids


def _pack_batches(
    token_iter: Iterator[int],
    batch_size: int,
    seq_len: int,
    n_batches: int,
) -> Iterator[Tensor]:
    """Pack a token stream into up to `n_batches` tensors of shape (batch_size, seq_len)."""
    tokens_per_batch = batch_size * seq_len
    buffer: list[int] = []
    emitted = 0
    for tid in token_iter:
        buffer.append(tid)
        if len(buffer) >= tokens_per_batch:
            flat = buffer[:tokens_per_batch]
            buffer = buffer[tokens_per_batch:]
            yield torch.tensor(flat, dtype=torch.long).reshape(batch_size, seq_len)
            emitted += 1
            if emitted >= n_batches:
                return


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


def multilang_ablation(
    model_path: ModelPath,
    components: str,
    tokens_per_lang: int = 10000,
    languages: str | None = None,
    batch_size: int | None = None,
    seq_len: int | None = None,
    output: str | None = None,
    output_sequences: str | None = None,
    invert: bool = False,
) -> tuple[Path, Path]:
    """Ablate the listed components and record per-position KL across programming languages.

    If `invert` is True, keep only the listed components on and disable everything else
    (other components and the delta).

    Writes two TSVs:
    - `output` (default `<run_dir>/multilang_ablation.tsv`): one row per (lang, prompt, pos).
    - `output_sequences` (default `<run_dir>/per_sequence_loss.tsv`): one row per (lang, prompt)
      with mean KL and the mean next-token cross-entropy of both models, for later comparison
      against training-loss curves.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spd_model, config, run_dir = load_spd_run(model_path)
    spd_model = spd_model.to(device)

    assert config.tokenizer_name is not None, "config.tokenizer_name is required"
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    decode = _make_decoder(tokenizer)

    assert isinstance(config.task_config, LMTaskConfig), "multilang_ablation only supports LM tasks"
    resolved_batch_size = batch_size if batch_size is not None else config.batch_size
    resolved_seq_len = seq_len if seq_len is not None else config.task_config.max_seq_len

    module_lookup = build_module_lookup(spd_model.target_module_paths)
    comps = _resolve_components(components, module_lookup)

    batch_shape = (resolved_batch_size, resolved_seq_len)
    fill_value = 0.0 if invert else 1.0
    set_value = 1.0 if invert else 0.0
    component_masks = {
        name: torch.full((*batch_shape, C), fill_value, device=device)
        for name, C in spd_model.module_to_c.items()
    }
    for module_name, component in comps:
        component_masks[module_name][..., component] = set_value
    if invert:
        deltas_and_masks = None
    else:
        weight_deltas = spd_model.calc_weight_deltas()
        delta_mask = torch.ones(batch_shape, device=device)
        deltas_and_masks = {name: (weight_deltas[name], delta_mask) for name in weight_deltas}
    mask_infos = make_mask_infos(component_masks, weight_deltas_and_masks=deltas_and_masks)

    langs = tuple(s.strip() for s in languages.split(",")) if languages else _DEFAULT_LANGUAGES

    tokens_per_batch = resolved_batch_size * resolved_seq_len
    n_batches = max(1, -(-tokens_per_lang // tokens_per_batch))

    output_path = Path(output).expanduser() if output else run_dir / "multilang_ablation.tsv"
    seq_path = (
        Path(output_sequences).expanduser()
        if output_sequences
        else run_dir / "per_sequence_loss.tsv"
    )
    for p in (output_path, seq_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    with (
        output_path.open("w", newline="") as f,
        seq_path.open("w", newline="") as seq_f,
        torch.no_grad(),
    ):
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        seq_writer = csv.DictWriter(seq_f, fieldnames=SEQ_FIELDS, delimiter="\t")
        writer.writeheader()
        seq_writer.writeheader()

        for lang in langs:
            token_iter = _iter_tokens(lang, tokenizer)
            batch_iter = _pack_batches(token_iter, resolved_batch_size, resolved_seq_len, n_batches)

            rows_written = 0
            prompt_offset = 0
            for batch in tqdm(batch_iter, desc=lang, total=n_batches):
                batch = batch.to(device)
                orig_logits = spd_model(batch)
                ablated_logits = spd_model(batch, mask_infos=mask_infos)
                assert isinstance(orig_logits, Tensor) and isinstance(ablated_logits, Tensor)

                orig_log_probs = F.log_softmax(orig_logits, dim=-1)
                orig_probs = orig_log_probs.exp()
                orig_prob_max, orig_pred = orig_probs.max(dim=-1)

                abl_log_probs = F.log_softmax(ablated_logits, dim=-1)
                abl_prob_max, abl_pred = abl_log_probs.exp().max(dim=-1)

                kl = (orig_probs * (orig_log_probs - abl_log_probs)).sum(dim=-1)

                targets = batch[:, 1:]
                orig_nll = -orig_log_probs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                abl_nll = -abl_log_probs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                n_ce_positions = targets.shape[1]
                mean_kl_per_seq = kl[:, :-1].mean(dim=-1).cpu().tolist()
                orig_ce_per_seq = orig_nll.mean(dim=-1).cpu().tolist()
                abl_ce_per_seq = abl_nll.mean(dim=-1).cpu().tolist()

                batch_cpu = batch.cpu().tolist()
                orig_pred_cpu = orig_pred.cpu().tolist()
                orig_prob_cpu = orig_prob_max.cpu().tolist()
                abl_pred_cpu = abl_pred.cpu().tolist()
                abl_prob_cpu = abl_prob_max.cpu().tolist()
                kl_cpu = kl.cpu().tolist()

                bsz, slen = batch.shape
                for b in range(bsz):
                    prompt_idx = prompt_offset + b
                    for t in range(slen):
                        writer.writerow(
                            {
                                "lang": lang,
                                "prompt": prompt_idx,
                                "pos": t,
                                "token_str": decode(batch_cpu[b][t]),
                                "orig_pred_str": decode(orig_pred_cpu[b][t]),
                                "orig_prob": orig_prob_cpu[b][t],
                                "ablated_pred_str": decode(abl_pred_cpu[b][t]),
                                "ablated_prob": abl_prob_cpu[b][t],
                                "kl": kl_cpu[b][t],
                            }
                        )
                    seq_writer.writerow(
                        {
                            "lang": lang,
                            "prompt": prompt_idx,
                            "n_positions": n_ce_positions,
                            "mean_kl": mean_kl_per_seq[b],
                            "orig_ce": orig_ce_per_seq[b],
                            "ablated_ce": abl_ce_per_seq[b],
                        }
                    )
                prompt_offset += bsz
                rows_written += bsz * slen

            if rows_written == 0:
                logger.warning(f"{lang}: dataset produced no batches (empty source?)")
            else:
                logger.info(f"{lang}: wrote {rows_written} rows")

    logger.info(f"Wrote {output_path} and {seq_path}")
    return output_path, seq_path


if __name__ == "__main__":
    fire.Fire(multilang_ablation)
