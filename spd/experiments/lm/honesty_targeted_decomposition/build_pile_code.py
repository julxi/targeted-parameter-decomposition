"""Build the code-lines control dataset: short lines of source code from the Pile's GitHub subset.

Purpose: target data for a control decomposition that has (almost) no truth content, so that its
CIs can be probed for statement truth on the tiu statements next to the real tiu arms. Writes
`data/pile_code/github_lines/<version>/` with `train.jsonl`, `test.jsonl`, `manifest.yaml` (see
data/README.md).

Matched to tiu v1 on one property only: the length histogram in Qwen tokens, per split (the same
number of lines of each length as tiu has statements of that length). So per-position training
signal and sequence length are the same as for the tiu arms; content is unrelated.

Source: `test.jsonl.zst` of monology/pile-uncopyrighted, a file that neither the tiu arms'
nontarget training stream (`train/*`) nor their nontarget eval (`val.jsonl.zst`) reads. Documents
are read in file order; only `pile_set_name == "Github"`.

Line filter (heuristic, aimed at keeping code and dropping English prose, which makes factual
claims): stripped, ASCII only, not starting with a comment marker, at least 2 code-symbol
characters, no run of 4+ plain words. Lines inside block comments that lack a leading `*` can
slip through when they contain symbols; the build log prints kept and dropped examples per rule.
Train/test are split by source document, and line texts are deduplicated across both splits.

    python spd/experiments/lm/honesty_targeted_decomposition/build_pile_code.py --version v1

An existing version is never overwritten; build a new version instead.
"""

import collections
import hashlib
import json
import random
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fire
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from spd.experiments.lm.prepared_datasets import load_prepared_split
from spd.settings import REPO_ROOT

PILE_REPO = "monology/pile-uncopyrighted"
PILE_REVISION = "3be90335b66f24456a5d6659d9c8d208c0357119"
PILE_FILE = "test.jsonl.zst"
TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
TIU_DIRS = sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "data" / "tiu").glob("*/v1"))

OUT_ROOT = REPO_ROOT / "data" / "pile_code" / "github_lines"
SCRIPT_PATH = Path(__file__).resolve()

COMMENT_PREFIXES = ("//", "/*", "*", "#!", "# ", "--", "<!--", ";", "%", '"""', "'''", "REM ")
CODE_SYMBOLS = set("(){}[];=<>&|_$@\\")
PROSE_RUN = re.compile(r"(?:\b[A-Za-z]+[,.]? ){3}[A-Za-z]+\b")


def reject_reason(line: str) -> str | None:
    """Why a stripped line is not kept as a code line, or None if it is kept."""
    if not line.isascii():
        return "non_ascii"
    if line == "#" or line.startswith(COMMENT_PREFIXES):
        return "comment"
    if sum(ch in CODE_SYMBOLS for ch in line) < 2:
        return "few_symbols"
    if PROSE_RUN.search(line):
        return "prose_run"
    return None


def _is_test(doc_index: int, test_frac: float, split_seed: int) -> bool:
    digest = hashlib.sha256(f"{split_seed}/{doc_index}".encode()).digest()
    return int.from_bytes(digest[:8]) / 2**64 < test_frac


def tiu_length_histograms() -> dict[str, collections.Counter[int]]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    hists = {}
    for split in ("train", "test"):
        texts = [t for d in TIU_DIRS for t in load_prepared_split(d, split)]
        hists[split] = collections.Counter(len(ids) for ids in tokenizer(texts)["input_ids"])
    assert len(TIU_DIRS) == 20, TIU_DIRS
    return hists


def build(
    max_lines_per_doc: int, max_docs: int, test_frac: float, split_seed: int, sample_seed: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Records per split and build statistics (incl. kept/dropped examples for inspection)."""
    targets = tiu_length_histograms()
    lengths_needed = set(targets["train"]) | set(targets["test"])
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    stream = load_dataset(
        PILE_REPO,
        data_files={"test": PILE_FILE},
        revision=PILE_REVISION,
        split="test",
        streaming=True,
    )
    rng = random.Random(sample_seed)
    pools: dict[str, dict[int, list[dict[str, Any]]]] = {"train": {}, "test": {}}
    seen_texts: set[str] = set()
    reasons: collections.Counter[str] = collections.Counter()
    dropped_examples: dict[str, list[str]] = collections.defaultdict(list)
    n_github = 0
    for doc_index, doc in enumerate(stream):
        if n_github == max_docs:
            break
        if doc["meta"]["pile_set_name"] != "Github":
            continue
        n_github += 1
        candidates = []
        for line_no, raw in enumerate(doc["text"].split("\n")):
            line = raw.strip()
            if not line:
                continue
            reason = reject_reason(line)
            if reason is None and len(tokenizer(line)["input_ids"]) not in lengths_needed:
                reason = "length"
            if reason is None and (line in seen_texts or line in {c[1] for c in candidates}):
                reason = "duplicate"
            reasons[reason or "passed_filters"] += 1
            if reason is not None:
                if len(dropped_examples[reason]) < 15 and rng.random() < 0.01:
                    dropped_examples[reason].append(line)
                continue
            candidates.append((line_no, line))
        for line_no, line in rng.sample(candidates, min(max_lines_per_doc, len(candidates))):
            seen_texts.add(line)
            split = "test" if _is_test(doc_index, test_frac, split_seed) else "train"
            n_tokens = len(tokenizer(line)["input_ids"])
            pools[split].setdefault(n_tokens, []).append(
                {
                    "text": line,
                    "n_tokens": n_tokens,
                    "pile_doc_index": doc_index,
                    "doc_line": line_no,
                }
            )

    records: dict[str, list[dict[str, Any]]] = {}
    for split, target in targets.items():
        chosen = []
        for n_tokens, count in sorted(target.items()):
            pool = pools[split].get(n_tokens, [])
            assert len(pool) >= count, (
                f"{split}: need {count} lines of {n_tokens} tokens, pool has {len(pool)}; "
                f"raise max_docs or max_lines_per_doc"
            )
            chosen += rng.sample(pool, count)
        rng.shuffle(chosen)
        records[split] = chosen
    stats = {
        "github_docs_read": n_github,
        "line_outcomes": dict(reasons),
        "pool_sizes": {s: {n: len(p) for n, p in sorted(pools[s].items())} for s in pools},
        "dropped_examples": dict(dropped_examples),
    }
    return records, stats


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
    path.write_bytes(content)
    return _sha256(content)


def write_dataset(
    out_dir: Path,
    version: str,
    records: dict[str, list[dict[str, Any]]],
    stats: dict[str, Any],
    params: dict[str, Any],
) -> None:
    assert not out_dir.exists(), f"{out_dir} exists; versions are immutable, build a new one"
    out_dir.mkdir(parents=True)
    splits = {}
    for split, recs in records.items():
        sha = _write_jsonl(out_dir / f"{split}.jsonl", recs)
        splits[split] = {"file": f"{split}.jsonl", "n_records": len(recs), "sha256": sha}
    repo_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    manifest = {
        "name": "pile_code/github_lines",
        "version": version,
        "format": "text",
        "description": (
            "Single stripped lines of source code from the Pile's GitHub subset (test file), "
            "comment and prose lines filtered out; per-split length histogram in Qwen2.5 tokens "
            "matches tiu v1. Control target with little truth content."
        ),
        "source": {
            "reference": "Gao et al. (2020). The Pile. arXiv:2101.00027",
            "url": f"https://huggingface.co/datasets/{PILE_REPO}",
            "commit": PILE_REVISION,
            "file": PILE_FILE,
            "license": "per-file licenses of the GitHub sources (Pile GitHub subset)",
        },
        "build": {
            "script": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
            "script_sha256": _sha256(SCRIPT_PATH.read_bytes()),
            "repo_commit": repo_commit,
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "params": params,
            "github_docs_read": stats["github_docs_read"],
            "line_outcomes": stats["line_outcomes"],
        },
        "splits": splits,
    }
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))


def main(
    version: str,
    max_lines_per_doc: int = 10,
    max_docs: int = 30_000,
    test_frac: float = 0.2,
    split_seed: int = 0,
    sample_seed: int = 0,
) -> None:
    assert re.fullmatch(r"v\d+", version), f"version must look like v1, v2, ...; got {version!r}"
    params = {
        "max_lines_per_doc": max_lines_per_doc,
        "max_docs": max_docs,
        "test_frac": test_frac,
        "split_seed": split_seed,
        "sample_seed": sample_seed,
        "tokenizer": TOKENIZER,
        "length_matched_to": TIU_DIRS,
    }
    records, stats = build(max_lines_per_doc, max_docs, test_frac, split_seed, sample_seed)
    print(json.dumps({k: v for k, v in stats.items() if k != "dropped_examples"}, indent=1))
    for reason, examples in stats["dropped_examples"].items():
        print(f"--- dropped ({reason}):")
        for e in examples:
            print(f"    {e!r}")
    print("--- kept (first 40 train):")
    for r in records["train"][:40]:
        print(f"    {r['text']!r}")
    write_dataset(OUT_ROOT / version, version, records, stats, params)


if __name__ == "__main__":
    fire.Fire(main)
