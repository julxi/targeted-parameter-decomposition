"""Build prepared datasets from the Universal Truthfulness Hyperplane (UTH) data (Liu et al. 2024).

Writes one dataset per upstream UTH dataset to `data/uth/<dataset>/<version>/` (layout in
data/README.md). The datasets fall into two roles, following the paper's cross-task protocol:

- **train task** (the paper's 14 training task categories): `train` split = the upstream train file,
  `test` split = up to `n_heldout` samples of the upstream `vali` file. Both splits only keep
  samples of at most `max_tokens` Qwen tokens, because the prompt loader pads every target sequence
  to `max_seq_len` and refuses longer ones. A train-task dataset is written only if at least
  `min_coverage` of its train samples fit; otherwise it is skipped whole. Its `test` split serves the
  decomposition's training eval and the tuning of probes; it never contains the test tasks.
- **test task** (the paper's 3 held-out categories: sentence completion, short-answer QA,
  summarization): `test` split only, up to `n_test` samples of the `vali` file, *not* length-capped
  (these are only run through forward passes for probing). There is deliberately no `train` split,
  so a config that lists a test-task dataset for training crashes in the loader instead of leaking
  the held-out tasks into the decomposition.

Cleaning, applied per upstream dataset before any sampling:
- a text that occurs with both labels (train and vali files together) is dropped everywhere: one of
  the two labels must be wrong;
- in `PAIRED_DATASETS`, only intact pairs are kept (see the comment there);
- exact duplicate texts within a split are kept once;
- held-out (`test`) samples of train tasks whose text also occurs in the dataset's train file are
  dropped.

Upstream records are `{"data": str, "label": 0|1}` or `{"question": str, "answer": str, "label":
0|1}`; the latter are rendered with the UTH repo's own template (`general_qa_prompt` in its
`src/utils.py`: "Question: {q}\nAnswer: {a}"), so the model sees what the paper's probes saw.
`data` records are used as they are. Label 1 = true. The upstream repo states no license; its
component datasets carry their own.

    python spd/experiments/lm/honesty_targeted_decomposition/build_uth.py --version v1

An existing version is never overwritten; build a new version instead.
"""

import collections
import hashlib
import json
import random
import re
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import fire
import yaml
from transformers import AutoTokenizer

from spd.settings import REPO_ROOT

UPSTREAM_REPO = "hkust-nlp/Universal_Truthfulness_Hyperplane"
UPSTREAM_URL = f"https://github.com/{UPSTREAM_REPO}"
UPSTREAM_COMMIT = "f509513016c5353bf8f5781635fbad925c0b7cea"
RAW_URL = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}"
TREE_URL = f"https://api.github.com/repos/{UPSTREAM_REPO}/git/trees/{UPSTREAM_COMMIT}?recursive=1"
REFERENCE = (
    "Liu, Chen, Cheng & He (2024). On the Universal Truthfulness Hyperplane Inside LLMs. "
    "EMNLP 2024. arXiv:2407.08582"
)
LICENSE = "none stated in the upstream repo; component datasets carry their own licenses"
TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
QA_TEMPLATE = "Question: {}\nAnswer: {}"

OUT_ROOT = REPO_ROOT / "data" / "uth"
SCRIPT_PATH = Path(__file__).resolve()

Role = Literal["train_task", "test_task"]

# Upstream builds these datasets as pairs per prompt (story / sentence): one record with the correct
# answer labelled 1, one with the wrong answer labelled 0. In about half of the pairs both records
# carry the *same* answer text (story_cloze vali: 960 of 1,871 pairs; definite_pronoun_resolution
# vali: 282 of 564), so one label of such a pair is wrong. In the train files most prompts appear
# once, and a record from a broken pair cannot be recognised. So intentionally only records of
# intact pairs (exactly two records, different answers, labels 0 and 1) are kept, which drops most
# of the train files. The prompt is everything before the last ANSWER_CUE.
PAIRED_DATASETS = {"story_cloze", "definite_pronoun_resolution"}
ANSWER_CUE = "\nAnswer: "

# Task categories as listed in the paper (Section 3.1 / its dataset table), upstream dataset names.
# easy_arc is not named in the paper's list; it is placed with arc (ARC-Easy vs ARC-Challenge).
CATEGORIES: dict[str, tuple[Role, list[str]]] = {
    "natural_language_inference": ("train_task", ["rte", "qnli", "anli"]),
    "summarization": ("test_task", ["cnn_dailymail_re", "xsum_re"]),
    "sentiment_analysis": ("train_task", ["imdb", "yelp_polarity"]),
    "topic_classification": ("train_task", ["ag_news", "dbpedia_14"]),
    "statement_fact_checking": (
        "train_task",
        [
            "counterfact",
            "creak",
            "animals",
            "capitals",
            "companies",
            "elements",
            "facts",
            "inventions",
        ],
    ),
    "paraphrase_identification": ("train_task", ["mrpc", "qqp", "paws"]),
    "short_answer_closed_book_qa": ("test_task", ["nq_re", "triva_qa_re", "sciq"]),
    "long_answer_closed_book_qa": ("train_task", ["nq_re_long", "triva_qa_re_long"]),
    "reading_comprehension_qa": ("train_task", ["multirc", "squad"]),
    "multiple_choice_reading_comprehension": ("train_task", ["boolq", "race", "dream"]),
    "sentence_completion": ("test_task", ["copa", "hellaswag", "story_cloze"]),
    "closed_book_multiple_choice_qa": (
        "train_task",
        ["commonsense_qa", "arc", "easy_arc", "piqa", "openbookqa"],
    ),
    "structure_to_text": ("train_task", ["e2e_nlg_cleaned", "web_nlg_re"]),
    "coreference": ("train_task", ["definite_pronoun_resolution", "winogrande", "wsc.fixed"]),
    "reading_and_commonsense": ("train_task", ["record", "cosmos_qa"]),
    "multi_step_reasoning_qa": ("train_task", ["hotpot_qa_re", "strategy_qa"]),
    "other": ("train_task", ["tqa", "arithmetic"]),
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def _upstream_datasets() -> list[str]:
    tree = json.loads(_fetch(TREE_URL))
    assert not tree["truncated"], "GitHub tree listing was truncated"
    return sorted(
        e["path"].removeprefix("data/")
        for e in tree["tree"]
        if e["type"] == "tree" and e["path"].startswith("data/") and e["path"].count("/") == 1
    )


def _parse(raw: bytes, upstream_file: str) -> list[tuple[str, bool]]:
    """(text, label) per upstream record, in file order."""
    rows = json.loads(raw)
    parsed = []
    for row in rows:
        match sorted(row):
            case ["data", "label"]:
                text = row["data"]
            case ["answer", "label", "question"]:
                text = QA_TEMPLATE.format(row["question"], row["answer"])
            case keys:
                raise AssertionError(f"{upstream_file}: unexpected record keys {keys}")
        assert row["label"] in (0, 1), f"{upstream_file}: label {row['label']!r}"
        parsed.append((text, bool(row["label"])))
    return parsed


def _broken_pair_texts(records: list[tuple[str, bool]], upstream_file: str) -> set[str]:
    """Texts of `records` that are not part of an intact pair (see PAIRED_DATASETS)."""
    groups: dict[str, list[tuple[str, str, bool]]] = collections.defaultdict(list)
    for text, label in records:
        assert ANSWER_CUE in text, f"{upstream_file}: no {ANSWER_CUE!r} in {text[:200]!r}"
        prompt, _, answer = text.rpartition(ANSWER_CUE)
        groups[prompt].append((text, answer, label))
    broken = set()
    for members in groups.values():
        answers = {answer for _, answer, _ in members}
        labels = sorted(label for _, _, label in members)
        if not (len(members) == 2 and len(answers) == 2 and labels == [False, True]):
            broken |= {text for text, _, _ in members}
    return broken


def _clean(
    train: list[tuple[str, bool]],
    vali: list[tuple[str, bool]],
    paired: bool,
    files: dict[str, str],
) -> tuple[list[int], list[int], dict[str, int]]:
    """Indices of the kept train and vali records, and counts of what was dropped (see docstring)."""
    broken: set[str] = set()
    if paired:
        broken = _broken_pair_texts(train, files["train"]) | _broken_pair_texts(vali, files["vali"])
    labels_of: dict[str, set[bool]] = collections.defaultdict(set)
    for text, label in train + vali:
        labels_of[text].add(label)
    conflicted = {t for t, ls in labels_of.items() if len(ls) > 1}

    def keep(records: list[tuple[str, bool]], exclude: set[str]) -> tuple[list[int], int]:
        seen: set[str] = set()
        kept, n_dup = [], 0
        for i, (text, _) in enumerate(records):
            if text in conflicted or text in broken or text in exclude:
                continue
            if text in seen:
                n_dup += 1
                continue
            seen.add(text)
            kept.append(i)
        return kept, n_dup

    train_texts = {text for text, _ in train}
    train_kept, train_dup = keep(train, exclude=set())
    vali_kept, vali_dup = keep(vali, exclude=train_texts)
    counts = {
        "label_conflict_texts": len(conflicted),
        "not_intact_pair_texts": len(broken),
        "train_duplicates": train_dup,
        "vali_duplicates": vali_dup,
        "vali_in_train": sum(1 for t, _ in vali if t in train_texts and t not in conflicted),
    }
    return train_kept, vali_kept, counts


def _record(
    text: str, label: bool, n_tokens: int, dataset: str, category: str, file: str, index: int
) -> dict[str, Any]:
    return {
        "text": text,
        "label": label,
        "dataset": dataset,
        "category": category,
        "n_tokens": n_tokens,
        "upstream_file": file,
        "upstream_index": index,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
    path.write_bytes(content)
    return _sha256(content)


def build_dataset(
    dataset: str,
    category: str,
    role: Role,
    tokenizer: Any,
    max_tokens: int,
    min_coverage: float,
    n_heldout: int,
    n_test: int,
    rng: random.Random,
) -> tuple[dict[str, list[dict[str, Any]]] | None, dict[str, Any], dict[str, bytes]]:
    """Splits of one dataset (None if a train task is skipped for coverage), build stats, raw files."""
    files = {s: f"data/{dataset}/{dataset}_probe_{s}.json" for s in ("train", "vali")}
    raw = {s: _fetch(f"{RAW_URL}/{f}") for s, f in files.items()}
    train, vali = _parse(raw["train"], files["train"]), _parse(raw["vali"], files["vali"])
    train_kept, vali_kept, counts = _clean(train, vali, dataset in PAIRED_DATASETS, files)
    stats: dict[str, Any] = {
        "upstream_train": len(train),
        "upstream_vali": len(vali),
        **counts,
    }

    def n_tokens(texts: list[str]) -> list[int]:
        return [len(ids) for ids in tokenizer(texts)["input_ids"]]

    match role:
        case "train_task":
            train_lens = n_tokens([train[i][0] for i in train_kept])
            coverage = sum(n <= max_tokens for n in train_lens) / len(train_lens)
            stats["train_coverage"] = round(coverage, 4)
            if coverage < min_coverage:
                return None, stats, raw
            train_recs = [
                _record(*train[i], n, dataset, category, files["train"], i)
                for i, n in zip(train_kept, train_lens, strict=True)
                if n <= max_tokens
            ]
            vali_lens = n_tokens([vali[i][0] for i in vali_kept])
            vali_fit = [
                (i, n) for i, n in zip(vali_kept, vali_lens, strict=True) if n <= max_tokens
            ]
            chosen = sorted(rng.sample(vali_fit, min(n_heldout, len(vali_fit))))
            test_recs = [
                _record(*vali[i], n, dataset, category, files["vali"], i) for i, n in chosen
            ]
            stats["vali_within_cap"] = len(vali_fit)
            splits = {"train": train_recs, "test": test_recs}
        case "test_task":
            chosen_idx = sorted(rng.sample(vali_kept, min(n_test, len(vali_kept))))
            lens = n_tokens([vali[i][0] for i in chosen_idx])
            test_recs = [
                _record(*vali[i], n, dataset, category, files["vali"], i)
                for i, n in zip(chosen_idx, lens, strict=True)
            ]
            splits = {"test": test_recs}
    for split, recs in splits.items():
        assert recs, f"{dataset}: empty {split} split"
        labels = [r["label"] for r in recs]
        assert 0 < sum(labels) < len(labels), f"{dataset} {split}: only one label"
    return splits, stats, raw


def write_dataset(
    dataset: str,
    category: str,
    role: Role,
    version: str,
    splits: dict[str, list[dict[str, Any]]],
    stats: dict[str, Any],
    raw: dict[str, bytes],
    params: dict[str, Any],
    built_at: str,
) -> Path:
    out_dir = OUT_ROOT / dataset / version
    assert not out_dir.exists(), f"{out_dir} exists; versions are immutable, build a new one"
    out_dir.mkdir(parents=True)
    split_info = {
        split: {
            "file": f"{split}.jsonl",
            "n_records": len(recs),
            "n_true": sum(r["label"] for r in recs),
            "max_tokens": max(r["n_tokens"] for r in recs),
            "sha256": _write_jsonl(out_dir / f"{split}.jsonl", recs),
        }
        for split, recs in splits.items()
    }
    repo_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    role_text = {
        "train_task": "training task (train + held-out test split, length-capped)",
        "test_task": "held-out test task (test split only, not length-capped)",
    }[role]
    manifest = {
        "name": f"uth/{dataset}",
        "version": version,
        "format": "text",
        "description": (
            f"UTH dataset {dataset} (category {category}), true and false samples mixed "
            f"(record field 'label'); role: {role_text}."
        ),
        "role": role,
        "category": category,
        "source": {
            "reference": REFERENCE,
            "url": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "files": {
                f"data/{dataset}/{dataset}_probe_{s}.json": {"sha256": _sha256(b)}
                for s, b in raw.items()
            },
            "license": LICENSE,
        },
        "build": {
            "script": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
            "script_sha256": _sha256(SCRIPT_PATH.read_bytes()),
            "repo_commit": repo_commit,
            "built_at": built_at,
            "params": params,
            "stats": stats,
        },
        "splits": split_info,
    }
    (out_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    )
    return out_dir


def main(
    version: str,
    max_tokens: int = 128,
    min_coverage: float = 0.8,
    n_heldout: int = 200,
    n_test: int = 1000,
    seed: int = 0,
) -> None:
    assert re.fullmatch(r"v\d+", version), f"version must look like v1, v2, ...; got {version!r}"
    assigned = [d for _, ds in CATEGORIES.values() for d in ds]
    assert len(assigned) == len(set(assigned)), "a dataset is listed in two categories"
    upstream = _upstream_datasets()
    assert sorted(assigned) == upstream, (
        f"category table and upstream differ: missing {sorted(set(upstream) - set(assigned))}, "
        f"unknown {sorted(set(assigned) - set(upstream))}"
    )
    params = {
        "max_tokens": max_tokens,
        "min_coverage": min_coverage,
        "n_heldout": n_heldout,
        "n_test": n_test,
        "seed": seed,
        "tokenizer": TOKENIZER,
    }
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    built_at = datetime.now(UTC).isoformat(timespec="seconds")

    written: dict[Role, list[str]] = {"train_task": [], "test_task": []}
    for category, (role, datasets) in CATEGORIES.items():
        for dataset in datasets:
            # One rng per dataset, so adding or skipping a dataset never changes another's sample.
            rng = random.Random(f"{seed}/{dataset}")
            splits, stats, raw = build_dataset(
                dataset, category, role, tokenizer, max_tokens, min_coverage, n_heldout, n_test, rng
            )
            print(f"=== {dataset} ({category}, {role}): {json.dumps(stats)}")
            if splits is None:
                print(f"    SKIPPED: train coverage {stats['train_coverage']} < {min_coverage}")
                continue
            for split, recs in splits.items():
                print(f"    {split}: {len(recs)} records, {sum(r['label'] for r in recs)} true")
                for r in recs[:2]:
                    print(f"      [{r['label']}] {r['text'][:200]!r}")
            out_dir = write_dataset(
                dataset, category, role, version, splits, stats, raw, params, built_at
            )
            written[role].append(str(out_dir.relative_to(REPO_ROOT)))
    for role, dirs in written.items():
        print(f"--- {role}: {len(dirs)} datasets")
        for d in dirs:
            print(f"    - {d}")


if __name__ == "__main__":
    fire.Fire(main)
