"""Build prepared datasets from the Truth-is-Universal statements (Bürger et al., NeurIPS 2024).

Writes one dataset per (upstream file, label) to `data/tiu/<file>_<true|false>/<version>/`, e.g.
`data/tiu/cities_true/v1/` and `data/tiu/neg_cities_false/v1/`. Each holds `train.jsonl`,
`test.jsonl` and `manifest.yaml` (see data/README.md).

All datasets of one build share a train/test split by subject (the city, the Spanish word, the
element, the animal, or the unordered number pair): every statement about a subject, whether true,
false or negated, lands in the same split.

    python data/tiu/build_tiu.py --version v1

An existing version is never overwritten; build a new version instead.
"""

import hashlib
import io
import json
import re
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fire
import pandas as pd
import yaml

UPSTREAM_URL = "https://github.com/sciai-lab/Truth_is_Universal"
UPSTREAM_COMMIT = "605ef00514415deb4806969a172f6e13e0798df7"
RAW_URL = f"https://raw.githubusercontent.com/sciai-lab/Truth_is_Universal/{UPSTREAM_COMMIT}"
REFERENCE = (
    "Bürger, Hamprecht & Nadler (2024). Truth is Universal: Robust Detection of Lies in LLMs. "
    "NeurIPS 2024. arXiv:2407.12831"
)
LICENSE = "MIT (Copyright (c) 2024 Scientific AI)"

OUT_ROOT = Path(__file__).parent
REPO_ROOT = OUT_ROOT.parent.parent

# upstream file -> subject namespace (files sharing a namespace share the split of each subject)
UPSTREAM_FILES = {
    "cities": "city",
    "neg_cities": "city",
    "sp_en_trans": "spanish_word",
    "neg_sp_en_trans": "spanish_word",
    "element_symb": "element",
    "neg_element_symb": "element",
    "animal_class": "animal",
    "neg_animal_class": "animal",
    "larger_than": "number_pair",
    "smaller_than": "number_pair",
}

SUBJECT_PATTERNS = {
    "spanish_word": r"^The Spanish word '(.+)' (?:means|does not mean) '",
    "element": r"^(.+?) (?:has|does not have) the symbol ",
    "animal": r"^The (.+?) is (?:not )?an? ",
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _subject(namespace: str, row: dict[str, Any]) -> str:
    match namespace:
        case "city":
            return str(row["city"])
        case "number_pair":
            n1, n2 = int(row["n1"]), int(row["n2"])
            return f"{min(n1, n2)}-{max(n1, n2)}"
        case _:
            match = re.match(SUBJECT_PATTERNS[namespace], row["statement"])
            assert match is not None, f"{namespace}: no subject in {row['statement']!r}"
            return match.group(1)


def _is_test(namespace: str, subject: str, test_frac: float, split_seed: int) -> bool:
    digest = hashlib.sha256(f"{split_seed}/{namespace}/{subject}".encode()).digest()
    return int.from_bytes(digest[:8]) / 2**64 < test_frac


def _records(
    df: pd.DataFrame, namespace: str, upstream_file: str
) -> list[tuple[str, dict[str, Any]]]:
    """(subject, record) per row; extra upstream columns (e.g. country, n1) are kept as metadata."""
    records = []
    for row_idx, row in zip(df.index, df.to_dict("records"), strict=True):
        subject = _subject(namespace, row)
        extra = {k: v for k, v in row.items() if k not in ("statement", "label")}
        record = {
            "text": row["statement"],
            "label": bool(row["label"]),
            "subject": subject,
            "upstream_file": f"datasets/{upstream_file}.csv",
            "upstream_row": int(row_idx),
            **extra,
        }
        records.append((subject, record))
    return records


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
    path.write_bytes(content)
    return _sha256(content)


def main(version: str, test_frac: float = 0.2, split_seed: int = 0) -> None:
    assert re.fullmatch(r"v\d+", version), f"version must look like v1, v2, ...; got {version!r}"
    script_sha256 = _sha256(Path(__file__).read_bytes())
    built_at = datetime.now(UTC).isoformat(timespec="seconds")

    for upstream_file, namespace in UPSTREAM_FILES.items():
        with urllib.request.urlopen(f"{RAW_URL}/datasets/{upstream_file}.csv") as resp:
            csv_bytes = resp.read()
        df = pd.read_csv(io.BytesIO(csv_bytes))
        assert set(df["label"].unique()) <= {0, 1}, f"{upstream_file}: non-binary labels"
        n_duplicates = int(df["statement"].duplicated().sum())
        df = df.drop_duplicates("statement")

        for label in (True, False):
            name = f"{upstream_file}_{str(label).lower()}"
            out_dir = OUT_ROOT / name / version
            assert not out_dir.exists(), f"{out_dir} exists; versions are immutable"

            splits: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
            for subject, record in _records(
                df[df["label"] == int(label)], namespace, upstream_file
            ):
                split = "test" if _is_test(namespace, subject, test_frac, split_seed) else "train"
                splits[split].append(record)
            assert splits["train"] and splits["test"], f"{name}: empty split"

            out_dir.mkdir(parents=True)
            split_info = {
                split: {
                    "file": f"{split}.jsonl",
                    "n_records": len(records),
                    "sha256": _write_jsonl(out_dir / f"{split}.jsonl", records),
                }
                for split, records in splits.items()
            }
            manifest = {
                "name": f"tiu/{name}",
                "version": version,
                "format": "text",
                "description": (
                    f"{'True' if label else 'False'} statements from Truth-is-Universal "
                    f"datasets/{upstream_file}.csv, split train/test by {namespace}."
                ),
                "source": {
                    "reference": REFERENCE,
                    "url": UPSTREAM_URL,
                    "commit": UPSTREAM_COMMIT,
                    "file": f"datasets/{upstream_file}.csv",
                    "file_sha256": _sha256(csv_bytes),
                    "license": LICENSE,
                },
                "build": {
                    "script": "data/tiu/build_tiu.py",
                    "script_sha256": script_sha256,
                    "repo_commit": _git_commit(),
                    "built_at": built_at,
                    "params": {
                        "label": label,
                        "test_frac": test_frac,
                        "split_seed": split_seed,
                        "split_by": namespace,
                    },
                    "dropped_duplicate_statements": n_duplicates,
                },
                "splits": split_info,
            }
            (out_dir / "manifest.yaml").write_text(
                yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
            )
            print(
                f"{out_dir.relative_to(REPO_ROOT)}: "
                + ", ".join(f"{s} {info['n_records']}" for s, info in split_info.items())
            )


if __name__ == "__main__":
    fire.Fire(main)
