"""Build prepared datasets from the Truth-is-Universal statements (Bürger et al., NeurIPS 2024).

Writes one dataset per (upstream file, label) to `data/tiu/<file>_<true|false>/<version>/`, e.g.
`data/tiu/cities_true/v1/` and `data/tiu/neg_cities_false/v1/`. Each holds `train.jsonl`,
`test.jsonl` and `manifest.yaml` (see data/README.md). This script lives in the `spd` package, not
next to its outputs, because the project keeps all code under `spd/`; each manifest's `build.script`
records the script's path.

All in-distribution datasets of one build share a train/test split by subject (the city, the
Spanish word, the element, the animal, or the unordered number pair): every statement about a
subject, whether true, false or negated, lands in the same split.

With `val_frac > 0` a third split `val.jsonl` holds a further share of the subjects, taken from what
would otherwise be train: for tuning hyperparameters without touching test. The subject's hash
decides all three splits (u < test_frac: test; u < test_frac + val_frac: val), so for the same
test_frac and split_seed the test split is identical to a build without val, and train + val equals
that build's train.

With `train_cap` every in-distribution dataset keeps at most that many train statements, so that no
family dominates training (uncapped, cities and the two comparison families are 83% of tiu v2's
train split). The cap only drops train statements; val and test are those of the uncapped build.
Statements are kept in the order of a subject hash, so a kept subject is kept in every dataset of
its namespace (cities_true, cities_false, neg_cities_*): the trimmed train split still holds the
true, false and negated statements about the same subjects.

Every build also writes the out-of-distribution (OOD) datasets of OOD_FILES (role `ood_test`): the
upstream files no training config uses, with a test split only, so they cannot be trained on by
accident. They are not split by subject, so their subjects (e.g. the cities of cities_de and
cities_conj) may occur in the in-distribution train splits. Builds up to v2 predate them.

    python spd/experiments/lm/honesty_targeted_decomposition/build_tiu.py --version v1
    python spd/experiments/lm/honesty_targeted_decomposition/build_tiu.py --version v2 --val_frac 0.12
    python spd/experiments/lm/honesty_targeted_decomposition/build_tiu.py --version v3 \\
        --val_frac 0.12 --train_cap 130

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
from typing import Any, Literal

import fire
import pandas as pd
import yaml

from spd.settings import REPO_ROOT

UPSTREAM_URL = "https://github.com/sciai-lab/Truth_is_Universal"
UPSTREAM_COMMIT = "605ef00514415deb4806969a172f6e13e0798df7"
RAW_URL = f"https://raw.githubusercontent.com/sciai-lab/Truth_is_Universal/{UPSTREAM_COMMIT}"
REFERENCE = (
    "Bürger, Hamprecht & Nadler (2024). Truth is Universal: Robust Detection of Lies in LLMs. "
    "NeurIPS 2024. arXiv:2407.12831"
)
LICENSE = "MIT (Copyright (c) 2024 Scientific AI)"

OUT_ROOT = REPO_ROOT / "data" / "tiu"
SCRIPT_PATH = Path(__file__).resolve()

Role = Literal["in_distribution", "ood_test"]

# in-distribution upstream file -> subject namespace (files sharing a namespace share the split of
# each subject)
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

# Upstream files not used for training, each testing a different distance from the training
# families: new facts in known sentence forms (inventors, facts and their negations), the training
# families written in German (`_de`, 50 statements each upstream), and compositions of two
# statements ("It is the case both that ... and that ...", "either ... or ..."), which upstream has
# only in affirmative form. Left out: common_claim_true_false and counterfact_true_false (larger,
# less curated).
_OOD_BASES = ["cities", "sp_en_trans", "element_symb", "animal_class", "inventors", "facts"]
OOD_FILES = [
    "inventors",
    "neg_inventors",
    "facts",
    "neg_facts",
    *[f"{neg}{base}_de" for base in _OOD_BASES for neg in ("", "neg_")],
    *[f"{base}_{kind}" for base in _OOD_BASES for kind in ("conj", "disj")],
]

SUBJECT_PATTERNS = {
    "spanish_word": r"^The Spanish word '(.+)' (?:means|does not mean) '",
    "element": r"^(.+?) (?:has|does not have) the symbol ",
    "animal": r"^The (.+?) is (?:not )?an? ",
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hash_unit(key: str) -> float:
    """Deterministic pseudo-uniform number in [0, 1) for a string key."""
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8]) / 2**64


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


def _split_of(
    namespace: str, subject: str, test_frac: float, val_frac: float, split_seed: int
) -> str:
    u = _hash_unit(f"{split_seed}/{namespace}/{subject}")
    if u < test_frac:
        return "test"
    return "val" if u < test_frac + val_frac else "train"


def _record(
    row_idx: int, row: dict[str, Any], upstream_file: str, subject: str | None
) -> dict[str, Any]:
    """One output record; extra upstream columns (e.g. country, n1) are kept as metadata.

    `subject` is None for the OOD files, which are not split by subject. The key order is part of
    the split files' sha256, so it must stay as it is for builds to stay comparable.
    """
    extra = {k: v for k, v in row.items() if k not in ("statement", "label")}
    return {
        "text": row["statement"],
        "label": bool(row["label"]),
        **({"subject": subject} if subject is not None else {}),
        "upstream_file": f"datasets/{upstream_file}.csv",
        "upstream_row": row_idx,
        **extra,
    }


def _download(upstream_file: str) -> tuple[pd.DataFrame, bytes, int]:
    """Upstream CSV without duplicate statements, its raw bytes, and the number of duplicates."""
    with urllib.request.urlopen(f"{RAW_URL}/datasets/{upstream_file}.csv") as resp:
        csv_bytes = resp.read()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    assert set(df["label"].unique()) <= {0, 1}, f"{upstream_file}: non-binary labels"
    n_duplicates = int(df["statement"].duplicated().sum())
    return df.drop_duplicates("statement"), csv_bytes, n_duplicates


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
    path.write_bytes(content)
    return _sha256(content)


def _write_dataset(
    upstream_file: str,
    label: bool,
    version: str,
    role: Role,
    splits: dict[str, list[dict[str, Any]]],
    description: str,
    csv_bytes: bytes,
    build: dict[str, Any],
) -> None:
    name = f"{upstream_file}_{str(label).lower()}"
    out_dir = OUT_ROOT / name / version
    assert not out_dir.exists(), f"{out_dir} exists; versions are immutable"
    assert all(splits.values()), f"{name}: empty split"
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
        "description": description,
        "role": role,
        "source": {
            "reference": REFERENCE,
            "url": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "file": f"datasets/{upstream_file}.csv",
            "file_sha256": _sha256(csv_bytes),
            "license": LICENSE,
        },
        "build": build,
        "splits": split_info,
    }
    (out_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    )
    print(
        f"{out_dir.relative_to(REPO_ROOT)}: "
        + ", ".join(f"{s} {info['n_records']}" for s, info in split_info.items())
    )


def main(
    version: str,
    test_frac: float = 0.2,
    val_frac: float = 0.0,
    train_cap: int | None = None,
    split_seed: int = 0,
) -> None:
    assert re.fullmatch(r"v\d+", version), f"version must look like v1, v2, ...; got {version!r}"
    assert val_frac >= 0 and test_frac + val_frac < 1, (test_frac, val_frac)
    assert train_cap is None or train_cap > 0, train_cap
    build_base = {
        "script": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "script_sha256": _sha256(SCRIPT_PATH.read_bytes()),
        "repo_commit": _git_commit(),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    split_names = ["train", "val", "test"] if val_frac > 0 else ["train", "test"]

    for upstream_file, namespace in UPSTREAM_FILES.items():
        df, csv_bytes, n_duplicates = _download(upstream_file)
        for label in (True, False):
            splits: dict[str, list[dict[str, Any]]] = {s: [] for s in split_names}
            rows = df.loc[df["label"] == int(label)]
            for row_idx, row in zip(rows.index, rows.to_dict("records"), strict=True):
                subject = _subject(namespace, row)
                split = _split_of(namespace, subject, test_frac, val_frac, split_seed)
                splits[split].append(_record(int(row_idx), row, upstream_file, subject))
            n_train_uncapped = len(splits["train"])
            if train_cap is not None:
                # kept statements stay in upstream order, only the selection follows the hash
                kept = sorted(
                    splits["train"],
                    key=lambda r: (
                        _hash_unit(f"{split_seed}/cap/{namespace}/{r['subject']}"),
                        r["text"],
                    ),
                )[:train_cap]
                kept_texts = {r["text"] for r in kept}
                splits["train"] = [r for r in splits["train"] if r["text"] in kept_texts]
            description = (
                f"{'True' if label else 'False'} statements from Truth-is-Universal "
                f"datasets/{upstream_file}.csv, split {'/'.join(split_names)} by {namespace}"
                + (f", train capped at {train_cap} statements." if train_cap is not None else ".")
            )
            build = {
                **build_base,
                "params": {
                    "label": label,
                    "test_frac": test_frac,
                    **({"val_frac": val_frac} if val_frac > 0 else {}),
                    **({"train_cap": train_cap} if train_cap is not None else {}),
                    "split_seed": split_seed,
                    "split_by": namespace,
                },
                "dropped_duplicate_statements": n_duplicates,
                **({"n_train_uncapped": n_train_uncapped} if train_cap is not None else {}),
            }
            _write_dataset(
                upstream_file,
                label,
                version,
                "in_distribution",
                splits,
                description,
                csv_bytes,
                build,
            )

    for upstream_file in OOD_FILES:
        df, csv_bytes, n_duplicates = _download(upstream_file)
        for label in (True, False):
            rows = df.loc[df["label"] == int(label)]
            records = [
                _record(int(row_idx), row, upstream_file, None)
                for row_idx, row in zip(rows.index, rows.to_dict("records"), strict=True)
            ]
            description = (
                f"{'True' if label else 'False'} statements from Truth-is-Universal "
                f"datasets/{upstream_file}.csv; out-of-distribution test set (test split only, "
                "not split by subject, never trained on)."
            )
            build = {
                **build_base,
                "params": {"label": label},
                "dropped_duplicate_statements": n_duplicates,
            }
            _write_dataset(
                upstream_file,
                label,
                version,
                "ood_test",
                {"test": records},
                description,
                csv_bytes,
                build,
            )


if __name__ == "__main__":
    fire.Fire(main)
