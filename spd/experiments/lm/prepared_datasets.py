"""Loading of the git-tracked prepared datasets under `data/` (layout described in data/README.md)."""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from spd.log import logger
from spd.settings import REPO_ROOT

Split = Literal["train", "test"]


class SplitFile(BaseModel):
    file: str
    n_records: int
    sha256: str


class PreparedDatasetManifest(BaseModel):
    """The fields of manifest.yaml that loading relies on; provenance fields are not parsed."""

    name: str
    version: str
    format: Literal["text"]
    splits: dict[Split, SplitFile]


def resolve_dataset_dir(dataset_dir: str) -> Path:
    """Dataset dirs in configs are relative to the repo root (e.g. data/tiu/cities_true/v1)."""
    path = Path(dataset_dir)
    return path if path.is_absolute() else REPO_ROOT / path


def load_prepared_split(dataset_dir: str, split: Split) -> list[str]:
    """Texts of one split of one prepared dataset, verified against the manifest's hash."""
    path = resolve_dataset_dir(dataset_dir)
    manifest_path = path / "manifest.yaml"
    assert manifest_path.exists(), f"No manifest.yaml in {path}"
    manifest = PreparedDatasetManifest(**yaml.safe_load(manifest_path.read_text()))
    split_file = manifest.splits[split]

    content = (path / split_file.file).read_bytes()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    assert actual_sha256 == split_file.sha256, (
        f"{path / split_file.file} does not match its manifest (sha256 {actual_sha256} != "
        f"{split_file.sha256}); prepared dataset versions are immutable, build a new version"
    )
    texts = [json.loads(line)["text"] for line in content.decode().splitlines()]
    assert len(texts) == split_file.n_records, f"{path / split_file.file}: wrong record count"
    return texts


def load_prepared_datasets(dataset_dirs: list[str], split: Split) -> list[str]:
    """Concatenation of one split across prepared datasets."""
    texts = []
    for dataset_dir in dataset_dirs:
        dataset_texts = load_prepared_split(dataset_dir, split)
        logger.info(f"{dataset_dir}: {len(dataset_texts)} {split} texts")
        texts.extend(dataset_texts)
    return texts
