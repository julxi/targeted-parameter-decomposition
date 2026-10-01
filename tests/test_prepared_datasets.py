import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from spd.configs import AllPositions, LastKPositions, TokenPositions
from spd.experiments.lm.prepared_datasets import load_prepared_datasets, load_prepared_split
from spd.experiments.lm.prompts_dataset import build_position_mask
from spd.settings import REPO_ROOT
from spd.utils.general_utils import calc_sum_recon_loss_lm, calc_sum_recon_loss_lm_at_positions


def test_build_position_mask() -> None:
    n_tokens = torch.tensor([3, 5])
    assert build_position_mask(n_tokens, 6, AllPositions()) is None

    tokens = build_position_mask(n_tokens, 6, TokenPositions(type="tokens"))
    assert tokens is not None
    assert tokens.tolist() == [
        [True, True, True, False, False, False],
        [True, True, True, True, True, False],
    ]

    last_2 = build_position_mask(n_tokens, 6, LastKPositions(type="last_k", k=2))
    assert last_2 is not None
    assert last_2.tolist() == [
        [False, True, True, False, False, False],
        [False, False, False, True, True, False],
    ]

    with pytest.raises(AssertionError):
        build_position_mask(n_tokens, 6, LastKPositions(type="last_k", k=4))


def test_recon_loss_at_positions_ignores_unselected_positions() -> None:
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 7)
    target = torch.randn(2, 4, 7)
    position_mask = torch.tensor([[True, False, False, True], [False, False, True, False]])

    loss, n = calc_sum_recon_loss_lm_at_positions(pred, target, "kl", position_mask)

    expected = calc_sum_recon_loss_lm(pred[position_mask], target[position_mask], "kl")
    assert n == 3
    assert torch.allclose(loss, expected)

    pred_changed_elsewhere = pred.clone()
    pred_changed_elsewhere[~position_mask] = 100.0
    loss_changed, _ = calc_sum_recon_loss_lm_at_positions(
        pred_changed_elsewhere, target, "kl", position_mask
    )
    assert torch.allclose(loss, loss_changed)


def _write_dataset(dataset_dir: Path, splits: dict[str, list[str]]) -> None:
    dataset_dir.mkdir(parents=True)
    split_info = {}
    for split, texts in splits.items():
        content = "".join(json.dumps({"text": t, "label": True}) + "\n" for t in texts).encode()
        (dataset_dir / f"{split}.jsonl").write_bytes(content)
        split_info[split] = {
            "file": f"{split}.jsonl",
            "n_records": len(texts),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    manifest = {"name": "toy", "version": "v1", "format": "text", "splits": split_info}
    (dataset_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest))


def test_load_prepared_datasets_concatenates_splits(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "a" / "v1", {"train": ["a1", "a2"], "test": ["a3"]})
    _write_dataset(tmp_path / "b" / "v1", {"train": ["b1"], "test": ["b2", "b3"]})
    dirs = [str(tmp_path / "a" / "v1"), str(tmp_path / "b" / "v1")]

    assert load_prepared_datasets(dirs, "train") == ["a1", "a2", "b1"]
    assert load_prepared_datasets(dirs, "test") == ["a3", "b2", "b3"]


def test_load_prepared_dataset_rejects_edited_split(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "a" / "v1"
    _write_dataset(dataset_dir, {"train": ["a1"], "test": ["a2"]})
    (dataset_dir / "train.jsonl").write_text(json.dumps({"text": "edited"}) + "\n")

    with pytest.raises(AssertionError, match="does not match its manifest"):
        load_prepared_datasets([str(dataset_dir)], "train")


def test_committed_tiu_datasets_verify_and_split_by_subject() -> None:
    dataset_dirs = sorted((REPO_ROOT / "data" / "tiu").glob("*/v1"))
    assert len(dataset_dirs) == 20

    split_by_subject: dict[tuple[str, str], str] = {}
    for dataset_dir in dataset_dirs:
        manifest = yaml.safe_load((dataset_dir / "manifest.yaml").read_text())
        namespace = manifest["build"]["params"]["split_by"]
        for split in ("train", "test"):
            assert load_prepared_split(str(dataset_dir), split)
            for line in (dataset_dir / f"{split}.jsonl").read_text().splitlines():
                key = (namespace, json.loads(line)["subject"])
                assert split_by_subject.setdefault(key, split) == split, key
