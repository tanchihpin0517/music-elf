"""Tests for the GigaMIDI dataset."""

from pathlib import Path

import pytest
from torch.utils.data import Dataset

from music_elf import GigaMidiDataset


@pytest.fixture
def gigamidi_root(tmp_path: Path) -> Path:
    (tmp_path / "nested").mkdir()
    (tmp_path / "first.mid").write_bytes(b"MThd-first")
    (tmp_path / "nested" / "second.mid").write_bytes(b"MThd-second")
    (tmp_path / "files.txt").write_text(
        "first.mid\n\nnested/second.mid\n",
        encoding="utf-8",
    )
    return tmp_path


def test_dataset_indexes_manifest_paths(gigamidi_root: Path) -> None:
    dataset = GigaMidiDataset(gigamidi_root)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset[0] == gigamidi_root / "first.mid"
    assert dataset[1] == gigamidi_root / "nested" / "second.mid"
    assert dataset[-1] == gigamidi_root / "nested" / "second.mid"


def test_dataset_applies_transform(gigamidi_root: Path) -> None:
    dataset = GigaMidiDataset(gigamidi_root, transform=Path.read_bytes)

    assert dataset[0] == b"MThd-first"


@pytest.mark.parametrize("index", [-3, 2])
def test_dataset_rejects_out_of_range_index(
    gigamidi_root: Path,
    index: int,
) -> None:
    dataset = GigaMidiDataset(gigamidi_root)

    with pytest.raises(IndexError, match="index out of range"):
        dataset[index]


@pytest.mark.parametrize("unsafe_path", ["../outside.mid", "/tmp/outside.mid"])
def test_dataset_rejects_paths_outside_root(tmp_path: Path, unsafe_path: str) -> None:
    (tmp_path / "files.txt").write_text(f"{unsafe_path}\n", encoding="utf-8")
    dataset = GigaMidiDataset(tmp_path)

    with pytest.raises(ValueError, match="within the dataset root"):
        dataset[0]


def test_dataset_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        GigaMidiDataset(tmp_path)


def test_dataset_defaults_to_dataset_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "sample.mid").write_bytes(b"MThd-sample")
    (root / "files.txt").write_text("sample.mid\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    dataset = GigaMidiDataset()

    assert dataset.root == root
    assert dataset[0] == root / "sample.mid"
