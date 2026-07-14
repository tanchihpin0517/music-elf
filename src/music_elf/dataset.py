"""PyTorch datasets for MIDI files."""

from array import array
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, overload

from torch.utils.data import Dataset

Sample = TypeVar("Sample")

DEFAULT_GIGAMIDI_ROOT = Path("./dataset")


class GigaMidiDataset(Dataset[Path | Sample]):
    """Access GigaMIDI files listed in ``files.txt``.

    The manifest is indexed by byte offset instead of loading more than two
    million paths into memory. MIDI files are not opened unless ``transform``
    does so.
    """

    @overload
    def __init__(
        self,
        root: str | Path = DEFAULT_GIGAMIDI_ROOT,
        *,
        file_list: str | Path = "files.txt",
        transform: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        root: str | Path = DEFAULT_GIGAMIDI_ROOT,
        *,
        file_list: str | Path = "files.txt",
        transform: Callable[[Path], Sample],
    ) -> None: ...

    def __init__(
        self,
        root: str | Path = DEFAULT_GIGAMIDI_ROOT,
        *,
        file_list: str | Path = "files.txt",
        transform: Callable[[Path], Sample] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest = Path(file_list).expanduser()
        self.file_list = manifest if manifest.is_absolute() else self.root / manifest
        self.transform = transform
        self._offsets = self._index_file_list()

    def _index_file_list(self) -> array[int]:
        offsets = array("Q")
        with self.file_list.open("rb") as manifest:
            while line := manifest.readline():
                if line.strip():
                    offsets.append(manifest.tell() - len(line))
        return offsets

    def __len__(self) -> int:
        return len(self._offsets)

    def _path_at(self, index: int) -> Path:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("GigaMIDI dataset index out of range")

        with self.file_list.open("rb") as manifest:
            manifest.seek(self._offsets[index])
            relative_path = Path(manifest.readline().decode("utf-8").strip())

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Manifest path must stay within the dataset root: {relative_path}")
        return self.root / relative_path

    def __getitem__(self, index: int) -> Path | Sample:
        path = self._path_at(index)
        return path if self.transform is None else self.transform(path)
