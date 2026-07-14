"""Rebuild a MIDI file for side-by-side listening."""

import argparse
import shutil
from pathlib import Path

from music_elf import GigaMidiDataset, Midi
from music_elf.dataset import DEFAULT_GIGAMIDI_ROOT

DEFAULT_OUTPUT = Path("test_outputs/earcheck/music-elf-earcheck.mid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and rebuild a MIDI file for an ear check.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="MIDI file to rebuild; when omitted, select one from the dataset.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_GIGAMIDI_ROOT,
        help=f"GigaMIDI root (default: {DEFAULT_GIGAMIDI_ROOT}).",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        default=Path("files.txt"),
        help="Manifest path, absolute or relative to --root (default: files.txt).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Manifest index to rebuild when no path is given (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Rebuilt MIDI path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.path is None:
        dataset = GigaMidiDataset(args.root, file_list=args.file_list)
        source_path = dataset[args.index]
        print(f"Dataset entries: {len(dataset):,}")
        print(f"Dataset index:   {args.index}")
    else:
        source_path = args.path.expanduser().resolve()

    output_path = args.output.expanduser().resolve()
    if output_path == source_path:
        raise ValueError("Output path must differ from the source MIDI path")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_copy_path = output_path.parent / "source.mid"
    if source_copy_path == output_path:
        source_copy_path = output_path.parent / "original.mid"
    if source_copy_path != source_path:
        shutil.copy2(source_path, source_copy_path)

    source = Midi.load(source_path)
    source.save(output_path)
    restored = Midi.load(output_path)

    paired_notes = zip(source.notes, restored.notes)
    timing_errors = [
        (
            abs(original.start - rebuilt.start),
            abs(original.duration - rebuilt.duration),
        )
        for original, rebuilt in paired_notes
    ]
    max_start_error = max((error[0] for error in timing_errors), default=0.0)
    max_duration_error = max((error[1] for error in timing_errors), default=0.0)

    print(f"Source:          {source_path}")
    print(f"Source copy:     {source_copy_path}")
    print(f"Rebuilt:         {output_path}")
    print(f"Source notes:    {len(source.notes):,}")
    print(f"Rebuilt notes:   {len(restored.notes):,}")
    print(f"Max start error: {max_start_error * 1000:.3f} ms")
    print(f"Max duration err:{max_duration_error * 1000:>7.3f} ms")
    print()
    print("Play the source and rebuilt files with the same MIDI player and soundfont.")


if __name__ == "__main__":
    main()
