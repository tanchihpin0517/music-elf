"""Print canonical notes from a GigaMIDI sample for manual inspection."""

import argparse
from pathlib import Path

from music_elf import GigaMidiDataset, Midi
from music_elf.dataset import DEFAULT_GIGAMIDI_ROOT


def positive_int(value: str) -> int:
    """Parse a positive integer for an argparse option."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a MIDI file and print its canonical notes.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="MIDI file to inspect; when omitted, select a sample from the dataset.",
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
        help="Manifest index to inspect when no path is given (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=30,
        help="Maximum number of notes to print (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.path is None:
        dataset = GigaMidiDataset(args.root, file_list=args.file_list)
        midi_path = dataset[args.index]
        print(f"Dataset entries: {len(dataset):,}")
        print(f"Dataset index:   {args.index}")
    else:
        midi_path = args.path.expanduser().resolve()

    midi = Midi.load(midi_path)
    print(f"MIDI file:       {midi_path}")
    print(f"Notes:           {len(midi.notes):,}")
    print()

    print(f"Notes (showing {min(len(midi.notes), args.limit):,}):")
    for index, note in enumerate(midi.notes[: args.limit]):
        print(f"NOTE {index:>5}: time={note.start:.6f}s {note}")


if __name__ == "__main__":
    main()
