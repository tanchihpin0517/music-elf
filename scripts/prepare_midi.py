#!/usr/bin/env python
"""Preprocess MIDI files into an HF Arrow dataset with input_ids for ELF training."""

import argparse
from pathlib import Path

from datasets import Dataset
from miditok import REMI, TokenizerConfig
from tqdm import tqdm

from elf.utils.music_tokenizer import _tokenize_ids


def build_tokenizer() -> REMI:
    cfg = TokenizerConfig(
        use_chords=True,
        use_tempos=True,
        one_token_stream_for_programs=True,
    )
    return REMI(cfg)


def iter_midi_paths(midi_dir: Path):
    for pattern in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
        yield from midi_dir.rglob(pattern)


def main():
    parser = argparse.ArgumentParser(description="MIDI -> HF dataset for ELF music training")
    parser.add_argument("--midi_dir", type=str, required=True, help="Directory of .mid/.midi files")
    parser.add_argument("--out", type=str, default="data/music_dataset", help="Output save_to_disk dir")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--min_length", type=int, default=32)
    parser.add_argument("--tokenizer_out", type=str, default=None,
                        help="Tokenizer json path (default: <out>/tokenizer.json)")
    args = parser.parse_args()

    midi_dir = Path(args.midi_dir)
    out_dir = Path(args.out)
    tokenizer_path = Path(args.tokenizer_out) if args.tokenizer_out else out_dir / "tokenizer.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    tok = build_tokenizer()
    rows = []
    midi_paths = sorted(iter_midi_paths(midi_dir))
    if not midi_paths:
        raise SystemExit(f"No MIDI files found under {midi_dir}")

    for path in tqdm(midi_paths, desc="Tokenizing MIDI"):
        try:
            ids = _tokenize_ids(tok, path)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue
        if not ids:
            continue
        for start in range(0, len(ids), args.max_length):
            chunk = ids[start : start + args.max_length]
            if len(chunk) >= args.min_length:
                rows.append({"input_ids": chunk})

    if not rows:
        raise SystemExit("No tokenized sequences met min_length; check MIDI files.")

    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(out_dir))
    tok.save(str(tokenizer_path))

    print(f"Saved {len(ds)} sequences to {out_dir}")
    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"vocab_size = {len(tok)}")


if __name__ == "__main__":
    main()
