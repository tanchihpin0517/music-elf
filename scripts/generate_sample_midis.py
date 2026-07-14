#!/usr/bin/env python
"""Generate a handful of tiny MIDI files for smoke testing when no corpus is available."""

import argparse
from pathlib import Path

from symusic import Note, Score, Tempo, TimeSignature, Track


def make_score(seed: int) -> Score:
    score = Score()
    score.time_signatures = [TimeSignature(time=0, numerator=4, denominator=4)]
    score.tempos = [Tempo(time=0, qpm=100 + (seed % 40))]

    notes = []
    t = 0
    for i in range(16 + seed % 8):
        pitch = 60 + (i * 2 + seed) % 12
        dur = 240 + (i % 3) * 120
        notes.append(Note(time=t, pitch=pitch, velocity=80, duration=dur))
        t += dur

    track = Track(program=0, notes=notes, name=f"seed_{seed}")
    score.tracks = [track]
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/sample_midis")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        path = out_dir / f"sample_{i:03d}.mid"
        make_score(i).dump_midi(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
