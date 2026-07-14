# Music ELF

An empty command-line scaffold for ELF model training and inference.

## Setup

Install the project and its development dependencies with
[uv](https://docs.astral.sh/uv/):

```console
uv sync
```

Inspect the CLI:

```console
uv run music-elf --help
```

## Commands

The six command combinations are:

```console
uv run music-elf train elf
uv run music-elf train encoder
uv run music-elf train lm
uv run music-elf infer elf
uv run music-elf infer encoder
uv run music-elf infer lm
```

Each command is currently a placeholder. It reports that it is not implemented
and exits with status 1.

## Dataset

`GigaMidiDataset` reads MIDI paths from `./dataset/files.txt` by default. For
local development, `./dataset` can be a symlink to the GigaMIDI directory:

```console
ln -s /home/tanch/dataset/gigamidi dataset
```

```python
from music_elf import GigaMidiDataset

dataset = GigaMidiDataset()
first_midi_path = dataset[0]
```

Pass a transform to load or tokenize each MIDI file on demand:

```python
from music_elf import Midi

dataset = GigaMidiDataset(transform=Midi.load)
midi = dataset[0]  # Midi(notes=[Note(...), ...])
```

`Midi.load` parses note events into canonical `Midi` and `Note` containers.
Explicit note-offs and velocity-zero note-ons are normalized to the same note
representation. `MidiTokenizer` remains a placeholder for the later vocabulary
and token-ID stage.

The representation retains notes only. Note start and duration values are in
seconds; source tempo and all other events are discarded after conversion:

```python
midi = Midi.load("input.mid")
midi.save("restored.mid")  # Fresh 120 BPM, 480-TPB type-0 MIDI
```

Eyeball a dataset sample and its first 50 notes with:

```console
uv run python scripts/eyeball_midi.py
```

Choose another manifest entry or inspect a file directly:

```console
uv run python scripts/eyeball_midi.py --index 1000 --limit 100
uv run python scripts/eyeball_midi.py path/to/example.mid --limit 100
```

Rebuild a sample for side-by-side listening:

```console
uv run python scripts/earcheck_midi.py --index 0
uv run python scripts/earcheck_midi.py path/to/example.mid --output test_outputs/earcheck/rebuilt.mid
```

The script copies the original to `test_outputs/earcheck/source.mid`, writes the
rebuilt MIDI next to it, and prints both paths, note counts, and the maximum
timing quantization error. Listen to both files with the same MIDI player and
soundfont.

Run the tests with:

```console
uv run pytest
```
