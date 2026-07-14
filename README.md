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

Run the tests with:

```console
uv run pytest
```
