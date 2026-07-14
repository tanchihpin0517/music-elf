# ELF

An empty command-line scaffold for ELF model training and inference.

## Setup

Install the project and its development dependencies with
[uv](https://docs.astral.sh/uv/):

```console
uv sync
```

Inspect the CLI:

```console
uv run elf --help
```

## Commands

The six command combinations are:

```console
uv run elf train elf
uv run elf train encoder
uv run elf train lm
uv run elf infer elf
uv run elf infer encoder
uv run elf infer lm
```

Each command is currently a placeholder. It reports that it is not implemented
and exits with status 1.

Run the tests with:

```console
uv run pytest
```
