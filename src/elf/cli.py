"""Command-line interface for ELF."""

from typing import NoReturn

import typer

app = typer.Typer(help="Train and run inference with ELF components.")
train_app = typer.Typer(help="Train an ELF component.")
infer_app = typer.Typer(help="Run inference with an ELF component.")

app.add_typer(train_app, name="train")
app.add_typer(infer_app, name="infer")


def _not_implemented(operation: str, component: str) -> NoReturn:
    """Report a placeholder command and exit unsuccessfully."""
    typer.echo(f"Not implemented: elf {operation} {component}", err=True)
    raise typer.Exit(code=1)


@train_app.command(name="elf")
def train_elf() -> None:
    """Train the ELF model."""
    _not_implemented("train", "elf")


@train_app.command(name="encoder")
def train_encoder() -> None:
    """Train the encoder."""
    _not_implemented("train", "encoder")


@train_app.command(name="lm")
def train_lm() -> None:
    """Train the language model."""
    _not_implemented("train", "lm")


@infer_app.command(name="elf")
def infer_elf() -> None:
    """Run inference with the ELF model."""
    _not_implemented("infer", "elf")


@infer_app.command(name="encoder")
def infer_encoder() -> None:
    """Run inference with the encoder."""
    _not_implemented("infer", "encoder")


@infer_app.command(name="lm")
def infer_lm() -> None:
    """Run inference with the language model."""
    _not_implemented("infer", "lm")
