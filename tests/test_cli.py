"""Tests for the ELF command-line interface."""

import pytest
from typer.testing import CliRunner

from music_elf.cli import app

runner = CliRunner()


def test_root_help_lists_command_groups() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "infer" in result.stdout


@pytest.mark.parametrize("operation", ["train", "infer"])
def test_group_help_lists_components(operation: str) -> None:
    result = runner.invoke(app, [operation, "--help"])

    assert result.exit_code == 0
    assert "elf" in result.stdout
    assert "encoder" in result.stdout
    assert "lm" in result.stdout


@pytest.mark.parametrize("operation", ["train", "infer"])
@pytest.mark.parametrize("component", ["elf", "encoder", "lm"])
def test_placeholder_commands(operation: str, component: str) -> None:
    result = runner.invoke(app, [operation, component])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == f"Not implemented: music-elf {operation} {component}\n"


@pytest.mark.parametrize(
    "arguments",
    [
        ["build"],
        ["train", "transformer"],
        ["infer", "transformer"],
    ],
)
def test_unknown_commands_are_usage_errors(arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    assert "No such command" in result.stderr
