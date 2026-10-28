"""
Integration Tests for Click CLI Commands.
"""

import pytest
from click.testing import CliRunner
from microquant.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Aetheris-MicroQuant" in result.output


def test_cli_clean():
    runner = CliRunner()
    result = runner.invoke(main, ["clean"])
    assert result.exit_code == 0
    assert "Cleaned data saved" in result.output


def test_cli_features():
    runner = CliRunner()
    result = runner.invoke(main, ["features"])
    assert result.exit_code == 0
    assert "Features saved" in result.output


def test_cli_resample():
    runner = CliRunner()
    result = runner.invoke(main, ["resample", "--tf", "1min"])
    assert result.exit_code == 0
    assert "Resampled candles saved" in result.output
