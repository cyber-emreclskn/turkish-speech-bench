from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from tsbench.cli import cli


def test_version() -> None:
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "turkish-speech-bench" in result.output


def test_models() -> None:
    result = CliRunner().invoke(cli, ["models"])
    assert result.exit_code == 0
    assert "faster-whisper:small" in result.output


def test_normalize_command() -> None:
    result = CliRunner().invoke(cli, ["normalize", "İSTANBUL'da IĞDIR!"])
    assert result.output.strip() == "istanbulda ığdır"


def test_run_and_report(manifest: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "-m",
            "fake:a",
            "-m",
            "fake:b",
            "-d",
            str(manifest),
            "-t",
            "1",
            "--label",
            "ci",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    files = sorted(out.rglob("*.json"))
    assert len(files) == 2
    assert "| `fake:a` | 12.5% |" in result.output

    md = runner.invoke(cli, ["report", str(out)])
    assert md.exit_code == 0
    assert "### ci · manifest:mini" in md.output

    as_json = json.loads(runner.invoke(cli, ["report", str(out), "--format", "json"]).output)
    assert {r["model"] for r in as_json} == {"fake:a", "fake:b"}

    csv_out = runner.invoke(cli, ["report", str(out), "--format", "csv"]).output
    assert csv_out.splitlines()[0].startswith("hardware,")


def test_run_unknown_model(manifest: Path) -> None:
    result = CliRunner().invoke(cli, ["run", "-m", "nope:x", "-d", str(manifest)])
    assert result.exit_code == 2
    assert "unknown backend" in result.output


def test_run_unknown_dataset() -> None:
    result = CliRunner().invoke(cli, ["run", "-m", "fake:x", "-d", "librispeech"])
    assert result.exit_code == 2


def test_report_empty(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["report", str(tmp_path)])
    assert result.exit_code == 1
    assert "no result files" in result.output
