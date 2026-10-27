"""Command-line interface: ``tsb``."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskID, TimeRemainingColumn
from rich.table import Table

from . import __version__

if TYPE_CHECKING:
    from .datasets import Dataset

console = Console(stderr=True)

SUGGESTED = [
    ("faster-whisper:tiny", "whisper", "39M params, fastest Whisper"),
    ("faster-whisper:base", "whisper", "74M"),
    ("faster-whisper:small", "whisper", "244M, common on-device choice"),
    ("faster-whisper:large-v3-turbo", "whisper", "809M, best accuracy that is still CPU-usable"),
    ("vosk:vosk-model-small-tr-0.3", "vosk", "Kaldi, ~35 MB, streaming"),
    ("hf-ctc:mpoyraz/wav2vec2-xls-r-300m-cv7-turkish", "ctc", "300M wav2vec2 fine-tuned on CV7"),
]

_RESULT_FILE = "TSB_RESULT_FILE"  # set by the parent when running a model in a child
_ERROR_FILE = "TSB_ERROR_FILE"

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CT2_NUM_THREADS")


def _pin_threads(threads: int) -> None:
    # Must happen before numerical backends are imported.
    for var in _THREAD_VARS:
        os.environ[var] = str(threads)


@click.group()
@click.version_option(__version__, prog_name="turkish-speech-bench")
def cli() -> None:
    """Benchmark Turkish speech recognition on CPU."""
    # Third-party backends: TSB_PLUGINS="my_pkg.backends,other" imports modules that
    # call tsbench.models.register(). Also applies to isolated child processes.
    for module in filter(None, os.environ.get("TSB_PLUGINS", "").split(",")):
        importlib.import_module(module.strip())


@cli.command("models")
def models_cmd() -> None:
    """List backends and suggested models."""
    from .models import backends

    table = Table(title="Suggested models")
    table.add_column("Spec", style="cyan")
    table.add_column("Extra")
    table.add_column("Notes", style="dim")
    for spec, extra, notes in SUGGESTED:
        table.add_row(spec, f"[{extra}]", notes)
    Console().print(table)
    Console().print(f"Backends: {', '.join(backends())}. Any 'backend:name' works.")


def _measure(
    spec: str,
    ds: Dataset,
    *,
    threads: int,
    limit: int | None,
    warmup: int,
    label: str | None,
    out: Path,
    keep_predictions: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """Benchmark one model in this process. Returns (result, error)."""
    from . import runner
    from .models import create
    from .report import save

    console.print(f"[bold]▶ {spec}[/bold] on {ds.name}")
    try:
        model = create(spec, threads)
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as bar:
            task = bar.add_task(spec, total=None)

            def on_progress(i: int, n: int, task: TaskID = task, bar: Progress = bar) -> None:
                bar.update(task, completed=i, total=n)

            result = runner.run(
                model,
                ds,
                limit=limit,
                warmup=warmup,
                hardware_label=label,
                keep_predictions=keep_predictions,
                progress=on_progress,
            )
    except Exception as exc:  # one broken model must not lose the others' results
        error = f"{type(exc).__name__}: {exc}"
        console.print(f"[red]✗ {spec} failed: {error}[/red]")
        return None, error
    path = save(result, out)
    m = result["metrics"]
    console.print(
        f"  WER {m['wer']:.2%} · CER {m['cer']:.2%} · RTFx {m['rtfx']} · "
        f"peak RAM {m['peak_rss_mb']} MB → [dim]{path}[/dim]"
    )
    if os.environ.get(_RESULT_FILE):
        Path(os.environ[_RESULT_FILE]).write_text(str(path), encoding="utf-8")
    return result, None


def _measure_isolated(spec: str, child_args: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    """Run one model in a fresh interpreter so memory left behind by the previous
    model (allocators rarely return it to the OS) can't inflate the next one's RAM."""
    with tempfile.TemporaryDirectory() as tmp:
        result_file, error_file = Path(tmp) / "result", Path(tmp) / "error"
        env = {**os.environ, _RESULT_FILE: str(result_file), _ERROR_FILE: str(error_file)}
        proc = subprocess.run(
            [sys.executable, "-m", "tsbench.cli", "run", "-m", spec, *child_args],
            env=env,
            stdout=subprocess.DEVNULL,  # child's summary table; the parent prints a combined one
            check=False,
        )
        if result_file.exists():
            path = Path(result_file.read_text(encoding="utf-8"))
            return json.loads(path.read_text(encoding="utf-8")), None
        if error_file.exists():
            return None, error_file.read_text(encoding="utf-8")
        return None, f"exited with code {proc.returncode}"


@cli.command()
@click.option(
    "-m", "--model", "models", multiple=True, required=True, help="backend:name (repeatable)"
)
@click.option(
    "-d",
    "--dataset",
    default="fleurs-tr",
    show_default=True,
    help="fleurs-tr[:split] or manifest.jsonl",
)
@click.option("-n", "--limit", type=int, help="Only the first N utterances (deterministic)")
@click.option("-t", "--threads", type=int, default=4, show_default=True)
@click.option("--warmup", type=int, default=1, show_default=True)
@click.option("--label", help="Human-readable hardware name, e.g. 'kiosk-n100'")
@click.option(
    "-o", "--out", type=click.Path(path_type=Path), default=Path("results"), show_default=True
)
@click.option("--no-predictions", is_flag=True, help="Don't store per-utterance transcripts")
@click.option(
    "--isolate/--no-isolate",
    default=True,
    show_default=True,
    help="Run each model in its own process (accurate memory numbers)",
)
def run(
    models: tuple[str, ...],
    dataset: str,
    limit: int | None,
    threads: int,
    warmup: int,
    label: str | None,
    out: Path,
    no_predictions: bool,
    isolate: bool,
) -> None:
    """Benchmark one or more models."""
    _pin_threads(threads)
    from .datasets import get_dataset
    from .models import create
    from .report import markdown

    for spec in models:  # validate every spec before spending time on any of them
        try:
            create(spec, threads)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
    try:
        ds = get_dataset(dataset)  # also downloads it once, before any child starts
    except (ValueError, FileNotFoundError) as exc:
        raise click.UsageError(str(exc)) from exc

    child_args = ["-d", dataset, "-t", str(threads), "--warmup", str(warmup), "-o", str(out)]
    child_args += ["--no-isolate"]
    if limit is not None:
        child_args += ["-n", str(limit)]
    if label:
        child_args += ["--label", label]
    if no_predictions:
        child_args += ["--no-predictions"]

    results: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for spec in models:
        if isolate and len(models) > 1:
            result, error = _measure_isolated(spec, child_args)
        else:
            result, error = _measure(
                spec,
                ds,
                threads=threads,
                limit=limit,
                warmup=warmup,
                label=label,
                out=out,
                keep_predictions=not no_predictions,
            )
        if result is not None:
            results.append(result)
        else:
            failures.append((spec, error or "unknown error"))
            if os.environ.get(_ERROR_FILE):
                Path(os.environ[_ERROR_FILE]).write_text(error or "", encoding="utf-8")

    if results:
        click.echo(markdown(results))
    if failures:
        click.echo("\n**Failed models**\n")
        for spec, error in failures:
            click.echo(f"- `{spec}`: {error}")
        sys.exit(1)


@cli.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format", "fmt", type=click.Choice(["md", "csv", "json"]), default="md", show_default=True
)
def report(paths: tuple[Path, ...], fmt: str) -> None:
    """Summarize result files (default: ./results)."""
    import csv
    import json

    from .report import load_results, markdown, rows

    results = load_results(paths or (Path("results"),))
    if not results:
        raise click.ClickException("no result files found")
    if fmt == "md":
        click.echo(markdown(results))
    elif fmt == "json":
        click.echo(json.dumps(rows(results), ensure_ascii=False, indent=2))
    else:
        table = rows(results)
        writer = csv.DictWriter(sys.stdout, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


@cli.command()
@click.argument("text")
def normalize(text: str) -> None:
    """Show how a transcript is normalized before scoring."""
    from .normalize import normalize as norm

    click.echo(norm(text))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
