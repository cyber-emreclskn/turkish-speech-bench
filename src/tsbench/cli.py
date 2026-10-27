"""Command-line interface: ``tsb``."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskID, TimeRemainingColumn
from rich.table import Table

from . import __version__

console = Console(stderr=True)

SUGGESTED = [
    ("faster-whisper:tiny", "whisper", "39M params, fastest Whisper"),
    ("faster-whisper:base", "whisper", "74M"),
    ("faster-whisper:small", "whisper", "244M, common on-device choice"),
    ("faster-whisper:large-v3-turbo", "whisper", "809M, best accuracy that is still CPU-usable"),
    ("vosk:vosk-model-small-tr-0.3", "vosk", "Kaldi, ~35 MB, streaming"),
    ("hf-ctc:mpoyraz/wav2vec2-xls-r-300m-cv7-turkish", "ctc", "300M wav2vec2 fine-tuned on CV7"),
]

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CT2_NUM_THREADS")


def _pin_threads(threads: int) -> None:
    # Must happen before numerical backends are imported.
    for var in _THREAD_VARS:
        os.environ[var] = str(threads)


@click.group()
@click.version_option(__version__, prog_name="turkish-speech-bench")
def cli() -> None:
    """Benchmark Turkish speech recognition on CPU."""


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
def run(
    models: tuple[str, ...],
    dataset: str,
    limit: int | None,
    threads: int,
    warmup: int,
    label: str | None,
    out: Path,
    no_predictions: bool,
) -> None:
    """Benchmark one or more models."""
    _pin_threads(threads)
    from . import runner
    from .datasets import get_dataset
    from .models import MissingBackendError, create
    from .report import markdown, save

    try:
        ds = get_dataset(dataset)
    except (ValueError, FileNotFoundError) as exc:
        raise click.UsageError(str(exc)) from exc

    results = []
    failed = False
    for spec in models:
        try:
            model = create(spec, threads)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        console.print(f"[bold]▶ {spec}[/bold] on {ds.name}")
        try:
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
                    keep_predictions=not no_predictions,
                    progress=on_progress,
                )
        except MissingBackendError as exc:
            console.print(f"[red]✗ {exc}[/red]")
            failed = True
            continue
        except Exception as exc:  # one broken model must not lose the others' results
            console.print(f"[red]✗ {spec} failed: {type(exc).__name__}: {exc}[/red]")
            failed = True
            continue
        finally:
            del model
            gc.collect()  # free the previous model before measuring the next one
        path = save(result, out)
        m = result["metrics"]
        console.print(
            f"  WER {m['wer']:.2%} · CER {m['cer']:.2%} · RTFx {m['rtfx']} · "
            f"peak RAM {m['peak_rss_mb']} MB → [dim]{path}[/dim]"
        )
        results.append(result)

    if results:
        click.echo(markdown(results))
    if failed:
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
