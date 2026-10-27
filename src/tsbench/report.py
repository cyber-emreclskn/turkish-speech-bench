"""Load result files and render leaderboards."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .runner import SCHEMA_VERSION

Result = dict[str, Any]


def result_filename(result: Result) -> Path:
    """Stable path for a result: one file per (hardware, dataset, model, threads, subset)."""

    def slug(s: str) -> str:
        return re.sub(r"[^a-z0-9.]+", "-", s.lower()).strip("-")

    sys_, ds, model, run = result["system"], result["dataset"], result["model"], result["run"]
    name = (
        f"{slug(ds['name'])}__{slug(model['spec'])}__t{run['threads']}__{slug(ds['subset'])}.json"
    )
    return Path(sys_["hardware_id"]) / name


def save(result: Result, results_dir: Path) -> Path:
    path = Path(results_dir) / result_filename(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_results(paths: Iterable[Path]) -> list[Result]:
    """Load result JSON files (directories are searched recursively)."""
    results = []
    for p in paths:
        p = Path(p)
        files = sorted(p.rglob("*.json")) if p.is_dir() else [p]
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("schema_version") != SCHEMA_VERSION or "metrics" not in data:
                continue
            results.append(data)
    return results


def rows(results: list[Result]) -> list[dict[str, Any]]:
    """Flatten results into leaderboard rows, sorted by hardware then WER."""
    out = []
    for r in results:
        m, model, sys_, ds = r["metrics"], r["model"], r["system"], r["dataset"]
        out.append(
            {
                "hardware": sys_["label"],
                "hardware_id": sys_["hardware_id"],
                "dataset": ds["name"],
                "subset": ds["subset"],
                "utterances": ds["utterances"],
                "model": model["spec"],
                "params_m": model.get("params_millions"),
                "threads": r["run"]["threads"],
                "wer": m["wer"],
                "cer": m["cer"],
                "rtfx": m["rtfx"],
                "latency_p90_s": m["latency_p90_s"],
                "peak_rss_mb": m["peak_rss_mb"],
                "disk_mb": model.get("disk_mb"),
                "date": r["created_at"][:10],
            }
        )
    return sorted(out, key=lambda x: (x["hardware"], x["dataset"], x["wer"]))


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "–"
    if pct:
        return f"{v * 100:.1f}%"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def markdown(results: list[Result]) -> str:
    """One table per (hardware, dataset), best WER first."""
    lines: list[str] = []
    current: tuple[str, str, str] | None = None
    for row in rows(results):
        key = (row["hardware"], row["dataset"], row["subset"])
        if key != current:
            current = key
            lines += [
                "",
                f"### {row['hardware']} · {row['dataset']} ({row['subset']}, {row['utterances']} utt.)",
                "",
                "| Model | WER ↓ | CER ↓ | RTFx ↑ | p90 latency | Peak RAM | Threads |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        lines.append(
            f"| `{row['model']}` | {_fmt(row['wer'], True)} | {_fmt(row['cer'], True)} "
            f"| {_fmt(row['rtfx'])} | {_fmt(row['latency_p90_s'])} s "
            f"| {_fmt(row['peak_rss_mb'])} MB | {row['threads']} |"
        )
    return "\n".join(lines).strip() + "\n"
