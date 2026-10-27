"""Run one model over one dataset and produce a result record."""

from __future__ import annotations

import gc
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import psutil

from . import __version__
from .datasets import Dataset, Utterance
from .metrics import cer, percentile, rtf, wer
from .models import STTModel
from .normalize import normalize
from .system import fingerprint

SCHEMA_VERSION = 1

ProgressFn = Callable[[int, int], None]


class PeakMemory:
    """Samples this process's resident memory in a background thread."""

    def __init__(self, interval: float = 0.02) -> None:
        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak = self.current()

    def current(self) -> int:
        return int(self._proc.memory_info().rss)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.peak = max(self.peak, self.current())

    def __enter__(self) -> PeakMemory:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.current())


def _mb(n: int) -> float:
    return round(n / 1024**2, 1)


def run(
    model: STTModel,
    dataset: Dataset,
    *,
    limit: int | None = None,
    warmup: int = 1,
    hardware_label: str | None = None,
    keep_predictions: bool = True,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Benchmark ``model`` on ``dataset``.

    Timing covers only ``model.transcribe`` (audio is decoded beforehand), so the
    numbers reflect the model, not file I/O. The first ``warmup`` utterances are
    transcribed once untimed to exclude one-off JIT/allocation costs, and then again
    as part of the timed run so every utterance is scored.
    """
    utterances: list[Utterance] = list(dataset.utterances(limit))
    if not utterances:
        raise ValueError("no utterances to evaluate")

    model.prepare()  # downloads happen here, outside every measurement
    gc.collect()
    with PeakMemory() as mem:
        baseline = mem.current()
        t0 = time.perf_counter()
        model.load()
        load_s = time.perf_counter() - t0
        after_load = mem.current()

        for utt in utterances[:warmup]:
            model.transcribe(utt.audio)

        rows: list[dict[str, Any]] = []
        refs: list[str] = []
        hyps: list[str] = []
        latencies: list[float] = []
        audio_s = 0.0
        for i, utt in enumerate(utterances, 1):
            t = time.perf_counter()
            hyp = model.transcribe(utt.audio)
            elapsed = time.perf_counter() - t
            refs.append(utt.text)
            hyps.append(hyp)
            latencies.append(elapsed)
            audio_s += utt.duration
            row: dict[str, Any] = {
                "id": utt.id,
                "ref": utt.text,
                "hyp": hyp,
                "duration_s": round(utt.duration, 3),
                "latency_s": round(elapsed, 4),
            }
            if keep_predictions:
                row["ref_norm"] = normalize(utt.text)
                row["hyp_norm"] = normalize(hyp)
            rows.append(row)
            if progress:
                progress(i, len(utterances))

    proc_s = sum(latencies)
    model_files = model.files()
    model.close()

    run_rtf = rtf(proc_s, audio_s)
    return {
        "schema_version": SCHEMA_VERSION,
        "tsbench_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": fingerprint(hardware_label),
        "dataset": {
            "name": dataset.name,
            "license": dataset.license,
            "utterances": len(rows),
            "subset": "full" if limit is None or limit >= len(dataset) else f"first-{limit}",
            "audio_seconds": round(audio_s, 2),
        },
        "model": {
            **asdict(model.info()),
            "disk_mb": _mb(sum(p.stat().st_size for p in model_files)) if model_files else None,
        },
        "run": {"threads": model.threads, "warmup": warmup},
        "metrics": {
            "wer": round(wer(refs, hyps), 4),
            "cer": round(cer(refs, hyps), 4),
            "rtf": round(run_rtf, 4),
            "rtfx": round(1 / run_rtf, 2) if run_rtf else None,
            "latency_p50_s": round(percentile(latencies, 50), 4),
            "latency_p90_s": round(percentile(latencies, 90), 4),
            "load_s": round(load_s, 2),
            "model_ram_mb": _mb(max(after_load - baseline, 0)),
            "peak_rss_mb": _mb(mem.peak),
        },
        "predictions": rows if keep_predictions else [],
    }
