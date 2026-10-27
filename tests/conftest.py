"""Fixtures: a tiny synthetic dataset and a fake model, so the whole pipeline
can be tested without downloading anything."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tsbench.datasets import SAMPLE_RATE, ManifestDataset
from tsbench.models import base as models_base

from .fake_plugin import SAMPLES, FakeModel, install

__all__ = ["SAMPLES", "FakeModel"]

ROOT = Path(__file__).resolve().parent.parent


def tone(seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    lines = []
    for uid, (ref, _hyp, dur) in SAMPLES.items():
        # write utt3 at 8 kHz stereo to exercise resampling/downmix
        if uid == "utt3":
            data = np.stack([tone(dur, 8000)] * 2, axis=1)
            sf.write(audio_dir / f"{uid}.wav", data, 8000)
        else:
            sf.write(audio_dir / f"{uid}.wav", tone(dur), SAMPLE_RATE)
        lines.append(
            json.dumps({"id": uid, "audio": f"audio/{uid}.wav", "text": ref}, ensure_ascii=False)
        )
    path = tmp_path / "mini.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def dataset(manifest: Path) -> ManifestDataset:
    return ManifestDataset(manifest)


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    install()
    # make the fake backends available to isolated child processes too
    monkeypatch.setenv("TSB_PLUGINS", "tests.fake_plugin")
    monkeypatch.setenv("PYTHONPATH", str(ROOT), prepend=os.pathsep)
    yield
    for name in ("fake", "boom"):
        models_base._REGISTRY.pop(name, None)
