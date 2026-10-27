"""Fixtures: a tiny synthetic dataset and a fake model, so the whole pipeline
can be tested without downloading anything."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tsbench.datasets import SAMPLE_RATE, Audio, ManifestDataset
from tsbench.models import STTModel, register
from tsbench.models import base as models_base

# id -> (reference, what the fake model "hears", duration seconds)
SAMPLES = {
    "utt1": ("İstanbul'da hava güzel.", "istanbulda hava güzel", 1.0),
    "utt2": ("Kâğıt kalemi ver.", "kağıt kalem ver", 1.5),  # 1 word error
    "utt3": ("ISPARTA'ya gidiyoruz", "ıspartaya gidiyoruz", 2.0),
}


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


class FakeModel(STTModel):
    """Returns a scripted transcript based on audio length; sleeps 10% of audio time."""

    backend = "fake"

    def __init__(self, name: str, threads: int) -> None:
        super().__init__(name, threads)
        self.loaded = False
        self.calls = 0
        self._by_len = {int(d * SAMPLE_RATE): hyp for _ref, hyp, d in SAMPLES.values()}

    def prepare(self) -> None:
        time.sleep(1.0)  # pretend to download; must not show up in load_s

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, audio: Audio) -> str:
        assert self.loaded, "transcribe() before load()"
        assert audio.dtype == np.float32
        self.calls += 1
        time.sleep(len(audio) / SAMPLE_RATE * 0.1)
        return self._by_len[len(audio)]


@pytest.fixture(autouse=True)
def fake_backend() -> Iterator[None]:
    register("fake", FakeModel)
    yield
    models_base._REGISTRY.pop("fake", None)
