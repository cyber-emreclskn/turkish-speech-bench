"""Vosk (Kaldi) — a classic lightweight CPU baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets import SAMPLE_RATE, Audio
from .base import MissingBackendError, ModelInfo, STTModel, dir_files

CHUNK = 8000  # samples fed per call (0.5 s), mimicking streaming use


class Vosk(STTModel):
    """``vosk:<model-name>``, e.g. ``vosk:vosk-model-small-tr-0.3`` (auto-downloaded)."""

    backend = "vosk"
    extra = "vosk"

    def __init__(self, name: str, threads: int) -> None:
        super().__init__(name, threads)
        self._model: Any = None
        self._vosk: Any = None

    def load(self) -> None:
        try:
            import vosk
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise MissingBackendError(self.backend, self.extra) from exc
        vosk.SetLogLevel(-1)
        self._vosk = vosk
        if Path(self.name).is_dir():
            self._model = vosk.Model(model_path=self.name)
        else:
            self._model = vosk.Model(model_name=self.name)

    def transcribe(self, audio: Audio) -> str:
        rec = self._vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        parts = []
        for start in range(0, len(pcm), CHUNK):
            if rec.AcceptWaveform(pcm[start : start + CHUNK].tobytes()):
                parts.append(json.loads(rec.Result()).get("text", ""))
        parts.append(json.loads(rec.FinalResult()).get("text", ""))
        return " ".join(p for p in parts if p)

    def info(self) -> ModelInfo:
        return ModelInfo(
            spec=self.spec,
            backend=self.backend,
            name=self.name,
            license="Apache-2.0",
            settings={"chunk_samples": CHUNK},
        )

    def files(self) -> list[Path]:
        # vosk.Model doesn't expose its path; check the locations vosk itself searches.
        roots = [Path(self.name)]
        roots += [Path(d) / self.name for d in getattr(self._vosk, "MODEL_DIRS", []) if d]
        for root in roots:
            if root.is_dir():
                return dir_files(root)
        return []
