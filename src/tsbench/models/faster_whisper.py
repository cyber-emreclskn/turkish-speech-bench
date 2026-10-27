"""Whisper via CTranslate2 (faster-whisper), INT8 on CPU by default."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..datasets import Audio
from .base import MissingBackendError, ModelInfo, STTModel, dir_files

# Approximate parameter counts of the OpenAI checkpoints (millions).
_PARAMS = {
    "tiny": 39,
    "base": 74,
    "small": 244,
    "medium": 769,
    "large-v2": 1550,
    "large-v3": 1550,
    "large-v3-turbo": 809,
    "turbo": 809,
}


class FasterWhisper(STTModel):
    backend = "faster-whisper"
    extra = "whisper"

    def __init__(
        self, name: str, threads: int, compute_type: str = "int8", beam_size: int = 5
    ) -> None:
        super().__init__(name, threads)
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model: Any = None
        self._path: Path | None = None

    def prepare(self) -> None:
        try:
            from faster_whisper.utils import download_model
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise MissingBackendError(self.backend, self.extra) from exc
        # Resolve the local directory ourselves so we can report the model's size on disk.
        local = Path(self.name)
        self._path = local if local.is_dir() else Path(download_model(self.name))

    def load(self) -> None:
        from faster_whisper import WhisperModel

        if self._path is None:
            self.prepare()
        self._model = WhisperModel(
            str(self._path), device="cpu", compute_type=self.compute_type, cpu_threads=self.threads
        )

    def transcribe(self, audio: Audio) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language="tr",
            task="transcribe",
            beam_size=self.beam_size,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segments)

    def info(self) -> ModelInfo:
        return ModelInfo(
            spec=self.spec,
            backend=self.backend,
            name=self.name,
            license="MIT",
            params_millions=_PARAMS.get(self.name),
            settings={"compute_type": self.compute_type, "beam_size": self.beam_size},
        )

    def files(self) -> list[Path]:
        return dir_files(self._path) if self._path else []
