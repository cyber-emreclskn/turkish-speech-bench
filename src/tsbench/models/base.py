"""Model interface and registry."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..datasets import Audio


class MissingBackendError(RuntimeError):
    """Raised when a backend's optional dependencies are not installed."""

    def __init__(self, backend: str, extra: str) -> None:
        super().__init__(
            f"backend {backend!r} needs extra packages: pip install 'turkish-speech-bench[{extra}]'"
        )


@dataclass
class ModelInfo:
    spec: str
    backend: str
    name: str
    license: str = "unknown"
    params_millions: float | None = None
    settings: dict[str, Any] = field(default_factory=dict)


class STTModel(ABC):
    """A speech-to-text system under test.

    Implementations must do all heavy work (downloads, weight loading) in
    :meth:`load` so it can be timed separately from transcription.
    """

    backend: str = ""
    extra: str = ""

    def __init__(self, name: str, threads: int) -> None:
        self.name = name
        self.threads = threads

    @property
    def spec(self) -> str:
        return f"{self.backend}:{self.name}"

    def prepare(self) -> None:  # noqa: B027 - optional hook
        """Download weights etc. Not timed, so network speed never affects results."""

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory. Timed as ``load_s``."""

    @abstractmethod
    def transcribe(self, audio: Audio) -> str: ...

    def info(self) -> ModelInfo:
        return ModelInfo(spec=self.spec, backend=self.backend, name=self.name)

    def files(self) -> list[Path]:
        """Model files on disk, used to report download size. Empty if unknown."""
        return []

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""


ModelFactory = Callable[[str, int], STTModel]
_REGISTRY: dict[str, ModelFactory] = {}

# backend name -> "module:Class", imported lazily so heavy deps load only when used
_BUILTIN = {
    "faster-whisper": "tsbench.models.faster_whisper:FasterWhisper",
    "vosk": "tsbench.models.vosk:Vosk",
    "hf-ctc": "tsbench.models.hf_ctc:HFCTC",
}


def register(backend: str, factory: ModelFactory) -> None:
    """Register a backend (also how plugins and tests add models)."""
    _REGISTRY[backend] = factory


def backends() -> list[str]:
    return sorted({*_BUILTIN, *_REGISTRY})


def create(spec: str, threads: int) -> STTModel:
    """Build a model from ``backend:name``, e.g. ``faster-whisper:small``."""
    backend, sep, name = spec.partition(":")
    if not sep or not name:
        raise ValueError(f"model spec must be 'backend:name', got {spec!r}")
    if backend in _REGISTRY:
        return _REGISTRY[backend](name, threads)
    if backend in _BUILTIN:
        module, _, cls = _BUILTIN[backend].partition(":")
        factory: ModelFactory = getattr(importlib.import_module(module), cls)
        return factory(name, threads)
    raise ValueError(f"unknown backend {backend!r}; available: {', '.join(backends())}")


def dir_files(path: Path) -> list[Path]:
    return [p for p in Path(path).rglob("*") if p.is_file()]
