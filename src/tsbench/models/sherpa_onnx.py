"""Transducer models (Zipformer, Conformer…) through sherpa-onnx / ONNX Runtime.

Covers the small streaming models that are built for on-device use, which is
exactly the hardware this benchmark cares about.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets import SAMPLE_RATE, Audio
from .base import MissingBackendError, ModelInfo, STTModel, dir_files

_PARTS = ("encoder", "decoder", "joiner")
# Streaming models only emit their last tokens once they have seen some trailing audio.
TAIL_PADDING_S = 0.66


def _pick(root: Path, part: str) -> Path:
    """Find ``<part>*.onnx``, preferring the INT8 export like the other CPU backends."""
    candidates = sorted(root.rglob(f"{part}*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"no {part}*.onnx in {root}")
    int8 = [p for p in candidates if "int8" in p.name]
    return (int8 or candidates)[0]


class SherpaOnnx(STTModel):
    """``sherpa-onnx:<hf-repo-or-dir>``, e.g. ``sherpa-onnx:mihuai/turkish-stt-66m-v2``.

    The directory needs ``tokens.txt`` plus encoder, decoder and joiner ``.onnx`` files.
    If it ships a ``config.json`` (``streaming``, ``decoding_method``, ``max_active_paths``,
    ``feature_dim``, ``files``), those settings are used, so a model runs the way its
    authors intended. Otherwise a model counts as streaming when its name says so
    (the k2-fsa naming convention) and is decoded with greedy search.
    """

    backend = "sherpa-onnx"
    extra = "sherpa"

    def __init__(self, name: str, threads: int) -> None:
        super().__init__(name, threads)
        self._path: Path | None = None
        self._files: dict[str, Path] = {}
        self._config: dict[str, Any] = {}
        self._license = "unknown"
        self._recognizer: Any = None
        self.streaming = False
        self.decoding_method = "greedy_search"
        self.max_active_paths = 4
        self.feature_dim = 80

    def prepare(self) -> None:
        local = Path(self.name)
        if local.is_dir():
            self._path = local
        else:
            from huggingface_hub import model_info, snapshot_download

            self._path = Path(
                snapshot_download(
                    self.name,
                    allow_patterns=["*.onnx", "tokens.txt", "config.json", "LICENSE*"],
                )
            )
            try:
                card = model_info(self.name).card_data
                self._license = str(getattr(card, "license", None) or "unknown")
            except Exception:  # the license is nice to have, never worth failing a run
                self._license = "unknown"
        self._resolve()

    def _resolve(self) -> None:
        root = self._path
        assert root is not None
        config_file = root / "config.json"
        if config_file.is_file():
            self._config = json.loads(config_file.read_text(encoding="utf-8"))
        named: dict[str, str] = self._config.get("files", {})

        def locate(part: str) -> Path:
            if part in named and (root / named[part]).is_file():
                return root / named[part]
            if part == "tokens":
                found = sorted(root.rglob("tokens.txt"))
                if not found:
                    raise FileNotFoundError(f"no tokens.txt in {root}")
                return found[0]
            return _pick(root, part)

        self._files = {part: locate(part) for part in ("tokens", *_PARTS)}
        self.streaming = bool(self._config.get("streaming", "streaming" in self.name.lower()))
        self.decoding_method = str(self._config.get("decoding_method", self.decoding_method))
        self.max_active_paths = int(self._config.get("max_active_paths", self.max_active_paths))
        self.feature_dim = int(self._config.get("feature_dim", self.feature_dim))

    def load(self) -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise MissingBackendError(self.backend, self.extra) from exc
        if self._path is None:
            self.prepare()
        factory = sherpa_onnx.OnlineRecognizer if self.streaming else sherpa_onnx.OfflineRecognizer
        self._recognizer = factory.from_transducer(
            tokens=str(self._files["tokens"]),
            encoder=str(self._files["encoder"]),
            decoder=str(self._files["decoder"]),
            joiner=str(self._files["joiner"]),
            num_threads=self.threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=self.feature_dim,
            decoding_method=self.decoding_method,
            max_active_paths=self.max_active_paths,
            provider="cpu",
        )

    def transcribe(self, audio: Audio) -> str:
        rec = self._recognizer
        stream = rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        if not self.streaming:
            rec.decode_stream(stream)
            return str(stream.result.text).strip()
        stream.accept_waveform(
            SAMPLE_RATE, np.zeros(int(TAIL_PADDING_S * SAMPLE_RATE), dtype=np.float32)
        )
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        result = rec.get_result(stream)
        return str(getattr(result, "text", result)).strip()

    def info(self) -> ModelInfo:
        params = str(self._config.get("num_parameters", "")).upper().removesuffix("M")
        return ModelInfo(
            spec=self.spec,
            backend=self.backend,
            name=self.name,
            license=self._license,
            params_millions=float(params) if params.replace(".", "", 1).isdigit() else None,
            settings={
                "streaming": self.streaming,
                "decoding_method": self.decoding_method,
                "max_active_paths": self.max_active_paths,
                "quantization": "int8" if "int8" in self._files["encoder"].name else "none",
            }
            if self._files
            else {},
        )

    def files(self) -> list[Path]:
        # Only the files that are actually loaded: a repo may ship several exports.
        return list(self._files.values()) if self._files else dir_files(self._path or Path())
