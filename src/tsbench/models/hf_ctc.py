"""Any CTC model from the Hugging Face Hub (wav2vec2, HuBERT, XLS-R...) with greedy decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..datasets import SAMPLE_RATE, Audio
from .base import MissingBackendError, ModelInfo, STTModel, dir_files


class HFCTC(STTModel):
    """``hf-ctc:<repo-id>``, e.g. ``hf-ctc:mpoyraz/wav2vec2-xls-r-300m-cv7-turkish``."""

    backend = "hf-ctc"
    extra = "ctc"

    def __init__(self, name: str, threads: int) -> None:
        super().__init__(name, threads)
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    def prepare(self) -> None:
        from huggingface_hub import snapshot_download

        if not Path(self.name).is_dir():
            snapshot_download(self.name)

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise MissingBackendError(self.backend, self.extra) from exc
        torch.set_num_threads(self.threads)
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.name)
        self._model = AutoModelForCTC.from_pretrained(self.name).eval()

    def transcribe(self, audio: Audio) -> str:
        inputs = self._processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
        ids = logits.argmax(dim=-1)
        return str(self._processor.batch_decode(ids)[0])

    def info(self) -> ModelInfo:
        params = None
        if self._model is not None:
            params = round(sum(p.numel() for p in self._model.parameters()) / 1e6, 1)
        return ModelInfo(
            spec=self.spec,
            backend=self.backend,
            name=self.name,
            params_millions=params,
            settings={"decoding": "greedy", "dtype": "float32"},
        )

    def files(self) -> list[Path]:
        path = getattr(getattr(self._model, "config", None), "_name_or_path", None)
        if path and Path(path).is_dir():
            return dir_files(Path(path))
        try:
            from huggingface_hub import snapshot_download

            return dir_files(Path(snapshot_download(self.name, local_files_only=True)))
        except Exception:  # pragma: no cover - cache layout varies
            return []
