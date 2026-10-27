"""Evaluation datasets.

Every dataset yields :class:`Utterance` objects with 16 kHz mono float32 audio.
Subsets are deterministic (sorted by utterance id) so that ``--limit N`` means the
same N utterances on every machine.
"""

from __future__ import annotations

import csv
import json
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import soundfile as sf
import soxr

SAMPLE_RATE = 16_000

Audio = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Utterance:
    id: str
    text: str
    audio: Audio

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


@dataclass(frozen=True)
class _Entry:
    id: str
    text: str
    path: Path


class Dataset(Protocol):
    name: str
    license: str

    def __len__(self) -> int: ...

    def utterances(self, limit: int | None = None) -> Iterator[Utterance]: ...


def load_audio(path: Path) -> Audio:
    """Read an audio file as 16 kHz mono float32."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        mono = soxr.resample(mono, sr, SAMPLE_RATE)
    return np.ascontiguousarray(mono, dtype=np.float32)


class _FileDataset:
    name = "files"
    license = "unknown"

    def __init__(self, entries: list[_Entry]) -> None:
        if not entries:
            raise ValueError("dataset is empty")
        ids = [e.id for e in entries]
        if len(set(ids)) != len(ids):
            raise ValueError("utterance ids must be unique")
        self._entries = sorted(entries, key=lambda e: e.id)

    def __len__(self) -> int:
        return len(self._entries)

    def utterances(self, limit: int | None = None) -> Iterator[Utterance]:
        entries = self._entries if limit is None else self._entries[:limit]
        for e in entries:
            yield Utterance(id=e.id, text=e.text, audio=load_audio(e.path))


class ManifestDataset(_FileDataset):
    """Your own data: a JSONL file with one ``{"audio": ..., "text": ...}`` per line.

    Audio paths are resolved relative to the manifest. An optional ``"id"`` field is
    used as the utterance id (defaults to the audio path).
    """

    def __init__(
        self, manifest: Path, name: str | None = None, dataset_license: str = "unknown"
    ) -> None:
        manifest = Path(manifest)
        entries = []
        with manifest.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    audio_rel, text = row["audio"], row["text"]
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f"{manifest}:{lineno}: expected audio and text") from exc
                path = (manifest.parent / audio_rel).resolve()
                entries.append(_Entry(str(row.get("id", audio_rel)), text, path))
        super().__init__(entries)
        self.name = name or f"manifest:{manifest.stem}"
        self.license = dataset_license


class FleursTurkish(_FileDataset):
    """Google FLEURS, Turkish (tr_tr) — CC-BY-4.0. Downloaded from the Hugging Face Hub."""

    license = "CC-BY-4.0"
    REPO_ID = "google/fleurs"
    CONFIG = "tr_tr"

    def __init__(self, split: str = "test", cache_dir: Path | None = None) -> None:
        self.name = f"fleurs-tr/{split}"
        tsv, audio_dir = self._download(split, cache_dir)
        super().__init__(self.parse_tsv(tsv, audio_dir))

    @classmethod
    def _download(cls, split: str, cache_dir: Path | None) -> tuple[Path, Path]:
        from huggingface_hub import hf_hub_download

        def fetch(filename: str) -> Path:
            return Path(
                hf_hub_download(
                    repo_id=cls.REPO_ID, filename=filename, repo_type="dataset", cache_dir=cache_dir
                )
            )

        tsv = fetch(f"data/{cls.CONFIG}/{split}.tsv")
        archive = fetch(f"data/{cls.CONFIG}/audio/{split}.tar.gz")
        target = archive.parent / f"{split}-extracted"
        marker = target / ".complete"
        if not marker.exists():
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as tar:
                if hasattr(tarfile, "data_filter"):  # safe extraction (3.10.12+)
                    tar.extractall(target, filter="data")
                else:  # pragma: no cover
                    tar.extractall(target)
            marker.touch()
        return tsv, target

    @staticmethod
    def parse_tsv(tsv: Path, audio_dir: Path) -> list[_Entry]:
        """Parse a FLEURS split file.

        Columns: id, file_name, raw_transcription, transcription, words, num_samples, gender.
        The *raw* transcription (with casing, punctuation and digits) is used as the
        reference; our normalizer then treats reference and hypothesis identically.
        """
        files = {p.name: p for p in audio_dir.rglob("*.wav")}
        entries = []
        with tsv.open(encoding="utf-8", newline="") as f:
            for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) < 3 or not row[1].endswith(".wav"):
                    raise ValueError(f"unexpected FLEURS row format: {row[:3]}")
                file_name, raw = row[1], row[2]
                if file_name not in files:
                    raise FileNotFoundError(f"{file_name} listed in {tsv.name} but not extracted")
                entries.append(_Entry(Path(file_name).stem, raw, files[file_name]))
        return entries


DATASETS = {"fleurs-tr": "FLEURS Turkish test split (CC-BY-4.0)"}


def get_dataset(spec: str, cache_dir: Path | None = None) -> Dataset:
    """Resolve ``fleurs-tr``, ``fleurs-tr:validation`` or a path to a ``.jsonl`` manifest."""
    if spec.endswith(".jsonl"):
        return ManifestDataset(Path(spec))
    name, _, split = spec.partition(":")
    if name == "fleurs-tr":
        return FleursTurkish(split or "test", cache_dir)
    raise ValueError(
        f"unknown dataset {spec!r}; choose from {sorted(DATASETS)} or a .jsonl manifest"
    )
