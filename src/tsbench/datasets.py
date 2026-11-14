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


# Column names used for the transcript by the datasets people actually publish.
_TEXT_COLUMNS = ("sentence", "text", "transcription", "raw_transcription", "transcript")


class HubParquetDataset(_FileDataset):
    """A Hugging Face dataset stored as Parquet with embedded audio.

    The files are downloaded once and every clip is written to the cache as its own
    file, so later runs (and ``--limit``) never touch the Parquet data again. Utterance
    ids come from the embedded file name when it is unique, otherwise from the row
    position, so they are the same on every machine.
    """

    def __init__(
        self,
        repo_id: str,
        pattern: str,
        *,
        name: str,
        dataset_license: str = "unknown",
        cache_dir: Path | None = None,
    ) -> None:
        self.name = name
        self.license = dataset_license
        super().__init__(self._extract(self._download(repo_id, pattern, cache_dir)))

    @staticmethod
    def _download(repo_id: str, pattern: str, cache_dir: Path | None) -> list[Path]:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                repo_id, repo_type="dataset", allow_patterns=[pattern], cache_dir=cache_dir
            )
        )
        files = sorted(root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no files matching {pattern!r} in dataset {repo_id}")
        return files

    @staticmethod
    def _extract(parquet_files: list[Path]) -> list[_Entry]:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "this dataset needs pyarrow: pip install 'turkish-speech-bench[parquet]'"
            ) from exc

        target = parquet_files[0].parent / f"{parquet_files[0].stem}-clips"
        index = target / "index.jsonl"
        if index.exists():
            rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
            return [_Entry(r["id"], r["text"], target / r["file"]) for r in rows]

        target.mkdir(parents=True, exist_ok=True)
        rows = []
        for file_no, parquet in enumerate(parquet_files):
            pf = pq.ParquetFile(parquet)
            schema = pf.schema_arrow
            audio_col = next(
                (
                    f.name
                    for f in schema
                    if pa.types.is_struct(f.type) and f.type.get_field_index("bytes") >= 0
                ),
                None,
            )
            text_col = next((c for c in _TEXT_COLUMNS if c in schema.names), None)
            if audio_col is None or text_col is None:
                raise ValueError(
                    f"{parquet.name}: need an audio column with embedded bytes and one of "
                    f"{_TEXT_COLUMNS}; found columns {schema.names}"
                )
            row_no = 0
            for batch in pf.iter_batches(batch_size=64, columns=[audio_col, text_col]):
                for audio, text in zip(
                    batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
                ):
                    position = f"{file_no:02d}-{row_no:06d}"
                    row_no += 1
                    if not audio or not audio.get("bytes") or not text:
                        continue  # a row without audio or transcript cannot be scored
                    embedded = Path(audio.get("path") or "")
                    file = f"{position}{embedded.suffix or '.wav'}"
                    (target / file).write_bytes(audio["bytes"])
                    rows.append({"id": embedded.stem or position, "text": text, "file": file})
        if len({r["id"] for r in rows}) != len(rows):  # embedded names repeat: use positions
            for r in rows:
                r["id"] = Path(r["file"]).stem
        index.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        return [_Entry(r["id"], r["text"], target / r["file"]) for r in rows]


# name -> (repo id, file pattern with {split}, default split, license, description)
_HUB_DATASETS = {
    "cv17-tr": (
        "ysdede/commonvoice_17_tr_fixed",
        "data/{split}-*.parquet",
        "test",
        "CC0-1.0",
        "Common Voice 17 Turkish, community read speech (CC0-1.0)",
    ),
    "mediaspeech-tr": (
        "ymoslem/MediaSpeech",
        "tr/{split}-*.parquet",
        "train",
        "CC-BY-4.0",
        "MediaSpeech Turkish, 10 h of broadcast speech (CC-BY-4.0)",
    ),
}

DATASETS = {
    "fleurs-tr": "FLEURS Turkish test split (CC-BY-4.0)",
    **{name: spec[4] for name, spec in _HUB_DATASETS.items()},
}


def get_dataset(spec: str, cache_dir: Path | None = None) -> Dataset:
    """Resolve a dataset name with an optional split (``fleurs-tr``, ``cv17-tr:validation``),
    ``hf:<repo>:<parquet glob>`` or a path to a ``.jsonl`` manifest."""
    if spec.endswith(".jsonl"):
        return ManifestDataset(Path(spec))
    if spec.startswith("hf:"):
        repo_id, _, pattern = spec[3:].partition(":")
        if not repo_id or not pattern:
            raise ValueError(
                "expected hf:<repo>:<parquet glob>, e.g. hf:org/data:data/test-*.parquet"
            )
        return HubParquetDataset(repo_id, pattern, name=f"hf:{repo_id}", cache_dir=cache_dir)
    name, _, split = spec.partition(":")
    if name == "fleurs-tr":
        return FleursTurkish(split or "test", cache_dir)
    if name in _HUB_DATASETS:
        repo_id, pattern, default_split, dataset_license, _ = _HUB_DATASETS[name]
        split = split or default_split
        return HubParquetDataset(
            repo_id,
            pattern.format(split=split),
            name=f"{name}/{split}",
            dataset_license=dataset_license,
            cache_dir=cache_dir,
        )
    raise ValueError(
        f"unknown dataset {spec!r}; choose from {sorted(DATASETS)}, "
        "hf:<repo>:<parquet glob> or a .jsonl manifest"
    )
