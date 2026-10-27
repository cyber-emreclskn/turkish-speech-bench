from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tsbench import datasets
from tsbench.datasets import SAMPLE_RATE, FleursTurkish, ManifestDataset, get_dataset

from .conftest import tone


def test_manifest_loads_sorted_16k_mono(dataset: ManifestDataset) -> None:
    utts = list(dataset.utterances())
    assert [u.id for u in utts] == ["utt1", "utt2", "utt3"]
    for u in utts:
        assert u.audio.dtype == np.float32
        assert u.audio.ndim == 1
    # utt3 was written at 8 kHz stereo; it must come back as 2 s at 16 kHz mono
    assert utts[2].duration == pytest.approx(2.0, abs=0.01)


def test_limit_is_deterministic(dataset: ManifestDataset) -> None:
    assert [u.id for u in dataset.utterances(limit=2)] == ["utt1", "utt2"]
    assert len(dataset) == 3


def test_manifest_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"audio": "x.wav"}\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        ManifestDataset(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        ManifestDataset(empty)


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    sf.write(tmp_path / "a.wav", tone(0.1), SAMPLE_RATE)
    m = tmp_path / "dup.jsonl"
    m.write_text(
        '{"id": "x", "audio": "a.wav", "text": "a"}\n{"id": "x", "audio": "a.wav", "text": "b"}\n'
    )
    with pytest.raises(ValueError, match="unique"):
        ManifestDataset(m)


def test_get_dataset(manifest: Path) -> None:
    assert get_dataset(str(manifest)).name == "manifest:mini"
    with pytest.raises(ValueError, match="unknown dataset"):
        get_dataset("librispeech")


def _fake_fleurs(tmp_path: Path) -> tuple[Path, Path]:
    """Build files shaped like google/fleurs data/tr_tr/{test.tsv,audio/test.tar.gz}."""
    rows = [
        ("1", "b.wav", "Ankara'ya gittik.", "ankaraya gittik", "a|n|k", "16000", "FEMALE"),
        ("1", "a.wav", "Ankara'ya gittik.", "ankaraya gittik", "a|n|k", "16000", "MALE"),
    ]
    tsv = tmp_path / "test.tsv"
    tsv.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")
    archive = tmp_path / "test.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("a.wav", "b.wav"):
            buf = io.BytesIO()
            sf.write(buf, tone(1.0), SAMPLE_RATE, format="WAV")
            info = tarfile.TarInfo(f"test/{name}")
            info.size = buf.tell()
            buf.seek(0)
            tar.addfile(info, buf)
    return tsv, archive


def test_fleurs_download_and_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tsv, archive = _fake_fleurs(tmp_path)
    calls = []

    def fake_download(filename: str, **kw: object) -> str:
        calls.append((filename, kw["repo_id"], kw["repo_type"]))
        return str(tsv if filename.endswith(".tsv") else archive)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    ds = FleursTurkish("test")
    assert ("data/tr_tr/test.tsv", "google/fleurs", "dataset") in calls
    assert ("data/tr_tr/audio/test.tar.gz", "google/fleurs", "dataset") in calls
    utts = list(ds.utterances())
    # same sentence by two speakers -> two utterances, keyed by file, sorted
    assert [u.id for u in utts] == ["a", "b"]
    assert utts[0].text == "Ankara'ya gittik."  # raw transcription is the reference
    assert ds.name == "fleurs-tr/test"
    assert ds.license == "CC-BY-4.0"
    # second load reuses the extracted audio
    FleursTurkish("test")


def test_fleurs_rejects_unexpected_format(tmp_path: Path) -> None:
    tsv = tmp_path / "x.tsv"
    tsv.write_text("id\tnot-a-wav\ttext\n")
    with pytest.raises(ValueError, match="FLEURS row format"):
        datasets.FleursTurkish.parse_tsv(tsv, tmp_path)


def test_fleurs_missing_audio(tmp_path: Path) -> None:
    tsv = tmp_path / "x.tsv"
    tsv.write_text("1\tmissing.wav\tmetin\n")
    with pytest.raises(FileNotFoundError, match=r"missing\.wav"):
        datasets.FleursTurkish.parse_tsv(tsv, tmp_path)
