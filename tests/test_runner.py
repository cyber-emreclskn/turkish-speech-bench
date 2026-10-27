from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsbench import runner
from tsbench.datasets import ManifestDataset
from tsbench.models import create
from tsbench.report import load_results, markdown, result_filename, rows, save

from .conftest import FakeModel


@pytest.fixture
def result(dataset: ManifestDataset) -> dict:
    return runner.run(create("fake:scripted", 2), dataset, hardware_label="test-box")


def test_accuracy_metrics(result: dict) -> None:
    m = result["metrics"]
    # 8 reference words, one substitution ("kalemi" -> "kalem"); casing/apostrophes don't count
    assert m["wer"] == pytest.approx(1 / 8, abs=1e-4)
    assert 0 < m["cer"] < m["wer"]


def test_speed_metrics(result: dict) -> None:
    m = result["metrics"]
    # the fake model sleeps 10% of audio duration -> RTF ≈ 0.1, RTFx ≈ 10
    assert m["rtf"] == pytest.approx(0.1, rel=0.3)
    assert m["rtfx"] == pytest.approx(1 / m["rtf"], rel=0.01)
    assert m["latency_p50_s"] <= m["latency_p90_s"]
    assert m["peak_rss_mb"] > 0
    assert 0 <= m["load_s"] < 0.2  # prepare() (0.3 s "download") is not timed


def test_metadata(result: dict) -> None:
    assert result["schema_version"] == runner.SCHEMA_VERSION
    assert result["dataset"] == {
        "name": "manifest:mini",
        "license": "unknown",
        "utterances": 3,
        "subset": "full",
        "audio_seconds": pytest.approx(4.5, abs=0.01),
    }
    assert result["model"]["spec"] == "fake:scripted"
    assert result["run"] == {"threads": 2, "warmup": 1}
    sys_ = result["system"]
    assert sys_["label"] == "test-box"
    assert len(sys_["hardware_id"]) == 12
    assert sys_["cores_logical"] >= 1


def test_predictions_are_normalized(result: dict) -> None:
    p = {r["id"]: r for r in result["predictions"]}
    assert p["utt1"]["ref_norm"] == p["utt1"]["hyp_norm"] == "istanbulda hava güzel"
    assert p["utt3"]["ref_norm"] == "ıspartaya gidiyoruz"


def test_warmup_is_extra_untimed_call(dataset: ManifestDataset) -> None:
    model = FakeModel("x", 1)
    result = runner.run(model, dataset, warmup=2, keep_predictions=False)
    assert model.calls == 3 + 2
    assert result["predictions"] == []


def test_limit_marks_subset(dataset: ManifestDataset) -> None:
    result = runner.run(create("fake:x", 1), dataset, limit=2)
    assert result["dataset"]["subset"] == "first-2"
    assert result["dataset"]["utterances"] == 2


def test_progress_callback(dataset: ManifestDataset) -> None:
    seen = []
    runner.run(create("fake:x", 1), dataset, progress=lambda i, n: seen.append((i, n)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_save_and_load_roundtrip(result: dict, tmp_path: Path) -> None:
    path = save(result, tmp_path)
    assert path == tmp_path / result_filename(result)
    assert path.name == "manifest-mini__fake-scripted__t2__full.json"
    assert "İstanbul" in path.read_text(encoding="utf-8")  # not \u-escaped
    (tmp_path / "junk.json").write_text(json.dumps({"hello": 1}))
    loaded = load_results([tmp_path])
    assert len(loaded) == 1
    assert loaded[0]["metrics"] == result["metrics"]


def test_report(result: dict) -> None:
    other = json.loads(json.dumps(result))
    other["model"]["spec"] = "fake:better"
    other["metrics"]["wer"] = 0.01
    table = rows([result, other])
    assert [r["model"] for r in table] == ["fake:better", "fake:scripted"]  # best WER first
    md = markdown([result, other])
    assert "### test-box · manifest:mini (full, 3 utt.)" in md
    assert "| `fake:better` | 1.0% |" in md
    assert "| `fake:scripted` | 12.5% |" in md
