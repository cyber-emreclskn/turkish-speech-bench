from __future__ import annotations

import pytest

from tsbench.models import MissingBackendError, backends, create
from tsbench.models.faster_whisper import FasterWhisper
from tsbench.models.hf_ctc import HFCTC
from tsbench.models.vosk import Vosk


def test_builtin_backends_listed() -> None:
    assert {"faster-whisper", "vosk", "hf-ctc", "sherpa-onnx", "fake"} <= set(backends())


@pytest.mark.parametrize(
    ("spec", "cls", "name"),
    [
        ("faster-whisper:small", FasterWhisper, "small"),
        ("vosk:vosk-model-small-tr-0.3", Vosk, "vosk-model-small-tr-0.3"),
        ("hf-ctc:org/model-name", HFCTC, "org/model-name"),
    ],
)
def test_create_is_lazy(spec: str, cls: type, name: str) -> None:
    # Creating must not import heavy backends or download anything.
    model = create(spec, 3)
    assert isinstance(model, cls)
    assert model.name == name
    assert model.threads == 3
    assert model.spec == spec


@pytest.mark.parametrize("spec", ["faster-whisper", "faster-whisper:", ":small"])
def test_bad_spec(spec: str) -> None:
    with pytest.raises(ValueError, match=r"backend:name|unknown backend"):
        create(spec, 1)


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="available: "):
        create("nemo:x", 1)


def test_info() -> None:
    info = create("faster-whisper:large-v3-turbo", 1).info()
    assert info.params_millions == 809
    assert info.license == "MIT"
    assert info.settings == {"compute_type": "int8", "beam_size": 5, "temperature": 0.0}


def test_missing_backend_message() -> None:
    assert "pip install 'turkish-speech-bench[whisper]'" in str(
        MissingBackendError("faster-whisper", "whisper")
    )


# ---------------------------------------------------------------- sherpa-onnx
class _FakeStream:
    def __init__(self) -> None:
        self.samples = 0
        self.finished = False
        self.result = type("R", (), {"text": " offline metin "})()

    def accept_waveform(self, sample_rate: int, samples: object) -> None:
        assert sample_rate == 16000
        self.samples += len(samples)  # type: ignore[arg-type]

    def input_finished(self) -> None:
        self.finished = True


class _FakeRecognizer:
    kwargs: dict[str, object]

    @classmethod
    def from_transducer(cls, **kwargs: object) -> _FakeRecognizer:
        rec = cls()
        rec.kwargs = kwargs
        rec.pending = 2
        return rec

    def create_stream(self) -> _FakeStream:
        self.stream = _FakeStream()
        return self.stream

    def is_ready(self, stream: _FakeStream) -> bool:
        return self.pending > 0

    def decode_stream(self, stream: _FakeStream) -> None:
        self.pending -= 1

    def get_result(self, stream: _FakeStream) -> str:
        return " akan metin "


def _sherpa_dir(tmp_path, *, config: dict | None = None, names: tuple[str, ...] = ()):  # type: ignore[no-untyped-def]
    import json

    for name in names or ("encoder.onnx", "encoder.int8.onnx", "decoder.onnx", "joiner.onnx"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "tokens.txt").write_text("a 0\n")
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.fixture
def fake_sherpa(monkeypatch: pytest.MonkeyPatch) -> type:
    import sys
    import types

    module = types.ModuleType("sherpa_onnx")
    online = type("OnlineRecognizer", (_FakeRecognizer,), {})
    offline = type("OfflineRecognizer", (_FakeRecognizer,), {})
    module.OnlineRecognizer = online  # type: ignore[attr-defined]
    module.OfflineRecognizer = offline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    return online


def test_sherpa_streaming_uses_shipped_config(tmp_path, fake_sherpa: type) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    config = {
        "streaming": True,
        "decoding_method": "modified_beam_search",
        "max_active_paths": 4,
        "num_parameters": "66M",
        "files": {"encoder": "encoder.int8.onnx"},
    }
    model = create(f"sherpa-onnx:{_sherpa_dir(tmp_path, config=config)}", 2)
    model.prepare()
    model.load()
    rec = model._recognizer  # type: ignore[attr-defined]
    assert isinstance(rec, fake_sherpa)
    assert rec.kwargs["encoder"].endswith("encoder.int8.onnx")
    assert rec.kwargs["decoding_method"] == "modified_beam_search"
    assert rec.kwargs["num_threads"] == 2

    assert model.transcribe(np.zeros(16000, dtype=np.float32)) == "akan metin"
    assert rec.stream.finished
    assert rec.stream.samples == 16000 + int(0.66 * 16000)  # audio + tail padding

    info = model.info()
    assert info.params_millions == 66
    assert info.settings["streaming"] is True
    assert info.settings["quantization"] == "int8"
    assert {p.name for p in model.files()} == {
        "tokens.txt",
        "encoder.int8.onnx",
        "decoder.onnx",
        "joiner.onnx",
    }


def test_sherpa_offline_without_config(tmp_path, fake_sherpa: type) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    model = create(f"sherpa-onnx:{_sherpa_dir(tmp_path)}", 1)
    model.load()  # load() prepares on its own
    rec = model._recognizer  # type: ignore[attr-defined]
    assert not isinstance(rec, fake_sherpa)
    assert rec.kwargs["encoder"].endswith("encoder.int8.onnx")  # INT8 preferred
    assert rec.kwargs["decoding_method"] == "greedy_search"
    assert model.transcribe(np.zeros(800, dtype=np.float32)) == "offline metin"
    assert rec.stream.samples == 800


def test_sherpa_missing_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = create(f"sherpa-onnx:{_sherpa_dir(tmp_path, names=('encoder.onnx',))}", 1)
    with pytest.raises(FileNotFoundError, match="decoder"):
        model.prepare()
