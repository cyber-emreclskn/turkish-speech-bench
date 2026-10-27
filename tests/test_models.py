from __future__ import annotations

import pytest

from tsbench.models import MissingBackendError, backends, create
from tsbench.models.faster_whisper import FasterWhisper
from tsbench.models.hf_ctc import HFCTC
from tsbench.models.vosk import Vosk


def test_builtin_backends_listed() -> None:
    assert {"faster-whisper", "vosk", "hf-ctc", "fake"} <= set(backends())


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
