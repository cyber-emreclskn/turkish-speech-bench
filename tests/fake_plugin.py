"""Test backends, loadable in child processes via TSB_PLUGINS=tests.fake_plugin."""

from __future__ import annotations

import time

import numpy as np

from tsbench.datasets import SAMPLE_RATE, Audio
from tsbench.models import STTModel, register

# id -> (reference, what the fake model "hears", duration seconds)
SAMPLES = {
    "utt1": ("İstanbul'da hava güzel.", "istanbulda hava güzel", 1.0),
    "utt2": ("Kâğıt kalemi ver.", "kağıt kalem ver", 1.5),  # 1 word error
    "utt3": ("ISPARTA'ya gidiyoruz", "ıspartaya gidiyoruz", 2.0),
}


class FakeModel(STTModel):
    """Returns a scripted transcript based on audio length; sleeps 10% of audio time."""

    backend = "fake"

    def __init__(self, name: str, threads: int) -> None:
        super().__init__(name, threads)
        self.loaded = False
        self.calls = 0
        self._by_len = {int(d * SAMPLE_RATE): hyp for _ref, hyp, d in SAMPLES.values()}

    def prepare(self) -> None:
        time.sleep(1.0)  # pretend to download; must not show up in load_s

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, audio: Audio) -> str:
        assert self.loaded, "transcribe() before load()"
        assert audio.dtype == np.float32
        self.calls += 1
        time.sleep(len(audio) / SAMPLE_RATE * 0.1)
        return self._by_len[len(audio)]


class BoomModel(FakeModel):
    backend = "boom"

    def transcribe(self, audio: Audio) -> str:
        raise RuntimeError("model exploded")


def install() -> None:
    register("fake", FakeModel)
    register("boom", BoomModel)


install()
