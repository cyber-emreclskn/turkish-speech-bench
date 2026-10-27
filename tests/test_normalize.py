from __future__ import annotations

import pytest

from tsbench.normalize import NormalizerConfig, normalize, turkish_lower, turkish_upper


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ISPARTA", "ısparta"),  # dotless capital I -> ı (str.lower gives "isparta")
        ("İSTANBUL", "istanbul"),  # dotted capital İ -> i (str.lower gives "i̇stanbul")
        ("IĞDIR", "ığdır"),
        ("ŞÇÖÜĞ", "şçöüğ"),
        ("i̇stanbul", "istanbul"),  # already mangled by a non-Turkish lower()
    ],
)
def test_turkish_lower(text: str, expected: str) -> None:
    assert turkish_lower(text) == expected


def test_turkish_upper_roundtrip() -> None:
    assert turkish_upper("istanbul ılıca") == "İSTANBUL ILICA"
    assert turkish_lower(turkish_upper("ığdır iğne")) == "ığdır iğne"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("İstanbul'da yaşıyorum.", "istanbulda yaşıyorum"),
        ("Ankara’ya gittik!", "ankaraya gittik"),  # typographic apostrophe
        ("Kâğıt, kalem; silgi?", "kağıt kalem silgi"),
        ("  çok    boşluk\tvar\n", "çok boşluk var"),
        ("Fiyat 2.500 TL — %20 indirim", "fiyat 2 500 tl 20 indirim"),
        ('"Merhaba" dedi.', "merhaba dedi"),
        ("", ""),
    ],
)
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_nfd_and_nfc_inputs_match() -> None:
    nfc = "çiğdem"
    nfd = "çiğdem"
    assert normalize(nfd) == normalize(nfc) == "çiğdem"


@pytest.mark.parametrize("text", ["İstanbul'da KÂĞIT, ılık su!", "IĞDIR’a 2024'te"])
def test_idempotent(text: str) -> None:
    once = normalize(text)
    assert normalize(once) == once


def test_options_can_be_disabled() -> None:
    cfg = NormalizerConfig(
        lowercase=False, remove_apostrophes=False, remove_punctuation=False, strip_circumflex=False
    )
    assert normalize("Kâğıt'ı  aldım.", cfg) == "Kâğıt'ı aldım."
