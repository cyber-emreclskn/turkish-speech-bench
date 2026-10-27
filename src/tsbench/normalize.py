"""Turkish-aware text normalization for ASR scoring.

Generic normalizers silently inflate Turkish WER:

* ``"I".lower()`` gives ``"i"`` (should be ``"ı"``), and ``"İ".lower()`` gives
  ``"i̇"`` (``i`` + U+0307 COMBINING DOT ABOVE), which no longer matches ``"i"``.
* Suffixes attached with an apostrophe (``İstanbul'da``) are written inconsistently
  by humans and models alike.
* Circumflex vowels (``kâğıt`` / ``kağıt``) are an orthographic choice, not a
  recognition error.

Both reference and hypothesis go through the *same* function before scoring.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_APOSTROPHES = "'’‘`´ʼ′"
_CIRCUMFLEX = str.maketrans({"â": "a", "î": "i", "û": "u", "Â": "A", "Î": "İ", "Û": "U"})
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizerConfig:
    """Options for :func:`normalize`. Defaults are the leaderboard settings."""

    lowercase: bool = True
    remove_apostrophes: bool = True  # "İstanbul'da" -> "istanbulda"
    remove_punctuation: bool = True
    strip_circumflex: bool = True  # "kâğıt" -> "kağıt"


DEFAULT = NormalizerConfig()


def turkish_lower(text: str) -> str:
    """Lowercase using Turkish casing rules (I -> ı, İ -> i)."""
    text = text.replace("I", "ı").replace("İ", "i")
    # Fix text that was already lowercased the non-Turkish way: "i̇" -> "i"
    return text.lower().replace("i̇", "i")


def turkish_upper(text: str) -> str:
    """Uppercase using Turkish casing rules (i -> İ, ı -> I)."""
    return text.replace("i", "İ").replace("ı", "I").upper()


def _strip_punctuation(text: str) -> str:
    # Replace any punctuation or symbol with a space; keep letters, marks and digits.
    return "".join(" " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in text)


def normalize(text: str, config: NormalizerConfig = DEFAULT) -> str:
    """Normalize a transcript for scoring. Idempotent."""
    text = unicodedata.normalize("NFC", text)
    if config.strip_circumflex:
        text = text.translate(_CIRCUMFLEX)
    if config.lowercase:
        text = turkish_lower(text)
    if config.remove_apostrophes:
        text = "".join("" if ch in _APOSTROPHES else ch for ch in text)
    if config.remove_punctuation:
        text = _strip_punctuation(text)
    return _WHITESPACE.sub(" ", text).strip()
