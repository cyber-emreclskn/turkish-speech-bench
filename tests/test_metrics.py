from __future__ import annotations

import pytest

from tsbench.metrics import cer, percentile, rtf, wer


def test_perfect_match_after_normalization() -> None:
    refs = ["İstanbul'da hava güzel."]
    hyps = ["istanbulda hava güzel"]
    assert wer(refs, hyps) == 0.0
    assert cer(refs, hyps) == 0.0


def test_turkish_casing_is_not_an_error() -> None:
    # A naive lowercaser would count "ISPARTA" vs "ısparta" as a substitution.
    assert wer(["ISPARTA güzel"], ["ısparta güzel"]) == 0.0


def test_corpus_level_wer_weights_by_reference_length() -> None:
    refs = ["bir iki üç dört", "beş"]
    hyps = ["bir iki üç dört", "altı"]
    # 1 error over 5 reference words, not mean(0, 1) = 0.5
    assert wer(refs, hyps) == pytest.approx(0.2)


def test_insertions_and_deletions() -> None:
    assert wer(["bir iki üç"], ["bir üç"]) == pytest.approx(1 / 3)
    assert wer(["bir iki"], ["bir iki üç dört"]) == pytest.approx(1.0)


def test_cer() -> None:
    assert cer(["kedi"], ["kedı"]) == pytest.approx(0.25)


def test_length_mismatch() -> None:
    with pytest.raises(ValueError, match="hypotheses"):
        wer(["a"], [])


def test_empty_reference_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        wer(["..."], ["bir"])


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([5], 90) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == pytest.approx(9.1)


def test_rtf() -> None:
    assert rtf(2.0, 10.0) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="positive"):
        rtf(1.0, 0)
