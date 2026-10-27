"""Accuracy and speed metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import jiwer

from .normalize import DEFAULT, NormalizerConfig, normalize


def _prepare(
    refs: Sequence[str], hyps: Sequence[str], config: NormalizerConfig
) -> tuple[list[str], list[str]]:
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    if not refs:
        raise ValueError("cannot score an empty set")
    nrefs = [normalize(r, config) for r in refs]
    nhyps = [normalize(h, config) for h in hyps]
    if any(not r for r in nrefs):
        raise ValueError("a reference is empty after normalization")
    return nrefs, nhyps


def wer(refs: Sequence[str], hyps: Sequence[str], config: NormalizerConfig = DEFAULT) -> float:
    """Corpus-level word error rate (total edits / total reference words)."""
    r, h = _prepare(refs, hyps, config)
    return float(jiwer.wer(r, h))


def cer(refs: Sequence[str], hyps: Sequence[str], config: NormalizerConfig = DEFAULT) -> float:
    """Corpus-level character error rate (spaces count as characters)."""
    r, h = _prepare(refs, hyps, config)
    return float(jiwer.cer(r, h))


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def rtf(processing_seconds: float, audio_seconds: float) -> float:
    """Real-time factor: processing time / audio duration (lower is faster)."""
    if audio_seconds <= 0:
        raise ValueError("audio duration must be positive")
    return processing_seconds / audio_seconds
