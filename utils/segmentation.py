"""Segment a reasoning trace into discrete steps.

We prefer finer granularity than raw paragraph splitting so that the
redundancy ratio rho = 1 - k*/N has reasonable resolution. Strategy:

  1. First split on blank lines (paragraph breaks).
  2. Within each paragraph, further split on sentence boundaries AND on
     logical connectors that typically introduce a new reasoning move
     ("So", "Therefore", "Thus", "Wait", "Actually", "Let me", "Hmm",
     "Alternatively", etc.).
  3. Merge pieces shorter than MIN_STEP_WORDS with the previous step.

The result is a list of short, semantically coherent reasoning moves.
"""
from __future__ import annotations
import re

MIN_STEP_WORDS = 12
MAX_STEP_WORDS = 80  # purely advisory; no hard cut

_CONNECTORS = [
    "So ", "So,", "Therefore", "Thus", "Hence",
    "Wait", "Actually", "Hmm", "Let me", "Let's",
    "Alternatively", "Alternatively,", "But wait",
    "Now", "Next", "First", "Second", "Third", "Finally",
    "However", "On the other hand",
    "To verify", "To check", "Checking", "Verifying",
    "Going back", "Back to",
]
_CONNECTOR_RE = re.compile(
    r"(?<=[.!?])\s+(?=(?:" + "|".join(re.escape(c) for c in _CONNECTORS) + r"))"
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\\])")


def _wc(s: str) -> int:
    return len(s.split())


def _split_one_paragraph(para: str) -> list[str]:
    # First split on logical connectors; each slice is a candidate step
    parts = _CONNECTOR_RE.split(para)
    # Then split long slices further on sentence boundaries
    refined: list[str] = []
    for p in parts:
        if _wc(p) <= MAX_STEP_WORDS:
            refined.append(p)
            continue
        sents = _SENTENCE_RE.split(p)
        refined.extend(sents)
    return [s.strip() for s in refined if s.strip()]


def segment_trace(trace: str) -> list[str]:
    if not trace or not trace.strip():
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", trace) if p.strip()]
    # If the trace has no blank lines, treat the whole thing as one paragraph.
    if len(paras) <= 1:
        paras = [trace.strip()]
    pieces: list[str] = []
    for p in paras:
        pieces.extend(_split_one_paragraph(p))
    # Merge tiny pieces forward into the previous one so every step is
    # >= MIN_STEP_WORDS (except possibly the very first).
    merged: list[str] = []
    for p in pieces:
        if merged and _wc(p) < MIN_STEP_WORDS:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    # Also merge trailing tiny tail into the last real step.
    if len(merged) >= 2 and _wc(merged[-1]) < MIN_STEP_WORDS:
        tail = merged.pop()
        merged[-1] = merged[-1] + " " + tail
    return merged
