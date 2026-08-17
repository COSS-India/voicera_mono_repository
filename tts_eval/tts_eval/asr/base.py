"""ASR backend contract for round-trip intelligibility, plus text normalisation.

Round-trip evaluation — synthesise text, transcribe the audio, compare — is the
standard objective proxy for pronunciation accuracy, and the normalisation step is
where most implementations quietly go wrong. Two rules are enforced here:

*   **The ASR's own error rate is part of the measurement.** A CER of 0.18 on
    Santali may mean the TTS mispronounced it or that the ASR cannot transcribe
    it. Every backend therefore reports its identity into the run record, and
    the framework never claims an absolute intelligibility figure — only a
    comparison between models measured with the *same* ASR.
*   **Normalisation must be script-aware and symmetric.** Both sides go through
    the identical pipeline. Applying Latin lowercasing to one side, or stripping
    Devanagari punctuation from one side only, manufactures error rates that look
    like model defects.
"""
from __future__ import annotations

import abc
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

from ..types import AudioBuffer

# Punctuation to drop before scoring. Includes the Indic danda/double danda, the
# Arabic full stop and question mark used by Urdu, and the zero-width joiners that
# ASR output and source text disagree about constantly in Indic scripts.
_PUNCT_CHARS = (
    "।॥"          # Devanagari danda, double danda
    "۔؟،؛"        # Urdu/Arabic full stop, question mark, comma, semicolon
    "?!.,;:"      # ASCII
    "\"'`“”‘’«»"  # quotes
    "()[]{}<>"    # brackets
    "-–—_/\\|*&^~+=@#$%"
    "…"
)
_PUNCT_RE = re.compile("[" + re.escape(_PUNCT_CHARS) + "]")
# ZWSP, ZWNJ, ZWJ, word joiner, BOM/ZWNBSP, soft hyphen. Invisible, semantically
# inert for scoring, and a frequent source of phantom character errors when the
# source text and the ASR output disagree about them. Written as escapes because
# the literal characters are unreviewable in source.
_INVISIBLE_RE = re.compile("[​‌‍⁠﻿­]")
_WS_RE = re.compile(r"\s+")


def _fold_digits(text: str) -> str:
    """Map any script's decimal digits to ASCII, so "১২" and "12" match.

    Uses the Unicode decimal property rather than a hand-listed table of digit
    blocks: that covers every Indic script (and any future one) without this
    function needing to know which languages exist.

    Representation only — it does NOT verbalise numbers. Turning "12,450" into
    spoken words is the job of ``TestCase.expected_transcript``.
    """
    out: list[str] = []
    for ch in text:
        if ch.isascii():
            out.append(ch)
            continue
        value = unicodedata.decimal(ch, None)
        out.append(str(value) if value is not None else ch)
    return "".join(out)


def normalise_text(text: str, *, language: str = "") -> str:
    """Canonical form used on BOTH the reference and the hypothesis.

    Steps, in order and for stated reasons:
      1. NFC — Indic text arrives both composed and decomposed depending on the
         source; without this, identical strings differ character-by-character.
      2. Drop invisible joiners.
      3. Fold native digits to ASCII.
      4. Casefold — a no-op for Indic scripts, and correct for Latin/Urdu
         transliteration where ASR casing is arbitrary.
      5. Strip punctuation — TTS is not being tested on comma placement, and ASR
         punctuation is a property of the ASR's decoder, not of the audio.
      6. Collapse whitespace.
    """
    t = unicodedata.normalize("NFC", text)
    t = _INVISIBLE_RE.sub("", t)
    t = _fold_digits(t)
    t = t.casefold()
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Edit distance with O(min(len)) memory.

    Written out rather than pulled from jiwer/editdistance to keep the ASR tier
    free of extra dependencies — the sequences here are one sentence long, so the
    pure-Python cost is irrelevant.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,        # deletion
                    current[j - 1] + 1,     # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class ErrorRate:
    rate: float
    edits: int
    reference_length: int


def character_error_rate(reference: str, hypothesis: str, *, language: str = "") -> ErrorRate:
    """CER over normalised text, ignoring spaces.

    Spaces are excluded because Indic ASR word segmentation is inconsistent —
    the same utterance transcribed twice can differ only in where spaces fall,
    which would show up as a real error rate for a model that pronounced
    everything correctly.
    """
    ref = normalise_text(reference, language=language).replace(" ", "")
    hyp = normalise_text(hypothesis, language=language).replace(" ", "")
    if not ref:
        return ErrorRate(rate=float("nan"), edits=0, reference_length=0)
    edits = _levenshtein(list(ref), list(hyp))
    return ErrorRate(rate=edits / len(ref), edits=edits, reference_length=len(ref))


def word_error_rate(reference: str, hypothesis: str, *, language: str = "") -> ErrorRate:
    ref = normalise_text(reference, language=language).split()
    hyp = normalise_text(hypothesis, language=language).split()
    if not ref:
        return ErrorRate(rate=float("nan"), edits=0, reference_length=0)
    edits = _levenshtein(ref, hyp)
    return ErrorRate(rate=edits / len(ref), edits=edits, reference_length=len(ref))


def slot_hits(hypothesis: str, required: tuple[str, ...], *, language: str = "") -> tuple[int, list[str]]:
    """Count required tokens present in the transcript; return (hits, missing).

    Matching is on normalised, space-stripped text so "O T P", "OTP" and "otp"
    all count — the model said the letters, which is what the check is about.
    """
    hyp = normalise_text(hypothesis, language=language).replace(" ", "")
    missing: list[str] = []
    hits = 0
    for token in required:
        needle = normalise_text(token, language=language).replace(" ", "")
        if needle and needle in hyp:
            hits += 1
        else:
            missing.append(token)
    return hits, missing


def _is_latin(text: str) -> bool:
    """True when the token's letters are all Latin (e.g. "OTP", "KYC").

    Used to spot a slot token an Indic ASR cannot emit: the model may pronounce
    "OTP" perfectly, but an ASR that only outputs native script will never write
    the Latin letters, so the round-trip check cannot witness it.
    """
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all("a" <= c <= "z" for c in letters)


def slot_evaluation(
    hypothesis: str, required: tuple[str, ...], *, language: str = ""
) -> tuple[int, list[str], list[str]]:
    """Like ``slot_hits`` but separates genuine misses from *unverifiable* tokens.

    A token is unverifiable when the ASR could not have emitted its script at all
    -- the evidence being that the transcript contains no character of that script.
    A Latin token like "OTP" scored against an Indic ASR whose output is entirely
    native script is the motivating case: counting it as a miss reports a
    pronunciation failure the transcript cannot actually witness (a false 0), so it
    is reported unverifiable instead. If the ASR *did* emit Latin somewhere in the
    transcript, the token is checked normally and a real miss still counts.

    Returns ``(hits, missing, unverifiable)``.
    """
    hyp = normalise_text(hypothesis, language=language).replace(" ", "")
    hyp_has_latin = any("a" <= c <= "z" for c in hyp)
    hits = 0
    missing: list[str] = []
    unverifiable: list[str] = []
    for token in required:
        needle = normalise_text(token, language=language).replace(" ", "")
        if not needle:
            continue
        if needle in hyp:
            hits += 1
        elif _is_latin(needle) and not hyp_has_latin:
            unverifiable.append(token)
        else:
            missing.append(token)
    return hits, missing, unverifiable


class ASRBackend(abc.ABC):
    """Transcribes audio for round-trip scoring."""

    name: str = ""

    def __init__(self, options: Mapping[str, Any] | None = None):
        self.options = dict(options or {})

    def available(self) -> tuple[bool, str]:
        return True, self.name

    def prepare(self) -> None:
        """Load the model. Called once per run."""

    def teardown(self) -> None:
        """Release the model."""

    def describe(self) -> dict[str, Any]:
        """Identity recorded in the run record.

        Comparing CER across runs is only valid when this is identical, so the
        comparison engine checks it and warns on a mismatch.
        """
        return {"backend": self.name, "version": self.available()[1], **self._describe_extra()}

    def _describe_extra(self) -> dict[str, Any]:
        return {}

    @abc.abstractmethod
    def transcribe(self, audio: AudioBuffer, language: str) -> str:
        """Return the transcript. Raise on failure; the caller records it."""


__all__ = [
    "ASRBackend",
    "ErrorRate",
    "character_error_rate",
    "normalise_text",
    "slot_hits",
    "word_error_rate",
]
