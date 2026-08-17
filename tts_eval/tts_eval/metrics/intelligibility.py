"""Pronunciation accuracy via round-trip ASR: CER, WER and slot accuracy.

Three metrics rather than one, because they fail in different directions:

*   ``cer``  — the primary signal for Indic scripts. Word-level scoring is noisy
    there because ASR word segmentation is inconsistent, so CER is what per-
    language verification is gated on.
*   ``wer``  — reported for comparability with published English benchmarks.
*   ``slot_accuracy`` — did the required tokens survive at all. A model that drops
    a 4-digit OTP from a 40-character sentence still scores ~0.90 CER-wise; for a
    voice agent that utterance is a total failure. CER cannot express that, so
    this does.

The transcript is stored on every metric's ``extra`` payload. That is not
decoration: a CER figure nobody can audit is not evidence, and having the
hypothesis in the record is what lets a reviewer tell "the model mispronounced it"
from "the ASR could not transcribe it".
"""
from __future__ import annotations

from typing import Mapping

from ..asr.base import (
    ASRBackend,
    character_error_rate,
    normalise_text,
    slot_evaluation,
    word_error_rate,
)
from ..datasets.loader import TestCase
from ..types import MetricStatus, MetricValue, SynthesisResult
from .base import MetricContext, UtteranceBackend, make_value, missing_value, register_backend


@register_backend
class IntelligibilityBackend(UtteranceBackend):
    name = "intelligibility"
    provides = ("cer", "wer", "slot_accuracy")

    def available(self) -> tuple[bool, str]:
        # The ASR instance is injected into the context, which does not exist yet
        # at availability time. Report availability optimistically and let
        # prepare() do the real check, so the reason lands in the run record.
        return True, "pending: depends on the configured ASR backend"

    def prepare(self, ctx: MetricContext) -> None:
        asr: ASRBackend | None = ctx.asr
        if asr is None:
            raise RuntimeError(
                "no ASR backend configured; set `asr:` in the suite config "
                "(e.g. backend: http_asr, url: http://localhost:8001/transcribe) "
                "or run with --metrics core to skip intelligibility"
            )
        ok, info = asr.available()
        if not ok:
            raise RuntimeError(f"ASR backend {asr.name!r} unavailable: {info}")
        asr.prepare()
        self._asr = asr
        self._version = info

    def teardown(self) -> None:
        asr = getattr(self, "_asr", None)
        if asr is not None:
            asr.teardown()

    def version(self) -> str:
        return getattr(self, "_version", "unknown")

    # ------------------------------------------------------------------
    def compute(
        self, case: TestCase, result: SynthesisResult, ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        audio = result.audio
        assert audio is not None

        try:
            hypothesis = self._asr.transcribe(audio, case.language)
        except Exception as e:  # noqa: BLE001
            reason = f"ASR failed: {type(e).__name__}: {e}"
            return {name: missing_value(name, reason, MetricStatus.ERROR) for name in self.provides}

        # `reference_text` is expected_transcript when the case defines one — the
        # spoken form of digits, currency and abbreviations. Scoring "₹12,450"
        # against a correct spoken rendering would otherwise report a ~90% CER for
        # a model that did exactly the right thing.
        reference = case.reference_text
        used_spoken_form = case.expected_transcript is not None

        cer = character_error_rate(reference, hypothesis, language=case.language)
        wer = word_error_rate(reference, hypothesis, language=case.language)

        shared_extra = {
            "hypothesis": hypothesis,
            "reference": reference,
            "reference_normalised": normalise_text(reference, language=case.language),
            "hypothesis_normalised": normalise_text(hypothesis, language=case.language),
            "asr_backend": self._asr.describe(),
            "scored_against_spoken_form": used_spoken_form,
        }

        # Cases with digits/symbols but no spoken form are flagged rather than
        # silently mis-scored, so a reviewer knows which numbers to distrust.
        caveat: str | None = None
        if not used_spoken_form and _has_non_verbal_tokens(case.text):
            caveat = (
                "case text contains digits or symbols but no `expected_transcript`; "
                "round-trip CER here also measures text normalisation mismatch, not only "
                "pronunciation"
            )
            ctx.warn(f"{case.id}: {caveat}")

        out: dict[str, MetricValue] = {}
        if cer.reference_length == 0:
            out["cer"] = missing_value("cer", "empty reference text after normalisation")
        else:
            out["cer"] = make_value(
                "cer",
                cer.rate,
                detail=caveat,
                extra={**shared_extra, "edits": cer.edits, "reference_chars": cer.reference_length},
            )
        if wer.reference_length == 0:
            out["wer"] = missing_value("wer", "empty reference text after normalisation")
        else:
            out["wer"] = make_value(
                "wer",
                wer.rate,
                detail=caveat,
                extra={"edits": wer.edits, "reference_words": wer.reference_length},
            )

        if not case.must_contain:
            out["slot_accuracy"] = missing_value(
                "slot_accuracy",
                "case declares no `must_contain` tokens",
                MetricStatus.NOT_APPLICABLE,
            )
        else:
            hits, missing, unverifiable = slot_evaluation(
                hypothesis, case.must_contain, language=case.language
            )
            verifiable = len(case.must_contain) - len(unverifiable)
            if verifiable == 0:
                # Every required token is in a script this ASR cannot emit (e.g.
                # a Latin "OTP" through an Indic-only ASR). Scoring 0 here would
                # read as "the model dropped every OTP" when the transcript simply
                # cannot witness it — so report not-applicable, like the en/CER gap.
                out["slot_accuracy"] = missing_value(
                    "slot_accuracy",
                    f"required token(s) {', '.join(unverifiable)} are in a script the "
                    f"ASR ({self._asr.name}) does not emit for language {case.language!r}; "
                    "round-trip slot verification is not possible",
                    MetricStatus.NOT_APPLICABLE,
                )
            else:
                details = []
                if missing:
                    details.append(f"missing from transcript: {', '.join(missing)}")
                if unverifiable:
                    details.append(
                        f"unverifiable (ASR cannot emit their script): {', '.join(unverifiable)}"
                    )
                out["slot_accuracy"] = make_value(
                    "slot_accuracy",
                    hits / verifiable,
                    extra={
                        "required": list(case.must_contain),
                        "missing": missing,
                        "unverifiable": unverifiable,
                        "hypothesis": hypothesis,
                    },
                    detail="; ".join(details) if details else None,
                )
        return out


def _has_non_verbal_tokens(text: str) -> bool:
    """True when the text contains something a model must expand to speak.

    Digits and currency/at/percent symbols are the reliable markers; punctuation
    is excluded because it is not verbalised and does not affect the transcript.
    """
    if any(ch.isdigit() for ch in text):
        return True
    return any(ch in text for ch in "₹$€£%@&#+")


__all__ = ["IntelligibilityBackend"]
