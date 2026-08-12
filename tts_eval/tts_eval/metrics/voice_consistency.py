"""Voice consistency: does the model sound like the same person every turn?

This is a real product failure for a voice agent and it is invisible to every
per-utterance metric. Each reply can be individually clean — good SNR, low CER,
excellent MOS — while pitch and timbre drift between turns, so the caller hears
the agent change identity mid-conversation. Reference-based speaker similarity
does not catch it either: every turn can sit at 0.72 against the reference and
still differ from each *other*.

Measured here as **dispersion across utterances of the same voice** in four
features that track perceived identity:

*   median F0            — perceived pitch
*   spectral centroid    — perceived brightness/timbre
*   loudness (RMS dBFS)  — perceived level
*   speaking rate        — perceived tempo

Features are extracted per utterance (cheap, numpy-only) and carried forward on
the ``intra_utterance_f0_cv`` metric's ``extra`` payload, so the run-level pass
needs no access to the audio — which by then has been written to disk and freed.
When the optional ``speaker`` backend is installed its embedding-based consistency
is reported alongside, and the report shows both rather than silently preferring
one.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from ..audio import estimate_f0, magnitude_spectrogram, spectral_centroid
from ..datasets.loader import TestCase
from ..types import MetricValue, SynthesisResult, UtteranceRecord
from .base import (
    MetricContext,
    RunBackend,
    UtteranceBackend,
    make_value,
    missing_value,
    register_backend,
)

# Relative weight of each feature in the consistency score. Pitch and timbre carry
# most of perceived identity; level and tempo shift identity less but still make a
# conversation feel like two different agents, so they contribute at lower weight.
_FEATURE_WEIGHTS = {
    "f0_median_hz": 0.35,
    "centroid_hz": 0.30,
    "loudness_dbfs": 0.20,
    "speaking_rate_cps": 0.15,
}

# Dispersion at which each feature scores zero consistency. These are NOT one
# shared number, because the features have very different natural variation for a
# single real speaker saying different sentences:
#
#   * median F0 is fairly stable within a speaker (~5-10% across utterances), so a
#     15% swing already sounds like a different person.
#   * spectral centroid is heavily content-driven — a sentence full of fricatives
#     is genuinely brighter than one full of vowels — so it needs a wider band or
#     every model looks inconsistent.
#   * speaking rate varies with phrasing and punctuation.
#   * loudness is in dB, so it uses absolute spread rather than a ratio: 6 dB of
#     swing between turns is a clearly audible level change.
#
# Calibration caveat, repeated in the report: this is a *relative* signal. Use it
# to rank models against each other on one test set, not against an absolute
# target. ``speaker_consistency`` (embedding-based, optional) is the calibrated
# version, and both are reported side by side when available.
_CV_AT_ZERO = {
    "f0_median_hz": 0.15,
    "centroid_hz": 0.30,
    "speaking_rate_cps": 0.30,
}
_LOUDNESS_DB_AT_ZERO = 6.0

# Minimum utterances per voice before a dispersion number means anything.
_MIN_GROUP = 3


@register_backend
class VoiceFeatureBackend(UtteranceBackend):
    """Per-utterance voice features and within-utterance pitch stability."""

    name = "voice_consistency"
    provides = ("intra_utterance_f0_cv",)

    def compute(
        self, case: TestCase, result: SynthesisResult, ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        audio = result.audio
        assert audio is not None

        f0 = estimate_f0(audio.samples, audio.sample_rate)
        voiced = f0[np.isfinite(f0)]

        spec, freqs = magnitude_spectrogram(audio.samples, audio.sample_rate)
        centroid = spectral_centroid(spec, freqs)
        centroid_mean = float(centroid.mean()) if centroid.size else None

        rms = float(np.sqrt(np.mean(audio.samples.astype(np.float64) ** 2)))
        loudness = 20.0 * np.log10(max(rms, 1e-12))
        rate = len(case.text) / audio.duration_s if audio.duration_s > 0 else None

        features: dict[str, Any] = {
            "f0_median_hz": float(np.median(voiced)) if voiced.size else None,
            "centroid_hz": centroid_mean,
            "loudness_dbfs": float(loudness),
            "speaking_rate_cps": rate,
            "voice": result.request.voice,
            "n_voiced_frames": int(voiced.size),
        }

        if voiced.size < 5:
            # No usable pitch track, but the other features (timbre, loudness,
            # rate) are still valid — so emit a not_computed value that *carries
            # the features anyway*, letting run-level consistency use what exists
            # instead of discarding the utterance.
            base = missing_value(
                "intra_utterance_f0_cv",
                f"only {int(voiced.size)} voiced frame(s) detected; pitch stability "
                "cannot be estimated",
            )
            return {
                "intra_utterance_f0_cv": replace(base, extra={"voice_features": features}),
            }

        cv = float(np.std(voiced) / max(np.mean(voiced), 1e-9))
        return {
            "intra_utterance_f0_cv": make_value(
                "intra_utterance_f0_cv",
                cv,
                extra={
                    "voice_features": features,
                    "f0_mean_hz": round(float(np.mean(voiced)), 2),
                    "f0_std_hz": round(float(np.std(voiced)), 2),
                    "voiced_frame_ratio": round(float(voiced.size / max(f0.size, 1)), 3),
                },
            )
        }


@register_backend
class VoiceConsistencyBackend(RunBackend):
    """Run-level cross-utterance voice stability, per voice and overall."""

    name = "voice_consistency_run"
    provides = ("voice_consistency",)

    def compute(
        self, records: Sequence[UtteranceRecord], ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            if not rec.result.ok:
                continue
            mv = rec.metrics.get("intra_utterance_f0_cv")
            feats = (mv.extra or {}).get("voice_features") if mv is not None else None
            if not feats:
                continue
            voice = feats.get("voice") or rec.result.request.voice or "(default)"
            groups.setdefault(str(voice), []).append(feats)

        if not groups:
            return {
                "voice_consistency": missing_value(
                    "voice_consistency",
                    "no per-utterance voice features available (all utterances failed, "
                    "or the voice_consistency utterance backend did not run)",
                )
            }

        per_voice: dict[str, Any] = {}
        scored: list[tuple[int, float]] = []
        for voice, feature_list in sorted(groups.items()):
            if len(feature_list) < _MIN_GROUP:
                per_voice[voice] = {
                    "n": len(feature_list),
                    "score": None,
                    "reason": f"needs at least {_MIN_GROUP} utterances, has {len(feature_list)}",
                }
                continue
            score, detail = _dispersion_score(feature_list)
            per_voice[voice] = {"n": len(feature_list), "score": round(score, 4), **detail}
            scored.append((len(feature_list), score))

        if not scored:
            return {
                "voice_consistency": missing_value(
                    "voice_consistency",
                    f"no voice has the {_MIN_GROUP} utterances needed to measure dispersion",
                    )
            }

        # Weight by group size so a voice with 40 utterances is not outvoted by one
        # with 3.
        total_n = sum(n for n, _ in scored)
        overall = sum(n * s for n, s in scored) / total_n

        return {
            "voice_consistency": make_value(
                "voice_consistency",
                overall,
                extra={"per_voice": per_voice, "n_voices": len(groups)},
                detail=(
                    "feature-dispersion estimate (F0, timbre, loudness, rate). Install the "
                    "'speaker' backend for embedding-based consistency."
                ),
            )
        }


def _dispersion_score(feature_list: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, Any]]:
    """Weighted consistency in [0, 1] from per-feature coefficients of variation."""
    detail: dict[str, Any] = {}
    weighted_sum = 0.0
    weight_used = 0.0

    for feature, weight in _FEATURE_WEIGHTS.items():
        values = [
            float(f[feature])
            for f in feature_list
            if f.get(feature) is not None and np.isfinite(float(f[feature]))
        ]
        if len(values) < _MIN_GROUP:
            detail[f"{feature}_cv"] = None
            continue
        arr = np.asarray(values, dtype=np.float64)
        if feature == "loudness_dbfs":
            # dBFS is negative and logarithmic, so a ratio-based CV is meaningless.
            # Use absolute spread in dB against its own zero point.
            dispersion = float(np.std(arr))
            normalised = dispersion / _LOUDNESS_DB_AT_ZERO
            detail[f"{feature}_std_db"] = round(dispersion, 3)
        else:
            mean = float(np.mean(arr))
            dispersion = float(np.std(arr) / abs(mean)) if abs(mean) > 1e-9 else 1.0
            normalised = dispersion / _CV_AT_ZERO[feature]
            detail[f"{feature}_cv"] = round(dispersion, 4)
        weighted_sum += weight * min(1.0, normalised)
        weight_used += weight

    if weight_used == 0:
        return 0.0, {**detail, "reason": "no usable features"}
    return float(max(0.0, 1.0 - weighted_sum / weight_used)), detail


__all__ = ["VoiceConsistencyBackend", "VoiceFeatureBackend"]
