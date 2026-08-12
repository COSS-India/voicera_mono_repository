"""Signal-level audio-quality metrics and degenerate-output detection.

Core backend: numpy only, runs at the provider's native sample rate (never on
resampled audio, which would measure our resampler as much as the model).

Scope, stated plainly: these are **objective signal checks, not a MOS**. They
catch the failures that make audio unusable — clipping, buzz, dead air,
truncation, looping — and they screen a large run down to the utterances a human
should actually listen to. They cannot tell you whether a voice sounds pleasant.
``utmos``/``dnsmos`` predict that, and ``subjective_mos`` measures it; the report
keeps all three visibly separate rather than folding them into one "quality"
number.

The degeneracy detector earns its place because autoregressive TTS (Indic-Mio is
LLM-based) fails in ways latency and CER both miss: it returns HTTP 200, streams
the expected number of bytes, and the content is a loop, a buzz, or half a
sentence. A harness without this check records those as successes.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from ..audio import (
    SILENCE_FLOOR,
    frame_rms,
    magnitude_spectrogram,
    spectral_flatness,
    trim_silence,
)
from ..datasets.loader import TestCase
from ..types import MetricValue, SynthesisResult
from .base import MetricContext, UtteranceBackend, make_value, missing_value, register_backend

# Analysis framing. 25 ms / 10 ms is the standard speech-processing choice and
# matches what the ASR and MOS backends use internally, so frame counts line up
# when cross-referencing.
_FRAME_MS = 25.0
_HOP_MS = 10.0

# A sample within this of full scale counts as clipped. Not exactly 1.0 because
# int16 round-tripping lands values at 32767/32768 = 0.99997.
_CLIP_THRESHOLD = 0.999

# Lag band searched for self-similarity when detecting loops, in seconds. Below
# 0.2 s normal syllabic rhythm dominates; above 1.5 s a repeat is long enough
# that CER would already have caught it.
_LOOP_LAG_MIN_S = 0.20
_LOOP_LAG_MAX_S = 1.50
# Repetition is measured on *mean-removed* spectra. Raw magnitude spectra are all
# dominated by the same spectral tilt (energy concentrated low), so their cosine
# similarity is high between any two frames of anything — which makes a raw
# measure flag clean speech and miss real loops. Subtracting the utterance's mean
# spectrum leaves only what varies over time, and then the three cases separate
# cleanly:
#   * looping generation -> the variation pattern repeats, similarity -> ~1
#   * clean speech       -> successive phonemes differ, similarity stays low
#   * steady buzz/tone   -> almost no variation, residual is noise, similarity ~0
_LOOP_SIMILARITY_THRESHOLD = 0.70

# RMS frame-to-frame deviation of the log spectrum (in nats) below which the audio
# is treated as spectrally static, so repetition is not assessed at all. Real
# speech moves well above this as phonemes change; a held tone sits far below.
_MIN_SPECTRAL_MOVEMENT = 0.15

# Dynamic range kept when log-compressing the spectrum for the repetition search.
# 80 dB below peak is far below audibility and keeps empty bins from contributing
# log-amplified numerical noise.
_LOG_FLOOR_DB = 80.0

# Frame pairs a candidate lag must leave to be averaged over. At a 10 ms hop this
# means repetition is only assessed on utterances of roughly 0.5 s or more; shorter
# clips report the check as skipped rather than a spuriously clean 0.0.
_MIN_FRAME_PAIRS = 25


@register_backend
class AudioQualityBackend(UtteranceBackend):
    name = "audio_quality"
    provides = (
        "snr_db",
        "clipping_pct",
        "silence_ratio",
        "leading_silence_ms",
        "trailing_silence_ms",
        "loudness_dbfs",
        "dynamic_range_db",
        "dc_offset",
        "spectral_flatness",
        "length_ratio",
        "degeneracy_score",
        "audio_quality_score",
    )

    def compute(
        self, case: TestCase, result: SynthesisResult, ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        audio = result.audio
        assert audio is not None
        x = audio.samples.astype(np.float64)
        sr = audio.sample_rate
        out: dict[str, MetricValue] = {}

        frame_len = max(16, int(sr * _FRAME_MS / 1000.0))
        hop_len = max(1, int(sr * _HOP_MS / 1000.0))
        rms = frame_rms(audio.samples, frame_len, hop_len).astype(np.float64)

        if rms.size == 0:
            # Audio shorter than one analysis frame: real (a 30 ms blip) but
            # unmeasurable. Say so rather than emitting numbers from one frame.
            for name in self.provides:
                out[name] = missing_value(
                    name, f"audio too short to analyse ({audio.n_samples} samples)"
                )
            out["length_ratio"] = self._length_ratio(case, audio.duration_s, ctx)
            out["degeneracy_score"] = make_value(
                "degeneracy_score", 1.0, detail="audio shorter than one analysis frame"
            )
            return out

        # --- silence structure -------------------------------------------
        speech_mask = rms > SILENCE_FLOOR
        silence_ratio = float(1.0 - speech_mask.mean())
        out["silence_ratio"] = make_value("silence_ratio", silence_ratio)

        _, n_lead, n_trail = trim_silence(audio.samples)
        out["leading_silence_ms"] = make_value("leading_silence_ms", n_lead / sr * 1000.0)
        out["trailing_silence_ms"] = make_value("trailing_silence_ms", n_trail / sr * 1000.0)

        all_silent = not bool(speech_mask.any())
        if all_silent:
            for name in ("snr_db", "loudness_dbfs", "dynamic_range_db", "spectral_flatness"):
                out[name] = missing_value(name, "audio contains no speech above the silence floor")
            out["clipping_pct"] = make_value("clipping_pct", 0.0)
            out["dc_offset"] = make_value("dc_offset", float(abs(x.mean())))
            out["length_ratio"] = self._length_ratio(case, audio.duration_s, ctx)
            out["degeneracy_score"] = make_value(
                "degeneracy_score", 1.0, detail="silent output: server returned audio with no speech"
            )
            out["audio_quality_score"] = make_value("audio_quality_score", 0.0)
            return out

        # --- level and headroom ------------------------------------------
        speech_rms = rms[speech_mask]
        rms_db = 20.0 * np.log10(np.maximum(speech_rms, 1e-12))
        out["loudness_dbfs"] = make_value(
            "loudness_dbfs",
            float(20.0 * np.log10(max(np.sqrt(np.mean(speech_rms**2)), 1e-12))),
            extra={"peak_dbfs": round(float(20.0 * np.log10(max(np.abs(x).max(), 1e-12))), 2)},
        )
        # p95 - p5 over speech frames only. Including silence would report the
        # noise floor as "dynamic range" and make dead air look expressive.
        dyn = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
        out["dynamic_range_db"] = make_value("dynamic_range_db", dyn)

        n_clipped = int(np.count_nonzero(np.abs(x) >= _CLIP_THRESHOLD))
        out["clipping_pct"] = make_value(
            "clipping_pct", 100.0 * n_clipped / max(x.size, 1), extra={"n_clipped_samples": n_clipped}
        )
        out["dc_offset"] = make_value("dc_offset", float(abs(x.mean())))

        # --- SNR ----------------------------------------------------------
        # Percentile method (NIST-style): speech level = p95 of frame energy,
        # noise floor = p5. Robust without a VAD, and a VAD would itself need a
        # model — which the core tier must not require.
        speech_level_db = float(np.percentile(20.0 * np.log10(np.maximum(rms, 1e-12)), 95))
        noise_level_db = float(np.percentile(20.0 * np.log10(np.maximum(rms, 1e-12)), 5))
        out["snr_db"] = make_value(
            "snr_db",
            speech_level_db - noise_level_db,
            extra={
                "speech_level_dbfs": round(speech_level_db, 2),
                "noise_level_dbfs": round(noise_level_db, 2),
            },
            detail="percentile estimate (p95 speech vs p5 noise), not a reference-based SNR",
        )

        # --- spectral character ------------------------------------------
        spec, _freqs = magnitude_spectrogram(audio.samples, sr, frame_ms=_FRAME_MS, hop_ms=_HOP_MS)
        flat_all = spectral_flatness(spec)
        # Restrict to speech frames: silence is noise by definition and would
        # dominate the mean, flagging every clean utterance with a long pause.
        usable = min(flat_all.size, speech_mask.size)
        voiced_flat = flat_all[:usable][speech_mask[:usable]]
        flatness = float(voiced_flat.mean()) if voiced_flat.size else float("nan")
        out["spectral_flatness"] = (
            make_value("spectral_flatness", flatness)
            if np.isfinite(flatness)
            else missing_value("spectral_flatness", "no voiced frames to analyse")
        )

        # --- duration plausibility ---------------------------------------
        length = self._length_ratio(case, audio.duration_s, ctx)
        out["length_ratio"] = length

        # --- degeneracy ---------------------------------------------------
        loop_score, loop_lag_s, loop_contrast = _self_similarity(spec, hop_len / sr)
        degeneracy, reasons = _degeneracy(
            length_ratio=length.value,
            silence_ratio=silence_ratio,
            flatness=flatness,
            clipping_pct=out["clipping_pct"].value or 0.0,
            loop_similarity=loop_score,
            loop_contrast=loop_contrast,
            thresholds=ctx.thresholds,
        )
        # Which sub-checks actually ran. A short clip leaves no room for the
        # repetition search, and a score of 0.0 must not be read as "verified clean"
        # when a check was skipped.
        checks_skipped: list[str] = []
        if loop_score is None:
            checks_skipped.append(
                "repetition (audio too short, or spectrum too static to have a trajectory)"
            )
        if not np.isfinite(flatness):
            checks_skipped.append("spectral flatness (no voiced frames)")

        out["degeneracy_score"] = make_value(
            "degeneracy_score",
            degeneracy,
            detail=(
                "; ".join(reasons)
                if reasons
                else (
                    "no failure evidence; skipped checks: " + ", ".join(checks_skipped)
                    if checks_skipped
                    else None
                )
            ),
            extra={
                "loop_similarity": None if loop_score is None else round(loop_score, 4),
                "loop_lag_s": None if loop_lag_s is None else round(loop_lag_s, 3),
                "loop_contrast": None if loop_contrast is None else round(loop_contrast, 4),
                "checks_skipped": checks_skipped,
            },
        )

        out["audio_quality_score"] = make_value(
            "audio_quality_score",
            _composite_score(
                snr_db=out["snr_db"].value,
                clipping_pct=out["clipping_pct"].value,
                silence_ratio=silence_ratio,
                dynamic_range_db=dyn,
                flatness=flatness,
                dc_offset=out["dc_offset"].value,
                degeneracy=degeneracy,
            ),
            detail="composite of signal checks; screening aid, not a MOS estimate",
        )
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _length_ratio(case: TestCase, duration_s: float, ctx: MetricContext) -> MetricValue:
        """Actual duration vs. the duration this much text should take.

        The expectation is a coarse chars-per-second rate, so the *value* is not
        precise — but the tails are unambiguous: 0.4 means the model stopped
        halfway, 2.5 means it looped. Those are the cases this exists to catch.
        """
        cps = ctx.chars_per_second(case.language)
        if cps <= 0 or not case.text:
            return missing_value("length_ratio", "cannot estimate expected duration for empty text")
        # Affine, not proportional: a fixed overhead for onset/offset/padding plus a
        # per-character rate. See Thresholds.duration_overhead_s for why the
        # intercept matters — without it, truncation of short utterances is invisible.
        overhead = ctx.thresholds.duration_overhead_s
        expected = overhead + len(case.text) / cps
        return make_value(
            "length_ratio",
            duration_s / expected,
            extra={
                "expected_duration_s": round(expected, 3),
                "actual_duration_s": round(duration_s, 3),
                "chars": len(case.text),
                "chars_per_second_assumed": cps,
                "duration_overhead_s": overhead,
            },
        )


def _self_similarity(
    spec: np.ndarray, hop_s: float
) -> tuple[float | None, float | None, float | None]:
    """Repetition strength from mean-removed spectral self-similarity.

    Returns ``(peak_similarity, peak_lag_seconds, contrast)``.

    Each frame's log-magnitude spectrum has the utterance mean subtracted, so what
    remains is how the spectrum *changes* over time. Cosine similarity of that
    residual at lag L answers "does the spectral trajectory repeat after L
    seconds". ``contrast`` (peak minus median across lags) is retained as a
    diagnostic in the metric's ``extra``: a high peak with near-zero contrast
    means uniform periodicity, which is useful when reading a flagged utterance,
    but the flag itself is driven by the peak.

    Log magnitude rather than linear: it compresses the huge dynamic range between
    formant peaks and valleys, so similarity reflects spectral shape rather than
    being dominated by whichever frames happen to be loudest.
    """
    if spec.shape[0] < 8 or hop_s <= 0:
        return None, None, None

    # Floor the spectrum relative to its own peak before taking the log, NOT with a
    # fixed absolute epsilon. A pure tone has near-zero energy in most bins, and
    # `log(tiny + 1e-8)` turns float noise in those bins into log swings of several
    # nats — which then dominate the residual and make a held tone look like it
    # varies wildly. Clamping 80 dB below peak (well below anything audible) makes
    # every empty bin land on the same value, so the residual reflects real spectral
    # change only.
    spec64 = spec.astype(np.float64)
    floor = float(spec64.max()) * (10.0 ** (-_LOG_FLOOR_DB / 20.0))
    log_spec = np.log(np.maximum(spec64, max(floor, 1e-12)))
    residual = log_spec - log_spec.mean(axis=0, keepdims=True)

    # Bail out when the spectrum barely moves at all. A held tone or a steady buzz
    # has a near-zero residual whose *tiny* remaining values are nonetheless
    # identical frame to frame — so after L2 normalisation the cosine similarity is
    # ~1.0 at every lag and the audio reads as a perfect loop. Normalising
    # floating-point noise into a confident match is the trap here, and a small
    # absolute epsilon does not close it because the residual is consistently
    # small rather than randomly small.
    #
    # RMS log deviation is the scale-free test: real speech moves 0.5-3 nats
    # frame to frame as phonemes change; a steady tone sits below 0.1. Static
    # output is not a loop, and it is already caught by spectral_flatness and the
    # silence checks, so returning "no trajectory" here is both correct and safe.
    rms_log_deviation = float(np.sqrt(np.mean(residual**2)))
    if rms_log_deviation < _MIN_SPECTRAL_MOVEMENT:
        return None, None, None

    norms = np.linalg.norm(residual, axis=1, keepdims=True)
    unit = np.where(norms > 1e-9, residual / np.maximum(norms, 1e-12), 0.0)

    lag_min = max(1, int(_LOOP_LAG_MIN_S / hop_s))
    # A lag is only trustworthy if enough frame pairs remain to average over.
    # Allowing a lag that leaves 5 pairs would let a sub-second utterance produce a
    # confident-looking repetition number from almost no evidence — better to report
    # the check as unassessable and let `checks_skipped` say so.
    lag_max = min(spec.shape[0] - _MIN_FRAME_PAIRS, int(_LOOP_LAG_MAX_S / hop_s))
    if lag_max <= lag_min:
        return None, None, None

    # Every lag in the band, one frame apart. A coarse grid is tempting for speed
    # but wrong: at a 10 ms hop, being one or two frames off the true loop period
    # already decorrelates the comparison, so a subsampled search reports a low
    # similarity for audio that is looping exactly. The band holds at most ~130
    # lags, which is cheap enough to scan exhaustively.
    lags = list(range(lag_min, lag_max + 1))
    sims = np.asarray(
        [float(np.mean(np.sum(unit[:-lag] * unit[lag:], axis=1))) for lag in lags],
        dtype=np.float64,
    )
    if sims.size == 0:
        return None, None, None
    peak_idx = int(np.argmax(sims))
    contrast = float(sims[peak_idx] - np.median(sims))
    return float(sims[peak_idx]), lags[peak_idx] * hop_s, contrast


def _degeneracy(
    *,
    length_ratio: float | None,
    silence_ratio: float,
    flatness: float,
    clipping_pct: float,
    loop_similarity: float | None,
    loop_contrast: float | None,
    thresholds,
) -> tuple[float, list[str]]:
    """Combine failure evidence into a 0-1 score plus human-readable reasons.

    Additive rather than multiplicative, and capped at 1.0: several weak signals
    together should raise suspicion, but any single unambiguous signal (a loop, a
    hard truncation) should already push past the 0.5 flag line on its own.
    """
    score = 0.0
    reasons: list[str] = []

    if length_ratio is not None:
        if length_ratio < thresholds.length_ratio_min:
            score += 0.6
            reasons.append(f"truncated: audio is {length_ratio:.2f}x expected duration")
        elif length_ratio > thresholds.length_ratio_max:
            score += 0.5
            reasons.append(f"over-long: audio is {length_ratio:.2f}x expected duration")

    if loop_similarity is not None and loop_similarity > _LOOP_SIMILARITY_THRESHOLD:
        score += 0.6
        reasons.append(
            f"repetitive: mean-removed spectral self-similarity {loop_similarity:.3f}"
            + (f" (contrast {loop_contrast:.3f})" if loop_contrast is not None else "")
        )

    if np.isfinite(flatness) and flatness > 0.30:
        score += 0.3
        reasons.append(f"noise-like/buzzy: spectral flatness {flatness:.3f}")

    if silence_ratio > 0.70:
        score += 0.4
        reasons.append(f"mostly silent: {silence_ratio:.0%} of frames below the silence floor")

    if clipping_pct > 1.0:
        score += 0.2
        reasons.append(f"heavy clipping: {clipping_pct:.2f}% of samples at full scale")

    return min(1.0, score), reasons


def _composite_score(
    *,
    snr_db: float | None,
    clipping_pct: float | None,
    silence_ratio: float,
    dynamic_range_db: float,
    flatness: float,
    dc_offset: float | None,
    degeneracy: float,
) -> float:
    """Weighted 0-1 composite of the signal checks.

    Weights are a documented judgement call for *screening*, not a calibrated
    perceptual model. SNR and degeneracy dominate because they map most directly
    onto "a caller would notice"; DC offset is included at low weight because it
    is cheap to detect and always a bug.
    """
    def band(value: float, good: float, bad: float) -> float:
        """Map a value onto [0, 1], 1 at `good`, 0 at `bad`, linear between."""
        if good == bad:
            return 1.0
        t = (value - bad) / (good - bad)
        return float(min(1.0, max(0.0, t)))

    parts = [
        (0.30, band(snr_db if snr_db is not None else 0.0, good=35.0, bad=12.0)),
        (0.15, band(-(clipping_pct or 0.0), good=0.0, bad=-1.0)),
        (0.10, band(-silence_ratio, good=-0.10, bad=-0.60)),
        (0.15, band(dynamic_range_db, good=25.0, bad=6.0)),
        (0.15, band(-(flatness if np.isfinite(flatness) else 1.0), good=-0.05, bad=-0.35)),
        (0.05, band(-(dc_offset or 0.0), good=0.0, bad=-0.05)),
    ]
    base = sum(w * v for w, v in parts) / sum(w for w, _ in parts)
    # Degeneracy is a gate, not a term: an utterance that looped is not "70% good".
    return float(max(0.0, base * (1.0 - degeneracy)))


__all__ = ["AudioQualityBackend"]
