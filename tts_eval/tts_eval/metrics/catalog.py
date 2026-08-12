"""The metric catalogue — one table that is simultaneously the schema, the
documentation, and the acceptance-criteria traceability matrix.

Every metric the framework can emit is declared here with its unit, its polarity
(which direction is better) and the acceptance criterion it serves. Three things
depend on this being centralised:

*   **Aggregation** needs the polarity even when a metric was not computed, so a
    report can say "UTMOS: not installed (higher is better)" instead of a blank.
*   **Comparison** cannot call a delta an improvement or a regression without
    polarity. Hard-coding that per call site is how a benchmark ends up reporting
    a latency increase as a win.
*   **Coverage of the story's acceptance criteria** is auditable: ``ac_matrix()``
    prints which metrics satisfy which criterion, so "the evaluation captures, at
    a minimum, ..." is verified mechanically rather than by reading code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..types import Direction

# Acceptance criteria from the platform-administrator story, used as tags below.
AC_NATURALNESS = "Speech Naturalness"
AC_PRONUNCIATION = "Pronunciation Accuracy"
AC_LATENCY = "Response Latency"
AC_VOICE_CONSISTENCY = "Voice Consistency"
AC_LANGUAGE_COVERAGE = "Language Coverage"
AC_AUDIO_QUALITY = "Audio Quality"
AC_INFERENCE_TIME = "Inference Time"
# Not named in the story but required to make the others trustworthy: a model
# that fails 40% of requests can post excellent latency on the 60% that survive.
AC_RELIABILITY = "Reliability"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    direction: Direction
    criterion: str
    scope: str  # "utterance" | "run"
    summary: str
    # Backend that produces it; "core" means always available.
    backend: str = "core"
    # Range hint used by the HTML report to draw bars sensibly.
    good: float | None = None
    bad: float | None = None
    # Smallest absolute change worth calling an improvement or a regression.
    #
    # Statistical significance alone is not enough. With enough paired samples a
    # bootstrap will happily certify a 0.001 ms difference in chunk gap as real —
    # and it *is* real, it is just meaningless. Without this floor a benchmark
    # reports noise as wins, and a reader loses the ability to tell a genuine
    # regression from a rounding artefact. When unset, a relative floor applies
    # (see compare.MIN_RELATIVE_EFFECT).
    min_effect: float | None = None


_H = Direction.HIGHER_IS_BETTER
_L = Direction.LOWER_IS_BETTER
_N = Direction.NEUTRAL


CATALOG: dict[str, MetricSpec] = {
    # ---------------- Response Latency ----------------
    "ttfb_ms": MetricSpec(
        "ttfb_ms", "ms", _L, AC_LATENCY, "utterance",
        "Time from request send to the first audio byte of any kind.",
        good=150, bad=800, min_effect=10.0
    ),
    "first_audible_ms": MetricSpec(
        "first_audible_ms", "ms", _L, AC_LATENCY, "utterance",
        "Time to the first sample above the silence floor — what a caller actually "
        "perceives as the reply starting. Diverges from TTFB when a server pads "
        "with leading silence.",
        good=200, bad=900, min_effect=10.0
    ),
    "stream_starvation_ms": MetricSpec(
        "stream_starvation_ms", "ms", _L, AC_LATENCY, "utterance",
        "Worst deficit between audio delivered and audio consumable at real-time "
        "playback. Above ~0 the playout buffer would have run dry mid-sentence.",
        good=0, bad=300, min_effect=5.0
    ),
    "stream_chunk_gap_p95_ms": MetricSpec(
        "stream_chunk_gap_p95_ms", "ms", _L, AC_LATENCY, "utterance",
        "95th percentile gap between consecutive audio chunks: streaming jitter.",
        good=120, bad=600, min_effect=5.0
    ),

    # ---------------- Inference Time ----------------
    "inference_time_ms": MetricSpec(
        "inference_time_ms", "ms", _L, AC_INFERENCE_TIME, "utterance",
        "Total wall-clock to synthesise the complete utterance.",
        good=1000, bad=6000, min_effect=20.0
    ),
    "rtf": MetricSpec(
        "rtf", "x", _L, AC_INFERENCE_TIME, "utterance",
        "Real-time factor: synthesis time divided by audio duration. Below 1.0 the "
        "model generates faster than playback, which is the requirement for live use.",
        good=0.3, bad=1.0, min_effect=0.02
    ),
    "chars_per_second": MetricSpec(
        "chars_per_second", "char/s", _H, AC_INFERENCE_TIME, "utterance",
        "Input characters synthesised per second of wall-clock.", min_effect=1.0
    ),
    "audio_duration_s": MetricSpec(
        "audio_duration_s", "s", _N, AC_INFERENCE_TIME, "utterance",
        "Duration of the produced audio. Context for RTF, not a quality signal.", min_effect=0.05
    ),
    "throughput_utt_per_min": MetricSpec(
        "throughput_utt_per_min", "utt/min", _H, AC_INFERENCE_TIME, "run",
        "Completed utterances per minute at the run's concurrency level.", min_effect=1.0
    ),

    # ---------------- Audio Quality ----------------
    "snr_db": MetricSpec(
        "snr_db", "dB", _H, AC_AUDIO_QUALITY, "utterance",
        "Estimated speech-to-noise ratio from the energy distribution "
        "(high-percentile frames vs. noise-floor frames).",
        good=35, bad=15, min_effect=1.0
    ),
    "clipping_pct": MetricSpec(
        "clipping_pct", "%", _L, AC_AUDIO_QUALITY, "utterance",
        "Share of samples at or beyond full scale. Non-zero means audible distortion.",
        good=0.0, bad=0.5, min_effect=0.05
    ),
    "silence_ratio": MetricSpec(
        "silence_ratio", "ratio", _L, AC_AUDIO_QUALITY, "utterance",
        "Fraction of frames below the silence floor. High values mean dead air or "
        "a partially failed generation.",
        good=0.15, bad=0.5, min_effect=0.02
    ),
    "leading_silence_ms": MetricSpec(
        "leading_silence_ms", "ms", _L, AC_AUDIO_QUALITY, "utterance",
        "Silence before speech starts. Directly adds to perceived response latency.",
        good=50, bad=400, min_effect=10.0
    ),
    "trailing_silence_ms": MetricSpec(
        "trailing_silence_ms", "ms", _L, AC_AUDIO_QUALITY, "utterance",
        "Silence after speech ends. Adds dead air before the agent can listen again.",
        good=80, bad=500, min_effect=10.0
    ),
    "loudness_dbfs": MetricSpec(
        "loudness_dbfs", "dBFS", _N, AC_AUDIO_QUALITY, "utterance",
        "RMS level over speech frames. Judged on consistency, not absolute value; "
        "wide spread across utterances means the caller keeps adjusting volume.", min_effect=0.5
    ),
    "dynamic_range_db": MetricSpec(
        "dynamic_range_db", "dB", _H, AC_AUDIO_QUALITY, "utterance",
        "Spread between loud and quiet speech frames. Very low values indicate "
        "over-compressed, lifeless output.",
        good=25, bad=8, min_effect=1.0
    ),
    "dc_offset": MetricSpec(
        "dc_offset", "", _L, AC_AUDIO_QUALITY, "utterance",
        "Mean sample value. Non-zero DC wastes headroom and can click on playback.",
        good=0.0, bad=0.02, min_effect=0.001
    ),
    "spectral_flatness": MetricSpec(
        "spectral_flatness", "ratio", _L, AC_AUDIO_QUALITY, "utterance",
        "Noise-likeness of voiced frames. High values indicate vocoder buzz or a "
        "codec collapse rather than clean speech.",
        good=0.05, bad=0.3, min_effect=0.01
    ),
    "length_ratio": MetricSpec(
        "length_ratio", "x", _N, AC_AUDIO_QUALITY, "utterance",
        "Actual duration divided by the duration expected for this text length. "
        "Well below 1 means truncation; well above 1 means the model looped or "
        "dragged. Both are silent failures that latency and CER can miss.",
        good=1.0, bad=0.5, min_effect=0.05
    ),
    "degeneracy_score": MetricSpec(
        "degeneracy_score", "score", _L, AC_AUDIO_QUALITY, "utterance",
        "0-1 evidence that the generation collapsed (looping, buzz, all-silence, "
        "truncation). The characteristic autoregressive-TTS failure mode.",
        good=0.0, bad=0.5, min_effect=0.05
    ),
    "audio_quality_score": MetricSpec(
        "audio_quality_score", "score", _H, AC_AUDIO_QUALITY, "utterance",
        "0-1 composite of the signal-level checks above. A screening aid — it is "
        "not a MOS substitute and is never presented as one.",
        good=0.9, bad=0.6, min_effect=0.02
    ),

    # ---------------- Pronunciation Accuracy ----------------
    "cer": MetricSpec(
        "cer", "ratio", _L, AC_PRONUNCIATION, "utterance",
        "Character error rate of a round-trip ASR transcript against the input text. "
        "The primary intelligibility metric for Indic scripts, where word "
        "segmentation is unreliable and WER is correspondingly noisy.",
        good=0.05, bad=0.25, backend="asr", min_effect=0.005
    ),
    "wer": MetricSpec(
        "wer", "ratio", _L, AC_PRONUNCIATION, "utterance",
        "Word error rate of the round-trip transcript. Reported for comparability "
        "with published English benchmarks.",
        good=0.08, bad=0.35, backend="asr", min_effect=0.005
    ),
    "slot_accuracy": MetricSpec(
        "slot_accuracy", "ratio", _H, AC_PRONUNCIATION, "utterance",
        "Fraction of the case's required tokens (OTP, account number, product name) "
        "present in the transcript. Catches the dropped-entity failure that CER "
        "hides: losing a 4-digit code in a 40-character sentence still scores 0.9.",
        good=1.0, bad=0.8, backend="asr", min_effect=0.02
    ),

    # ---------------- Speech Naturalness ----------------
    "utmos": MetricSpec(
        "utmos", "MOS", _H, AC_NATURALNESS, "utterance",
        "UTMOS predicted naturalness (1-5). The de-facto objective standard, but "
        "documented to rank-invert against human listeners — always read next to "
        "subjective_mos.",
        good=4.0, bad=3.0, backend="utmos", min_effect=0.05
    ),
    "dnsmos_ovrl": MetricSpec(
        "dnsmos_ovrl", "MOS", _H, AC_NATURALNESS, "utterance",
        "DNSMOS P.835 overall quality (1-5).", good=4.0, bad=3.0, backend="dnsmos", min_effect=0.05
    ),
    "dnsmos_sig": MetricSpec(
        "dnsmos_sig", "MOS", _H, AC_NATURALNESS, "utterance",
        "DNSMOS P.835 signal quality (1-5).", good=4.0, bad=3.0, backend="dnsmos", min_effect=0.05
    ),
    "dnsmos_bak": MetricSpec(
        "dnsmos_bak", "MOS", _H, AC_NATURALNESS, "utterance",
        "DNSMOS P.835 background intrusiveness (1-5).", good=4.0, bad=3.0, backend="dnsmos", min_effect=0.05
    ),
    "subjective_mos": MetricSpec(
        "subjective_mos", "MOS", _H, AC_NATURALNESS, "utterance",
        "Mean human naturalness rating from an ingested listening test. The only "
        "metric here that measures naturalness rather than predicting it.",
        good=4.0, bad=3.0, backend="subjective", min_effect=0.1
    ),
    "subjective_mushra": MetricSpec(
        "subjective_mushra", "0-100", _H, AC_NATURALNESS, "utterance",
        "Mean human MUSHRA score (0-100), the scale used by IndicVoices-R and "
        "RASMALAI for Indic TTS.",
        good=80, bad=50, backend="subjective", min_effect=2.0
    ),
    "subjective_cmos": MetricSpec(
        "subjective_cmos", "-3..+3", _H, AC_NATURALNESS, "utterance",
        "Mean human comparative preference against the paired system. Positive means "
        "listeners preferred this system.",
        good=1.0, bad=-1.0, backend="subjective", min_effect=0.2
    ),
    "subjective_smos": MetricSpec(
        "subjective_smos", "MOS", _H, AC_VOICE_CONSISTENCY, "utterance",
        "Mean human speaker-similarity rating against the reference voice (1-5). The "
        "human counterpart to speaker_similarity.",
        good=4.0, bad=3.0, backend="subjective", min_effect=0.1
    ),
    "ttsds2_overall": MetricSpec(
        "ttsds2_overall", "score", _H, AC_NATURALNESS, "run",
        "TTSDS2 weighted overall score: distributional distance from real speech "
        "across prosody, speaker, intelligibility and generic-quality categories.",
        good=80, bad=50, backend="ttsds2", min_effect=1.0
    ),

    # ---------------- Voice Consistency ----------------
    "voice_consistency": MetricSpec(
        "voice_consistency", "score", _H, AC_VOICE_CONSISTENCY, "run",
        "0-1 stability of voice identity across utterances of the same voice, from "
        "pitch, timbre, loudness and rate dispersion (or speaker embeddings when "
        "that backend is installed). Low values mean the caller hears a different "
        "person between turns.",
        good=0.9, bad=0.7, min_effect=0.02
    ),
    "intra_utterance_f0_cv": MetricSpec(
        "intra_utterance_f0_cv", "ratio", _L, AC_VOICE_CONSISTENCY, "utterance",
        "Coefficient of variation of F0 within one utterance. Very high values "
        "indicate pitch instability or a voice switching mid-sentence.",
        good=0.2, bad=0.5, min_effect=0.01
    ),
    "speaker_similarity": MetricSpec(
        "speaker_similarity", "cosine", _H, AC_VOICE_CONSISTENCY, "utterance",
        "Cosine similarity between the output's speaker embedding and the reference "
        "voice's. The SIM/SECS metric used in published TTS work.",
        good=0.7, bad=0.4, backend="speaker", min_effect=0.02
    ),
    "speaker_consistency": MetricSpec(
        "speaker_consistency", "cosine", _H, AC_VOICE_CONSISTENCY, "run",
        "Mean pairwise speaker-embedding similarity among utterances of the same "
        "voice. The embedding-based counterpart to voice_consistency; both are "
        "reported so it is visible when the dependency-free estimate misleads.",
        good=0.85, bad=0.6, backend="speaker_consistency", min_effect=0.02
    ),

    # ---------------- Language Coverage ----------------
    "coverage_ratio": MetricSpec(
        "coverage_ratio", "ratio", _H, AC_LANGUAGE_COVERAGE, "run",
        "Verified languages divided by languages claimed in the model card. A card "
        "claim is not evidence; this is measured.",
        good=1.0, bad=0.7, min_effect=0.001
    ),
    "languages_verified": MetricSpec(
        "languages_verified", "count", _H, AC_LANGUAGE_COVERAGE, "run",
        "Languages that both synthesised successfully and passed the intelligibility "
        "threshold.", min_effect=0.5
    ),
    "languages_attempted": MetricSpec(
        "languages_attempted", "count", _N, AC_LANGUAGE_COVERAGE, "run",
        "Languages present in the test set for this run.", min_effect=0.5
    ),

    # ---------------- Reliability ----------------
    "success_rate": MetricSpec(
        "success_rate", "ratio", _H, AC_RELIABILITY, "run",
        "Utterances that returned usable audio. Reported first because every other "
        "aggregate is conditioned on it.",
        good=1.0, bad=0.95, min_effect=0.005
    ),
    "degenerate_rate": MetricSpec(
        "degenerate_rate", "ratio", _L, AC_RELIABILITY, "run",
        "Utterances that returned audio but whose degeneracy_score exceeded the "
        "threshold: nominal successes that a caller would hear as a failure.",
        good=0.0, bad=0.05, min_effect=0.005
    ),
}


def spec(name: str) -> MetricSpec:
    """Look up a metric, tolerating unknown names.

    Unknown metrics are given a neutral spec rather than raising, so a plugin can
    emit an extra measurement without a catalogue edit. It just will not get
    polarity-aware treatment in reports, which is the intended nudge to declare it.
    """
    found = CATALOG.get(name)
    if found is not None:
        return found
    return MetricSpec(name, "", _N, "Uncategorised", "utterance", "Plugin-provided metric.")


def unit(name: str) -> str:
    return spec(name).unit


def direction(name: str) -> Direction:
    return spec(name).direction


def by_criterion() -> dict[str, list[MetricSpec]]:
    out: dict[str, list[MetricSpec]] = {}
    for s in CATALOG.values():
        out.setdefault(s.criterion, []).append(s)
    return {k: sorted(v, key=lambda s: s.name) for k, v in out.items()}


def criteria_order() -> list[str]:
    """Report ordering: reliability first (it conditions everything), then the
    story's criteria in the order a reviewer reads them."""
    return [
        AC_RELIABILITY,
        AC_NATURALNESS,
        AC_PRONUNCIATION,
        AC_LATENCY,
        AC_INFERENCE_TIME,
        AC_AUDIO_QUALITY,
        AC_VOICE_CONSISTENCY,
        AC_LANGUAGE_COVERAGE,
    ]


def ac_matrix(available_backends: Iterable[str] | None = None) -> list[dict[str, object]]:
    """Traceability matrix: criterion -> metrics -> whether they can be computed now.

    Surfaced by ``tts-eval metrics`` and embedded in every report so a reviewer
    can see at a glance that each required criterion is covered, and by what.
    """
    have = set(available_backends) if available_backends is not None else None
    rows: list[dict[str, object]] = []
    grouped = by_criterion()
    for criterion in criteria_order() + [c for c in sorted(grouped) if c not in criteria_order()]:
        for s in grouped.get(criterion, []):
            rows.append(
                {
                    "criterion": criterion,
                    "metric": s.name,
                    "unit": s.unit,
                    "scope": s.scope,
                    "backend": s.backend,
                    "available": None if have is None else (s.backend in have),
                    "summary": s.summary,
                }
            )
    return rows


__all__ = [
    "CATALOG",
    "MetricSpec",
    "spec",
    "unit",
    "direction",
    "by_criterion",
    "criteria_order",
    "ac_matrix",
    "AC_NATURALNESS",
    "AC_PRONUNCIATION",
    "AC_LATENCY",
    "AC_VOICE_CONSISTENCY",
    "AC_LANGUAGE_COVERAGE",
    "AC_AUDIO_QUALITY",
    "AC_INFERENCE_TIME",
    "AC_RELIABILITY",
]
