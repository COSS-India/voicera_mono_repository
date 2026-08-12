"""Response-latency and inference-time metrics. Core: no dependencies.

Streaming TTS latency is not one number. This backend separates four things that
a single "latency" figure conflates, because they fail independently:

*   ``ttfb_ms`` — the server started answering.
*   ``first_audible_ms`` — the caller heard speech. Later than TTFB by however
    much leading silence the model pads with.
*   ``inference_time_ms`` / ``rtf`` — total cost to produce the utterance. This is
    the throughput/capacity number, not the perceived-latency number.
*   ``stream_starvation_ms`` — whether delivery kept ahead of playback. A model
    can post a 120 ms TTFB and still stutter, and only this catches it.

Starvation is measured against a real-time playout schedule rather than by
looking at raw inter-chunk gaps: a long gap is harmless if the previous chunk
contained enough audio to cover it. That distinction is what makes the number
actionable instead of alarming.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..datasets.loader import TestCase
from ..types import MetricStatus, MetricValue, SynthesisResult, UtteranceRecord
from .base import MetricContext, RunBackend, UtteranceBackend, make_value, missing_value, register_backend


@register_backend
class LatencyBackend(UtteranceBackend):
    name = "latency"
    provides = (
        "ttfb_ms",
        "first_audible_ms",
        "inference_time_ms",
        "rtf",
        "chars_per_second",
        "audio_duration_s",
        "stream_starvation_ms",
        "stream_chunk_gap_p95_ms",
    )

    def compute(
        self, case: TestCase, result: SynthesisResult, ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        audio = result.audio
        assert audio is not None  # engine guarantees this
        duration = audio.duration_s
        out: dict[str, MetricValue] = {}

        # Replayed audio has no meaningful timings unless the original run's
        # timings.json travelled with it. Fabricating them from file size would
        # produce a plausible-looking number that means nothing.
        replayed_without_timings = (
            result.provider_meta.get("replayed") and not result.provider_meta.get("replayed_timings")
        )

        if replayed_without_timings:
            for name in ("ttfb_ms", "first_audible_ms", "inference_time_ms", "rtf", "chars_per_second"):
                out[name] = missing_value(
                    name, "replayed audio carries no original timings", MetricStatus.NOT_APPLICABLE
                )
        else:
            out["ttfb_ms"] = (
                make_value("ttfb_ms", result.ttfb_ms)
                if result.ttfb_ms is not None
                else missing_value("ttfb_ms", "server sent no audio frame")
            )
            out["first_audible_ms"] = (
                make_value("first_audible_ms", result.first_audible_ms)
                if result.first_audible_ms is not None
                else missing_value(
                    "first_audible_ms", "no sample above the silence floor (audio is silent)"
                )
            )
            out["inference_time_ms"] = make_value("inference_time_ms", result.total_ms)
            out["rtf"] = (
                make_value(
                    "rtf",
                    (result.total_ms / 1000.0) / duration,
                    extra={"audio_duration_s": round(duration, 4)},
                )
                if duration > 0
                else missing_value("rtf", "zero-length audio")
            )
            out["chars_per_second"] = (
                make_value("chars_per_second", len(case.text) / (result.total_ms / 1000.0))
                if result.total_ms > 0
                else missing_value("chars_per_second", "zero elapsed time")
            )

        out["audio_duration_s"] = make_value("audio_duration_s", duration)

        # --- streaming behaviour -----------------------------------------
        if not ctx.capabilities.streaming:
            for name in ("stream_starvation_ms", "stream_chunk_gap_p95_ms"):
                out[name] = missing_value(
                    name, "provider is not streaming", MetricStatus.NOT_APPLICABLE
                )
            return out

        timings = result.chunk_timings
        if len(timings) < 2 or replayed_without_timings:
            reason = (
                "replayed audio carries no chunk timings"
                if replayed_without_timings
                else f"only {len(timings)} chunk(s) received; no streaming behaviour to measure"
            )
            for name in ("stream_starvation_ms", "stream_chunk_gap_p95_ms"):
                out[name] = missing_value(name, reason, MetricStatus.NOT_APPLICABLE)
            return out

        gaps = np.diff(np.asarray([t.offset_ms for t in timings], dtype=np.float64))
        out["stream_chunk_gap_p95_ms"] = make_value(
            "stream_chunk_gap_p95_ms",
            float(np.percentile(gaps, 95)),
            extra={
                "n_chunks": len(timings),
                "max_gap_ms": round(float(gaps.max()), 2),
                "mean_gap_ms": round(float(gaps.mean()), 2),
            },
        )
        out["stream_starvation_ms"] = _starvation(result, audio.sample_rate)
        return out


def _starvation(result: SynthesisResult, sample_rate: int) -> MetricValue:
    """Worst-case playout-buffer deficit, in ms.

    Model: playback starts when the first chunk arrives and then runs at real
    time. After each chunk we know how much audio has been *delivered* and how
    much would have been *consumed* by then. A positive deficit means the buffer
    was empty and the caller heard a gap.

        deficit_i = (arrival_i - arrival_0) - delivered_before_i

    Reported as the maximum deficit; 0 means delivery always stayed ahead.
    """
    timings = result.chunk_timings
    if not timings or sample_rate <= 0:
        return missing_value("stream_starvation_ms", "no chunk timings")

    start = timings[0].offset_ms
    delivered_ms = 0.0
    worst = 0.0
    worst_at = 0.0
    for chunk in timings:
        elapsed = chunk.offset_ms - start
        deficit = elapsed - delivered_ms
        if deficit > worst:
            worst = deficit
            worst_at = elapsed
        delivered_ms += (chunk.n_samples / sample_rate) * 1000.0

    return make_value(
        "stream_starvation_ms",
        max(0.0, worst),
        extra={
            "worst_at_ms": round(worst_at, 2),
            "audio_delivered_ms": round(delivered_ms, 2),
            "wall_clock_ms": round(timings[-1].offset_ms - start, 2),
        },
    )


@register_backend
class ThroughputBackend(RunBackend):
    """Run-level throughput at the configured concurrency.

    Separate from per-utterance latency because it answers a different question:
    not "how fast is one reply" but "how many concurrent calls can one instance
    carry". Both are needed — a model can be fast per utterance and still saturate
    a GPU at four concurrent streams.
    """

    name = "throughput"
    provides = ("throughput_utt_per_min",)

    def compute(
        self, records: Sequence[UtteranceRecord], ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        wall_ms = ctx.options.get("wall_clock_ms")
        ok = [r for r in records if r.result.ok]
        if not wall_ms or not ok:
            return {
                "throughput_utt_per_min": missing_value(
                    "throughput_utt_per_min",
                    "no wall-clock measured for the run" if not wall_ms else "no successful utterances",
                )
            }
        minutes = float(wall_ms) / 60000.0
        total_audio_s = sum(
            (r.result.audio.duration_s if r.result.audio is not None else 0.0) for r in ok
        )
        return {
            "throughput_utt_per_min": make_value(
                "throughput_utt_per_min",
                len(ok) / minutes if minutes > 0 else None,
                extra={
                    "concurrency": ctx.options.get("concurrency"),
                    "wall_clock_s": round(float(wall_ms) / 1000.0, 2),
                    "n_ok": len(ok),
                    # Aggregate RTF across the whole run: total audio produced per
                    # second of wall-clock. Above 1 the instance sustains more than
                    # real time and can serve concurrent calls.
                    "audio_seconds_per_wall_second": (
                        round(total_audio_s / (float(wall_ms) / 1000.0), 3) if wall_ms else None
                    ),
                },
            )
        }


__all__ = ["LatencyBackend", "ThroughputBackend"]
