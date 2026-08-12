"""Deterministic offline adapter — no server, no GPU, no network.

This is not a toy. It is load-bearing for three things the acceptance criteria
demand and that a real model cannot give you:

1.  **Reproducibility testing.** Its output is a pure function of
    ``(text, voice, language, seed)``, so the framework's own reproducibility
    guarantee (identical inputs -> identical fingerprint -> identical metrics)
    can be asserted in CI.
2.  **Metric verification.** Failure modes can be injected on demand —
    degenerate babble, truncation, clipping, silence, dropouts — so we can prove
    the audio-quality and coverage metrics actually fire instead of hoping they
    would.
3.  **Onboarding.** ``tts-eval run --model mock`` produces a complete benchmark
    report on a laptop, so the workflow can be reviewed before any GPU is booked.

The synthesised signal is speech-*shaped* (harmonic stack at a voice-dependent
F0, syllable-rate amplitude envelope, inter-word pauses) rather than a sine tone,
so DSP metrics land in realistic ranges. It is deliberately NOT intelligible:
round-trip ASR on mock audio is expected to score badly, and the suite reports
that honestly rather than pretending.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Mapping

import numpy as np

from ..errors import SynthesisFailed
from ..types import Capabilities, Determinism, SynthesisRequest
from .base import TTSAdapter, _Capture, register_adapter

# Rough syllables-per-second, used to size synthetic duration. Indic scripts pack
# more phonemes per character than Latin, so a single chars/sec constant would
# make Hindi audio implausibly long relative to English.
_CHARS_PER_SECOND = {"en": 15.0, "default": 11.0}


@register_adapter
class MockAdapter(TTSAdapter):
    name = "mock"
    requires = ()

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self._sample_rate = int(self.config.get("sample_rate") or 24000)
        self._ttfb_ms = float(self.config.get("ttfb_ms") or 40.0)
        self._chunk_ms = float(self.config.get("chunk_ms") or 200.0)
        # Multiplier on real time spent per second of audio produced. 0 keeps the
        # test suite fast; ~0.3 imitates a GPU model for demo reports.
        self._realtime_factor = float(self.config.get("realtime_factor") or 0.0)
        # Deterministic fault injection, keyed off the utterance id hash so the
        # *same* utterances fail on every run (a random failure set would make
        # runs incomparable, defeating the purpose).
        self._fail_rate = float(self.config.get("fail_rate") or 0.0)
        self._degenerate_rate = float(self.config.get("degenerate_rate") or 0.0)
        self._truncate_rate = float(self.config.get("truncate_rate") or 0.0)
        self._clip_rate = float(self.config.get("clip_rate") or 0.0)
        self._silent_rate = float(self.config.get("silent_rate") or 0.0)
        # Languages the mock "does not support", to exercise coverage gating.
        self._unsupported = set(self.config.get("unsupported_languages") or ())

    def _build_capabilities(self, config: Mapping[str, Any]) -> Capabilities:
        base = super()._build_capabilities(config)
        return Capabilities(
            streaming=True,
            voices=base.voices or ("mock_female", "mock_male"),
            languages=base.languages,
            # Read from the card rather than hardcoded, so a test can model a
            # provider that ignores seeds and check that the runner warns.
            supports_seed=base.supports_seed,
            supports_emotion=base.supports_emotion,
            native_sample_rate=int(config.get("sample_rate") or 24000),
            # The whole point: bit-identical output for identical input.
            determinism=Determinism.DETERMINISTIC,
        )

    # ------------------------------------------------------------------
    def _fault(self, h: float) -> str | None:
        """Which fault (if any) this utterance gets.

        Faults occupy *disjoint* bands of [0, 1) so exactly one can apply. An
        earlier cumulative-threshold version let a rate of 1.0 on one fault switch
        on every later fault too, producing "truncated AND clipped AND looping"
        for every utterance — which silently invalidated any test that asserted a
        single failure mode.
        """
        cursor = 0.0
        for name, rate in (
            ("fail", self._fail_rate),
            ("silent", self._silent_rate),
            ("truncate", self._truncate_rate),
            ("clip", self._clip_rate),
            ("degenerate", self._degenerate_rate),
        ):
            if rate <= 0:
                continue
            if cursor <= h < cursor + rate:
                return name
            cursor += rate
        return None

    async def _synthesise(self, request: SynthesisRequest, capture: _Capture) -> None:
        capture.sample_rate = self._sample_rate
        capture.meta(sample_rate=self._sample_rate, model="mock", channels=1)

        h = _hash_unit(f"{request.utterance_id}|{request.text}|{request.seed}")

        if request.language in self._unsupported:
            raise SynthesisFailed(f"mock: language {request.language!r} not supported by this model")

        fault = self._fault(h)
        if fault:
            capture.meta(injected_fault=fault)
        if fault == "fail":
            raise SynthesisFailed("mock: injected synthesis failure")

        if self._ttfb_ms > 0:
            await asyncio.sleep(self._ttfb_ms / 1000.0)

        if fault == "silent":
            # Emit real frames of pure silence: exercises the "server said done
            # but produced nothing audible" path, which a naive harness scores as
            # a success.
            capture.chunk(np.zeros(int(self._sample_rate * 1.0), dtype=np.float32))
            return

        self._last_fault_applied = True
        samples = self._render(request, degenerate=(fault == "degenerate"))
        if fault and not self._last_fault_applied:
            capture.meta(fault_applied=False)
        elif fault:
            capture.meta(fault_applied=True)

        if fault == "truncate":
            samples = samples[: max(1, int(samples.size * 0.45))]
        elif fault == "clip":
            samples = np.clip(samples * 6.0, -1.0, 1.0)

        chunk_len = max(1, int(self._sample_rate * self._chunk_ms / 1000.0))
        for start in range(0, samples.size, chunk_len):
            part = samples[start : start + chunk_len]
            if self._realtime_factor > 0:
                await asyncio.sleep((part.size / self._sample_rate) * self._realtime_factor)
            capture.chunk(part)

    # ------------------------------------------------------------------
    def _render(self, request: SynthesisRequest, *, degenerate: bool) -> np.ndarray:
        """Build a speech-shaped waveform deterministically from the request."""
        sr = self._sample_rate
        cps = _CHARS_PER_SECOND.get(request.language, _CHARS_PER_SECOND["default"])
        duration = max(0.35, len(request.text) / cps)

        # Voice identity drives F0 and timbre, so utterances sharing a voice are
        # genuinely consistent and utterances with different voices genuinely are
        # not — which is what voice-consistency metrics need to be testable.
        voice_key = request.voice or "default"
        vh = _hash_unit(f"voice:{voice_key}")
        f0 = 105.0 + 130.0 * vh  # ~105-235 Hz
        n_harmonics = 12

        t = np.arange(int(sr * duration), dtype=np.float64) / sr
        rng = np.random.default_rng(_hash_int(f"{request.utterance_id}|{request.seed}"))

        # Slow F0 contour (declination + intonation) keeps frames voiced but not
        # monotone, so F0-stability metrics see realistic within-voice variance.
        contour = 1.0 + 0.06 * np.sin(2 * np.pi * 0.35 * t) - 0.10 * (t / max(t[-1], 1e-6))
        phase = 2 * np.pi * f0 * np.cumsum(contour) / sr

        # Moving formants. Without these the spectrum is static, and a static
        # spectrum is not what speech looks like — it reads as a held tone to
        # every spectral metric, so the mock would trip the very degeneracy
        # detector it exists to test. Three resonances walk piecewise-linearly
        # between per-syllable targets: a crude but structurally correct model of
        # vowel transitions, and enough to make each syllable spectrally distinct.
        syllable_hz = 4.2
        n_syllables = max(2, int(duration * syllable_hz) + 1)
        formant_ranges = ((350.0, 850.0), (900.0, 2100.0), (2300.0, 3200.0))
        # Jittered breakpoints: real syllable timing is irregular, and perfectly
        # even spacing would make the envelope exactly periodic — itself a
        # repetition signal.
        jitter = rng.uniform(-0.35, 0.35, size=n_syllables) / syllable_hz
        breakpoints = np.clip(np.linspace(0.0, duration, n_syllables) + jitter, 0.0, duration)
        breakpoints = np.maximum.accumulate(breakpoints)  # np.interp needs monotonic x
        formants = [
            np.interp(t, breakpoints, rng.uniform(low, high, size=n_syllables))
            for low, high in formant_ranges
        ]

        signal = np.zeros_like(t)
        for k in range(1, n_harmonics + 1):
            harmonic_hz = k * f0
            # 1/k^1.2 rolloff approximates a glottal source spectrum...
            gain = 1.0 / k**1.2
            # ...then each harmonic is shaped by its distance to the moving
            # formants, so its amplitude varies over time and the spectrum evolves.
            shaping = np.zeros_like(t)
            for centre, bandwidth in zip(formants, (180.0, 260.0, 340.0)):
                shaping += np.exp(-0.5 * ((harmonic_hz - centre) / bandwidth) ** 2)
            signal += gain * (0.25 + shaping) * np.sin(k * phase + vh * k)

        # Syllable-rate envelope with genuine inter-word gaps, so silence-ratio
        # and speaking-rate metrics have something real to measure.
        env = 0.5 * (1.0 - np.cos(2 * np.pi * syllable_hz * t))
        gap_positions = rng.random(max(1, int(duration * 1.6)))
        for g in gap_positions:
            centre = g * duration
            env *= 1.0 - 0.95 * np.exp(-0.5 * ((t - centre) / 0.045) ** 2)

        # Jittered syllable envelope with genuine inter-word gaps, so silence-ratio
        # and speaking-rate metrics have something real to measure.
        env = 0.5 * (1.0 - np.cos(2 * np.pi * syllable_hz * (t + jitter[0])))
        gap_positions = rng.random(max(1, int(duration * 1.6)))
        for g in gap_positions:
            centre = g * duration
            env *= 1.0 - 0.95 * np.exp(-0.5 * ((t - centre) / 0.045) ** 2)

        # Light breath noise, kept low: enough that voiced frames are not
        # perfectly tonal, not enough to read as a noisy channel.
        noise = 0.004 * rng.standard_normal(t.size)
        out = 0.28 * signal / max(np.abs(signal).max(), 1e-9) * env + noise * env

        self._last_fault_applied = True
        if degenerate:
            # Simulates the characteristic autoregressive-TTS collapse: the model
            # locks onto a segment and emits it over and over. Built by tiling a
            # real slice of the utterance rather than by generating a synthetic
            # tone, because that is what the failure actually looks like — locally
            # plausible speech, globally repeating — and it is the case the
            # repetition detector must catch.
            #
            # A loop needs at least two full periods inside the detector's search
            # band to be detectable at all, so an utterance shorter than that cannot
            # carry this fault. Say so via `fault_applied` instead of silently
            # returning clean audio, which would make a test think the detector
            # missed something it was never given.
            loop_len = int(sr * 0.42)
            if out.size >= 2 * loop_len:
                start = (out.size - loop_len) // 2
                segment = out[start : start + loop_len]
                repeats = int(np.ceil(out.size / loop_len))
                out = np.tile(segment, repeats)[: out.size]
            else:
                self._last_fault_applied = False

        # Short lead-in/lead-out silence, as real servers produce.
        pad = np.zeros(int(sr * 0.03), dtype=np.float64)
        return np.concatenate([pad, out, pad]).astype(np.float32)


def _hash_int(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _hash_unit(key: str) -> float:
    """Stable pseudo-random float in [0, 1) from a string.

    Used instead of ``random`` so fault injection is identical across runs,
    processes and machines — a prerequisite for the reproducibility tests.
    """
    return _hash_int(key) / float(1 << 64)


__all__ = ["MockAdapter"]
