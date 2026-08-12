"""Provider-agnostic synthesis: deterministic mock, honest failures, replay."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from tts_eval.adapters import available_adapters, build_adapter
from tts_eval.audio import AudioBuffer, write_wav
from tts_eval.errors import ConfigError
from tts_eval.types import SynthesisRequest


class TestAdapters:
    def test_all_builtin_adapters_registered(self):
        assert set(available_adapters()) == {"http_rest", "mock", "replay", "websocket_pcm"}

    def test_mock_is_bit_deterministic(self):
        adapter = build_adapter("mock", {"ttfb_ms": 0})
        req = SynthesisRequest(utterance_id="u1", text="नमस्ते", language="hi", voice="v", seed=7)
        a = asyncio.run(adapter.synthesize(req))
        b = asyncio.run(adapter.synthesize(req))
        assert np.array_equal(a.audio.samples, b.audio.samples)

    def test_different_seed_changes_audio(self):
        adapter = build_adapter("mock", {"ttfb_ms": 0})
        base = dict(utterance_id="u1", text="नमस्ते", language="hi", voice="v")
        a = asyncio.run(adapter.synthesize(SynthesisRequest(**base, seed=1)))
        b = asyncio.run(adapter.synthesize(SynthesisRequest(**base, seed=2)))
        assert not np.array_equal(a.audio.samples, b.audio.samples)

    def test_timings_are_populated_and_ordered(self):
        adapter = build_adapter("mock", {"ttfb_ms": 5})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u", text="hello there", language="en"))
        )
        assert result.ttfb_ms is not None and result.first_audible_ms is not None
        assert result.first_audible_ms >= result.ttfb_ms
        offsets = [c.offset_ms for c in result.chunk_timings]
        assert offsets == sorted(offsets)

    def test_first_audible_is_later_than_ttfb_when_padded(self):
        """The mock pads 30 ms of silence; TTFB alone would flatter it."""
        adapter = build_adapter("mock", {"ttfb_ms": 0})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u", text="hello", language="en"))
        )
        assert result.first_audible_ms > result.ttfb_ms

    def test_injected_failure_is_recorded_not_raised(self):
        adapter = build_adapter("mock", {"ttfb_ms": 0, "fail_rate": 1.0})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u", text="hi", language="en"))
        )
        assert not result.ok
        assert "injected synthesis failure" in result.error

    def test_unsupported_language_fails_that_utterance(self):
        adapter = build_adapter("mock", {"ttfb_ms": 0, "unsupported_languages": ["sat"]})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u", text="hi", language="sat"))
        )
        assert not result.ok and "not supported" in result.error

    def test_faults_are_disjoint(self):
        """Overlapping bands once made every utterance truncated AND clipped."""
        adapter = build_adapter("mock", {"ttfb_ms": 0, "truncate_rate": 1.0, "clip_rate": 0.0})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u", text="hello world", language="en"))
        )
        assert result.provider_meta.get("injected_fault") == "truncate"

    def test_silent_output_counts_as_failure_not_success(self):
        """A clean `done` with no audible samples is a failed request."""
        adapter = build_adapter("mock", {"ttfb_ms": 0, "silent_rate": 1.0})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u", text="hi", language="en"))
        )
        assert result.provider_meta.get("all_silent") is True

    def test_replay_reads_back_written_audio(self, tmp_path):
        sr = 24000
        samples = (0.3 * np.sin(2 * np.pi * 180 * np.arange(sr) / sr)).astype(np.float32)
        write_wav(tmp_path / "u1.wav", AudioBuffer(samples=samples, sample_rate=sr))
        adapter = build_adapter("replay", {"audio_dir": str(tmp_path)})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u1", text="x", language="en"))
        )
        assert result.ok
        assert result.audio.n_samples == sr
        # No sidecar timings present, so latency must be absent, not invented.
        assert result.ttfb_ms is None

    def test_replay_missing_file_fails_cleanly(self, tmp_path):
        adapter = build_adapter("replay", {"audio_dir": str(tmp_path)})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="nope", text="x", language="en"))
        )
        assert not result.ok and "no replay audio" in result.error

    def test_adapter_config_redacts_secrets(self):
        adapter = build_adapter("mock", {"api_key": "super-secret", "ttfb_ms": 0})
        assert adapter.describe()["config"]["api_key"] == "***redacted***"

    def test_http_rest_requires_url(self):
        with pytest.raises(ConfigError, match="requires 'url'"):
            build_adapter("http_rest", {})
