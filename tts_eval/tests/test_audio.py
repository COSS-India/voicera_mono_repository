"""WAV I/O and DSP primitives the whole metric layer stands on."""
from __future__ import annotations

import numpy as np
import pytest

from tts_eval.audio import AudioBuffer, estimate_f0, read_wav, resample, write_wav


class TestAudio:
    def test_wav_roundtrip_preserves_signal(self, tmp_path):
        sr = 24000
        samples = (0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
        path = write_wav(tmp_path / "a.wav", AudioBuffer(samples=samples, sample_rate=sr))
        back = read_wav(path)
        assert back.sample_rate == sr
        assert np.allclose(back.samples, samples, atol=1e-4)

    def test_resample_preserves_duration(self):
        buf = AudioBuffer(samples=np.zeros(24000, dtype=np.float32), sample_rate=24000)
        assert resample(buf, 16000).duration_s == pytest.approx(1.0, abs=1e-3)

    def test_f0_tracks_a_known_tone(self):
        sr = 16000
        t = np.arange(sr) / sr
        f0 = estimate_f0((0.6 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), sr)
        voiced = f0[np.isfinite(f0)]
        assert voiced.size > 5
        assert np.median(voiced) == pytest.approx(150, rel=0.05)

    def test_f0_reports_silence_as_unvoiced(self):
        f0 = estimate_f0(np.zeros(16000, dtype=np.float32), 16000)
        assert not np.isfinite(f0).any()

    def test_write_wav_clips_instead_of_wrapping(self, tmp_path):
        """Integer wraparound would turn a loud passage into a click we then score."""
        loud = np.array([3.0, -3.0, 0.0], dtype=np.float32)
        path = write_wav(tmp_path / "loud.wav", AudioBuffer(samples=loud, sample_rate=8000))
        back = read_wav(path).samples
        assert back.max() <= 1.0 and back.min() >= -1.0
