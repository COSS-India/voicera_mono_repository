"""Audio I/O and DSP primitives, numpy-only.

Deliberately avoids scipy/librosa/soundfile. Those are the usual choices, but
they would make the *core* install heavy, and the core has to run in CI and on
laptops (see the layered-metrics contract in ``tts_eval/__init__.py``). Stdlib
``wave`` plus numpy covers everything the core metrics need; heavy plugins bring
their own loaders.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .types import AudioBuffer

# Amplitude below which a sample is treated as silence. -60 dBFS is the usual
# broadcast noise-floor convention and comfortably below any real speech, while
# still above the numerical dither that neural vocoders emit in "silent" regions.
SILENCE_FLOOR = 10 ** (-60.0 / 20.0)  # ~0.001


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def write_wav(path: str | Path, audio: AudioBuffer, *, bit_depth: int = 16) -> Path:
    """Write mono PCM WAV. Returns the path written.

    Clipping is applied explicitly rather than left to integer wraparound: a
    wrapped sample turns a loud passage into a loud *click*, which would then be
    scored as an audio-quality artefact caused by our own writer.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if bit_depth == 16:
        sampwidth, dtype, peak = 2, np.int16, 32767.0
    elif bit_depth == 32:
        sampwidth, dtype, peak = 4, np.int32, 2147483647.0
    else:
        raise ValueError(f"unsupported bit_depth {bit_depth}, expected 16 or 32")

    pcm = (np.clip(audio.samples, -1.0, 1.0) * peak).astype(dtype)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sampwidth)
        wf.setframerate(audio.sample_rate)
        wf.writeframes(pcm.tobytes())
    return p


def read_wav(path: str | Path) -> AudioBuffer:
    """Read a WAV into a mono float32 AudioBuffer.

    Multi-channel input is averaged to mono because every metric here is
    defined on a single channel and TTS output is mono in practice; averaging is
    the least surprising reduction.
    """
    p = Path(path)
    with wave.open(str(p), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        # 8-bit WAV is unsigned by spec.
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 3:
        # 24-bit: no numpy dtype, so widen each 3-byte little-endian frame.
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
        ints = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)).astype(np.int32)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported sample width {sampwidth} bytes in {p}")

    if n_channels > 1:
        usable = (data.size // n_channels) * n_channels
        data = data[:usable].reshape(-1, n_channels).mean(axis=1)

    return AudioBuffer(samples=np.ascontiguousarray(data, dtype=np.float32), sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------
def resample(audio: AudioBuffer, target_rate: int) -> AudioBuffer:
    """Linear-interpolation resample.

    Linear interpolation is not transparent — it attenuates highs and aliases on
    downsampling — so it is used ONLY to feed models that demand a fixed input
    rate (ASR at 16 kHz, MOS predictors at 16 kHz). It is never used on audio
    whose quality is being measured; ``audio_quality`` metrics always run at the
    provider's native rate. Backends that ship a proper resampler (torchaudio)
    use their own instead.
    """
    if target_rate == audio.sample_rate or audio.samples.size == 0:
        return AudioBuffer(samples=audio.samples, sample_rate=target_rate)

    duration = audio.samples.size / audio.sample_rate
    n_out = max(1, int(round(duration * target_rate)))
    src_idx = np.linspace(0.0, audio.samples.size - 1, n_out, dtype=np.float64)
    out = np.interp(src_idx, np.arange(audio.samples.size, dtype=np.float64), audio.samples)
    return AudioBuffer(samples=out.astype(np.float32), sample_rate=target_rate)


# ---------------------------------------------------------------------------
# framing / energy
# ---------------------------------------------------------------------------
def frame_signal(samples: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Split into overlapping frames -> shape (n_frames, frame_len).

    Returns an empty (0, frame_len) array when the signal is shorter than one
    frame so every caller can rely on a 2-D result and skip length guards.
    """
    if samples.size < frame_len or frame_len <= 0 or hop_len <= 0:
        return np.zeros((0, max(frame_len, 1)), dtype=np.float32)
    n_frames = 1 + (samples.size - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    return samples[idx]


def frame_rms(samples: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    frames = frame_signal(samples, frame_len, hop_len)
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)).astype(np.float32)


def dbfs(x: float | np.ndarray) -> np.ndarray | float:
    """Amplitude -> dBFS, with a floor so silence yields -inf-free numbers."""
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))


def first_audible_index(samples: np.ndarray, floor: float = SILENCE_FLOOR) -> int | None:
    """Index of the first sample above the silence floor, or None if all silent.

    Used to separate "server responded" from "speech started", which is the
    distinction that matters for perceived latency in a live conversation.
    """
    above = np.flatnonzero(np.abs(samples) > floor)
    return int(above[0]) if above.size else None


def trim_silence(samples: np.ndarray, floor: float = SILENCE_FLOOR) -> tuple[np.ndarray, int, int]:
    """Return (trimmed, n_leading_silent, n_trailing_silent)."""
    above = np.flatnonzero(np.abs(samples) > floor)
    if above.size == 0:
        return samples[:0], samples.size, 0
    start, end = int(above[0]), int(above[-1]) + 1
    return samples[start:end], start, samples.size - end


# ---------------------------------------------------------------------------
# spectral helpers (numpy FFT only)
# ---------------------------------------------------------------------------
def magnitude_spectrogram(
    samples: np.ndarray, sample_rate: int, *, frame_ms: float = 25.0, hop_ms: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed magnitude spectrogram -> (spec[n_frames, n_bins], freqs)."""
    frame_len = max(16, int(sample_rate * frame_ms / 1000.0))
    hop_len = max(1, int(sample_rate * hop_ms / 1000.0))
    frames = frame_signal(samples, frame_len, hop_len)
    if frames.shape[0] == 0:
        return np.zeros((0, frame_len // 2 + 1), dtype=np.float32), np.zeros(frame_len // 2 + 1)
    window = np.hanning(frame_len).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate)
    return spec, freqs


def spectral_centroid(spec: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Per-frame spectral centroid in Hz. A cheap, robust timbre proxy — used to
    detect voice drift across utterances without loading a speaker model."""
    if spec.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    total = spec.sum(axis=1)
    safe = np.where(total > 0, total, 1.0)
    return ((spec * freqs[None, :]).sum(axis=1) / safe).astype(np.float32)


def spectral_flatness(spec: np.ndarray) -> np.ndarray:
    """Per-frame spectral flatness (geometric/arithmetic mean ratio).

    Near 1 means noise-like, near 0 means tonal. High flatness over voiced
    regions is a strong signal of vocoder buzz or a codec collapse, which is a
    real Indic-Mio failure mode when token generation degenerates.
    """
    if spec.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    p = spec.astype(np.float64) + 1e-10
    geo = np.exp(np.mean(np.log(p), axis=1))
    arith = np.mean(p, axis=1)
    return (geo / np.maximum(arith, 1e-12)).astype(np.float32)


def estimate_f0(
    samples: np.ndarray,
    sample_rate: int,
    *,
    fmin: float = 60.0,
    fmax: float = 400.0,
    frame_ms: float = 40.0,
    hop_ms: float = 20.0,
) -> np.ndarray:
    """Frame-wise F0 via normalised autocorrelation; NaN for unvoiced frames.

    A full pitch tracker (pYIN/CREPE) would be better but needs librosa/torch.
    Autocorrelation is adequate here because F0 is used for *relative* stability
    across utterances of the same voice, not for absolute pitch reporting — and
    octave errors that recur consistently do not inflate the variance we measure.
    """
    frame_len = max(64, int(sample_rate * frame_ms / 1000.0))
    hop_len = max(1, int(sample_rate * hop_ms / 1000.0))
    frames = frame_signal(samples, frame_len, hop_len)
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(frame_len - 1, int(sample_rate / fmin))
    if max_lag <= min_lag:
        return np.full(frames.shape[0], np.nan, dtype=np.float32)

    out = np.full(frames.shape[0], np.nan, dtype=np.float32)
    for i, frame in enumerate(frames.astype(np.float64)):
        frame = frame - frame.mean()
        energy = float(np.dot(frame, frame))
        if energy <= 1e-9:
            continue  # silent frame -> unvoiced
        # Full autocorrelation via FFT, then search the plausible lag band.
        n_fft = 1 << int(np.ceil(np.log2(2 * frame_len)))
        spec = np.fft.rfft(frame, n=n_fft)
        acf = np.fft.irfft(spec * np.conj(spec), n=n_fft)[:frame_len]
        band = acf[min_lag : max_lag + 1]
        if band.size == 0:
            continue
        lag = int(np.argmax(band)) + min_lag
        # Normalised peak height doubles as a voicing decision. 0.3 is a
        # deliberately permissive threshold: missing voiced frames biases the
        # stability estimate more than admitting a few weak ones.
        if acf[0] <= 0 or band.max() / acf[0] < 0.3:
            continue
        out[i] = sample_rate / float(lag)
    return out
