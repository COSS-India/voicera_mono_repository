"""
Audio decoding and resampling for the OpenAI-compatible API.

Agent stacks do not speak our native 16 kHz format. OpenAI's Realtime `pcm16` is
**24 kHz**, and telephony bridges send G.711 at 8 kHz. Feeding any of those to a model
expecting 16 kHz produces fluent-looking nonsense rather than an error -- the exact
failure this project already spent a round diagnosing -- so conversion happens here,
once, at the edge.
"""
import io
import subprocess

import numpy as np
from scipy.signal import resample_poly

TARGET_RATE = 16000

# G.711 is a byte-per-sample companding law; decode via a 256-entry table rather than
# the deprecated `audioop`, which is gone in Python 3.13.
def _ulaw_table() -> np.ndarray:
    code = np.arange(256, dtype=np.int32) ^ 0xFF
    sign = code & 0x80
    exponent = (code >> 4) & 0x07
    mantissa = code & 0x0F
    value = ((mantissa << 3) + 0x84) << exponent
    value -= 0x84
    return np.where(sign, -value, value).astype(np.int16)


def _alaw_table() -> np.ndarray:
    code = np.arange(256, dtype=np.int32) ^ 0x55
    sign = code & 0x80
    exponent = (code >> 4) & 0x07
    mantissa = code & 0x0F
    # np.where evaluates both branches, so the shift must stay non-negative even for
    # the exponent==0 rows that the other branch supplies.
    shift = np.maximum(exponent - 1, 0)
    value = np.where(exponent == 0, (mantissa << 4) + 8, ((mantissa << 4) + 0x108) << shift)
    # A-law inverts mu-law's sign convention: after the 0x55 xor, a SET bit 7 is positive.
    return np.where(sign, value, -value).astype(np.int16)


ULAW = _ulaw_table()
ALAW = _alaw_table()


def resample_to_16k(x: np.ndarray, rate: int) -> np.ndarray:
    if rate == TARGET_RATE:
        return x.astype(np.float32, copy=False)
    from math import gcd
    g = gcd(int(rate), TARGET_RATE)
    return resample_poly(x.astype(np.float32), TARGET_RATE // g, int(rate) // g).astype(np.float32)


def pcm_bytes_to_16k(raw: bytes, fmt: str, rate: int) -> bytes:
    """
    Convert one chunk of a realtime stream into 16 kHz mono int16 PCM.

    `fmt` follows OpenAI's naming: pcm16 (default 24 kHz), g711_ulaw, g711_alaw (8 kHz).
    """
    if fmt == "g711_ulaw":
        samples = ULAW[np.frombuffer(raw, dtype=np.uint8)]
        rate = 8000
    elif fmt == "g711_alaw":
        samples = ALAW[np.frombuffer(raw, dtype=np.uint8)]
        rate = 8000
    else:
        if len(raw) % 2:
            raw = raw[:-1]
        samples = np.frombuffer(raw, dtype=np.int16)

    if rate == TARGET_RATE:
        return samples.tobytes()
    resampled = resample_to_16k(samples.astype(np.float32) / 32768.0, rate)
    return (np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def decode_file_to_16k(data: bytes, filename: str = "") -> bytes:
    """
    Decode an uploaded audio file to 16 kHz mono int16 PCM.

    soundfile covers wav/flac/ogg; anything else (mp3, m4a, webm, mp4) goes through
    ffmpeg, which the image already installs.
    """
    try:
        import soundfile as sf
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        resampled = resample_to_16k(mono, rate)
        return (np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    except Exception:
        pass

    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(TARGET_RATE), "pipe:1"],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError(
            f"could not decode audio{' ' + filename if filename else ''}: "
            f"{proc.stderr.decode('utf-8', 'replace')[:200]}"
        )
    return proc.stdout
