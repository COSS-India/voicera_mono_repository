"""Chunked HTTP can split a 4-byte float; WebSocket messages never did.

The TTS client carries a remainder between reads. If that logic is wrong the
samples shift by a byte or two and the audio becomes noise -- audible
immediately, but only in a real call, and only sometimes. This reproduces the
client's framing against adversarial chunk splits.
"""
import numpy as np
import pytest


def reassemble(chunks):
    """The exact remainder-carrying logic from IndicParlerRESTTTSService.run_tts."""
    out = []
    remainder = b""
    for chunk in chunks:
        if not chunk:
            continue
        buf = remainder + chunk
        usable = len(buf) - (len(buf) % 4)
        remainder = buf[usable:]
        if not usable:
            continue
        out.append(np.frombuffer(buf[:usable], dtype=np.float32))
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


RNG = np.random.default_rng(11)
SAMPLES = (RNG.random(4096, dtype=np.float32) - 0.5).astype(np.float32)
RAW = SAMPLES.tobytes()


def _split(data, sizes):
    out, i = [], 0
    while i < len(data):
        n = sizes[len(out) % len(sizes)]
        out.append(data[i:i + n])
        i += n
    return out


@pytest.mark.parametrize(
    ("label", "sizes"),
    [
        ("aligned", [4096]),
        ("one byte at a time", [1]),
        ("misaligned by one", [1, 4096]),
        ("misaligned by three", [3, 4093]),
        ("prime sized", [7, 13, 31]),
        ("single huge chunk", [len(RAW)]),
    ],
)
def test_samples_survive_any_chunk_boundary(label, sizes):
    got = reassemble(_split(RAW, sizes))
    assert np.array_equal(got, SAMPLES), f"{label}: audio corrupted by chunk framing"


def test_trailing_partial_sample_is_held_not_emitted():
    # A stream cut mid-float must drop the fragment, never emit a garbage sample.
    got = reassemble(_split(RAW[:-2], [500]))
    assert len(got) == len(SAMPLES) - 1
    assert np.array_equal(got, SAMPLES[:-1])


def test_empty_chunks_are_ignored():
    chunks = _split(RAW, [64])
    with_gaps = []
    for c in chunks:
        with_gaps.extend([b"", c])
    assert np.array_equal(reassemble(with_gaps), SAMPLES)
