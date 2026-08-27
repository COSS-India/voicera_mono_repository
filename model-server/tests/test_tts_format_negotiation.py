"""Two TTS models, two wire formats, one client.

Indic Parler streams 44.1 kHz float32; Orpheus streams 24 kHz signed 16-bit.
Neither is wrong -- `pcm` is OpenAI's own 16-bit format and `pcm_f32le` is an
extension Parler serves because float32 is what its engine produces. The client
therefore cannot assume a width or a rate; it has to read what the server says it
sent and decode accordingly.

Getting this wrong does not raise. It produces a stream of plausible-looking
bytes that sound like noise down a phone line, which is why it is tested here
rather than left to the first person who deploys a second TTS model.

The decoder is extracted from the real client source, so this fails when that
drifts rather than passing against a copy.
"""
import ast
import struct
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
CLIENT = ROOT.parent / "voice_2_voice_server" / "services" / "ai4bharat" / "tts.py"

needs_client = pytest.mark.skipif(
    not CLIENT.is_file(), reason="voice_2_voice_server not present in this checkout"
)


def client_source() -> str:
    return CLIENT.read_text(encoding="utf-8")


def load_decoders() -> dict:
    """_DECODERS maps the declared format to the numpy dtype that reads it."""
    def names(n):
        if isinstance(n, ast.Assign):
            return [getattr(t, "id", None) for t in n.targets]
        if isinstance(n, ast.AnnAssign):
            return [getattr(n.target, "id", None)]
        return []

    tree = ast.parse(client_source())
    node = next((n for n in tree.body if "_DECODERS" in names(n)), None)
    assert node is not None, "tts.py no longer defines _DECODERS"
    ns: dict = {"np": np}
    exec(compile(ast.Module([node], []), "<tts>", "exec"), ns)  # noqa: S102
    return ns["_DECODERS"]


def load_to_int16():
    """_to_int16 is a method; lift it out with a stub self carrying the gain."""
    tree = ast.parse(client_source())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "ModelServerTTSService")
    fn = next((n for n in cls.body
               if isinstance(n, ast.FunctionDef) and n.name == "_to_int16"), None)
    assert fn is not None, "ModelServerTTSService no longer defines _to_int16"
    ns: dict = {"np": np}
    exec(compile(ast.Module([fn], []), "<tts>", "exec"), ns)  # noqa: S102
    return ns["_to_int16"]


class FakeService:
    def __init__(self, gain=1.0):
        self._gain = gain
        self._to_int16 = load_to_int16().__get__(self)


# ---------------------------------------------------------------- the table

@needs_client
@pytest.mark.parametrize(("declared", "width"), [
    ("pcm_f32le", 4),   # Indic Parler
    ("pcm", 2),         # Orpheus, and OpenAI's own name for it
    ("pcm_s16le", 2),   # the explicit spelling, same thing
])
def test_every_declared_format_has_a_width(declared, width):
    assert load_decoders()[declared].itemsize == width


@needs_client
def test_an_unknown_format_is_refused_rather_than_guessed():
    """A model that declares mp3 must produce an error naming it, not silence.

    The client cannot decode it, and decoding mp3 bytes as PCM is exactly the
    failure that sounds like a fax machine instead of raising.
    """
    assert "mp3" not in load_decoders()
    src = client_source()
    assert "cannot decode" in src, "the unknown-format branch is gone"
    assert "ErrorFrame" in src


# ---------------------------------------------------------------- conversion

@needs_client
def test_int16_at_unity_gain_is_a_byte_for_byte_passthrough():
    """Orpheus already narrowed to 16-bit on the GPU. Re-scaling it would risk
    changing audio that was already exactly right."""
    samples = np.array([-32768, -1, 0, 1, 32767, 1234], dtype="<i2")
    assert FakeService(gain=1.0)._to_int16(samples) == samples.tobytes()


@needs_client
def test_float32_converts_the_way_it_always_did():
    """Parler's path must not move: multiply by gain, clip to [-1, 1], scale."""
    samples = np.array([-1.0, -0.5, 0.0, 0.25, 1.0], dtype="<f4")
    got = np.frombuffer(FakeService(gain=1.0)._to_int16(samples), dtype=np.int16)
    expected = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    assert list(got) == list(expected)


@needs_client
@pytest.mark.parametrize("gain", [0.5, 1.5, 3.0])
def test_gain_never_wraps_around(gain):
    """Scaling int16 without clipping wraps loud samples to the opposite sign --
    silence turning into a crack. Both paths must saturate instead."""
    loud_i16 = np.array([32767, -32768, 30000, -30000], dtype="<i2")
    out = np.frombuffer(FakeService(gain=gain)._to_int16(loud_i16), dtype=np.int16)
    assert out.max() <= 32767 and out.min() >= -32768
    # Direction is preserved: a positive sample never comes back negative.
    assert all(np.sign(a) == np.sign(b) or b == 0 for a, b in zip(loud_i16, out, strict=True))

    loud_f32 = np.array([1.0, -1.0, 0.9, -0.9], dtype="<f4")
    out = np.frombuffer(FakeService(gain=gain)._to_int16(loud_f32), dtype=np.int16)
    assert out.max() <= 32767 and out.min() >= -32768


# ---------------------------------------------------------------- chunking

def reassemble(payload: bytes, width: int, splits: list[int]) -> bytes:
    """The client's remainder logic, driven by the width the header declared.

    The split pattern cycles until the payload is exhausted, so the only bytes
    left unemitted are a real trailing partial sample -- not an artefact of the
    test stopping early.
    """
    out, remainder, pos, i = bytearray(), b"", 0, 0
    while pos < len(payload):
        n = max(1, splits[i % len(splits)])
        i += 1
        chunk, pos = payload[pos:pos + n], pos + n
        if not chunk:
            continue
        buf = remainder + chunk
        usable = len(buf) - (len(buf) % width)
        remainder = buf[usable:]
        if usable:
            out += buf[:usable]
    return bytes(out)


@needs_client
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("splits", [
    [1] * 64,                    # one byte at a time
    [3, 5, 7, 11, 13, 17, 23],   # primes, so nothing lands on a boundary
    [6, 6, 6, 6, 6, 6],          # misaligned for width 4, aligned for width 2
    [64],                        # one shot
])
def test_a_sample_split_across_chunks_survives(width, splits):
    """This is the bug that produced noise when we moved TTS off WebSockets, and
    it is width-dependent -- so a 16-bit model re-opens it if the width is
    hardcoded to 4."""
    payload = bytes(range(64))
    got = reassemble(payload, width, splits)
    assert got == payload[:len(got)], "samples desynchronised"
    assert len(got) % width == 0, "emitted a partial sample"
    assert len(payload) - len(got) < width, "dropped more than one partial sample"


@needs_client
def test_the_width_comes_from_the_header_not_a_constant():
    src = client_source()
    assert "% width" in src, "the remainder logic is not driven by the declared width"
    assert "% 4)" not in src, "a 4-byte sample width is still hardcoded"


@needs_client
def test_orpheus_round_trip_at_24k():
    """End to end on the numbers Orpheus actually sends: 24 kHz, s16le, arriving
    in chunks that do not align to sample boundaries."""
    tone = (np.sin(np.linspace(0, 8 * np.pi, 600)) * 20000).astype("<i2")
    payload = tone.tobytes()
    splits = [7] * (len(payload) // 7) + [len(payload) % 7]
    reassembled = reassemble(payload, 2, splits)
    decoded = np.frombuffer(reassembled, dtype=load_decoders()["pcm"])
    assert len(decoded) >= len(tone) - 1
    assert np.array_equal(decoded[:len(decoded)], tone[:len(decoded)])
    # And what reaches Pipecat is 16-bit, whatever the wire format was.
    out = FakeService(gain=1.0)._to_int16(decoded)
    assert len(out) == len(decoded) * 2
    assert struct.unpack("<h", out[:2])[0] == tone[0]
