"""The multipart route must decode to exactly the samples the old base64 path did.

Extracted from the real stt/server.py rather than reimplemented, so the test
fails if the server changes.
"""
import ast
import base64
import io
import wave
from pathlib import Path

import numpy as np
import pytest

SERVER = Path(__file__).resolve().parent.parent / "stt" / "server.py"


def _load_decoders():
    ns = {"np": np, "base64": base64}
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    wanted = {"_decode_audio_b64", "_pcm_from_upload"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), "<server>", "exec"), ns)  # noqa: S102
    missing = wanted - ns.keys()
    assert not missing, f"stt/server.py no longer defines {missing}"
    return ns["_decode_audio_b64"], ns["_pcm_from_upload"]


OLD, NEW = _load_decoders()
RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    ("label", "pcm"),
    [
        ("silence", np.zeros(8000, dtype=np.int16)),
        ("speech_like", RNG.normal(0, 6000, 8000).astype(np.int16)),
        ("full_scale", np.array([-32768, 32767] * 4000, dtype=np.int16)),
        ("odd_length", RNG.integers(-3000, 3000, 4001).astype(np.int16)),
    ],
)
def test_raw_pcm_matches_base64_path(label, pcm):
    raw = pcm.tobytes()
    assert np.array_equal(OLD(base64.b64encode(raw).decode()), NEW(raw)), label


def test_wav_upload_matches_raw_pcm():
    pcm = RNG.normal(0, 5000, 8000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    assert np.array_equal(OLD(base64.b64encode(pcm.tobytes()).decode()), NEW(buf.getvalue()))
