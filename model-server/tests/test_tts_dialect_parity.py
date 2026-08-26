"""Both request dialects must reach the Parler runner identically.

The legacy shape is what the voice server sent before the revamp; speech.create
is the new contract. If these ever diverge, synthesis changes.
"""
import asyncio
import importlib.util
import json
import queue
from pathlib import Path

import numpy as np
import pytest

SERVER = Path(__file__).resolve().parent.parent / "tts" / "server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("parler_server", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


server = _load_server()


class _FakeWS:
    def __init__(self, first_message):
        self._first = first_message
        self.sent = []

    async def recv(self):
        return self._first

    async def send(self, msg):
        self.sent.append(msg)


async def _drive(request_json):
    ws = _FakeWS(request_json)
    prefill_q = queue.Queue()
    task = asyncio.create_task(server.handle_client(ws, None, prefill_q))
    for _ in range(100):
        await asyncio.sleep(0.01)
        if not prefill_q.empty():
            break
    req, out_q = prefill_q.get_nowait()
    out_q.put(("audio", np.zeros(128, dtype=np.float32)))
    out_q.put(("done", None))
    await asyncio.wait_for(task, timeout=5)
    return req, [json.loads(m) for m in ws.sent if isinstance(m, str)]


LEGACY = json.dumps(
    {"prompt": "नमस्ते", "description": "Divya. A clear, natural voice.", "language": "hi"}
)
V1 = json.dumps(
    {
        "type": "speech.create",
        "id": "abc123",
        "input": "नमस्ते",
        "voice": {"preset": "Divya", "description": "A clear, natural voice."},
        "language": "hi",
    }
)


@pytest.mark.asyncio
async def test_both_dialects_reach_the_runner_identically():
    legacy_req, _ = await _drive(LEGACY)
    v1_req, _ = await _drive(V1)
    assert legacy_req.prompt == v1_req.prompt
    assert legacy_req.description == v1_req.description == "Divya. A clear, natural voice."


@pytest.mark.asyncio
async def test_frame_names_follow_the_dialect_used():
    _, legacy_frames = await _drive(LEGACY)
    _, v1_frames = await _drive(V1)
    assert [f["type"] for f in legacy_frames] == ["meta", "done"]
    assert [f["type"] for f in v1_frames] == ["speech.meta", "speech.done"]
    assert v1_frames[0]["id"] == "abc123"
    # Clients must read the rate off the frame, never assume it.
    assert v1_frames[0]["sample_rate"] == server.AUDIO_SAMPLE_RATE
