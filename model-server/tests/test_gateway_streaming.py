"""The gateway must stream rather than buffer, and must let a client disconnect
reach the upstream -- that propagation is what makes barge-in free the TTS slot.
"""
import asyncio
import contextlib
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

CHUNKS, CHUNK_DELAY = 5, 0.20          # a 1.0s response, first byte immediately
STATE = {"cancelled": 0, "completed": 0}

upstream = FastAPI()


@upstream.get("/health")
def _health():
    return {"status": "ok"}


@upstream.post("/v1/audio/transcriptions")
async def _stt():
    async def gen():
        for i in range(CHUNKS):
            yield f"chunk{i}\n".encode()
            await asyncio.sleep(CHUNK_DELAY)

    return StreamingResponse(gen(), media_type="text/plain")


@upstream.websocket("/v1/audio/speech")
async def _speech(ws: WebSocket):
    await ws.accept()
    try:
        raw = await ws.receive_text()
    except WebSocketDisconnect:
        return                                   # health probe: connect and close
    req = json.loads(raw)
    await ws.send_text(json.dumps({"type": "speech.meta", "id": req["id"],
                                   "sample_rate": 44100, "format": "pcm_f32le", "channels": 1}))
    try:
        for _ in range(20):
            await ws.send_bytes(b"\x00\x01" * 160)
            await asyncio.sleep(0.05)
        await ws.send_text(json.dumps({"type": "speech.done", "id": req["id"]}))
        STATE["completed"] += 1
    except (WebSocketDisconnect, RuntimeError):
        STATE["cancelled"] += 1                  # the eviction path barge-in relies on


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app, port):
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        time.sleep(0.05)
        with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), 0.1):
            return server
    raise RuntimeError(f"server on {port} never came up")


@pytest.fixture(scope="module")
def gateway_url():
    up_port, gw_port = _free_port(), _free_port()
    _serve(upstream, up_port)

    import os
    os.environ.update(
        STT_MODEL="test-stt", TTS_MODEL="test-tts", LLM_MODEL="",
        STT_UPSTREAM=f"http://127.0.0.1:{up_port}",
        TTS_UPSTREAM=f"ws://127.0.0.1:{up_port}",
    )
    import app.config
    import app.main
    app.config.settings = app.config.Settings.from_env()
    app.main.settings = app.config.settings
    _serve(app.main.app, gw_port)
    return f"127.0.0.1:{gw_port}"


@pytest.mark.asyncio
async def test_http_response_is_streamed_not_buffered(gateway_url):
    total_expected = CHUNKS * CHUNK_DELAY
    async with httpx.AsyncClient(timeout=30) as c:
        t0 = time.perf_counter()
        ttfb = None
        async with c.stream("POST", f"http://{gateway_url}/v1/audio/transcriptions",
                            content=b"") as r:
            async for _ in r.aiter_raw():
                if ttfb is None:
                    ttfb = time.perf_counter() - t0
        total = time.perf_counter() - t0
    assert total >= total_expected * 0.7, "upstream did not actually stream slowly"
    # Buffering would push first-byte up to roughly the full duration.
    assert ttfb < total_expected * 0.4, f"looks buffered: ttfb={ttfb:.3f}s total={total:.3f}s"


@pytest.mark.asyncio
async def test_client_disconnect_evicts_upstream(gateway_url):
    before = dict(STATE)
    ws = await websockets.connect(f"ws://{gateway_url}/v1/audio/speech")
    await ws.send(json.dumps({"type": "speech.create", "id": "x1", "input": "hello",
                              "voice": {"preset": "Divya"}, "language": "hi"}))
    await ws.recv()                               # meta
    for _ in range(3):
        await ws.recv()                           # a few audio frames
    await ws.close()                              # barge-in
    for _ in range(60):
        await asyncio.sleep(0.05)
        if STATE["cancelled"] > before["cancelled"]:
            break
    assert STATE["cancelled"] == before["cancelled"] + 1, "upstream never saw the disconnect"
    assert STATE["completed"] == before["completed"], "upstream wrongly ran to completion"
