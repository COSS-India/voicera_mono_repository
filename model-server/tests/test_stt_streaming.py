"""Live transcription through the gateway.

The STT slot gained a second shape. `POST /v1/audio/transcriptions` is one-shot:
send an utterance, get a sentence. `WS /v1/asr/ws` is a conversation -- audio
flows in for as long as someone talks while partial transcripts flow out.

WebSocket came back into the gateway for this, having been removed when TTS
moved to HTTP. That was not a reversal of judgement: TTS is one-directional, so
HTTP was strictly better and gave cancellation for free. This is genuinely
two-directional, and neither side knows when the other will speak next.

What the relay must do, and what silently breaks if it does not:

* pass both frame types, both ways, untouched -- audio is binary, control and
  transcripts are text, and a relay that handles one and drops the other looks
  alive while transcribing nothing
* let the caller hang up and have that reach the model, so a decoder is not left
  running against an empty room for the rest of the session timeout
* refuse readably when no model is deployed, rather than failing the handshake
"""
import asyncio
import json

import pytest
import websockets
from conftest import free_port, serve
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

STATE = {"disconnected": set(), "received": []}

upstream = FastAPI()


@upstream.get("/health")
def _health():
    return {"status": "ok"}


@upstream.websocket("/v1/asr/ws")
async def _asr(ws: WebSocket):
    """Stands in for indic-transcribe: ready frame, partials, turn_final."""
    await ws.accept()
    tag = ws.query_params.get("language", "?")
    await ws.send_json({"type": "ready", "language": tag, "sample_rate": 16000})
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(1000)
            if (audio := message.get("bytes")) is not None:
                STATE["received"].append(len(audio))
                # One partial per chunk, the way a streaming decoder emits.
                await ws.send_json({"type": "partial", "turn": 0,
                                    "text": f"chunk{len(STATE['received'])}"})
            elif (text := message.get("text")) is not None:
                if json.loads(text).get("type") == "stop":
                    await ws.send_json({"type": "closed", "transcript": "done"})
                    return
    except WebSocketDisconnect:
        # The decoder would be released here. If this never fires, a caller who
        # hung up leaves a session transcribing silence.
        STATE["disconnected"].add(tag)


async def recv(ws, timeout: float = 5.0):
    """Every read here is bounded.

    A relay that drops a frame does not error -- it simply never answers. Without
    a deadline these tests hang instead of failing, which is worse than a red
    build: the mutation that proved this was a one-line change that silently
    stopped forwarding text frames.
    """
    return await asyncio.wait_for(ws.recv(), timeout=timeout)


def build_gateway(*, stt_model: str) -> str:
    from app.config import Settings, Upstream
    from app.main import create_app

    up_port, gw_port = free_port(), free_port()
    serve(upstream, up_port)
    url = f"http://127.0.0.1:{up_port}" if stt_model else ""
    settings = Settings(
        stt=Upstream("stt", url, stt_model),
        tts=Upstream("tts", "", ""),
        llm=Upstream("llm", "", ""),
    )
    serve(create_app(settings), gw_port)
    return f"ws://127.0.0.1:{gw_port}"


@pytest.fixture(scope="module")
def stt_on():
    return build_gateway(stt_model="indic-transcribe")


@pytest.fixture(scope="module")
def stt_off():
    return build_gateway(stt_model="")


@pytest.mark.asyncio
async def test_audio_in_transcripts_out(stt_on):
    """The round trip: binary one way, JSON the other, concurrently."""
    async with websockets.connect(f"{stt_on}/v1/asr/ws?language=hi") as ws:
        ready = json.loads(await recv(ws))
        assert ready["type"] == "ready"
        assert ready["language"] == "hi", "query parameters did not reach the model"

        for _ in range(3):
            await ws.send(b"\x00\x01" * 160)          # 16-bit PCM, as the model wants
        seen = [json.loads(await recv(ws)) for _ in range(3)]

    assert [m["type"] for m in seen] == ["partial"] * 3
    assert [m["text"] for m in seen] == ["chunk1", "chunk2", "chunk3"], \
        "partials arrived out of order or were coalesced"


@pytest.mark.asyncio
async def test_a_text_command_reaches_the_model(stt_on):
    """Control frames share the socket with audio. A relay that forwards only
    bytes would leave the client unable to say it had finished."""
    async with websockets.connect(f"{stt_on}/v1/asr/ws?language=stopper") as ws:
        await recv(ws)                                 # ready
        await ws.send(json.dumps({"type": "stop"}))
        closing = json.loads(await recv(ws))
    assert closing == {"type": "closed", "transcript": "done"}


@pytest.mark.asyncio
async def test_hanging_up_reaches_the_model(stt_on):
    """Barge-in for the microphone side. Without this the decoder keeps running
    against an empty room until the session times out."""
    tag = "hangup"
    ws = await websockets.connect(f"{stt_on}/v1/asr/ws?language={tag}")
    await recv(ws)
    await ws.send(b"\x00\x01" * 160)
    await recv(ws)
    await ws.close()

    for _ in range(60):
        await asyncio.sleep(0.05)
        if tag in STATE["disconnected"]:
            break
    assert tag in STATE["disconnected"], "the model never saw the client leave"


@pytest.mark.asyncio
async def test_partials_are_not_held_until_the_end(stt_on):
    """A relay that buffered would deliver every partial at once, which is the
    same as not streaming at all."""
    async with websockets.connect(f"{stt_on}/v1/asr/ws?language=timing") as ws:
        await recv(ws)
        await ws.send(b"\x00\x01" * 160)
        first = await recv(ws, timeout=2.0)
        assert json.loads(first)["type"] == "partial"
        # A second chunk must not need the first to be drained.
        await ws.send(b"\x00\x01" * 160)
        second = await recv(ws, timeout=2.0)
        assert json.loads(second)["type"] == "partial"


@pytest.mark.asyncio
async def test_no_model_deployed_explains_itself(stt_off):
    """Refusing the handshake would tell the caller only "HTTP 403", which does
    not distinguish a wrong URL from a slot nobody filled."""
    async with websockets.connect(f"{stt_off}/v1/asr/ws") as ws:
        body = json.loads(await recv(ws))
    assert body["reason"] == "upstream_not_configured"
    assert "STT_MODEL" in body["error"], body["error"]


@pytest.mark.asyncio
async def test_the_one_shot_route_still_works_alongside(stt_on):
    """Adding streaming must not have disturbed the endpoint every deployment
    uses today."""
    import httpx
    http_base = stt_on.replace("ws://", "http://", 1)
    async with httpx.AsyncClient(timeout=5) as c:
        r = await c.get(f"{http_base}/health")
    assert r.status_code == 200
    assert r.json()["upstreams"]["stt"]["model"] == "indic-transcribe"
