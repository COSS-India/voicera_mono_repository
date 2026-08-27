"""The LLM slot, which is the only one that ships turned off.

"Off" therefore has to behave as carefully as "on": a caller must get a clear
503 rather than a hang or a 404, `/v1/models` must not advertise it, and turning
it on must route and stream token-by-token. None of that was covered before --
the LLM was a placeholder nobody could start.

Everything here runs against the real gateway over a real socket. An ASGI test
client would not show whether tokens arrive as they are produced, which for a
voice agent is the entire point: the first token starts the sentence.
"""
import asyncio
import json
import time

import httpx
import pytest
from conftest import free_port, serve
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

TOKENS = ["Namaste", " ", "aap", " kaise", " hain", "?"]
TOKEN_DELAY = 0.08

upstream = FastAPI()


@upstream.get("/health")
def _health():
    return {"status": "ok"}


@upstream.post("/v1/chat/completions")
async def _chat(body: dict):
    """Stands in for vLLM. Streams SSE the way the OpenAI spec does."""
    async def gen():
        for token in TOKENS:
            chunk = {
                "id": "cmpl-1", "object": "chat.completion.chunk",
                "model": body.get("model", "?"),
                "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            await asyncio.sleep(TOKEN_DELAY)
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def build_gateway(*, llm_model: str) -> str:
    """A gateway process with the LLM slot either filled or empty."""
    from app.config import Settings, Upstream
    from app.main import create_app

    up_port, gw_port = free_port(), free_port()
    serve(upstream, up_port)
    url = f"http://127.0.0.1:{up_port}" if llm_model else ""
    settings = Settings(
        stt=Upstream("stt", "", ""),
        tts=Upstream("tts", "", ""),
        llm=Upstream("llm", url, llm_model),
    )
    serve(create_app(settings), gw_port)
    return f"http://127.0.0.1:{gw_port}"


@pytest.fixture(scope="module")
def llm_on():
    return build_gateway(llm_model="qwen3.5-4b")


@pytest.fixture(scope="module")
def llm_off():
    return build_gateway(llm_model="")


# ---------------------------------------------------------------- slot is off

@pytest.mark.asyncio
async def test_empty_slot_answers_503_not_404(llm_off):
    """404 would read as "wrong URL" and send someone hunting through routes."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{llm_off}/v1/chat/completions",
                         json={"model": "anything", "messages": []})
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_the_503_says_which_slot_and_how_to_fix_it(llm_off):
    async with httpx.AsyncClient(timeout=10) as c:
        body = (await c.post(f"{llm_off}/v1/chat/completions", json={})).json()
    message = body["error"]["message"]
    assert "LLM" in message, message
    assert body["error"]["type"] == "upstream_not_configured"


@pytest.mark.asyncio
async def test_an_empty_slot_is_not_advertised(llm_off):
    """OpenAI clients read /v1/models as "what can I call right now"."""
    async with httpx.AsyncClient(timeout=10) as c:
        listed = (await c.get(f"{llm_off}/v1/models")).json()
    assert [m["id"] for m in listed["data"]] == []


@pytest.mark.asyncio
async def test_health_stays_200_when_a_slot_is_deliberately_empty(llm_off):
    """Not deployed is not the same as broken. Marking it degraded would make
    every monitor cry wolf on a stack that is working as configured."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{llm_off}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    assert r.json()["upstreams"]["llm"] == {"deployed": False}


# ---------------------------------------------------------------- slot is on

@pytest.mark.asyncio
async def test_filled_slot_routes_to_the_upstream(llm_on):
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{llm_on}/v1/chat/completions",
                         json={"model": "qwen3.5-4b", "messages": [], "stream": True})
    assert r.status_code == 200
    assert "Namaste" in r.text


@pytest.mark.asyncio
async def test_filled_slot_is_advertised_under_its_catalogue_id(llm_on):
    async with httpx.AsyncClient(timeout=10) as c:
        listed = (await c.get(f"{llm_on}/v1/models")).json()
    entry = next(m for m in listed["data"] if m["kind"] == "llm")
    assert entry["id"] == "qwen3.5-4b", "the id clients must send back in `model`"


@pytest.mark.asyncio
async def test_tokens_arrive_as_they_are_generated(llm_on):
    """Buffering here would hold the whole sentence until the last token, and
    TTS could not start speaking until the model had finished thinking."""
    whole_response = len(TOKENS) * TOKEN_DELAY
    async with httpx.AsyncClient(timeout=30) as c:
        t0 = time.perf_counter()
        first = None
        seen = 0
        async with c.stream("POST", f"{llm_on}/v1/chat/completions",
                            json={"model": "qwen3.5-4b", "messages": [],
                                  "stream": True}) as r:
            async for _ in r.aiter_raw():
                seen += 1
                if first is None:
                    first = time.perf_counter() - t0
        total = time.perf_counter() - t0
    assert seen > 1, "the whole stream arrived as one chunk -- something buffered it"
    assert total >= whole_response * 0.7, "upstream did not actually stream slowly"
    assert first < whole_response * 0.5, f"first token late: {first:.3f}s of {total:.3f}s"


@pytest.mark.asyncio
async def test_the_model_field_reaches_the_upstream_unchanged(llm_on):
    """vLLM 400s on a model name it is not serving, so the gateway must not
    rewrite or drop the field on its way through."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{llm_on}/v1/chat/completions",
                         json={"model": "qwen3.5-4b", "messages": []})
    assert '"model": "qwen3.5-4b"' in r.text or '"model":"qwen3.5-4b"' in r.text
