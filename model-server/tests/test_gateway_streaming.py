"""The gateway must stream rather than buffer, and must let a client disconnect
reach the upstream -- that propagation is what makes barge-in free the TTS slot.
"""
import asyncio
import time

import httpx
import pytest
from conftest import free_port, serve
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

CHUNKS, CHUNK_DELAY = 5, 0.20          # a 1.0s response, first byte immediately

# Tagged by request, not counted. Two tests abort a speech stream, and the
# upstream's cleanup runs whenever the event loop gets to it -- with a shared
# counter one test's late cancellation lands inside another's measurement and
# the barge-in test fails for no reason. A flaky barge-in test is worse than
# none, because the first real failure gets waved through as "that one again".
STATE: dict[str, set[str]] = {"cancelled": set(), "completed": set()}

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


@upstream.post("/v1/audio/speech")
async def _speech(req: dict):
    """Stands in for the Parler server. The generator's finally block is exactly
    where the real one calls runner.evict(), so what happens here is the path
    barge-in depends on. The request's `input` doubles as its tag."""
    tag = req.get("input", "untagged")

    async def gen():
        completed = False
        try:
            for _ in range(40):
                yield b"\x00\x00\x80\x3f" * 40      # 40 float32 samples
                await asyncio.sleep(0.05)
            completed = True
            STATE["completed"].add(tag)
        finally:
            if not completed:
                STATE["cancelled"].add(tag)        # the eviction path barge-in relies on

    return StreamingResponse(gen(), media_type="audio/pcm",
                             headers={"X-Sample-Rate": "44100",
                                      "X-Audio-Format": "pcm_f32le"})


@pytest.fixture(scope="module")
def gateway_url():
    up_port, gw_port = free_port(), free_port()
    serve(upstream, up_port)

    from app.config import Settings, Upstream
    from app.main import create_app

    up = f"http://127.0.0.1:{up_port}"
    settings = Settings(
        stt=Upstream("stt", up, "test-stt"),
        tts=Upstream("tts", up, "test-tts"),
        llm=Upstream("llm", "", ""),
    )
    serve(create_app(settings), gw_port)
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
async def test_speech_streams_and_reports_the_sample_rate(gateway_url):
    """The client reads the rate off the header rather than assuming 44.1 kHz,
    so the gateway must not strip it."""
    async with (
        httpx.AsyncClient(timeout=30) as c,
        c.stream("POST", f"http://{gateway_url}/v1/audio/speech",
                 json={"input": "headers-test"}) as r,
    ):
        assert r.status_code == 200
        assert r.headers["x-sample-rate"] == "44100"
        assert r.headers["x-audio-format"] == "pcm_f32le"
        got = 0
        async for chunk in r.aiter_raw():
            got += len(chunk)
            if got >= 320:
                break
    assert got >= 320


@pytest.mark.asyncio
async def test_client_disconnect_evicts_upstream(gateway_url):
    tag = "disconnect-test"
    async with (
        httpx.AsyncClient(timeout=30) as c,
        c.stream("POST", f"http://{gateway_url}/v1/audio/speech",
                 json={"input": tag}) as r,
    ):
        frames = 0
        async for _ in r.aiter_raw():
            frames += 1
            if frames >= 3:
                break                              # barge-in: stop reading and hang up
    for _ in range(60):
        await asyncio.sleep(0.05)
        if tag in STATE["cancelled"]:
            break
    assert tag in STATE["cancelled"], "upstream never saw the disconnect"
    assert tag not in STATE["completed"], "upstream wrongly ran to completion"


def test_the_container_entrypoint_object_still_exists():
    """The image runs `uvicorn app.main:app`. Moving settings onto the app
    instance must not have turned that into a factory nobody calls."""
    import app.main as gateway
    from fastapi import FastAPI
    assert isinstance(gateway.app, FastAPI)
    # Built from the environment, which is where the container's settings arrive.
    assert gateway.app.state.settings is not None
    routes = {r.path for r in gateway.app.routes}
    for path in ("/health", "/models", "/v1/models", "/v1/audio/transcriptions",
                 "/v1/audio/speech", "/v1/chat/completions"):
        assert path in routes, f"{path} is not registered on the entrypoint app"
