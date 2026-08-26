"""VoicEra model-server gateway.

The single published port. Routes on modality, streams everything, and holds no
model-specific knowledge -- each model server speaks OpenAI spec natively, so
adding a model never touches this file.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse

from . import catalogue
from .config import Upstream, settings
from .proxy import forward_http, make_client, relay_ws

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gateway")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = make_client()
    for up in settings.all():
        log.info(
            "%s: %s", up.kind,
            f"{up.model or '?'} -> {up.url}" if up.enabled else "not deployed",
        )
    yield
    await app.state.http.aclose()


app = FastAPI(title="VoicEra model-server", lifespan=lifespan)


def _unavailable(up: Upstream) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": (
                    f"No {up.kind.upper()} model is deployed. Set {up.kind.upper()}_UPSTREAM "
                    f"and include the model in COMPOSE_PROFILES."
                ),
                "type": "upstream_not_configured",
            }
        },
    )


# ---------------------------------------------------------------- health / models

async def _probe(client: httpx.AsyncClient, up: Upstream) -> dict:
    """Health-check an upstream.

    The Parler TTS server is a raw websockets process with no HTTP surface, so a
    GET /health would report it permanently down. WebSocket upstreams get a TCP
    connect check instead.
    """
    if not up.enabled:
        return {"deployed": False}
    base = {"deployed": True, "model": up.model}
    try:
        if up.url.startswith(("ws://", "wss://")):
            # A bare TCP connect makes the raw-websockets Parler server log an
            # InvalidMessage traceback on every probe. Completing the handshake
            # and closing is both quieter and a truer check -- the server's
            # handler already returns cleanly on ConnectionClosed.
            ws = await asyncio.wait_for(
                websockets.connect(f"{up.url}/v1/audio/speech", open_timeout=2),
                timeout=3.0,
            )
            await ws.close()
            return {**base, "reachable": True}
        r = await client.get(f"{up.url}/health", timeout=2.0)
        return {**base, "reachable": r.status_code < 500}
    except Exception as exc:
        return {**base, "reachable": False, "error": str(exc)}


@app.get("/health")
async def health(request: Request):
    client: httpx.AsyncClient = request.app.state.http
    kinds = [up.kind for up in settings.all()]
    results = await asyncio.gather(*(_probe(client, up) for up in settings.all()))
    checks = dict(zip(kinds, results, strict=True))
    degraded = any(c.get("deployed") and not c.get("reachable") for c in checks.values())
    return JSONResponse(
        status_code=503 if degraded else 200,
        content={"status": "degraded" if degraded else "healthy", "upstreams": checks},
    )


@app.get("/models")
async def models():
    """Everything the server can host, and which slot each model is filling now.

    Distinct from /v1/models on purpose: that one is the OpenAI-compatible list
    of models you can call right now, so it must not advertise anything a client
    would get a 503 from.
    """
    live = {up.kind: up.model for up in settings.all() if up.enabled}
    entries = [
        {**m, "deployed": live.get(m["kind"]) == m["id"]}
        for m in catalogue.load()
    ]
    return {
        "object": "list",
        "data": entries,
        "deployed": {kind: live.get(kind) for kind in catalogue.KINDS},
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible: only models that can be called right now.

    See /models for the full catalogue including what is not deployed."""
    return {
        "object": "list",
        "data": [
            {"id": up.model, "object": "model", "owned_by": "voicera", "kind": up.kind}
            for up in settings.all() if up.enabled and up.model
        ],
    }


# ---------------------------------------------------------------- STT

async def _rest(request: Request, up: Upstream, path: str):
    """Every REST route is the same shape: refuse if the slot is empty, else
    stream through. Kept as one helper so adding a route stays a one-liner."""
    if not up.enabled:
        return _unavailable(up)
    return await forward_http(request.app.state.http, request, f"{up.url}{path}")


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    return await _rest(request, settings.stt, "/v1/audio/transcriptions")


# ---------------------------------------------------------------- TTS

@app.websocket("/v1/audio/speech")
async def speech(websocket: WebSocket):
    if not settings.tts.enabled:
        await websocket.close(code=1013, reason="No TTS model deployed")
        return
    await relay_ws(websocket, f"{settings.tts.url}/v1/audio/speech")


# ---------------------------------------------------------------- LLM (placeholder)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Passthrough to any OpenAI-compatible upstream. Disabled by default in pass 1;
    vLLM already speaks this spec, so enabling it is an env var, not code."""
    return await _rest(request, settings.llm, "/v1/chat/completions")
