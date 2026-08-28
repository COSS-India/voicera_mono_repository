"""Transparent streaming proxy.

One hard requirement runs through this module: **never buffer**. A proxy that
collects a full response before forwarding adds hundreds of milliseconds to TTS
time-to-first-byte, which is the difference between a natural phone call and an
awkward one. Request bodies and response bodies both stream.

Cancellation matters as much as throughput. When a caller is interrupted,
Pipecat cancels the task reading the response; that closes this connection,
which closes the upstream one, which frees the model's slot.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import httpx
import websockets
from fastapi import Request, WebSocket
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

# read/write are deliberately unbounded: a TTS generation or an SSE completion
# holds the response open for as long as the model takes. connect/pool stay short
# so an upstream that is down fails fast instead of hanging a call.
TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=None, pool=5.0)

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def _forwardable(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def make_client() -> httpx.AsyncClient:
    """One long-lived client. Per-request clients would rebuild the connection
    pool on every call and add a TCP handshake to every turn of the conversation."""
    return httpx.AsyncClient(
        timeout=TIMEOUT,
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
    )


async def forward_http(
    client: httpx.AsyncClient, request: Request, target_url: str
) -> StreamingResponse:
    """Forward a request upstream and stream the response straight back.

    The request body is passed as an async iterator rather than read into memory,
    and the response is handed to Starlette as `aiter_raw()` so bytes reach the
    caller as they arrive. `client.send(stream=True)` is what makes this possible:
    it returns once headers are in, before the body has been read.
    """
    upstream_req = client.build_request(
        request.method,
        target_url,
        headers=_forwardable(request.headers),
        content=request.stream(),
        params=request.query_params,
    )
    resp = await client.send(upstream_req, stream=True)
    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers=_forwardable(resp.headers),
        background=BackgroundTask(resp.aclose),
    )


log = logging.getLogger("gateway")

# WebSocket is back, and only here.
#
# It was removed when TTS moved to the OpenAI speech endpoint, and that was
# right: TTS is one-directional -- text in, audio out -- so HTTP did the same job
# with less machinery and got cancellation for free.
#
# Live transcription is not one-directional. Audio flows in for as long as the
# caller talks while partial transcripts flow out, and neither side knows when
# the other will speak next. That is what a socket is for. Faking it over HTTP
# would mean either long-polling for partials or a second connection carrying
# results, both worse than the thing WebSocket already does.
#
# The relay stays dumb: bytes and text frames pass through untouched, in both
# directions, and the protocol on top is entirely between the client and the
# model. The gateway learns nothing about either.


async def relay_ws(client: WebSocket, target_url: str) -> None:
    """Splice a client WebSocket onto an upstream one.

    Both directions run concurrently; whichever ends first tears the other down,
    so a caller hanging up mid-sentence closes the upstream session and frees the
    decoder rather than leaving it transcribing an empty room.
    """
    await client.accept()
    try:
        upstream = await websockets.connect(target_url, max_size=None, open_timeout=10)
    except Exception as exc:                                        # noqa: BLE001
        # Accept-then-explain, rather than refusing the handshake: a client whose
        # handshake fails is told only "HTTP 403" and cannot tell "wrong URL"
        # from "model still loading".
        log.warning("upstream websocket %s unreachable: %s", target_url, exc)
        with contextlib.suppress(Exception):
            await client.send_json({
                "type": "error", "reason": "upstream_unreachable",
                "error": f"streaming upstream is not reachable: {exc}",
            })
            await client.close(code=1013, reason="upstream unreachable")
        return

    async def to_upstream() -> None:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                return
            if (data := message.get("bytes")) is not None:
                await upstream.send(data)
            elif (text := message.get("text")) is not None:
                await upstream.send(text)

    async def to_client() -> None:
        async for message in upstream:
            if isinstance(message, (bytes, bytearray)):
                await client.send_bytes(bytes(message))
            else:
                await client.send_text(message)

    pumps = [asyncio.create_task(to_upstream()), asyncio.create_task(to_client())]
    try:
        done, pending = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            # Surface a genuine failure; a closed socket on either side is normal.
            exc = task.exception()
            if exc and not isinstance(exc, websockets.ConnectionClosed):
                log.warning("websocket relay ended on %s: %s", type(exc).__name__, exc)
    finally:
        with contextlib.suppress(Exception):
            await upstream.close()
        with contextlib.suppress(Exception):
            await client.close()
