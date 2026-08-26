"""Transparent streaming proxy.

One hard requirement runs through this whole module: **never buffer**. A proxy
that collects a full response before forwarding it adds hundreds of milliseconds
to TTS time-to-first-byte, which is the difference between a natural phone call
and an awkward one. Both directions of both transports stream.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx
import websockets
from fastapi import Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from starlette.websockets import WebSocket, WebSocketDisconnect

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


async def relay_ws(client_ws: WebSocket, target_url: str) -> None:
    """Bidirectional WebSocket relay.

    Barge-in depends on this: when the voice server drops the socket, the
    client-to-upstream pump ends, the `async with` closes the upstream connection,
    and the TTS runner sees the disconnect and evicts the request. If the close
    did not propagate, an abandoned generation would keep occupying one of Parler's
    24 KV slots and keep consuming decode steps for a caller who already
    interrupted -- a slow capacity leak under load.
    """
    await client_ws.accept()

    try:
        async with websockets.connect(
            target_url, max_size=None, ping_interval=20, ping_timeout=20,
            open_timeout=5,
        ) as upstream:

            async def client_to_upstream() -> None:
                while True:
                    msg = await client_ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        return
                    if (text := msg.get("text")) is not None:
                        await upstream.send(text)
                    elif (data := msg.get("bytes")) is not None:
                        await upstream.send(data)

            async def upstream_to_client() -> None:
                async for msg in upstream:
                    if isinstance(msg, (bytes, bytearray)):
                        await client_ws.send_bytes(bytes(msg))
                    else:
                        await client_ws.send_text(msg)

            pumps = {
                asyncio.create_task(client_to_upstream()),
                asyncio.create_task(upstream_to_client()),
            }
            done, pending = await asyncio.wait(
                pumps, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if (exc := task.exception()) and not isinstance(
                    exc, (WebSocketDisconnect, websockets.ConnectionClosed)
                ):
                    raise exc

    except WebSocketDisconnect:
        pass
    finally:
        # Already closed by the disconnect that got us here -- fine either way.
        with contextlib.suppress(RuntimeError):
            await client_ws.close()
