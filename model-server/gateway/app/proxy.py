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

import httpx
from fastapi import Request
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
