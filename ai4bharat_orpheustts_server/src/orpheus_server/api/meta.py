"""Liveness and metrics. Unauthenticated so probes and dashboards work."""
from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response, status

router = APIRouter()


@router.get("/health", tags=["meta"], summary="Liveness and readiness")
async def health(request: Request, response: Response):
    """``ready`` flips true only after the model is loaded and warmup has finished.

    Returns 503 until then, so orchestrators hold traffic back rather than sending
    it to an engine that is still capturing CUDA graphs.
    """
    engine = request.app.state.engine
    settings = request.app.state.settings
    ready = engine.ready
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if ready else "loading",
        "ready": ready,
        "model": settings.server.model_name,
        "model_path": engine.model_path,
        "quantization": settings.model.quantization,
        "max_num_seqs": settings.engine.max_num_seqs,
        "streams_active": engine.metrics.streams_active,
    }


@router.get("/metrics", tags=["meta"], summary="Aggregate runtime counters")
async def metrics(request: Request):
    """Process-wide counters only.

    Per-request latency is deliberately absent: with many concurrent streams a
    shared ``last_ttfa`` reports whichever request finished most recently, which
    is misleading. Each response carries its own ``X-TTFA-Ms`` / ``X-RTF``
    instead, and the WebSocket ``done`` frame carries the full per-stream summary.
    """
    engine = request.app.state.engine
    return {
        **engine.metrics.snapshot(),
        "ready": engine.ready,
        "uptime_seconds": (
            round(time.time() - engine.started_at, 1) if engine.started_at else None
        ),
    }
