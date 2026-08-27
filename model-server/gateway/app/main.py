"""VoicEra model-server gateway.

The single published port. Routes on modality, streams everything, and holds no
model-specific knowledge -- each model server speaks OpenAI spec natively, so
adding a model never touches this file.

Settings live on the app instance rather than as a module global. That is not
ceremony: two gateways with different configurations have to be able to exist in
one process, or tests end up reassigning a shared global and whichever ran last
silently reconfigures the other one's routes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import catalogue
from .config import Settings, Upstream
from .config import settings as env_settings
from .proxy import forward_http, make_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gateway")


def conf(request: Request) -> Settings:
    return request.app.state.settings


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = make_client()
    for up in app.state.settings.all():
        log.info(
            "%s: %s", up.kind,
            f"{up.model or '?'} -> {up.url}" if up.enabled else "not deployed",
        )
    yield
    await app.state.http.aclose()


def _unavailable(up: Upstream) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": (
                    f"No {up.kind.upper()} model is deployed. Set {up.kind.upper()}_MODEL "
                    f"in .env and include {up.kind} in COMPOSE_PROFILES."
                ),
                "type": "upstream_not_configured",
            }
        },
    )


async def _probe(client: httpx.AsyncClient, up: Upstream) -> dict:
    """Health-check an upstream. All three speak HTTP."""
    if not up.enabled:
        return {"deployed": False}
    base = {"deployed": True, "model": up.model}
    try:
        r = await client.get(f"{up.url}/health", timeout=2.0)
        return {**base, "reachable": r.status_code < 500}
    except Exception as exc:                                        # noqa: BLE001
        return {**base, "reachable": False, "error": str(exc)}


async def _rest(request: Request, up: Upstream, path: str):
    """Every REST route is the same shape: refuse if the slot is empty, else
    stream through. Kept as one helper so adding a route stays a one-liner."""
    if not up.enabled:
        return _unavailable(up)
    return await forward_http(request.app.state.http, request, f"{up.url}{path}")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a gateway bound to one configuration."""
    app = FastAPI(title="VoicEra model-server", lifespan=lifespan)
    app.state.settings = settings or env_settings

    # ------------------------------------------------------ health / catalogue

    @app.get("/health")
    async def health(request: Request):
        client: httpx.AsyncClient = request.app.state.http
        slots = conf(request).all()
        results = await asyncio.gather(*(_probe(client, up) for up in slots))
        checks = dict(zip((up.kind for up in slots), results, strict=True))
        # A slot nobody deployed is not a fault. Reporting it as degraded would
        # make every monitor cry wolf on a stack running exactly as configured.
        degraded = any(c.get("deployed") and not c.get("reachable") for c in checks.values())
        return JSONResponse(
            status_code=503 if degraded else 200,
            content={"status": "degraded" if degraded else "healthy", "upstreams": checks},
        )

    @app.get("/models")
    async def models(request: Request):
        """Everything the server can host, and which slot each model fills now.

        Distinct from /v1/models on purpose: that one is the OpenAI-compatible
        list of models you can call right now, so it must not advertise anything
        a client would get a 503 from.
        """
        live = {up.kind: up.model for up in conf(request).all() if up.enabled}
        entries = [{**m, "deployed": live.get(m["kind"]) == m["id"]} for m in catalogue.load()]
        return {
            "object": "list",
            "data": entries,
            "deployed": {kind: live.get(kind) for kind in catalogue.KINDS},
        }

    @app.get("/v1/models")
    async def list_models(request: Request):
        """OpenAI-compatible: only models that can be called right now.

        See /models for the full catalogue including what is not deployed."""
        return {
            "object": "list",
            "data": [
                {"id": up.model, "object": "model", "owned_by": "voicera", "kind": up.kind}
                for up in conf(request).all() if up.enabled and up.model
            ],
        }

    # ------------------------------------------------------ the three modalities

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request):
        return await _rest(request, conf(request).stt, "/v1/audio/transcriptions")

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        return await _rest(request, conf(request).tts, "/v1/audio/speech")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Passthrough to any OpenAI-compatible upstream. vLLM speaks this spec
        natively, so filling the LLM slot is a folder and an env var, not code."""
        return await _rest(request, conf(request).llm, "/v1/chat/completions")

    return app


# What the container runs: `uvicorn app.main:app`.
app = create_app()
