"""FastAPI application factory.

Run directly:      python -m orpheus_server
Run with uvicorn:  uvicorn orpheus_server.app:app --host 0.0.0.0 --port 9000
Run with Docker:   docker compose up -d
"""
from __future__ import annotations

import contextlib
import logging
import os

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings, check_hardware, load_settings
from .engine import TTSEngine
from .voices import load_roster

log = logging.getLogger("orpheus")

DESCRIPTION = """\
Streaming text-to-speech for 22 Indian languages (AI4Bharat Orpheus + SNAC), served
with vLLM continuous batching.

**OpenAI-compatible** — point any OpenAI client at `<base>/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9000/v1", api_key="not-needed")
client.audio.speech.create(model="orpheus", voice="Amit",
                           input="नमस्ते, आज मौसम बहुत अच्छा है।").write_to_file("out.mp3")
```

The speaker name selects the language — every speaker belongs to exactly one, so
`voice="Amit"` is Hindi and `voice="Anitha"` is Tamil. `GET /v1/voices` lists them all.

**Choosing an endpoint**

| Need | Use |
|---|---|
| Drop-in OpenAI replacement | `POST /v1/audio/speech` |
| Incremental audio over HTTP | `POST /v1/audio/speech` with `response_format` `pcm` or `mp3` — the only two formats that survive chunking |
| OpenAI SSE audio deltas | `POST /v1/audio/speech` with `stream_format: "sse"` |
| Lowest latency (voice agents) | `/v1/tts/ws` WebSocket — not listed below, since OpenAPI cannot describe WebSockets |
| A complete WAV, one call | `POST /v1/tts` |
| Playable URL for `<audio>` or curl | `GET /v1/tts/stream` |
"""


def openai_error(status_code: int, message: str, param: str | None = None) -> JSONResponse:
    """Render an error in OpenAI's envelope.

    OpenAI clients read ``error.message``; FastAPI's default ``{"detail": ...}``
    reaches them as an opaque blob instead, so a wrong voice name shows up in the
    SDK as an unhelpful string. Same status, same text, shape the clients parse.
    """
    kind = "invalid_request_error" if status_code < 500 else "server_error"
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": kind, "param": param, "code": None}},
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("ORPHEUS_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()

    notes: list[str] = []
    if settings is None:
        settings, notes = load_settings()

    roster = load_roster(settings.resolved_voices_file())

    @contextlib.asynccontextmanager
    async def lifespan(application: FastAPI):
        for note in notes:
            log.info("%s", note)
        for level, message in check_hardware(settings):
            log.log(logging.WARNING if level == "warning" else logging.INFO, "%s", message)
        log.info(
            "roster: %d languages, %d speakers, %d styles (template=%s)",
            len(roster.languages), len(roster.all_voices), len(roster.styles), roster.template,
        )
        ambiguous = roster.ambiguous_voices
        if ambiguous:
            log.warning(
                "these speaker names appear in more than one language, so OpenAI clients must send "
                "the 'language' extension for them: %s", ambiguous,
            )
        await engine.start()
        try:
            yield
        finally:
            await engine.stop()

    application = FastAPI(
        title="AI4Bharat Orpheus Indic TTS",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )

    engine = TTSEngine(settings, roster)
    application.state.settings = settings
    application.state.roster = roster
    application.state.engine = engine

    @application.exception_handler(StarletteHTTPException)
    async def _http_error(_request, exc: StarletteHTTPException):
        response = openai_error(exc.status_code, str(exc.detail))
        if exc.headers:
            # Starlette's default handler carries these; 405 needs Allow, 401
            # needs WWW-Authenticate.
            response.headers.update(exc.headers)
        return response

    @application.exception_handler(RequestValidationError)
    async def _validation_error(_request, exc: RequestValidationError):
        # OpenAI answers a malformed body with 400, not FastAPI's default 422.
        errors = exc.errors()
        first = errors[0] if errors else {}
        param = ".".join(str(p) for p in first.get("loc", ()) if p != "body") or None
        message = first.get("msg", "invalid request")
        return openai_error(400, f"{message} (at '{param}')" if param else message, param)

    if settings.server.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.server.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    from .api import meta, native, openai_speech

    application.include_router(meta.router)
    application.include_router(openai_speech.router, prefix="/v1")
    application.include_router(native.router, prefix="/v1")
    application.include_router(native.ws_router)
    return application


app = create_app()
