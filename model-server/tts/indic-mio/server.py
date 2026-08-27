"""Indic-Mio TTS, on the OpenAI speech endpoint.

Replaces the WebSocket protocol this server used to speak. The engine below is
untouched -- token generation is still delegated to vLLM and MioCodec still does
the decode. Only the transport changed:

    POST /v1/audio/speech    OpenAI-compatible, streams PCM as it is produced
    GET  /health             503 until the codec is loaded and vLLM answers
    GET  /v1/voices          the speaker roster, for building agent config

Why the change: every model in the model-server's TTS slot answers the same
endpoint, so the voice pipeline needs no per-model client. A WebSocket would
have been a second protocol for one model.

Two things HTTP gives us that the socket did not:

* **Cancellation is free.** `synthesize_stream` is an async generator; when the
  caller hangs up mid-sentence, Starlette closes it, GeneratorExit unwinds into
  the aiohttp response reading from vLLM, and that request is aborted. The old
  code had to notice a ConnectionClosed and return. Barge-in stops the GPU work
  rather than letting it run to the end of a sentence nobody is listening to.

* **Self-description.** The response says what is in it -- format, rate, channels
  -- so a client that also talks to Orpheus (24 kHz signed 16-bit) or Indic
  Parler (44.1 kHz float32) decodes each correctly without knowing which one it
  reached.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
from typing import Literal

import numpy as np
import uvicorn
from config import Config
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from tts_engine import MioTTSEngine, TTSGenerationError

logging.basicConfig(
    level=os.getenv("MIO_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("indic_mio.server")

#: What this server will encode into. `pcm_f32le` is what the codec produces, so
#: it is the default and involves no conversion; `pcm` is OpenAI's own name for
#: signed 16-bit, offered because standard OpenAI clients ask for it.
ResponseFormat = Literal["pcm_f32le", "pcm"]


class SpeechRequest(BaseModel):
    """Body of POST /v1/audio/speech. OpenAI's schema, plus two extensions."""

    input: str = Field(..., description="Text to speak, in its native script.")
    model: str | None = Field(None, description="Accepted for compatibility; one model is served.")
    voice: str | None = Field(None, description="Speaker id from GET /v1/voices.")
    response_format: ResponseFormat = "pcm_f32le"
    speed: float = Field(1.0, description="Accepted for compatibility; not applied.")
    # --- beyond the OpenAI schema ---
    instructions: str | None = Field(
        None, description="Extension: accepted for cross-model compatibility, not used here."
    )
    language: str | None = Field(
        None, description="Extension: informational. The speaker embedding carries the accent."
    )


def create_app(config: Config | None = None) -> FastAPI:
    """Build the server around one engine.

    A factory rather than a module-level app so the engine's lifetime is tied to
    the application's, and so two configurations can exist in one process --
    which is what makes this testable without a GPU.
    """
    cfg = config or Config.from_env()

    @contextlib.asynccontextmanager
    async def lifespan(application: FastAPI):
        engine: MioTTSEngine = application.state.engine
        # Loading the codec is CPU/GPU work and blocks; keep it off the loop so
        # the health endpoint can answer "loading" while it happens.
        await asyncio.to_thread(engine.load_codec)
        await engine.start()
        application.state.ready = True
        logger.info(
            "Indic-Mio TTS on :%d (vllm=%s model=%s codec=%s sr=%d)",
            cfg.port, cfg.llm_base_url, cfg.llm_model, cfg.codec_model_id, engine.sample_rate,
        )
        try:
            yield
        finally:
            application.state.ready = False
            await engine.close()

    app = FastAPI(title="Indic-Mio TTS", lifespan=lifespan)
    app.state.config = cfg
    app.state.engine = MioTTSEngine(cfg)
    app.state.ready = False

    @app.get("/health")
    async def health():
        """503 while loading, so the gateway holds traffic back rather than
        sending it to a codec that has not finished loading."""
        ready = app.state.ready
        body = {
            "status": "ok" if ready else "loading",
            "ready": ready,
            "model": cfg.llm_model,
            "codec": cfg.codec_model_id,
            "sample_rate": app.state.engine.sample_rate if ready else None,
        }
        return JSONResponse(body, status_code=200 if ready else 503)

    @app.get("/v1/voices")
    async def voices():
        """The speaker roster. Agent config needs the ids, and they come from a
        manifest rather than being hardcoded anywhere."""
        path = os.path.join(cfg.voices_dir, "manifest.json")
        try:
            with open(path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.warning("voice manifest unavailable (%s): %s", path, exc)
            return {"object": "list", "data": [], "default": cfg.default_voice or None}
        return {
            "object": "list",
            "data": [
                {"id": v.get("name"), "gender": v.get("gender"), "source": v.get("source")}
                for v in manifest.get("voices", [])
            ],
            "default": cfg.default_voice or manifest.get("default"),
        }

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        engine: MioTTSEngine = app.state.engine
        if not app.state.ready:
            return JSONResponse(
                {"error": {"message": "model still loading", "type": "not_ready"}},
                status_code=503,
            )

        to_int16 = req.response_format == "pcm"
        frame = max(1, cfg.frame_samples)

        async def stream():
            try:
                async for chunk in engine.synthesize_stream(req.input, voice=req.voice):
                    # Re-slice into modest frames so the caller hears audio in
                    # small steps rather than in whatever size the decoder chose.
                    for start in range(0, chunk.size, frame):
                        part = chunk[start:start + frame]
                        if part.size == 0:
                            continue
                        arr = part.astype(np.float32, copy=False)
                        if to_int16:
                            yield (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                        else:
                            yield arr.tobytes()
            except TTSGenerationError as exc:
                # The status line has already gone out, so the only honest signal
                # left is to stop writing. Log loudly; the client sees a short
                # stream, which its own error handling treats as a failure.
                logger.warning("generation failed: %s", exc)
            except asyncio.CancelledError:
                # Barge-in. Let it propagate so the generator closes and the vLLM
                # request behind it is aborted rather than running to completion.
                logger.info("client disconnected; generation cancelled")
                raise
            except Exception:
                logger.exception("unexpected error during synthesis")

        return StreamingResponse(
            stream(),
            media_type="audio/pcm",
            headers={
                # What is in the body. A client that also talks to a 24 kHz
                # 16-bit model reads these rather than assuming.
                "X-Audio-Format": req.response_format,
                "X-Sample-Rate": str(engine.sample_rate),
                "X-Channels": "1",
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Indic-Mio TTS (vLLM + MioCodec)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.from_env()
    if args.host:
        cfg = cfg.__class__(**{**cfg.__dict__, "host": args.host})
    if args.port:
        cfg = cfg.__class__(**{**cfg.__dict__, "port": args.port})

    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
