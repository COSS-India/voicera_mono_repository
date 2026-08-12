"""OpenAI-compatible audio endpoints.

Point any OpenAI client at this server and speech synthesis works unchanged:

    client = OpenAI(base_url="http://host:9000/v1", api_key="not-needed")
    client.audio.speech.create(model="orpheus", voice="Amit", input="...")

The one thing worth understanding is voice selection. OpenAI's schema has no
language field, so the speaker name carries it: every speaker in the roster
belongs to exactly one language, which makes ``voice="Amit"`` unambiguously
Hindi. A ``language`` field is accepted as an extension for rosters where a
speaker name is shared, but standard clients never need it.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .. import audio as audio_fmt
from ..engine import StreamStats, TTSEngine
from ..voices import Roster
from .deps import get_engine, get_roster, get_settings

log = logging.getLogger("orpheus.openai")

router = APIRouter()

ResponseFormat = Literal["wav", "pcm", "mp3", "flac", "opus"]
StreamFormat = Literal["audio", "sse"]


class SpeechRequest(BaseModel):
    """Body of ``POST /v1/audio/speech``. Mirrors OpenAI's schema."""

    model_config = ConfigDict(protected_namespaces=())

    model: str = Field("orpheus", description="Accepted for compatibility; the server serves one model.")
    input: str = Field(..., description="Text to synthesize, in the language's native script.",
                       examples=["नमस्ते, आज मौसम बहुत अच्छा है।"])
    voice: str = Field(..., description="Speaker name from GET /v1/voices. Determines the language.",
                       examples=["Amit"])
    response_format: ResponseFormat = Field(
        "mp3", description="Output format. OpenAI's default is mp3. 'pcm' is raw 24 kHz mono s16le.",
    )
    stream_format: StreamFormat = Field(
        "audio",
        description="'audio' returns the audio body (chunked while generating); 'sse' returns "
                    "server-sent speech.audio.delta events carrying base64 audio.",
    )
    speed: float = Field(
        1.0, ge=0.25, le=4.0,
        description="Accepted for compatibility but IGNORED: Orpheus has no rate control, and "
                    "resampling would shift pitch. A non-default value sets X-Speed-Ignored on the response.",
    )
    instructions: Optional[str] = Field(
        None,
        description="Used as the speaking style when it matches a name from GET /v1/styles "
                    "(e.g. 'NEWS'); otherwise ignored.",
    )
    # --- extensions beyond the OpenAI schema ---
    language: Optional[str] = Field(
        None, description="Extension: language code. Only needed if a speaker name is not unique.",
    )
    style: Optional[str] = Field(
        None, description="Extension: speaking style, e.g. 'CONV', 'NEWS'. Takes precedence over instructions.",
    )
    max_tokens: Optional[int] = Field(
        None, description="Extension: cap on generated audio tokens (~85 ms of audio each).",
    )


def _resolve_style(req: SpeechRequest, roster: Roster) -> Optional[str]:
    """Style from the extension field, else from `instructions` if it names one."""
    if req.style:
        return req.style
    if req.instructions and req.instructions.strip().upper() in {s.upper() for s in roster.styles}:
        return req.instructions.strip().upper()
    return None


@router.get("/models", tags=["openai"], summary="List models (OpenAI-compatible)")
async def list_models(request: Request):
    settings = get_settings(request)
    engine: TTSEngine = request.app.state.engine
    created = int(engine.started_at or time.time())
    return {
        "object": "list",
        "data": [{
            "id": settings.server.model_name,
            "object": "model",
            "created": created,
            "owned_by": "local",
        }],
    }


@router.post(
    "/audio/speech",
    tags=["openai"],
    summary="Synthesize speech (OpenAI-compatible, buffered or streaming)",
    response_class=Response,
    responses={200: {"content": {"audio/mpeg": {}, "audio/wav": {}, "audio/pcm": {},
                                 "audio/flac": {}, "audio/ogg": {}, "text/event-stream": {}},
                     "description": "Audio in the requested format, or an SSE event stream."}},
)
async def create_speech(
    req: SpeechRequest,
    engine: TTSEngine = Depends(get_engine),
    roster: Roster = Depends(get_roster),
):
    try:
        language, voice, style = roster.resolve(req.voice, req.language, _resolve_style(req, roster))
    except LookupError as exc:
        raise HTTPException(400, str(exc)) from exc

    max_tokens = engine.clamp_max_tokens(req.max_tokens)
    engine.metrics.requests_total += 1
    stats = StreamStats()

    headers = {"X-Language": language, "X-Voice": voice}
    if style:
        headers["X-Style"] = style
    if req.speed != 1.0:
        headers["X-Speed-Ignored"] = str(req.speed)

    def pcm_stream():
        return engine.stream_pcm(
            text=req.input, voice=voice, language=language,
            style=style, max_tokens=max_tokens, stats=stats,
        )

    if req.stream_format == "sse":
        return StreamingResponse(
            _sse_events(engine, pcm_stream, req.response_format, stats),
            media_type="text/event-stream",
            headers={**headers, "Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    # stream_format="audio". pcm and mp3 are chunked as they are produced, so a
    # client that reads incrementally hears audio in ~120 ms while one that calls
    # create() just buffers the body. wav, flac and opus are encoded in one pass
    # instead: their headers are only correct once the length is known, and a
    # client cannot be sent a correction for bytes it already has.
    if not audio_fmt.streams_incrementally(req.response_format):
        encoder = audio_fmt.make_encoder(req.response_format, streaming=False)
        body = bytearray()
        try:
            async for pcm in pcm_stream():
                body += encoder.feed(pcm)
            body += encoder.close()
        except Exception as exc:  # noqa: BLE001
            engine.metrics.errors_total += 1
            log.exception("synthesis failed")
            raise HTTPException(500, f"synthesis failed: {exc}") from exc
        if stats.frames == 0:
            engine.metrics.errors_total += 1
            raise HTTPException(500, "no audio produced")
        return Response(
            content=bytes(body),
            media_type=encoder.media_type,
            headers={**headers, **_timing_headers(stats)},
        )

    return StreamingResponse(
        _audio_chunks(engine, pcm_stream, req.response_format),
        media_type=audio_fmt.media_type(req.response_format),
        headers={**headers, "X-Accel-Buffering": "no"},
    )


def _timing_headers(stats: StreamStats) -> dict[str, str]:
    """This request's own measurements - never another concurrent stream's."""
    out = {
        "X-Audio-Duration-Sec": f"{stats.audio_ms / 1000.0:.2f}",
        "X-Generation-Ms": f"{stats.gen_ms:.1f}",
    }
    if stats.ttfa_ms is not None:
        out["X-TTFA-Ms"] = f"{stats.ttfa_ms:.1f}"
    if stats.rtf is not None:
        out["X-RTF"] = f"{stats.rtf:.3f}"
    return out


async def _audio_chunks(engine: TTSEngine, pcm_stream, fmt: str):
    """Chunked audio body. Errors mid-stream can only truncate: headers are gone."""
    encoder = audio_fmt.make_encoder(fmt, streaming=True)
    engine.metrics.streams_active += 1
    try:
        async for pcm in pcm_stream():
            chunk = encoder.feed(pcm)
            if chunk:
                yield chunk
        tail = encoder.close()
        if tail:
            yield tail
    except Exception:  # noqa: BLE001
        engine.metrics.errors_total += 1
        log.exception("streaming synthesis failed")
    finally:
        engine.metrics.streams_active -= 1


async def _sse_events(engine: TTSEngine, pcm_stream, fmt: str, stats: StreamStats):
    """OpenAI's SSE speech stream: audio deltas, then a terminal done event."""
    encoder = audio_fmt.make_encoder(fmt, streaming=True)
    engine.metrics.streams_active += 1

    def event(payload: dict) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode()

    try:
        async for pcm in pcm_stream():
            chunk = encoder.feed(pcm)
            if chunk:
                yield event({
                    "type": "speech.audio.delta",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                })
        tail = encoder.close()
        if tail:
            yield event({
                "type": "speech.audio.delta",
                "audio": base64.b64encode(tail).decode("ascii"),
            })
        yield event({
            "type": "speech.audio.done",
            "usage": {
                "input_tokens": 0,
                "output_tokens": stats.tokens,
                "total_tokens": stats.tokens,
            },
            "audio": {
                "duration_ms": round(stats.audio_ms, 1),
                "format": fmt,
                "sample_rate": audio_fmt.SAMPLE_RATE,
            },
            "timings": stats.summary(),
        })
    except Exception as exc:  # noqa: BLE001
        engine.metrics.errors_total += 1
        log.exception("sse synthesis failed")
        yield event({"type": "error", "error": {"message": str(exc), "type": "synthesis_error"}})
    finally:
        engine.metrics.streams_active -= 1
