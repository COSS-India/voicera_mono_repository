"""Native endpoints: the roster catalog, and the low-latency WebSocket path.

These sit alongside the OpenAI-compatible surface rather than replacing it.
The catalog endpoints exist because OpenAI's schema has nowhere to advertise 22
languages and 12 speaking styles. The WebSocket exists because it is the fastest
path to first audio and OpenAI has no equivalent - a voice agent that has to
barge in mid-sentence wants a socket, not a request.
"""
from __future__ import annotations

import contextlib
import json
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .. import audio as audio_fmt
from ..codec import SAMPLE_RATE
from ..engine import StreamStats, TTSEngine
from ..voices import Roster
from .deps import get_engine, get_roster

log = logging.getLogger("orpheus.native")

router = APIRouter()
ws_router = APIRouter()


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize, in the language's native script.",
                      examples=["नमस्ते, आज मौसम बहुत अच्छा है।"])
    voice: str = Field(..., description="Speaker name from GET /v1/voices.", examples=["Amit"])
    language: Optional[str] = Field(None, description="Optional: inferred from the speaker name.",
                                    examples=["hi"])
    style: Optional[str] = Field(None, description="Speaking style from GET /v1/styles.", examples=["CONV"])
    max_tokens: Optional[int] = Field(
        None, description="Cap on generated audio tokens (~12.2 ms of audio each).")


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
@router.get("/languages", tags=["catalog"], summary="List supported languages and their speakers")
async def languages(roster: Roster = Depends(get_roster)):
    return roster.catalog()


@router.get("/voices", tags=["catalog"], summary="List speakers, optionally filtered by language")
async def voices(
    language: Optional[str] = Query(None, description="Filter by language code, e.g. 'ta'."),
    roster: Roster = Depends(get_roster),
):
    if language is not None and language not in roster.by_code:
        raise HTTPException(404, f"unknown language '{language}'. See GET /v1/languages.")
    return {
        entry["code"]: {"name": entry["name"], "voices": entry["voices"]}
        for entry in roster.languages
        if language is None or entry["code"] == language
    }


@router.get("/styles", tags=["catalog"], summary="List speaking styles")
async def styles(roster: Roster = Depends(get_roster)):
    return {"styles": roster.styles, "default": roster.default_style}


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
@router.post("/tts", tags=["synthesis"], summary="Synthesize a complete WAV",
             response_class=Response,
             responses={200: {"content": {"audio/wav": {}},
                              "description": "24 kHz mono 16-bit WAV with timing headers."}})
async def tts(
    req: TTSRequest,
    engine: TTSEngine = Depends(get_engine),
    roster: Roster = Depends(get_roster),
):
    try:
        language, voice, style = roster.resolve(req.voice, req.language, req.style)
    except LookupError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        token_ids = engine.preflight(req.text, voice, style)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    stats = StreamStats()
    engine.metrics.requests_total += 1
    pcm = bytearray()
    engine.metrics.streams_active += 1
    try:
        async for chunk in engine.stream_pcm(
            text=req.text, voice=voice, language=language, style=style,
            max_tokens=engine.clamp_max_tokens(req.max_tokens), stats=stats,
            token_ids=token_ids,
        ):
            pcm += chunk
    except Exception as exc:  # noqa: BLE001
        engine.metrics.errors_total += 1
        log.exception("synthesis failed")
        raise HTTPException(500, f"synthesis failed: {exc}") from exc
    finally:
        engine.metrics.streams_active -= 1
    if not pcm:
        engine.metrics.errors_total += 1
        raise HTTPException(500, "no audio produced")

    duration = audio_fmt.pcm_duration_seconds(len(pcm))
    headers = {
        "X-Language": language,
        "X-Voice": voice,
        "X-Audio-Duration-Sec": f"{duration:.2f}",
        "X-Generation-Ms": f"{stats.gen_ms:.1f}",
    }
    if stats.ttfa_ms is not None:
        headers["X-TTFA-Ms"] = f"{stats.ttfa_ms:.1f}"
    if stats.rtf is not None:
        headers["X-RTF"] = f"{stats.rtf:.3f}"
    return Response(content=audio_fmt.wrap_wav(bytes(pcm)), media_type="audio/wav", headers=headers)


@router.get("/tts/stream", tags=["synthesis"],
            summary="Stream a WAV as it is generated (curl- and browser-friendly)")
async def tts_stream(
    text: str = Query(..., description="Text to synthesize."),
    voice: str = Query(..., description="Speaker name."),
    language: Optional[str] = Query(None, description="Optional: inferred from the speaker name."),
    style: Optional[str] = Query(None, description="Speaking style."),
    max_tokens: Optional[int] = Query(None, description="Cap on generated audio tokens."),
    engine: TTSEngine = Depends(get_engine),
    roster: Roster = Depends(get_roster),
):
    """A GET so it can be dropped straight into an `<audio src>` or `curl`.

    The WAV header declares the maximum legal length because the real one is not
    known until generation ends; players read to end-of-stream instead of trusting it.
    """
    try:
        resolved_language, resolved_voice, resolved_style = roster.resolve(voice, language, style)
    except LookupError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        token_ids = engine.preflight(text, resolved_voice, resolved_style)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    stats = StreamStats()
    engine.metrics.requests_total += 1
    clamped = engine.clamp_max_tokens(max_tokens)

    async def body():
        encoder = audio_fmt.make_encoder("wav", streaming=True)
        engine.metrics.streams_active += 1
        try:
            async for pcm in engine.stream_pcm(
                text=text, voice=resolved_voice, language=resolved_language,
                style=resolved_style, max_tokens=clamped, stats=stats,
                token_ids=token_ids,
            ):
                yield encoder.feed(pcm)
            tail = encoder.close()
            if tail:
                yield tail
        except Exception:  # noqa: BLE001
            engine.metrics.errors_total += 1
            log.exception("streaming synthesis failed")
        finally:
            engine.metrics.streams_active -= 1

    return StreamingResponse(
        body(),
        media_type="audio/wav",
        headers={"X-Language": resolved_language, "X-Voice": resolved_voice,
                 "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@ws_router.websocket("/v1/tts/ws")
async def tts_websocket(websocket: WebSocket):
    """Lowest-latency path. Not in the OpenAPI schema - OpenAPI cannot describe WebSockets.

    Protocol: the client sends one JSON request
    ``{"text": ..., "voice": ..., "language"?: ..., "style"?: ..., "max_tokens"?: ...}``.
    The server replies with a JSON ``start`` frame describing the audio format,
    then binary PCM frames (24 kHz mono s16le, one 85.33 ms frame each), then a
    final JSON ``done`` frame carrying this stream's own metrics. Failures arrive
    as ``{"type": "error", ...}``.

    One utterance per connection: the socket is closed once the stream ends, on
    success or failure alike. Clients open a connection per request.
    """
    engine: TTSEngine = websocket.app.state.engine
    roster: Roster = websocket.app.state.roster
    await websocket.accept()

    if not engine.ready:
        with contextlib.suppress(Exception):
            await websocket.send_text(json.dumps({"type": "error", "message": "engine is still loading"}))
            await websocket.close()
        return

    engine.metrics.streams_active += 1
    try:
        request = json.loads(await websocket.receive_text())
        language, voice, style = roster.resolve(
            request["voice"], request.get("language"), request.get("style")
        )
        # Checked before the start frame goes out, so a bad request is one error
        # frame rather than a start frame followed by silence.
        token_ids = engine.preflight(request.get("text", ""), voice, style)
        engine.metrics.requests_total += 1
        stats = StreamStats()

        await websocket.send_text(json.dumps({
            "type": "start", "sample_rate": SAMPLE_RATE, "format": "s16le",
            "channels": 1, "language": language, "voice": voice, "style": style,
        }))
        async for pcm in engine.stream_pcm(
            text=request["text"], voice=voice, language=language, style=style,
            max_tokens=engine.clamp_max_tokens(request.get("max_tokens")), stats=stats,
            token_ids=token_ids,
        ):
            await websocket.send_bytes(pcm)

        metrics = stats.summary()
        metrics["jitter_p99_ms"] = (
            round(float(np.percentile(stats.gaps_ms, 99)), 1) if stats.gaps_ms else 0.0
        )
        await websocket.send_text(json.dumps({"type": "done", "metrics": metrics}))
    except WebSocketDisconnect:
        pass
    except (KeyError, LookupError, ValueError) as exc:
        # Client-side mistakes: malformed JSON (a ValueError), a missing field, an
        # unknown voice, text that cannot fit the context. Not counted as server
        # errors. ValueError also covers everything preflight rejects.
        with contextlib.suppress(Exception):
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
    except Exception as exc:  # noqa: BLE001 - a bad request must not kill the socket
        engine.metrics.errors_total += 1
        log.exception("websocket synthesis failed")
        with contextlib.suppress(Exception):
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
    finally:
        engine.metrics.streams_active -= 1
        with contextlib.suppress(Exception):
            await websocket.close()
