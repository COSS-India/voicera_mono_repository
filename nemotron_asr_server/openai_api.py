"""
OpenAI-compatible transcription API.

Voice-agent stacks speak two shapes: `POST /v1/audio/transcriptions` for whole files,
and a Realtime WebSocket for live audio. Both are implemented here on top of the same
ASRSession the native endpoint uses, so batching and vocabulary slicing apply
identically.

Deliberate divergences from OpenAI, because the model differs:
  * `language` is required. This is a multisoftmax checkpoint whose output layer is
    sliced per language, and no reliable auto-detection exists: scoring slice
    probability mass sits at chance, and scoring each language's prompt picked Urdu for
    Hindi. An unnamed language defaults to ASR_DEFAULT_LANG rather than guessing.
  * Timestamps, logprobs, diarization and translation are not implemented.
"""
import asyncio
import base64
import json
import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from audio_io import decode_file_to_16k, pcm_bytes_to_16k
from session import ASRSession, WIRE_CHUNK_BYTES
from asr_engine import LANGUAGE_PROMPT_MAP, bhili_model, indic_model

router = APIRouter()

MODEL_IDS = {
    "indic-nemotron-600m": False,
    "bhili-nemotron-600m": True,
    # Convenience aliases so callers can point Whisper-shaped code at us unchanged.
    "whisper-1": False,
    "gpt-4o-transcribe": False,
    "gpt-4o-mini-transcribe": False,
}

DEFAULT_LANGUAGE = "hi"


class DeltaTracker:
    """
    Turn a growing, revisable hypothesis into monotonic deltas.

    RNNT does not only append -- it can rewrite the tail of what it already emitted as
    more audio arrives. Streaming protocols have no "unsay that" event, so only the
    prefix that two consecutive hypotheses agree on is treated as settled and emitted.
    The volatile tail is held back and lands in the final event.

    Guarantees the concatenated deltas are always a prefix of the final transcript.
    """

    def __init__(self):
        self._previous = ""
        self._emitted = ""

    @staticmethod
    def _common_prefix(a: str, b: str) -> str:
        limit = min(len(a), len(b))
        i = 0
        while i < limit and a[i] == b[i]:
            i += 1
        return a[:i]

    def push(self, text: str) -> str:
        """Delta to emit for this hypothesis, possibly empty."""
        stable = self._common_prefix(self._previous, text)
        self._previous = text
        if len(stable) > len(self._emitted):
            delta = stable[len(self._emitted):]
            self._emitted = stable
            return delta
        return ""

    def finish(self, final: str) -> str:
        """Whatever of the final transcript has not been emitted yet."""
        if final.startswith(self._emitted):
            tail = final[len(self._emitted):]
        else:
            tail = final
            self._emitted = ""
        self._emitted = final
        return tail


def _language_for(model: str, language: str | None) -> str:
    lang = (language or "").strip().lower() or (
        "bhb" if MODEL_IDS.get(model) else DEFAULT_LANGUAGE
    )
    if lang not in LANGUAGE_PROMPT_MAP:
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "message": f"Unsupported language {lang!r}. Supported: {sorted(LANGUAGE_PROMPT_MAP)}",
                "type": "invalid_request_error", "param": "language"}},
        )
    return lang


def _transcribe_pcm(pcm: bytes, language: str) -> str:
    """Drive one whole utterance through the streaming session and return the text."""
    # One file is one transcript: segmenting it into utterances would change the
    # contract of this endpoint.
    session = ASRSession(f"oai-{uuid.uuid4()}", language, vad=False)
    for off in range(0, len(pcm), WIRE_CHUNK_BYTES):
        session.append_audio(pcm[off:off + WIRE_CHUNK_BYTES])
        session.process_available()
    return session.flush()


@router.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {"object": "list", "data": [
        {"id": mid, "object": "model", "created": now, "owned_by": "ai4bharat"}
        for mid in MODEL_IDS
    ]}


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("indic-nemotron-600m"),
    language: str | None = Form(None),
    response_format: str = Form("json"),
    stream: bool = Form(False),
    prompt: str | None = Form(None),
    temperature: float | None = Form(None),
):
    lang = _language_for(model, language)
    data = await file.read()
    try:
        pcm = await asyncio.to_thread(decode_file_to_16k, data, file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": {
            "message": str(e), "type": "invalid_request_error", "param": "file"}})

    duration = len(pcm) / 2 / 16000

    if stream:
        async def events():
            # The model emits a growing hypothesis; deltas are the new suffix.
            session = ASRSession(f"oai-{uuid.uuid4()}", lang, vad=False)
            tracker = DeltaTracker()
            for off in range(0, len(pcm), WIRE_CHUNK_BYTES):
                session.append_audio(pcm[off:off + WIRE_CHUNK_BYTES])
                for text in await session.aprocess_available():
                    delta = tracker.push(text)
                    if delta:
                        yield "data: " + json.dumps(
                            {"type": "transcript.text.delta", "delta": delta}) + "\n\n"
            final = await session.aflush()
            tail = tracker.finish(final)
            if tail:
                yield "data: " + json.dumps(
                    {"type": "transcript.text.delta", "delta": tail}) + "\n\n"
            yield "data: " + json.dumps({"type": "transcript.text.done", "text": final}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    text = await asyncio.to_thread(_transcribe_pcm, pcm, lang)

    if response_format == "text":
        return JSONResponse(content=text, media_type="text/plain")
    if response_format == "verbose_json":
        return {"task": "transcribe", "language": lang, "duration": round(duration, 3),
                "text": text, "segments": []}
    return {"text": text}


@router.post("/v1/realtime/transcription_sessions")
async def create_transcription_session(payload: dict | None = None):
    payload = payload or {}
    sid = f"sess_{uuid.uuid4().hex[:24]}"
    return {
        "id": sid,
        "object": "realtime.transcription_session",
        "expires_at": int(time.time()) + 3600,
        # No auth on this deployment; the field exists so client SDKs do not choke.
        "client_secret": {"value": f"ek_{uuid.uuid4().hex}", "expires_at": int(time.time()) + 3600},
        "input_audio_format": payload.get("input_audio_format", "pcm16"),
        "input_audio_transcription": payload.get(
            "input_audio_transcription", {"model": "indic-nemotron-600m", "language": DEFAULT_LANGUAGE}),
        "turn_detection": payload.get("turn_detection"),
    }


def _event(kind: str, **fields) -> str:
    return json.dumps({"event_id": f"event_{uuid.uuid4().hex[:20]}", "type": kind, **fields})


@router.websocket("/v1/realtime")
async def realtime(websocket: WebSocket):
    """
    Realtime transcription over WebSocket.

    Audio arrives base64-encoded in `input_audio_buffer.append`. OpenAI's pcm16 is
    24 kHz and G.711 is 8 kHz, so everything is converted to the model's 16 kHz here.
    """
    await websocket.accept()
    session_id = f"sess_{uuid.uuid4().hex[:24]}"
    item_id = f"item_{uuid.uuid4().hex[:20]}"

    audio_format = "pcm16"
    input_rate = 24000
    language = DEFAULT_LANGUAGE
    session: ASRSession | None = None
    tracker = DeltaTracker()

    def start(lang: str) -> ASRSession:
        # Live audio: VAD on, so each pause closes an item the way server_vad does.
        return ASRSession(f"oai-rt-{uuid.uuid4()}", lang, vad=True)

    async def send(kind: str, **fields):
        await websocket.send_text(_event(kind, **fields))

    await send("transcription_session.created", session={
        "id": session_id, "object": "realtime.transcription_session",
        "input_audio_format": audio_format,
        "input_audio_transcription": {"model": "indic-nemotron-600m", "language": language},
        "turn_detection": None,
    })

    try:
        session = start(language)
        while True:
            raw = await websocket.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await send("error", error={"type": "invalid_request_error",
                                           "message": "event was not valid JSON"})
                continue
            kind = event.get("type")

            if kind in ("session.update", "transcription_session.update"):
                cfg = event.get("session", event)
                audio_format = cfg.get("input_audio_format", audio_format)
                input_rate = 8000 if audio_format.startswith("g711") else int(
                    cfg.get("input_audio_sample_rate", 24000))
                tr = cfg.get("input_audio_transcription") or {}
                new_lang = (tr.get("language") or language).strip().lower()
                if new_lang not in LANGUAGE_PROMPT_MAP:
                    await send("error", error={
                        "type": "invalid_request_error", "param": "input_audio_transcription.language",
                        "message": f"Unsupported language {new_lang!r}"})
                    continue
                language = new_lang
                session = start(language)
                tracker = DeltaTracker()
                await send("transcription_session.updated", session={
                    "id": session_id, "object": "realtime.transcription_session",
                    "input_audio_format": audio_format,
                    "input_audio_transcription": {"model": "indic-nemotron-600m", "language": language},
                })

            elif kind == "input_audio_buffer.append":
                chunk = base64.b64decode(event.get("audio") or b"")
                if not chunk:
                    continue
                pcm = pcm_bytes_to_16k(chunk, audio_format, input_rate)
                session.append_audio(pcm)
                partials = await session.aprocess_available()

                # An utterance the VAD closed becomes a completed item, which is what
                # a client using server-side turn detection expects.
                for final in getattr(session, "finals", []):
                    tail = tracker.finish(final)
                    if tail:
                        await send("conversation.item.input_audio_transcription.delta",
                                   item_id=item_id, content_index=0, delta=tail)
                    await send("input_audio_buffer.speech_stopped", item_id=item_id)
                    await send("conversation.item.input_audio_transcription.completed",
                               item_id=item_id, content_index=0, transcript=final)
                    tracker = DeltaTracker()
                    item_id = f"item_{uuid.uuid4().hex[:20]}"

                for text in partials:
                    delta = tracker.push(text)
                    if delta:
                        await send("conversation.item.input_audio_transcription.delta",
                                   item_id=item_id, content_index=0, delta=delta)

            elif kind == "input_audio_buffer.commit":
                final = await session.aflush()
                tail = tracker.finish(final)
                if tail:
                    await send("conversation.item.input_audio_transcription.delta",
                               item_id=item_id, content_index=0, delta=tail)
                await send("input_audio_buffer.committed", item_id=item_id)
                await send("conversation.item.input_audio_transcription.completed",
                           item_id=item_id, content_index=0, transcript=final)
                session = start(language)
                tracker = DeltaTracker()
                item_id = f"item_{uuid.uuid4().hex[:20]}"

            elif kind == "input_audio_buffer.clear":
                session = start(language)
                tracker = DeltaTracker()
                await send("input_audio_buffer.cleared")

            else:
                await send("error", error={"type": "invalid_request_error",
                                           "message": f"unsupported event type {kind!r}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await send("error", error={"type": "server_error", "message": str(e)})
        except Exception:
            pass
