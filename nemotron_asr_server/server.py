import asyncio
import hashlib
import json
import os
import time
import uuid
import wave
import uvloop
import numpy as np
import torch
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.requests import Request
from starlette.websockets import WebSocketState

from fastapi.middleware.cors import CORSMiddleware

from asr_engine import (
    indic_model,
    bhili_model,
    model_lock,
    StepRequest,
    start_scheduler,
    get_scheduler,
    scheduler_stats,
    mel_scheduler_stats,
    prompt_index,
    resolve_model_and_prompt,
    set_prompt,
    get_streaming_params,
    get_initial_cache,
    extract_transcription_text,
    DEVICE,
)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

app = FastAPI(title="AI4Bharat Nemotron Streaming ASR Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # Credentials cannot be combined with a "*" origin, and nothing here uses cookies
    # or auth headers, so the browser would reject the pairing anyway.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from session import (
    WIRE_CHUNK_SAMPLES,
    WIRE_CHUNK_BYTES,
    INACTIVITY_TIMEOUT_SEC,
    MEL_LEFT_CTX_FRAMES,
    MEL_RIGHT_CTX_FRAMES,
    SAMPLE_RATE,
    RATE_CHECK_MIN_AUDIO_SEC,
    RATE_RATIO_MIN,
    RATE_RATIO_MAX,
    DEBUG_CAPTURE,
    CAPTURE_DIR,
    CAPTURE_MAX_SECONDS,
    CAPTURE_MAX_BYTES,
    BUILD_ID,
    sessions,
    ASRSession,
)


async def session_reaper_task():
    """Drop streams abandoned without a clean WebSocket close."""
    while True:
        try:
            await asyncio.sleep(5)
            now = time.time()
            stale = [
                s for s in list(sessions.values())
                if now - s.last_active > INACTIVITY_TIMEOUT_SEC
            ]
            for s in stale:
                sessions.pop(s.session_id, None)
                # Closing the socket is what actually ends the stream; removing the
                # dict entry alone leaves the handler holding a live session.
                if s.websocket is not None:
                    try:
                        await s.websocket.close(code=1000, reason="idle timeout")
                    except Exception:
                        pass
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Reaper] Error in session reaper: {e}")


@app.on_event("startup")
async def on_startup():
    start_scheduler()
    asyncio.create_task(session_reaper_task())
    print("[Server] ASR Server started and session reaper active.")


@app.middleware("http")
async def no_store(request: Request, call_next):
    """
    The page and the AudioWorklet module must never be served from cache.

    A stale index.html (48 kHz AudioContext) paired with a fresh audio-processor.js
    (which no longer resamples) sends 48 kHz samples the server reads as 16 kHz --
    producing fluent-looking nonsense that is easily mistaken for a model fault.
    These files are a few KB on a demo server; caching them buys nothing.
    """
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


from openai_api import router as openai_router
app.include_router(openai_router)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    info = {
        "status": "healthy",
        "device": DEVICE,
        "build_id": BUILD_ID,
        "debug_capture": DEBUG_CAPTURE,
        "active_streams": len(sessions),
        "input_rates": {
            sid: {
                "ratio": round(s.last_rate_ratio, 3) if s.last_rate_ratio else None,
                "client_sample_rate": s.client_sample_rate,
                "samples_dropped": s.samples_dropped,
            }
            for sid, s in list(sessions.items())
        },
        "models_loaded": {
            "multilingual": indic_model is not None,
            "bhili": bhili_model is not None,
        },
    }
    info["batching"] = scheduler_stats()
    info["mel_batching"] = mel_scheduler_stats()
    slicer = getattr(indic_model, "_vocab_slicer", None)
    if slicer is not None:
        info["vocab_slicing"] = slicer.stats()
    if indic_model is not None:
        p = get_streaming_params(indic_model)
        info["streaming"] = {
            "att_context_size": list(indic_model.encoder.att_context_size),
            "chunk_frames": p["chunk_size"],
            "chunk_ms": p["chunk_size"][1] * p["hop_length"] * 1000 // 16000,
            "decoder": indic_model.cur_decoder,
        }
    return info


@app.websocket("/v1/asr/ws")
async def websocket_asr(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    language = websocket.query_params.get("language", "hi")

    try:
        session = ASRSession(session_id, language, websocket)
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(code=1011)
        return

    sessions[session_id] = session

    await websocket.send_json({
        "session_id": session_id,
        "status": "ready",
        "language": session.language_code,
        "model_chunk_ms": session.model_chunk_ms,
        "wire_chunk_ms": WIRE_CHUNK_SAMPLES * 1000 // SAMPLE_RATE,
        "expected_sample_rate": SAMPLE_RATE,
        "build_id": BUILD_ID,
    })

    try:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            # 1. Binary PCM audio stream
            if "bytes" in message and message["bytes"]:
                session.append_audio(message["bytes"])

                # Warn once per session if the audio arriving does not line up with
                # real time -- the signature of a client running at the wrong rate.
                if not session.rate_warned:
                    ratio = session.input_rate_ratio()
                    if ratio is not None and not (RATE_RATIO_MIN <= ratio <= RATE_RATIO_MAX):
                        session.rate_warned = True
                        implied = session.implied_sample_rate(ratio)
                        print(f"[Rate] session {session_id}: audio arriving at "
                              f"{ratio:.2f}x real time -- client is probably running at "
                              f"{implied} Hz, not {SAMPLE_RATE} Hz. Transcripts will be nonsense.")
                        await websocket.send_json({
                            "session_id": session_id,
                            "status": "audio_rate_warning",
                            "ratio": round(ratio, 3),
                            "implied_sample_rate": implied,
                            "expected_sample_rate": SAMPLE_RATE,
                        })

                t_start = time.perf_counter()
                # GPU work off the event loop: a step that overruns real time must not
                # stall the socket that is still feeding it.
                texts = await session.aprocess_available()
                latency_ms = (time.perf_counter() - t_start) * 1000.0

                if session.detected_lang and session.detected_lang != session.language_code:
                    session.language_code = session.detected_lang
                    await websocket.send_json({
                        "session_id": session_id,
                        "status": "language_detected",
                        "language": session.detected_lang,
                    })

                for final in getattr(session, "finals", []):
                    await websocket.send_json({
                        "session_id": session_id, "text": final, "is_final": True,
                        "latency_ms": round(latency_ms, 2), "language": session.language_code,
                        "endpoint": "silence",
                    })

                for text in texts:
                    await websocket.send_json({
                        "session_id": session_id,
                        "text": text,
                        "is_final": False,
                        "latency_ms": round(latency_ms, 2),
                        "language": session.language_code,
                    })

            # 2. JSON Control Messages
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                action = payload.get("action")

                if action == "hello":
                    # The client reports the rate its AudioContext actually got, so a
                    # mismatch is caught before a single audio frame is sent.
                    rate = payload.get("sampleRate")
                    session.client_sample_rate = rate
                    ok = rate is not None and abs(float(rate) - SAMPLE_RATE) < 1.0
                    if not ok:
                        print(f"[Rate] session {session_id}: client reports "
                              f"sampleRate={rate}, expected {SAMPLE_RATE}")
                    await websocket.send_json({
                        "session_id": session_id,
                        "status": "hello_ack" if ok else "sample_rate_mismatch",
                        "client_sample_rate": rate,
                        "expected_sample_rate": SAMPLE_RATE,
                    })

                elif action == "set_language":
                    new_lang = payload.get("language", "hi")
                    try:
                        model, target_lang, is_bhili = resolve_model_and_prompt(new_lang.strip().lower())
                        if model is None:
                            raise RuntimeError(f"no model for {new_lang!r}")
                        # Validate the prompt now rather than failing silently mid-stream.
                        with model_lock:
                            set_prompt(model, target_lang)
                    except Exception as e:
                        await websocket.send_json({
                            "session_id": session_id,
                            "status": "language_rejected",
                            "language": new_lang,
                            "error": str(e),
                        })
                        continue

                    session.language_code = new_lang.strip().lower()
                    session.model, session.target_lang, session.is_bhili = model, target_lang, is_bhili
                    session.reset_state()
                    await websocket.send_json({
                        "session_id": session_id,
                        "status": "language_updated",
                        "language": session.language_code,
                    })

                elif action in ("flush_eos", "eos"):
                    t_start = time.perf_counter()
                    text = await session.aflush()
                    latency_ms = (time.perf_counter() - t_start) * 1000.0
                    await websocket.send_json({
                        "session_id": session_id,
                        "text": text,
                        "is_final": True,
                        "latency_ms": round(latency_ms, 2),
                        "language": session.language_code,
                    })
                    session.reset_state()

                elif action == "reset":
                    session.reset_state()
                    await websocket.send_json({
                        "session_id": session_id,
                        "status": "session_reset",
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Error] Session {session_id}: {e}")
    finally:
        sessions.pop(session_id, None)
        if DEBUG_CAPTURE:
            try:
                session.write_capture()
            except Exception as e:
                print(f"[Capture] failed for {session_id}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1, log_level="info")
