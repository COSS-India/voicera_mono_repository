"""
FastAPI gateway for `indic-transcribe-core`.

Endpoints
---------
  WS   /v1/asr/ws                  raw PCM16 in, JSON partials/finals out
  POST /v1/audio/transcriptions    whole-file, OpenAI-shaped
  GET  /health                     503 while loading, 200 once restored AND warm
  GET  /metrics                    engine + batcher counters
  GET  /v1/languages               the 25 this checkpoint really supports
  GET  /                           the demo page

One uvicorn worker, always: the engine owns the GPU from a single thread (see batcher.py).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from batcher import BatcherConfig, GpuWorker
from core_engine import (PROMPT_LEN, SAMPLE_RATE, EngineConfig, StreamingEngine,
                         load_supported_languages)
from vad import VadConfig

logging.basicConfig(
    level=os.getenv("CORE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("core.app")

STATIC = Path(__file__).parent / "static"

#: Script per language, for the demo's selector. Wrong language => wrong script with no error,
#: so the UI shows the script the user should expect to see.
SCRIPTS = {
    "as": "Bengali-Assamese", "bhb": "Devanagari", "bho": "Devanagari", "bn": "Bengali",
    "brx": "Devanagari", "doi": "Devanagari", "en": "Latin", "gu": "Gujarati",
    "hi": "Devanagari", "kn": "Kannada", "kok": "Devanagari", "ks": "Perso-Arabic",
    "mai": "Devanagari", "ml": "Malayalam", "mni": "Meetei Mayek", "mr": "Devanagari",
    "ne": "Devanagari", "or": "Odia", "pa": "Gurmukhi", "sa": "Devanagari",
    "sat": "Ol Chiki", "sd": "Perso-Arabic", "ta": "Tamil", "te": "Telugu", "ur": "Perso-Arabic",
}
NAMES = {
    "as": "Assamese", "bhb": "Bhili", "bho": "Bhojpuri", "bn": "Bengali", "brx": "Bodo",
    "doi": "Dogri", "en": "Indian English", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
    "kok": "Konkani", "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam",
    "mni": "Manipuri", "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
    "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil", "te": "Telugu",
    "ur": "Urdu",
}
#: Identification is markedly worse for these; measured top-1 bho 0.047, hi 0.258, mai 0.356,
#: ur 0.490. We never auto-detect, but the demo warns that picking the wrong one of these
#: neighbours is easy and fails silently.
WEAK_LID = {"bho", "hi", "mai", "ur"}
#: 22 scheduled languages of the Eighth Schedule, for grouping the selector.
SCHEDULED = {"as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni",
             "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"}



#: Rotate a stream's decoder state on AUDIO DURATION, not token count.
#:
#: Duration is what actually governs both limits that matter. The audio buffer holds only
#: `left + chunk + right` = 11.44 s at the default geometry, so past that the oldest audio is
#: evicted while the decoder keeps accumulating token context for one ever-longer utterance --
#: cross-attention then pins to the window edge, AlignAtt's commit condition stops being
#: satisfied, and emission stalls until a pause lets attention drift back. That is the
#: "transcribed for ~19 s then stopped, resumed a little on each pause" report. Separately the
#: checkpoint trains at `max_duration: 30`, so one unbounded turn is outside its regime anyway.
#:
#: The earlier token-based trigger (300 tokens ~= minutes of speech) was far too late to help.
SAFE_ROLL_SECS = float(os.getenv("CORE_ROLL_SOFT_SECS", "12") or 12)
HARD_ROLL_SECS = float(os.getenv("CORE_ROLL_HARD_SECS", "20") or 20)
#: Silence long enough to be a word boundary, but far shorter than an endpoint decision.
ROLL_GAP_MS = 250.0
#: Secondary guard, unchanged in purpose: `pred_tokens_ids` is capped at `max_generation_length`
#: (512) and `decoder_mems_list` grows against the decoder's 1024-position limit. Overrunning
#: either is an out-of-bounds device write surfacing as a CUDA illegal memory access, so this
#: stays as a backstop even though duration should always trip first.
HARD_ROLL_TOKENS = 420
#: How much of the outgoing turn's text to carry into a seamless rotation. Must be large enough
#: to cover the audio window that is carried with it: the new decoder sees ~10.4 s of already
#: transcribed audio, and if the carried text covers less of it than the buffer does, the model
#: has audio it has no text for and tries to transcribe backwards. Measured at 24: turn-level
#: warm-up dropped to 0.37 s but the run lost 15% of its words. 0 disables the text carry.
CARRY_TOKENS = int(os.getenv("CORE_CARRY_TOKENS", "24") or 24)
#: Seconds of left context carried with it. 0 = the whole window.
CARRY_SECS = float(os.getenv("CORE_CARRY_SECS", "0") or 0)

#: Seamless rotation: hand the outgoing turn's audio window and text tail to the incoming one, so
#: it need not re-pay time-to-first-partial. OFF, because it was built, measured, and lost.
#:
#: The mechanism works -- a rotated turn committed in 0.37 s against 2.17 s cold -- but end to end
#: it is a regression every way it was configured, on the 43 s longform clip:
#:
#:   off (shipped)          95 commits  98% words  worst gap 2.39 s  tail 1.02 s
#:   on, full 10.4 s audio  58 commits  83% words  worst gap 6.50 s  tail 12.5 s
#:   on, 4 s audio          83 commits  97% words  worst gap 4.31 s  tail 1.07 s
#:   on, 2 s audio          52 commits  75% words  worst gap 3.74 s  tail 15.2 s
#:
#: Carried-token count made no difference at all (24/64/128 gave identical results), so the text
#: tail is not the lever -- the carried AUDIO is. A decoder handed a stretch it has already
#: transcribed stops producing rather than continuing, most plausibly predicting EOS on a context
#: that reads as a finished utterance. Kept behind this flag because it is measured and someone
#: will otherwise try it again.
SEAMLESS_ROTATION = os.getenv("CORE_SEAMLESS_ROTATION", "0") == "1"


def _needs_safety_roll(sess) -> bool:
    secs = sess._audio_secs_fed
    if secs >= HARD_ROLL_SECS or (sess._emitted_len - PROMPT_LEN) >= HARD_ROLL_TOKENS:
        return True
    if secs < SAFE_ROLL_SECS:
        return False
    # Past the soft threshold: take the first brief gap so the cut lands between words.
    return getattr(sess.vad, "_silence_ms", 0.0) >= ROLL_GAP_MS


def _env_float(k: str, d: float) -> float:
    v = os.getenv(k, "").strip()
    return float(v) if v else d


def _env_int(k: str, d: int) -> int:
    v = os.getenv(k, "").strip()
    return int(v) if v else d


def build_engine() -> tuple[StreamingEngine, GpuWorker]:
    token_budget = os.getenv("CORE_TOKEN_BUDGET", "").strip()
    cfg = EngineConfig(
        ckpt_path=os.getenv("CORE_CKPT", "/artifacts/indic_transcribe_core.nemo"),
        hf_dir=os.getenv("CORE_HF_DIR", "/models/core"),
        language=os.getenv("CORE_LANGUAGE", "hi"),
        left_context_secs=_env_float("CORE_LEFT_SECS", 10.0),
        chunk_secs=_env_float("CORE_CHUNK_SECS", 1.0),
        right_context_secs=_env_float("CORE_RIGHT_SECS", 0.5),
        streaming_policy=os.getenv("CORE_POLICY", "alignatt"),
        alignatt_thr=_env_int("CORE_ALIGNATT_THR", 8),
        token_budget=int(token_budget) if token_budget else None,
        compute_dtype=os.getenv("CORE_DTYPE", "bfloat16"),
        use_cuda_events=os.getenv("CORE_CUDA_EVENTS", "1") == "1",
        max_batch=_env_int("CORE_MAX_BATCH", 32),
        max_sessions=_env_int("CORE_MAX_SESSIONS", 8),
        realtime_capacity=_env_int("CORE_REALTIME_CAPACITY", 8),
        vad=VadConfig(
            enabled=os.getenv("CORE_VAD", "1") == "1",
            threshold=_env_float("CORE_VAD_THRESHOLD", 0.5),
            min_silence_ms=_env_int("CORE_VAD_SILENCE_MS", 800),
        ),
    )
    engine = StreamingEngine(cfg)
    worker = GpuWorker(engine, BatcherConfig(
        batch_window_ms=_env_float("CORE_BATCH_WINDOW_MS", 0.0),
        max_batch=cfg.max_batch,
        adaptive=os.getenv("CORE_ADAPTIVE", "0") == "1",
    ))
    return engine, worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine, worker = build_engine()
    app.state.engine = engine
    app.state.worker = worker
    app.state.load_error = None

    def _load():
        try:
            engine.load()
            worker.start()
        except Exception as e:  # surfaced by /health rather than killing the process
            log.exception("model load failed")
            app.state.load_error = repr(e)

    await asyncio.get_running_loop().run_in_executor(None, _load)
    try:
        yield
    finally:
        worker.stop()


app = FastAPI(title="indic-transcribe-core", lifespan=lifespan)


# ------------------------------------------------------------------------------------
# health / metrics / roster
# ------------------------------------------------------------------------------------
@app.get("/health")
def health():
    engine = app.state.engine
    if app.state.load_error:
        return JSONResponse({"status": "error", "error": app.state.load_error}, status_code=503)
    # A dead engine must FAIL its healthcheck. Reporting "ok" while every session errors is worse
    # than crashing: nothing restarts it and nothing alerts. Measured: the service sat healthy for
    # minutes, refusing every connection, after one session poisoned the CUDA context.
    if getattr(engine, "fatal", None):
        return JSONResponse({"status": "fatal", "error": engine.fatal}, status_code=503)
    if not engine.ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "sessions": len(engine.sessions)}


@app.get("/metrics")
def metrics():
    engine = app.state.engine
    if not engine.ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    m = engine.metrics()
    m.update(app.state.worker.metrics())
    return m


@app.post("/admin/batcher")
async def admin_batcher(cfg: dict):
    """Retune the batch-formation window at runtime.

    W is the experiment surface of this project (see batcher.py), and Run C sweeps it across
    six values x four max_batch settings. Restarting the service per cell would mean a 4.9 GB
    model reload and a cold CUDA cache each time -- which would dominate exactly the numbers
    the sweep is trying to measure. So the knobs are live.

    Accepts any of: batch_window_ms, max_batch, adaptive. Returns the resulting config.
    """
    b = app.state.worker.cfg
    for k in ("batch_window_ms", "max_batch", "adaptive"):
        if k in cfg:
            setattr(b, k, type(getattr(b, k))(cfg[k]))
    # max_batch lives in two places; the engine enforces it per tick.
    if "max_batch" in cfg:
        app.state.engine.cfg.max_batch = int(cfg["max_batch"])
    log.info("batcher retuned: %s", b.snapshot())
    return b.snapshot()


@app.post("/admin/reset_stats")
async def admin_reset_stats():
    """Drop accumulated counters so a sweep cell measures only itself.

    Without this every cell inherits the previous cell's percentile samples, and a sweep of
    six windows reports six increasingly-smeared versions of the first one.
    """
    from core_engine import EngineStats

    from batcher import BatcherStats

    app.state.engine.stats = EngineStats()
    app.state.worker.stats = BatcherStats()
    return {"status": "reset"}


@app.get("/v1/languages")
def languages():
    """The roster the demo populates itself from, so the UI cannot drift from the checkpoint.

    Served from the checkpoint's own tokenizer_config.json (25), NOT the wrapper's LANGUAGES
    tuple (27) -- bgc and hne are advertised there but absent from core's vocabulary.
    """
    engine = app.state.engine
    langs = engine.languages if engine.ready else load_supported_languages(
        os.getenv("CORE_HF_DIR", "/models/core"))

    def group(code: str) -> str:
        if code == "en":
            return "Indian English"
        if code in SCHEDULED:
            return "Scheduled languages"
        return "Other"

    return {
        "count": len(langs),
        "modes": ["native"],
        "mode_note": ("Only native script is served. <|itn|> and <|romanized|> exist in the "
                      "vocabulary but are untrained on this checkpoint, so they return fluent "
                      "wrong output instead of an error."),
        "auto_detect": False,
        "auto_detect_note": ("Language identification is not offered. A wrong language yields "
                             "confidently wrong SCRIPT rather than an obvious error."),
        "languages": [
            {
                "code": c,
                "name": NAMES.get(c, c),
                "script": SCRIPTS.get(c, "?"),
                "group": group(c),
                "weak_lid": c in WEAK_LID,
            }
            for c in langs
        ],
    }


@app.get("/")
def index():
    page = STATIC / "demo.html"
    if not page.exists():
        raise HTTPException(404, "demo page not installed")
    return FileResponse(page)


# ------------------------------------------------------------------------------------
# websocket streaming
# ------------------------------------------------------------------------------------
@app.websocket("/v1/asr/ws")
async def ws_asr(ws: WebSocket):
    engine = app.state.engine
    worker = app.state.worker

    lang = ws.query_params.get("language") or engine.cfg.language
    mode = ws.query_params.get("mode", "native")
    # Auto-segmentation is OFF by default: the CLIENT decides when a stream starts and stops.
    #
    # Deciding it from pause length is a policy guess, and a wrong guess is disruptive -- it
    # chops a sentence mid-thought and there is no way for the speaker to override it. Someone
    # collecting their thoughts for a second is not finished talking. Pass endpoint=1 to opt in
    # to pause-based turn segmentation; otherwise a stream runs until the client says `stop`.
    endpoint = ws.query_params.get("endpoint", "0") == "1"

    # Refuse a connection in a way the client can actually read.
    #
    # Closing before `accept()` looks like the tidy thing to do and is a trap: the handshake
    # never completes, so the close code and reason are discarded and the client is told only
    # `HTTP 403`. Measured -- a client refused for being over capacity and one refused for
    # asking for an unsupported language were indistinguishable, and 403 suggests neither.
    #
    # So: accept, send a structured error the client can branch on, then close with the right
    # code. It costs one completed handshake on a request being turned away, which is the
    # price of the caller knowing why.
    async def refuse(code: int, kind: str, message: str, **extra) -> None:
        await ws.accept()
        try:
            await ws.send_json({"type": "error", "error": message, "reason": kind, **extra})
        except Exception:
            pass
        await ws.close(code=code, reason=message[:120])

    if getattr(engine, "fatal", None):
        await refuse(1011, "engine_restarting",
                     "engine hit an unrecoverable CUDA error and is restarting; retry shortly")
        return
    if not engine.ready:
        await refuse(1013, "loading", "model is still loading; retry shortly")
        return
    try:
        engine.validate_request(lang, mode)
    except ValueError as e:
        await refuse(1008, "bad_request", str(e))
        return
    if len(engine.sessions) >= engine.cfg.max_sessions:
        # 1013 "try again later" -- the honest code for a server that is full rather than
        # broken. Both numbers are reported so a client that retries can tell a transient
        # burst from asking for more than this deployment can ever serve in real time.
        await refuse(
            1013, "at_capacity",
            f"at capacity: {engine.cfg.max_sessions} concurrent sessions",
            max_sessions=engine.cfg.max_sessions,
            realtime_capacity=engine.cfg.realtime_capacity,
            sessions_active=len(engine.sessions))
        return

    await ws.accept()
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue = asyncio.Queue()

    # ---- turns -------------------------------------------------------------------------
    #
    # A pause ends a TURN, not the STREAM.
    #
    # An engine session is single-shot: `request_finalize()` sets `_finalized`, after which
    # `ready()` is permanently False and the session can never decode again. Treating the VAD
    # endpoint as the end of the *connection* therefore meant one natural mid-sentence pause
    # killed the whole stream -- the client got a `final`, the socket closed, and the rest of
    # what the user said was never transcribed. That is not streaming.
    #
    # So the endpoint now rolls over: the finished session is finalised and retired, a fresh
    # one takes over immediately, and the socket stays open until the client actually leaves.
    # Each turn gets its own decoder state, which also stops long sessions from accumulating
    # decoder mems without bound.
    turns: dict[int, object] = {}
    started_at: dict[int, float] = {}
    first_commit_logged: set = set()
    state = {"turn": 0, "sess": None}
    committed: list[str] = []

    def make_sink(turn_idx: int):
        def sink(d):
            # Called from the GPU worker thread; hop back to the loop.
            loop.call_soon_threadsafe(outbox.put_nowait, (turn_idx, d))
        return sink

    def start_turn(inherit_from=None):
        # `inherit_from` makes a rotation seamless: the new turn adopts the outgoing turn's audio
        # window and a bounded tail of its text, so it can commit immediately instead of re-paying
        # ~2.4 s of time-to-first-partial. Omitted for the FIRST turn, which has nothing to
        # inherit and legitimately pays the cold start once.
        s = engine.create_session(
            lang=lang,
            inherit_from=inherit_from if SEAMLESS_ROTATION else None,
            carry_tokens=CARRY_TOKENS, carry_secs=CARRY_SECS)
        turns[state["turn"]] = s
        started_at[state["turn"]] = time.monotonic()
        worker.register(s.sid, make_sink(state["turn"]))
        state["sess"] = s
        return s

    def retire(turn_idx: int):
        s = turns.pop(turn_idx, None)
        if s is not None:
            worker.unregister(s.sid)
            engine.close_session(s.sid)

    try:
        start_turn()
    except RuntimeError as e:
        await ws.send_json({"type": "error", "error": str(e)})
        await ws.close(code=1013)
        return

    await ws.send_json({
        "type": "ready",
        "session": state["sess"].sid,
        "language": lang,
        "script": SCRIPTS.get(lang, "?"),
        "sample_rate": SAMPLE_RATE,
        "chunk_secs_effective": engine.chunk_eff,
        "right_secs_effective": engine.right_eff,
        "theoretical_latency_s": round(engine.theoretical_latency, 3),
        "token_budget": engine.token_budget,
        "endpointing": endpoint,
        "continuous": True,
    })

    async def pump_out():
        while True:
            turn_idx, d = await outbox.get()
            is_turn_end = d.is_final
            if is_turn_end:
                if d.full_text.strip():
                    committed.append(d.full_text.strip())
                retire(turn_idx)

            # ATTRIBUTION: how long did this turn take to produce its first word? For turn 0
            # that is time-to-first-partial; for a rotated turn it is the pause the user sees.
            # If they are the same number, the gap IS the cold start and nothing else.
            if turn_idx not in first_commit_logged and (d.text or "").strip():
                first_commit_logged.add(turn_idx)
                t_start = started_at.get(turn_idx)
                if t_start is not None:
                    log.info("turn %d first commit after %.2fs%s", turn_idx,
                             time.monotonic() - t_start,
                             " (cold start)" if turn_idx == 0 else " (rotation gap)")

            live = d.full_text.strip()
            transcript = " ".join(committed if is_turn_end else committed + ([live] if live else []))
            cur = state["sess"]
            await ws.send_json({
                # `turn_final`, not `final`: the turn is done, the stream is not. A client that
                # closes on this message reintroduces exactly the bug this replaced.
                "type": "turn_final" if is_turn_end else "partial",
                "turn": turn_idx,
                "text": d.text,
                "full_text": d.full_text,
                "transcript": transcript,
                "latency_ms": round(d.latency_ms, 1),
                "ttfp_ms": (round((time.monotonic() - cur.t_audio0) * 1000, 1)
                            if cur is not None and cur.t_audio0 else None),
                "n_partials": cur.n_partials if cur is not None else 0,
            })

    async def close_stream():
        """Finalise the current turn and wait briefly for its last words."""
        s = state["sess"]
        if s is not None and not s._pending_finalize:
            s.request_finalize()
        for _ in range(60):                       # up to ~3 s
            if not turns:
                break
            await asyncio.sleep(0.05)
        await ws.send_json({"type": "closed", "transcript": " ".join(committed)})

    out_task = asyncio.create_task(pump_out())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (b := msg.get("bytes")) is not None:
                pcm = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
                sess = state["sess"]
                sess.feed(pcm)
                # VAD endpoint: this turn is over. drop_pending because the tail is silence by
                # definition. Then hand straight over to a new session so the next word is
                # already being buffered -- no gap, no closed socket.
                if endpoint and sess.endpointed and not sess._pending_finalize:
                    sess.request_finalize(drop_pending=True)
                    state["turn"] += 1
                    start_turn(inherit_from=sess)
                elif not sess._pending_finalize and _needs_safety_roll(sess):
                    # SAFETY ROLLOVER — not a policy decision, a survival one.
                    #
                    # A session's decoder state grows with every decode step: pred_tokens_ids is
                    # capped at max_generation_length, and decoder_mems_list is concatenated with
                    # no bound at all against the decoder's 1024-position limit. A stream the user
                    # never stops would eventually run past both, and the failure mode is not a
                    # clean error -- it is an out-of-bounds device write surfacing as a CUDA
                    # illegal memory access from an unrelated kernel. That is exactly the crash
                    # measured at the T5 geometry on long clips.
                    #
                    # So the server rotates decoder state underneath a long stream. This is
                    # invisible: the transcript is cumulative across turns, and we prefer to cut
                    # at a brief silence so it lands between words rather than inside one.
                    sess.request_finalize()
                    state["turn"] += 1
                    start_turn(inherit_from=sess)
                    log.info("state rotation after %.1fs / %d tokens (turn %d -> %d)",
                             sess._audio_secs_fed, sess._emitted_len - PROMPT_LEN,
                             state["turn"] - 1, state["turn"])
            elif (t := msg.get("text")) is not None:
                try:
                    cmd = json.loads(t)
                except json.JSONDecodeError:
                    cmd = {"type": t.strip()}
                kind = cmd.get("type")
                if kind in ("stop", "eof", "close"):
                    await close_stream()
                    break
                if kind in ("finalize", "commit"):
                    # Commit the current turn on demand but KEEP streaming.
                    sess = state["sess"]
                    if sess is not None and not sess._pending_finalize:
                        sess.request_finalize()
                        state["turn"] += 1
                        start_turn(inherit_from=sess)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception:
        log.exception("ws stream failed")
    finally:
        out_task.cancel()
        for idx in list(turns):
            retire(idx)
        try:
            await ws.close()
        except Exception:
            pass


# ------------------------------------------------------------------------------------
# file transcription
# ------------------------------------------------------------------------------------
def decode_upload(raw: bytes) -> np.ndarray:
    """Decode an uploaded file to mono float32 16 kHz.

    No ffmpeg in this image on purpose (8 vCPUs; we never transcode per request), so this is
    soundfile's formats only: WAV/FLAC/OGG. Anything at another rate is rejected rather than
    silently resampled, because resampling parity matters for this model's front end.
    """
    import soundfile as sf

    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise HTTPException(
            415, f"audio is {sr} Hz; this service accepts {SAMPLE_RATE} Hz only "
                 "(no ffmpeg in the image, and resampling parity matters for the front end)")
    return wav


@app.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...),
                         language: str = Form(None),
                         model: str = Form(None),
                         response_format: str = Form("json")):
    engine = app.state.engine
    if not engine.ready:
        raise HTTPException(503, "model still loading")
    lang = language or engine.cfg.language
    try:
        engine.validate_request(lang, "native")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    wav = decode_upload(await file.read())
    sess = engine.create_session(lang=lang)
    done: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    app.state.worker.register(sess.sid, lambda d: loop.call_soon_threadsafe(done.put_nowait, d))
    try:
        sess.feed(wav)
        sess.request_finalize()
        text = ""
        while True:
            d = await asyncio.wait_for(done.get(), timeout=120)
            text = d.full_text or text
            if d.is_final:
                break
    finally:
        app.state.worker.unregister(sess.sid)
        engine.close_session(sess.sid)

    if response_format == "text":
        return JSONResponse(content=text, media_type="text/plain")
    return {"text": text, "language": lang, "model": model or os.getenv("CORE_MODEL_ID",
                                                                        "indic-transcribe-core")}
