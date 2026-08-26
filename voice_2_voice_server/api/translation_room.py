"""Live translation rooms: one presenter → many per-language listeners.

This module is fully self-contained and does not touch the existing 1:1 voice
pipeline. A ``TranslationRoom`` fans a presenter's speech out to any number of
listeners, translating once per active target language:

    presenter WS ─▶ transport.input → VAD → STT ─▶ source text
                                                       │
                          ┌────────────────────────────┼──────────────┐
                          ▼                             ▼              ▼
                    LangWorker[hi]              LangWorker[ta]   LangWorker[en]
              translate→TTS→fan-out       translate→TTS→fan-out  ...
                          │                             │
                 subscribers[hi] WS…          subscribers[ta] WS…

Cost scales with the number of *distinct active languages*, not listeners:
N listeners on one language share a single translate+TTS stream.

Multi-worker caveat: rooms are process-local. With VOICE_SERVER_NUM_WORKERS > 1
a presenter and a listener can land on different workers and never meet. Run
translation single-worker or route ``/translate/*`` by agent_id/share_token.
"""

import asyncio
import base64
import json
import os
import time

import aiohttp
from typing import Any, Dict, Optional

from loguru import logger
from fastapi import WebSocket

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.clocks.system_clock import SystemClock
from pipecat.audio.utils import create_stream_resampler
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from serializer.vobiz_serializer import VobizFrameSerializer
from .services import (
    ServiceCreationError,
    create_stt_service,
    create_tts_service,
)
from .translation_engines import TranslationEngine, create_translation_engine
from utils.call_management import try_acquire_call_slot, release_call_slot

# Listeners are always our browser client, which streams/plays 16 kHz L16 PCM.
LISTEN_SAMPLE_RATE = 16000
PUBLISH_SAMPLE_RATE = 16000
SESSION_TIMEOUT_SECS = int(os.getenv("TRANSLATION_SESSION_TIMEOUT_SECS", "3600"))
# Bound the per-language backlog so a slow language can't grow memory without limit.
MAX_SEGMENT_BACKLOG = 50
# Fan audio out at a fixed cadence: 20 ms mono int16 @ 16 kHz = 640 bytes/frame.
# A steady frame size keeps every listener's jitter buffer smooth instead of
# forwarding whatever variable-length blob the TTS backend happened to emit.
LISTEN_FRAME_MS = 20
LISTEN_CHUNK_BYTES = int(LISTEN_SAMPLE_RATE * LISTEN_FRAME_MS / 1000) * 2
# Per-listener outbound buffer (~10 s of audio). A slow socket drops its own
# oldest frames instead of stalling the whole language for everyone else.
# Sized in *audio* seconds, but filled in bursts: a language worker hands over a
# whole sentence as fast as TTS yields it, which for a long sentence is several
# seconds of audio in well under a second of wall clock. At 2 s this overflowed on
# ordinary sentences and quietly ate the head of each one.
LISTENER_SEND_BACKLOG = 500
# On overflow, drop a contiguous block rather than one frame per push. Dropping
# single 20 ms frames repeatedly punches a burst of holes through the middle of a
# word; dropping one block is a single clean skip the listener reads as a cut.
LISTENER_DROP_BLOCK = LISTENER_SEND_BACKLOG // 2
# Keep a room (and its language workers) alive briefly after the presenter drops,
# so a transient reconnect doesn't tear down every listener's session.
PRESENTER_GRACE_SECS = int(os.getenv("TRANSLATION_PRESENTER_GRACE_SECS", "20"))
# Silence (seconds) before VAD declares the utterance finished and the pipeline
# (STT → translate → TTS) fires. Tempting to shorten for "real time", but an
# ordinary mid-sentence breath is 200-300 ms: drop below that and clauses split
# in half. A translator handed half a clause cannot reorder it (English SVO into
# Hindi/Tamil SOV needs the whole thought), and each fragment costs its own VAD
# wait + LLM call + TTS call — so over-shortening this makes the pipeline both
# slower and worse. Env-tunable per speaker cadence.
VAD_STOP_SECS = float(os.getenv("TRANSLATION_VAD_STOP_SECS", "0.4"))

# Sentences waiting to be synthesised for this language. Translation of the next
# segment runs while the current one is still being spoken (see LangWorker), so
# this queue is what decouples the two stages; bounded so a slow TTS backend
# back-pressures translation instead of buffering the whole talk.
MAX_SENTENCE_BACKLOG = 8
# A stalled TTS provider must not mute a language for the rest of the broadcast.
# This is an *inactivity* limit (time since the last audio frame), not a
# total-duration limit, so a legitimately long sentence is never cut short. The
# per-engine translation timeouts live with the engines (translation_engines.py).
TTS_STALL_TIMEOUT_SECS = float(os.getenv("TRANSLATION_TTS_STALL_SECS", "20"))


# ---------------------------------------------------------------------------
# Frame processors
# ---------------------------------------------------------------------------

class TranscriptCollector(FrameProcessor):
    """Capture final STT transcripts from the presenter and hand them upstream.

    Some STT providers (Sarvam) decide segment boundaries server-side, so each
    "final" TranscriptionFrame is already a complete thought and translating it
    immediately is correct. Others (AI4Bharat's REST wrapper) segment on a short
    internal silence timer and hand back one TranscriptionFrame per fragment, not
    per sentence -- translating each fragment in isolation produces broken,
    context-free translations (a translator handed half a clause cannot reorder
    it). Buffering fragments and flushing only on the pipeline's own
    UserStoppedSpeakingFrame (real VAD, already wired to the transport)
    re-assembles the full utterance before it is translated; for a provider that
    already emits one final per utterance this is a one-fragment no-op.
    """

    def __init__(self, on_final):
        super().__init__()
        self._on_final = on_final
        self._buffer: list[str] = []

    async def _flush(self) -> None:
        if not self._buffer:
            return
        full_text = " ".join(self._buffer).strip()
        self._buffer.clear()
        if full_text:
            try:
                await self._on_final(full_text)
            except Exception as e:  # never let a handler error kill the pipeline
                logger.warning(f"translation: source-text handler failed: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            text = (getattr(frame, "text", "") or "").strip()
            if text:
                self._buffer.append(text)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._flush()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            # Don't drop a trailing fragment that never got a VAD stop signal.
            # EndFrame covers a clean stop; CancelFrame is what actually flows
            # here on presenter disconnect (PipelineTask.cancel() cancels, it
            # doesn't end, the pipeline) -- without this branch the last
            # buffered-but-unflushed fragment silently vanishes whenever the
            # presenter drops mid-utterance instead of pausing cleanly.
            await self._flush()
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Per-listener writer
# ---------------------------------------------------------------------------

class _Listener:
    """One subscriber socket with its own bounded send queue + writer task.

    Decoupling each socket behind its own queue means a slow listener skips its
    own oldest audio (one gap for them) instead of back-pressuring the TTS
    consumer and stalling every other listener on the language.
    """

    def __init__(self, websocket: WebSocket, label: str = ""):
        self.ws = websocket
        self._label = label
        self._alive = True
        self._queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=LISTENER_SEND_BACKLOG)
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                message = await self._queue.get()
                await self.ws.send_text(message)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # The writer is this socket's only path to audio, so a dead writer is
            # permanent silence. Close the socket so the receive loop unblocks and
            # cleans us up (and the browser client reconnects) instead of leaving
            # the listener staring at a "live" indicator hearing nothing.
            self._alive = False
            logger.warning(f"translation{self._label}: listener writer stopped: {e}")
            try:
                await self.ws.close(code=1011, reason="send failed")
            except Exception:
                pass

    def enqueue(self, message: str) -> None:
        if not self._alive:
            return
        try:
            self._queue.put_nowait(message)
            return
        except asyncio.QueueFull:
            pass
        # This socket is more than LISTENER_SEND_BACKLOG frames behind live speech.
        # Drop one contiguous block so the listener takes a single skip forward,
        # rather than shedding a frame per push and hearing the rest of the talk
        # riddled with 20 ms holes.
        dropped = 0
        for _ in range(LISTENER_DROP_BLOCK):
            try:
                self._queue.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        if dropped:
            logger.warning(
                f"translation{self._label}: listener too slow, skipped {dropped} "
                f"frames (~{dropped * LISTEN_FRAME_MS / 1000:.1f}s of audio)"
            )
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            pass

    async def close(self) -> None:
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# Per-language worker
# ---------------------------------------------------------------------------

class _SegmentTiming:
    """Per-segment stopwatch for one language.

    The lag a listener actually feels is the sum of VAD wait, STT, LLM
    time-to-first-token, TTS time-to-first-byte and their own jitter buffer.
    Without this the only way to tell which stage owns the delay is to guess from
    the audio, so every segment logs its own breakdown once it finishes.
    """

    __slots__ = (
        "chars", "t_recv", "engine", "t_first_token", "t_first_audio",
        "t_last_audio", "audio_bytes", "sentences", "_pending", "_closed",
        "_logged",
    )

    def __init__(self, text: str, t_recv: float, engine: str = "llm"):
        self.chars = len(text)
        self.t_recv = t_recv
        self.engine = engine
        self.t_first_token = 0.0
        self.t_first_audio = 0.0
        self.t_last_audio = 0.0
        self.audio_bytes = 0
        self.sentences = 0
        self._pending = 0
        self._closed = False
        self._logged = False

    def note_token(self) -> None:
        if not self.t_first_token:
            self.t_first_token = time.monotonic()

    def note_sentence(self) -> None:
        self.sentences += 1
        self._pending += 1

    def note_first_audio(self) -> None:
        if not self.t_first_audio:
            self.t_first_audio = time.monotonic()

    def note_sentence_done(self, emitted: int) -> None:
        self._pending -= 1
        if emitted:
            self.audio_bytes += emitted
            self.t_last_audio = time.monotonic()

    def close(self) -> None:
        self._closed = True

    def report_ready(self) -> bool:
        """True exactly once, when translation has ended and all audio is out."""
        if self._logged or not self._closed or self._pending > 0:
            return False
        self._logged = True
        return True

    def summary(self) -> str:
        audio_secs = self.audio_bytes / 2 / LISTEN_SAMPLE_RATE
        ttft = (self.t_first_token - self.t_recv) if self.t_first_token else -1.0
        ttfa = (self.t_first_audio - self.t_recv) if self.t_first_audio else -1.0
        synth_wall = (self.t_last_audio - self.t_first_audio) if self.t_last_audio else 0.0
        # >1 means this language synthesises slower than it is spoken, so every
        # listener's backlog grows for as long as the presenter keeps talking.
        rtf = (synth_wall / audio_secs) if audio_secs > 0 else 0.0
        return (
            f"segment done | {self.chars} chars → {self.sentences} chunk(s) | "
            f"{self.engine}_ttft={ttft:.2f}s tts_ttfa={ttfa:.2f}s | "
            f"audio={audio_secs:.1f}s synth={synth_wall:.1f}s rtf={rtf:.2f}"
        )


class LangWorker:
    """Translate → TTS → fan-out for a single target language.

    Two stages, decoupled by ``_synth_queue``: the translate stage pulls source
    segments and streams sentences out, the synth stage speaks them in order.
    Overlapping them hides one LLM time-to-first-token per segment, which would
    otherwise be dead air on every listener's clock. Ordering is preserved because
    a single synth task drains a FIFO.

    The TTS service is driven directly (``run_tts``) rather than through a
    Pipeline: awaiting the generator gives natural back-pressure, so when
    synthesis can't keep up the bounded queues fill and the source queue drops its
    oldest segment — keeping the stream close to live speech.
    """

    def __init__(self, room: "TranslationRoom", language: str):
        self.room = room
        self.language = language
        self.subscribers: "dict[WebSocket, _Listener]" = {}
        self._queue: "asyncio.Queue[tuple[str, float]]" = asyncio.Queue(
            maxsize=MAX_SEGMENT_BACKLOG
        )
        self._synth_queue: "asyncio.Queue[tuple[str, _SegmentTiming]]" = asyncio.Queue(
            maxsize=MAX_SENTENCE_BACKLOG
        )
        self._tts: Optional[Any] = None
        self._http_session: Optional[Any] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._synth_task: Optional[asyncio.Task] = None
        self._resampler = create_stream_resampler()
        self._slot_acquired = False
        self._start_time = 0.0

    async def start(self) -> bool:
        """Build the TTS service and start consuming. Returns False on capacity/error."""
        # Refuse a language the selected engine can't translate BEFORE taking a
        # call slot, so a listener gets a clear rejection instead of a worker
        # that acquires capacity and then emits silence every segment.
        reason = self.room.engine.unsupported(self.language)
        if reason:
            logger.warning(f"translation[{self.language}]: {reason}")
            return False
        if not await try_acquire_call_slot(self.room.org_id):
            logger.warning(
                f"translation[{self.language}]: capacity/budget exceeded, refusing worker"
            )
            return False
        self._slot_acquired = True
        self._start_time = time.monotonic()
        try:
            tts_config = dict(self.room.tts_config)
            tts_config["language"] = self.language
            # Voices are per-language for providers like AI4Bharat (Hindi speakers
            # are invalid for Tamil), so a translation agent stores one voice per
            # target language and we apply the matching one here.
            voice = self.room.target_voices.get(self.language)
            if voice:
                tts_config["speaker"] = voice
                tts_config["voice_id"] = voice
                # create_tts_service checks args["speaker"] before the top-level
                # key, so override it there too or the presenter's default voice
                # (stored in args by the backend) shadows the per-language one.
                if "args" in tts_config and isinstance(tts_config["args"], dict):
                    tts_config["args"] = dict(tts_config["args"])
                    tts_config["args"]["speaker"] = voice
            else:
                # No per-language voice configured (it's optional). Don't let the
                # agent's base/primary voice leak through as every listener
                # language's speaker -- it's picked for the presenter's own
                # language and provider, and may not even be a valid speaker for
                # this provider/language (e.g. an AI4Bharat-style display name
                # like "Kavitha" isn't in Sarvam's roster, which wants
                # lowercase "kavitha"). Clear it so create_tts_service falls back
                # to its own provider-correct default instead of failing every
                # segment on an invalid inherited speaker.
                tts_config.pop("speaker", None)
                tts_config.pop("voice_id", None)
                if "args" in tts_config and isinstance(tts_config["args"], dict):
                    tts_config["args"] = dict(tts_config["args"])
                    tts_config["args"].pop("speaker", None)
            # Own an aiohttp session so create_tts_service can hand back an HTTP TTS
            # variant whose run_tts yields audio for our direct drain (the streaming
            # services push audio out-of-band and don't fit). Created here, on the
            # loop, so the session binds to the running loop before being used.
            self._http_session = aiohttp.ClientSession()
            # create_tts_service does a blocking integration-key lookup for most
            # providers; keep it off the event loop so many listeners joining at
            # once can't serialise into a stall.
            self._tts = await asyncio.to_thread(
                create_tts_service,
                tts_config,
                LISTEN_SAMPLE_RATE,
                self.room.org_id,
                self._http_session,
            )
            # We drive run_tts directly, outside a pipeline, so the lifecycle frames
            # a pipeline delivers never arrive. Two steps still have to run:
            #   setup()  wires the TaskManager/clock a FrameProcessor needs before
            #            it can create any task (else "TaskManager is still not
            #            initialized").
            #   start()  runs the service's StartFrame handler, which resolves the
            #            output sample rate run_tts stamps onto its audio; skipping
            #            it leaves the rate at 0.
            tm = TaskManager()
            tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
            await self._tts.setup(
                FrameProcessorSetup(clock=SystemClock(), task_manager=tm)
            )
            await self._tts.start(
                StartFrame(audio_out_sample_rate=LISTEN_SAMPLE_RATE)
            )
            self._consumer_task = asyncio.create_task(self._consume())
            self._synth_task = asyncio.create_task(self._run_synth())
            logger.info(f"translation[{self.language}]: worker started")
            return True
        except ServiceCreationError as e:
            logger.error(f"translation[{self.language}]: TTS setup failed: {e}")
            await self._cleanup_on_start_failure()
            return False
        except Exception as e:
            logger.error(f"translation[{self.language}]: worker start failed: {e}")
            await self._cleanup_on_start_failure()
            return False

    async def _cleanup_on_start_failure(self) -> None:
        """Clean up resources allocated during start() when it fails partway through."""
        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception:
                pass
            self._http_session = None
        self._tts = None
        await self._release_slot()

    def enqueue(self, text: str) -> None:
        # Live interpretation: when we fall behind, drop the OLDEST pending
        # segment so the stream stays close to what the presenter is saying now.
        item = (text, time.monotonic())
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except asyncio.QueueFull:
                try:
                    dropped, _ = self._queue.get_nowait()
                    logger.warning(
                        f"translation[{self.language}]: backlog full, dropped oldest "
                        f"segment: {dropped[:40]!r}"
                    )
                except asyncio.QueueEmpty:
                    return

    async def _consume(self) -> None:
        """Translate stage: source segments → sentences on the synth queue.

        Deliberately does not wait for audio. The next segment starts translating
        while the current one is still being spoken, so the LLM's
        time-to-first-token overlaps synthesis instead of adding to it.
        """
        while True:
            text, t_recv = await self._queue.get()
            seg = _SegmentTiming(text, t_recv, self.room.engine.name)
            try:
                first = True
                async for sentence in self.room.engine.stream(
                    text, self.language, on_token=seg.note_token
                ):
                    # Push each sentence's text just before its audio so listeners
                    # see it even if TTS lags; tag the source only on the first
                    # sentence so the original isn't repeated on every chunk.
                    self.broadcast(
                        json.dumps(
                            {
                                "event": "transcript",
                                "role": "assistant",
                                "content": sentence,
                                "source": text if first else None,
                            }
                        )
                    )
                    first = False
                    seg.note_sentence()
                    await self._synth_queue.put((sentence, seg))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # One bad segment must not kill the language for everyone.
                logger.error(f"translation[{self.language}]: segment failed: {e}")
            finally:
                seg.close()
                # A segment that yielded no sentences never reaches the synth
                # stage, so it has to be reported from here.
                if seg.report_ready():
                    logger.info(f"translation[{self.language}]: {seg.summary()}")

    async def _run_synth(self) -> None:
        """Synth stage: speak queued sentences in order, one at a time."""
        while True:
            sentence, seg = await self._synth_queue.get()
            emitted = 0
            try:
                emitted = await self._synthesize(sentence, seg)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"translation[{self.language}]: synthesis failed: {e}")
            finally:
                # Mark where this chunk's audio ends in the listener's own stream.
                # A client that has fallen behind uses these to catch up between
                # sentences instead of cutting itself off mid-word.
                self.broadcast(json.dumps({"event": "audio_boundary"}))
                seg.note_sentence_done(emitted)
                if seg.report_ready():
                    logger.info(f"translation[{self.language}]: {seg.summary()}")

    async def _synthesize(self, text: str, seg: Optional[_SegmentTiming] = None) -> int:
        """Speak one chunk; returns the bytes fanned out.

        Retries once, but only when the failure happened before any audio went
        out: a provider that dies mid-chunk has already had part of the sentence
        spoken, and re-running it would repeat that part. A provider that dies at
        connect used to lose the chunk outright — the listener heard the sentence
        simply missing from the middle of the talk.
        """
        for attempt in range(2):
            emitted, error = await self._synthesize_once(text, seg)
            if error is None:
                return emitted
            if emitted:
                logger.warning(
                    f"translation[{self.language}]: TTS failed after {emitted} bytes "
                    f"({error}); chunk truncated"
                )
                return emitted
            if attempt == 0:
                logger.warning(
                    f"translation[{self.language}]: TTS failed before any audio "
                    f"({error}); retrying once"
                )
                await asyncio.sleep(0.2)
                continue
            logger.error(f"translation[{self.language}]: TTS failed, chunk dropped: {error}")
        return 0

    async def _synthesize_once(
        self, text: str, seg: Optional[_SegmentTiming] = None
    ) -> "tuple[int, Optional[str]]":
        """Stream TTS for one chunk, resampled to 16 kHz and paced into 20 ms frames.

        The pacing sleep between sends is load-bearing, not cosmetic: an HTTP TTS
        provider (Sarvam) hands back a whole sentence's audio in one response, so
        without it every 20 ms chunk for a 10 s sentence gets pushed onto the
        listener's queue and out over the socket in a single burst. The listener's
        own catch-up logic then reads that burst as "10 s ahead of live" and cuts
        nearly all of it, leaving only whatever chunk was scheduled last -- heard
        as just the last word or two of every sentence. Sending at the frame's own
        real-time rate keeps delivery looking like a live stream regardless of how
        the provider returned the audio.

        Returns ``(bytes_sent, error)``; ``error`` is None on a clean finish.
        """
        if self._tts is None:
            return 0, "no TTS service"
        buffer = bytearray()
        emitted = 0
        error: Optional[str] = None
        gen = self._tts.run_tts(text)
        try:
            while True:
                try:
                    # Inactivity bound, not a total-duration bound: a hung provider
                    # would otherwise hold this language silent for its socket
                    # timeout (ten minutes on the on-prem backend) with no recovery.
                    frame = await asyncio.wait_for(
                        gen.__anext__(), TTS_STALL_TIMEOUT_SECS
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    error = f"no audio for {TTS_STALL_TIMEOUT_SECS:.0f}s"
                    break
                if isinstance(frame, ErrorFrame):
                    error = str(frame.error)
                    break
                if not (isinstance(frame, AudioRawFrame) and frame.audio):
                    continue
                pcm = frame.audio
                in_rate = frame.sample_rate
                if in_rate <= 0:
                    # Defensive: a provider that never reported its rate would make
                    # the resampler divide by zero. Skip rather than crash the worker.
                    logger.warning(f"translation[{self.language}]: TTS frame missing sample_rate; dropped")
                    continue
                if in_rate != LISTEN_SAMPLE_RATE:
                    pcm = await self._resampler.resample(pcm, in_rate, LISTEN_SAMPLE_RATE)
                buffer.extend(pcm)
                while len(buffer) >= LISTEN_CHUNK_BYTES:
                    if seg is not None:
                        seg.note_first_audio()
                    self._send_audio(bytes(buffer[:LISTEN_CHUNK_BYTES]))
                    emitted += LISTEN_CHUNK_BYTES
                    del buffer[:LISTEN_CHUNK_BYTES]
                    # Real-time pacing, not just real-time-sized chunks (see
                    # docstring): without this sleep the loop drains as fast as
                    # the event loop allows.
                    await asyncio.sleep(LISTEN_FRAME_MS / 1000)
        finally:
            # Release the provider's socket/session even when we abandoned the
            # generator on a stall.
            try:
                await gen.aclose()
            except Exception:
                pass
        if buffer:
            if seg is not None:
                seg.note_first_audio()
            self._send_audio(bytes(buffer))
            emitted += len(buffer)
        return emitted, error

    def _send_audio(self, pcm: bytes) -> None:
        self.broadcast(
            json.dumps(
                {
                    "event": "playAudio",
                    "media": {
                        "contentType": "audio/x-l16",
                        "sampleRate": LISTEN_SAMPLE_RATE,
                        "payload": base64.b64encode(pcm).decode("utf-8"),
                    },
                }
            )
        )

    def add_subscriber(self, websocket: WebSocket) -> None:
        self.subscribers[websocket] = _Listener(websocket, f"[{self.language}]")

    async def remove_subscriber(self, websocket: WebSocket) -> None:
        listener = self.subscribers.pop(websocket, None)
        if listener is not None:
            await listener.close()

    def broadcast(self, message: str) -> None:
        """Hand one frame to every listener's own send queue (never blocks)."""
        for listener in self.subscribers.values():
            listener.enqueue(message)

    async def stop(self) -> None:
        for task in (self._consumer_task, self._synth_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._consumer_task = None
        self._synth_task = None
        if self._tts is not None:
            # Mirror the pipeline shutdown we bypassed at start: stop() closes the
            # Sarvam socket and cancels its receive/keepalive tasks; cleanup() tears
            # down the TaskManager. Both best-effort — teardown must not raise.
            try:
                await self._tts.stop(EndFrame())
            except Exception:
                pass
            try:
                await self._tts.cleanup()
            except Exception:
                pass
            self._tts = None
        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception:
                pass
            self._http_session = None
        for listener in list(self.subscribers.values()):
            await listener.close()
        self.subscribers.clear()
        await self._release_slot()
        logger.info(f"translation[{self.language}]: worker stopped")

    async def _release_slot(self) -> None:
        if self._slot_acquired:
            self._slot_acquired = False
            # Charge 0 minutes: every language is driven by the SAME presenter
            # speech, so the publisher already accounts for the wall-clock. Charging
            # each language its own duration would bill N× the real talk length and
            # exhaust the daily budget after one multi-language broadcast. The
            # concurrency slot is still released here.
            await release_call_slot(self.room.org_id, 0.0)


# ---------------------------------------------------------------------------
# Room
# ---------------------------------------------------------------------------

class TranslationRoom:
    """State for one translation agent: publisher + per-language workers."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.publisher: Optional[WebSocket] = None
        self.workers: "Dict[str, LangWorker]" = {}
        self.lock = asyncio.Lock()
        self.engine: TranslationEngine
        self._grace_task: Optional[asyncio.Task] = None
        self._derive_from_config(config)

    def _derive_from_config(self, config: Dict[str, Any]) -> None:
        """(Re)derive room fields from a config. Called on create and whenever a
        presenter claims the room, so a room first created by an early listener
        picks up the presenter's freshly fetched config instead of a stale one."""
        self.config = config
        self.org_id = config.get("org_id")
        self.source_language = config.get("source_language") or config.get("language")
        self.stt_config = dict(config.get("stt_model") or {})
        self.tts_config = dict(config.get("tts_model") or {})
        target_voices = config.get("target_voices")
        self.target_voices: Dict[str, str] = (
            {str(k): str(v) for k, v in target_voices.items() if v}
            if isinstance(target_voices, dict)
            else {}
        )
        self.extra_instructions = str(config.get("system_prompt") or "").strip()
        # Exactly one engine per room; the unselected one is never constructed.
        # Default "llm" keeps existing agents (no translation_engine key) intact.
        self.engine = create_translation_engine(
            config.get("translation_engine") or "llm",
            org_id=self.org_id,
            source_language=self.source_language,
            extra_instructions=self.extra_instructions,
        )

    async def on_source_text(self, text: str) -> None:
        """Fan a presenter transcript out to every active language worker."""
        logger.info(f"translation[{self.agent_id}]: source → {text[:120]}")
        for worker in list(self.workers.values()):
            worker.enqueue(text)

    async def announce_presenter_live(self) -> None:
        """Tell already-waiting listeners that the presenter just started."""
        notice = json.dumps({"event": "presenter_live"})
        for worker in list(self.workers.values()):
            worker.broadcast(notice)

    def cancel_grace(self) -> None:
        """Abort a pending teardown because a presenter (re)claimed the room."""
        if self._grace_task and not self._grace_task.done():
            self._grace_task.cancel()
        self._grace_task = None

    async def schedule_end_broadcast(self) -> None:
        """Defer teardown by a grace period so a brief presenter drop doesn't kill
        every listener's session. A reconnecting presenter cancels this."""
        async with self.lock:
            if self._grace_task and not self._grace_task.done():
                return
            self._grace_task = asyncio.create_task(self._grace_then_end())
        # Let waiting listeners know the presenter paused (kept, not closed).
        for worker in list(self.workers.values()):
            worker.broadcast(json.dumps({"event": "status", "presenter_online": False}))

    async def _grace_then_end(self) -> None:
        try:
            await asyncio.sleep(PRESENTER_GRACE_SECS)
        except asyncio.CancelledError:
            return
        if self.publisher is None:
            await self.end_broadcast()
            await _maybe_cleanup_room(self)

    async def end_broadcast(self) -> None:
        """Tear down every language worker once the presenter is gone.

        Without this, workers keep holding one call slot per active language for
        as long as a listener leaves their tab open — starving the org's real
        call concurrency. Listeners are told the session ended and disconnected
        so they can rejoin when the presenter returns.
        """
        async with self.lock:
            workers = list(self.workers.values())
            self.workers.clear()
        notice = json.dumps({"event": "session_ended"})
        for worker in workers:
            sockets = list(worker.subscribers)
            # Stop the per-listener writer tasks first so nothing else is writing
            # these sockets while we send the final notice and close them.
            await worker.stop()
            for ws in sockets:
                try:
                    await ws.send_text(notice)
                    await ws.close(code=1000, reason="broadcast ended")
                except Exception:
                    pass

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ROOMS: "Dict[str, TranslationRoom]" = {}
_ROOMS_LOCK = asyncio.Lock()


async def get_or_create_room(agent_id: str, config: Dict[str, Any]) -> TranslationRoom:
    async with _ROOMS_LOCK:
        room = _ROOMS.get(agent_id)
        if room is None:
            room = TranslationRoom(agent_id, config)
            _ROOMS[agent_id] = room
        return room


async def try_claim_publisher(
    agent_id: str, config: Dict[str, Any], websocket: WebSocket
) -> Optional[TranslationRoom]:
    """Atomically reserve the presenter slot for a room.

    Creating the room and setting ``publisher`` happen under a single lock so
    two presenters can't both pass the "is anyone broadcasting?" check, and so a
    concurrent listener teardown can't pop the room out of ``_ROOMS`` between the
    presenter creating it and claiming it (which would silently split presenter
    and listeners onto two different room objects). Returns None if a presenter
    is already active.
    """
    async with _ROOMS_LOCK:
        room = _ROOMS.get(agent_id)
        if room is None:
            room = TranslationRoom(agent_id, config)
            _ROOMS[agent_id] = room
        if room.publisher is not None:
            return None
        room.publisher = websocket
        room.cancel_grace()  # reconnect within the grace window keeps listeners alive
        room._derive_from_config(config)  # presenter's config wins over any stale one
        return room


async def _maybe_cleanup_room(room: TranslationRoom) -> None:
    async with _ROOMS_LOCK:
        if room.publisher is None and not room.workers:
            _ROOMS.pop(room.agent_id, None)
            logger.info(f"translation[{room.agent_id}]: room torn down")


# ---------------------------------------------------------------------------
# Connection handlers (called from server.py routes; websocket already accepted)
# ---------------------------------------------------------------------------

async def run_publisher(websocket: WebSocket, agent_id: str, config: Dict[str, Any]) -> None:
    """Run the presenter leg: transport.input → VAD → STT → source text bus."""
    room = await try_claim_publisher(agent_id, config, websocket)
    if room is None:
        await websocket.close(code=4409, reason="A presenter is already broadcasting")
        return

    start_time = time.monotonic()
    slot_acquired = False
    try:
        if not await try_acquire_call_slot(room.org_id):
            await websocket.close(code=1013, reason="capacity or budget exceeded")
            return
        slot_acquired = True

        # Pre-flight the selected engine up front: without this every segment
        # would fail mid-broadcast and listeners would just hear silence. The LLM
        # engine resolves its credential (off the event loop); the NMT engine
        # probes model readiness. Neither touches the other's dependencies, so an
        # NMT broadcast needs no OpenAI key and vice-versa.
        engine_error = await room.engine.prepare()
        if engine_error:
            logger.error(
                f"translation[{agent_id}]: {room.engine.name} engine not ready for "
                f"org={room.org_id}: {engine_error}"
            )
            # Close reason is surfaced verbatim in the broadcast dialog and is
            # capped at 123 bytes by the WS spec, so keep it short and specific
            # to the engine rather than echoing the full internal message.
            reason = (
                "NMT translation backend unreachable"
                if room.engine.name == "nmt"
                else "No OpenAI credential configured for translation"
            )
            await websocket.close(code=4402, reason=reason)
            return

        first_message = await websocket.receive_text()
        data = json.loads(first_message)
        if data.get("event") != "start":
            logger.warning(
                f"translation publish: expected 'start', got {data.get('event')}"
            )
            return
        start_info = data.get("start", {})
        stream_sid = start_info.get("streamSid") or start_info.get("streamId") or "publisher"
        call_sid = start_info.get("callSid") or start_info.get("callId") or "publisher"

        serializer = VobizFrameSerializer(
            stream_sid=stream_sid,
            call_sid=call_sid,
            params=VobizFrameSerializer.InputParams(
                vobiz_sample_rate=PUBLISH_SAMPLE_RATE,
                sample_rate=PUBLISH_SAMPLE_RATE,
                auto_hang_up=False,
            ),
        )
        vad_analyzer = SileroVADAnalyzer(
            sample_rate=PUBLISH_SAMPLE_RATE,
            params=VADParams(stop_secs=VAD_STOP_SECS, min_volume=0.5, confidence=0.3, start_secs=0.1),
        )
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=False,  # presenter hears nothing back ("mute bot")
                add_wav_header=False,
                vad_analyzer=vad_analyzer,
                serializer=serializer,
                audio_in_passthrough=True,
                session_timeout=SESSION_TIMEOUT_SECS,
            ),
        )

        stt_config = dict(room.stt_config)
        if room.source_language and not stt_config.get("language"):
            stt_config["language"] = room.source_language
        stt = await asyncio.to_thread(
            create_stt_service,
            stt_config,
            PUBLISH_SAMPLE_RATE,
            vad_analyzer=vad_analyzer,
            org_id=room.org_id,
        )
        collector = TranscriptCollector(room.on_source_text)

        task = PipelineTask(
            Pipeline([transport.input(), stt, collector]),
            params=PipelineParams(allow_interruptions=False, enable_metrics=False),
        )

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_transport, _client):
            logger.info("translation publish: presenter disconnected")
            await task.cancel()

        logger.info(f"translation[{agent_id}]: presenter connected")
        await room.announce_presenter_live()
        await PipelineRunner(handle_sigint=False).run(task)
    except Exception as e:
        logger.error(f"translation publish error: {e}")
    finally:
        # Compare-and-clear: never wipe a publisher that isn't us (defensive; the
        # atomic claim already prevents a second presenter from taking over).
        if room.publisher is websocket:
            room.publisher = None
        if slot_acquired:
            await release_call_slot(room.org_id, time.monotonic() - start_time)
        # Grace period before teardown so a brief presenter reconnect keeps every
        # listener's session (and warm language workers) alive.
        await room.schedule_end_broadcast()


async def run_listener(websocket: WebSocket, room: TranslationRoom, language: str) -> None:
    """Run a listener leg: subscribe to a language group and stream audio out."""
    async with room.lock:
        worker = room.workers.get(language)
        if worker is None:
            worker = LangWorker(room, language)
            if not await worker.start():
                await websocket.close(code=1013, reason="capacity or setup error")
                # This connection may have been what created the room.
                await _maybe_cleanup_room(room)
                return
            room.workers[language] = worker
        worker.add_subscriber(websocket)
    logger.info(f"translation[{room.agent_id}]: listener joined lang={language}")

    # Without this a listener who opens the link before the talk starts sees
    # "connected" and hears silence, which is indistinguishable from a fault.
    try:
        await websocket.send_text(
            json.dumps(
                {
                    "event": "status",
                    "presenter_online": room.publisher is not None,
                }
            )
        )
    except Exception:
        pass

    try:
        # Listener is playback-only; block on receive so we notice disconnects.
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        async with room.lock:
            await worker.remove_subscriber(websocket)
            # Only tear down if THIS worker is still the registered one. A
            # presenter restart (end_broadcast) or reconnect can replace it with
            # a new worker for the same language; popping by key would orphan the
            # replacement (leaking its slot and muting its listeners).
            if not worker.subscribers and room.workers.get(language) is worker:
                await worker.stop()
                room.workers.pop(language, None)
        await _maybe_cleanup_room(room)
