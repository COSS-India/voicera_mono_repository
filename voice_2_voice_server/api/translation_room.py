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
from typing import Any, Dict, Optional

from loguru import logger
from fastapi import WebSocket

from pipecat.frames.frames import AudioRawFrame, ErrorFrame, TranscriptionFrame, Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
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
    platform_key_fallback_enabled,
)
from utils.backend_utils import fetch_integration_key
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
# Per-listener outbound buffer (~2 s of audio). A slow socket drops its own
# oldest frames instead of stalling the whole language for everyone else.
LISTENER_SEND_BACKLOG = 100
# Keep a room (and its language workers) alive briefly after the presenter drops,
# so a transient reconnect doesn't tear down every listener's session.
PRESENTER_GRACE_SECS = int(os.getenv("TRANSLATION_PRESENTER_GRACE_SECS", "20"))


def _translation_model() -> str:
    return os.getenv("TRANSLATION_MODEL") or "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Frame processors
# ---------------------------------------------------------------------------

class TranscriptCollector(FrameProcessor):
    """Capture final STT transcripts from the presenter and hand them upstream."""

    def __init__(self, on_final):
        super().__init__()
        self._on_final = on_final

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            text = (getattr(frame, "text", "") or "").strip()
            if text:
                try:
                    await self._on_final(text)
                except Exception as e:  # never let a handler error kill the pipeline
                    logger.warning(f"translation: source-text handler failed: {e}")
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Per-listener writer
# ---------------------------------------------------------------------------

class _Listener:
    """One subscriber socket with its own bounded send queue + writer task.

    Decoupling each socket behind its own queue means a slow listener drops its
    own oldest audio (a brief gap for them) instead of back-pressuring the TTS
    consumer and stalling every other listener on the language.
    """

    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self._queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=LISTENER_SEND_BACKLOG)
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                message = await self._queue.get()
                await self.ws.send_text(message)
        except Exception:
            pass  # socket died; run_listener's receive loop will clean us up

    def enqueue(self, message: str) -> None:
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()  # drop oldest, keep up with live speech
            except asyncio.QueueEmpty:
                pass
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

class LangWorker:
    """Translate → TTS → fan-out for a single target language.

    The TTS service is driven directly (``run_tts``) rather than through a
    Pipeline: awaiting the generator gives natural back-pressure, so when
    synthesis can't keep up the bounded source queue fills and drops its oldest
    segment — keeping the stream close to live speech.
    """

    def __init__(self, room: "TranslationRoom", language: str):
        self.room = room
        self.language = language
        self.subscribers: "dict[WebSocket, _Listener]" = {}
        self._queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=MAX_SEGMENT_BACKLOG)
        self._tts: Optional[Any] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._resampler = create_stream_resampler()
        self._slot_acquired = False
        self._start_time = 0.0

    async def start(self) -> bool:
        """Build the TTS service and start consuming. Returns False on capacity/error."""
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
            # create_tts_service does a blocking integration-key lookup for most
            # providers; keep it off the event loop so many listeners joining at
            # once can't serialise into a stall.
            self._tts = await asyncio.to_thread(
                create_tts_service, tts_config, LISTEN_SAMPLE_RATE, self.room.org_id
            )
            self._consumer_task = asyncio.create_task(self._consume())
            logger.info(f"translation[{self.language}]: worker started")
            return True
        except ServiceCreationError as e:
            logger.error(f"translation[{self.language}]: TTS setup failed: {e}")
            await self._release_slot()
            return False
        except Exception as e:
            logger.error(f"translation[{self.language}]: worker start failed: {e}")
            await self._release_slot()
            return False

    def enqueue(self, text: str) -> None:
        # Live interpretation: when we fall behind, drop the OLDEST pending
        # segment so the stream stays close to what the presenter is saying now.
        while True:
            try:
                self._queue.put_nowait(text)
                return
            except asyncio.QueueFull:
                try:
                    dropped = self._queue.get_nowait()
                    logger.warning(
                        f"translation[{self.language}]: backlog full, dropped oldest "
                        f"segment: {dropped[:40]!r}"
                    )
                except asyncio.QueueEmpty:
                    return

    async def _consume(self) -> None:
        while True:
            text = await self._queue.get()
            try:
                translated = await self.room.translate(text, self.language)
                if not translated:
                    continue
                # Push the text first so listeners see it even if TTS lags.
                self.broadcast(
                    json.dumps(
                        {
                            "event": "transcript",
                            "role": "assistant",
                            "content": translated,
                            "source": text,
                        }
                    )
                )
                await self._synthesize(translated)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # One bad segment must not kill the language for everyone.
                logger.error(f"translation[{self.language}]: segment failed: {e}")

    async def _synthesize(self, text: str) -> None:
        """Stream TTS for one segment, resampled to 16 kHz and paced into 20 ms frames."""
        if self._tts is None:
            return
        buffer = bytearray()
        async for frame in self._tts.run_tts(text):
            if isinstance(frame, ErrorFrame):
                logger.warning(f"translation[{self.language}]: TTS error: {frame.error}")
                return
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
                self._send_audio(bytes(buffer[:LISTEN_CHUNK_BYTES]))
                del buffer[:LISTEN_CHUNK_BYTES]
        if buffer:
            self._send_audio(bytes(buffer))

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
        self.subscribers[websocket] = _Listener(websocket)

    async def remove_subscriber(self, websocket: WebSocket) -> None:
        listener = self.subscribers.pop(websocket, None)
        if listener is not None:
            await listener.close()

    def broadcast(self, message: str) -> None:
        """Hand one frame to every listener's own send queue (never blocks)."""
        for listener in self.subscribers.values():
            listener.enqueue(message)

    async def stop(self) -> None:
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except (asyncio.CancelledError, Exception):
                pass
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
        self._openai = None
        self._model = _translation_model()
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

    async def translate(self, text: str, target_language: str) -> Optional[str]:
        client = self.get_openai_client()
        if client is None:
            logger.error("translation: no OpenAI key available (org or platform)")
            return None
        system = (
            f"You are a translation engine. Translate the user's text from "
            f"{self.source_language} into {target_language}. Output only the "
            f"translation, with no commentary, labels or quotation marks."
        )
        # The agent's own prompt is extra style/domain guidance; the rules above
        # stay authoritative so the output is always just the translation.
        if self.extra_instructions:
            system = f"{system}\n\nAdditional guidance:\n{self.extra_instructions}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        # One retry: a single transient LLM hiccup (429/timeout) would otherwise
        # silently drop a whole segment of speech for this language.
        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                resp = await client.chat.completions.create(
                    model=self._model, messages=messages, temperature=0.2
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_error = e
                if attempt == 0:
                    await asyncio.sleep(0.3)
        logger.warning(f"translation to {target_language} failed: {last_error}")
        return None

    def get_openai_client(self):
        """Resolve (and cache) the translation client.

        Uses blocking integration lookup, so callers should warm this before the
        audio path rather than on the first transcript.
        """
        if self._openai is not None:
            return self._openai
        api_key = None
        if self.org_id:
            api_key = fetch_integration_key(self.org_id, "OpenAI")
        if not api_key and platform_key_fallback_enabled():
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        from openai import AsyncOpenAI

        self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai


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

        # Resolve the translation credential up front: without it every segment
        # would fail mid-broadcast and listeners would just hear silence. The
        # lookup is blocking (sync requests), so keep it off the event loop.
        if await asyncio.to_thread(room.get_openai_client) is None:
            logger.error(
                f"translation[{agent_id}]: no OpenAI key available for org={room.org_id} "
                "(configure an OpenAI Integration or enable ALLOW_PLATFORM_KEY_FALLBACK)"
            )
            await websocket.close(code=4402, reason="translation credential not configured")
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
            params=VADParams(stop_secs=0.4, min_volume=0.5, confidence=0.3, start_secs=0.1),
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
