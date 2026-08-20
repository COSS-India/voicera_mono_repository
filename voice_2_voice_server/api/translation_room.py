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

from pipecat.frames.frames import AudioRawFrame, TranscriptionFrame, TTSSpeakFrame, Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
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
from utils.bot_utils import FastPunctuationAggregator
from utils.backend_utils import fetch_integration_key
from utils.call_management import try_acquire_call_slot, release_call_slot

# Listeners are always our browser client, which streams/plays 16 kHz L16 PCM.
LISTEN_SAMPLE_RATE = 16000
PUBLISH_SAMPLE_RATE = 16000
SESSION_TIMEOUT_SECS = int(os.getenv("TRANSLATION_SESSION_TIMEOUT_SECS", "3600"))
# Bound the per-language backlog so a slow language can't grow memory without limit.
MAX_SEGMENT_BACKLOG = 50


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


class FanOutSink(FrameProcessor):
    """Terminal sink: serialise each TTS audio frame and broadcast it."""

    def __init__(self, broadcast):
        super().__init__()
        self._broadcast = broadcast

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame) and frame.audio:
            payload = base64.b64encode(frame.audio).decode("utf-8")
            # The browser honours media.sampleRate, so no server-side resample.
            message = json.dumps(
                {
                    "event": "playAudio",
                    "media": {
                        "contentType": "audio/x-l16",
                        "sampleRate": frame.sample_rate,
                        "payload": payload,
                    },
                }
            )
            await self._broadcast(message)
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Per-language worker
# ---------------------------------------------------------------------------

class LangWorker:
    """Translate → TTS → fan-out for a single target language."""

    def __init__(self, room: "TranslationRoom", language: str):
        self.room = room
        self.language = language
        self.subscribers: "set[WebSocket]" = set()
        self._queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=MAX_SEGMENT_BACKLOG)
        self._pipeline_task: Optional[PipelineTask] = None
        self._runner_task: Optional[asyncio.Task] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._slot_acquired = False
        self._start_time = 0.0

    async def start(self) -> bool:
        """Build the TTS pipeline and start consuming. Returns False on capacity/error."""
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
            tts = create_tts_service(
                tts_config, LISTEN_SAMPLE_RATE, org_id=self.room.org_id
            )
            tts._aggregate_sentences = True
            tts._text_aggregator = FastPunctuationAggregator()

            sink = FanOutSink(self.broadcast)
            self._pipeline_task = PipelineTask(
                Pipeline([tts, sink]),
                params=PipelineParams(
                    allow_interruptions=False,
                    enable_metrics=False,
                    # Providers that don't take an explicit rate fall back to this
                    # (pipecat defaults to 24 kHz); pin it so every language emits
                    # the same rate the listener page expects.
                    audio_out_sample_rate=LISTEN_SAMPLE_RATE,
                ),
            )
            self._runner_task = asyncio.create_task(
                PipelineRunner(handle_sigint=False).run(self._pipeline_task)
            )
            self._runner_task.add_done_callback(self._log_runner_exit)
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

    def _log_runner_exit(self, task: "asyncio.Task") -> None:
        """Surface a dead TTS pipeline instead of letting the language go silent."""
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(f"translation[{self.language}]: TTS pipeline died: {error}")

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
                if self._pipeline_task is not None:
                    await self._pipeline_task.queue_frames([TTSSpeakFrame(translated)])
                await self.broadcast(
                    json.dumps(
                        {
                            "event": "transcript",
                            "role": "assistant",
                            "content": translated,
                            "source": text,
                        }
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # One bad segment must not kill the language for everyone.
                logger.error(f"translation[{self.language}]: segment failed: {e}")

    async def broadcast(self, message: str) -> None:
        """Send one frame to every listener on this language, evicting dead sockets."""
        dead = []
        for ws in list(self.subscribers):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.subscribers.discard(ws)

    async def stop(self) -> None:
        if self._consumer_task is not None:
            self._consumer_task.cancel()
        # Cancel the pipeline first so the runner finishes on its own, then wait
        # for both tasks so a stopped worker leaves nothing running behind it.
        if self._pipeline_task is not None:
            try:
                await self._pipeline_task.cancel()
            except Exception as e:
                logger.debug(f"translation[{self.language}]: pipeline cancel: {e}")
        for task in (self._consumer_task, self._runner_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await self._release_slot()
        logger.info(f"translation[{self.language}]: worker stopped")

    async def _release_slot(self) -> None:
        if self._slot_acquired:
            self._slot_acquired = False
            await release_call_slot(
                self.room.org_id, time.monotonic() - self._start_time
            )


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
            await worker.broadcast(notice)

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
            for ws in list(worker.subscribers):
                try:
                    await ws.send_text(notice)
                    await ws.close(code=1000, reason="broadcast ended")
                except Exception:
                    pass
            worker.subscribers.clear()
            await worker.stop()

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
        try:
            resp = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"translation to {target_language} failed: {e}")
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
        stt = create_stt_service(
            stt_config, PUBLISH_SAMPLE_RATE, vad_analyzer=vad_analyzer, org_id=room.org_id
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
        await room.end_broadcast()
        await _maybe_cleanup_room(room)


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
        worker.subscribers.add(websocket)
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
            worker.subscribers.discard(websocket)
            # Only tear down if THIS worker is still the registered one. A
            # presenter restart (end_broadcast) or reconnect can replace it with
            # a new worker for the same language; popping by key would orphan the
            # replacement (leaking its slot and muting its listeners).
            if not worker.subscribers and room.workers.get(language) is worker:
                await worker.stop()
                room.workers.pop(language, None)
        await _maybe_cleanup_room(room)
