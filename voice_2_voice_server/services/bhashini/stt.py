"""Bhashini Socket.IO STT Service for Pipecat"""

import asyncio
import os
import traceback
from typing import AsyncGenerator, Optional
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    ErrorFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

try:
    import socketio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("Install with: pip install python-socketio[asyncio_client] aiohttp")
    raise Exception(f"Missing module: {e}")


class BhashiniSTTService(STTService):
    """Bhashini real-time STT using Socket.IO."""

    def __init__(
        self,
        *,
        api_key: str,
        socket_url: str = None,
        service_id: str = "bhashini/ai4bharat/conformer-multilingual-asr",
        language: str = "hi",
        sample_rate: int = 16000,
        response_frequency_secs: float = 1.0,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._socket_url = socket_url or os.getenv("BHASHINI_SOCKET_URL", "wss://dhruva-api.bhashini.gov.in")
        self._service_id = service_id
        self._language = language
        self._response_frequency_secs = response_frequency_secs

        self._sio: Optional[socketio.AsyncClient] = None

        self._is_connected = False
        self._is_ready = False
        self._is_speaking = False
        self._ready_event: Optional[asyncio.Event] = None

        logger.info(
            "BhashiniSTTService initialised | url={} service_id={} language={} sample_rate={}",
            self._socket_url, self._service_id, self._language, sample_rate,
        )
        if not self._api_key:
            logger.error("❌ STT [INIT] BHASHINI_API_KEY is empty or not set — connection will fail")
        else:
            logger.info("✅ STT [INIT] BHASHINI_API_KEY is present (length={})", len(self._api_key))

    def _build_task_sequence(self) -> list:
        task = [{
            "taskType": "asr",
            "config": {
                "serviceId": self._service_id,
                "language": {"sourceLanguage": self._language},
                "samplingRate": self.sample_rate,
                "audioFormat": "wav"
            }
        }]
        logger.debug("STT [CONFIG] Task sequence built: {}", task)
        return task

    async def start(self, frame: StartFrame):
        logger.info("STT [LIFECYCLE] start() called — pipeline StartFrame received")
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        logger.info("STT [LIFECYCLE] stop() called — pipeline EndFrame received")
        await self._send_end_of_stream()
        await self._disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        logger.info("STT [LIFECYCLE] cancel() called — pipeline CancelFrame received")
        await self._disconnect()
        await super().cancel(frame)

    async def _connect(self):
        logger.info("STT [CONNECT] Step 1/5 — Starting connection to Bhashini: {}", self._socket_url)

        self._ready_event = asyncio.Event()
        self._sio = socketio.AsyncClient(reconnection_attempts=5)

        @self._sio.event
        async def connect():
            sid = self._sio.get_sid()
            logger.info("STT [CONNECT] Step 2/5 — WebSocket connected, sid={}", sid)
            self._is_connected = True
            task_seq = self._build_task_sequence()
            logger.info(
                "STT [CONNECT] Step 3/5 — Emitting 'start' event with task sequence and responseFrequency={}s",
                self._response_frequency_secs,
            )
            await self._sio.emit("start", (
                task_seq,
                {"responseFrequencyInSecs": self._response_frequency_secs}
            ))
            logger.debug("STT [CONNECT] 'start' event emitted — waiting for server 'ready' response")

        @self._sio.event
        async def connect_error(data):
            logger.error(
                "STT [CONNECT] ❌ Step 2/5 FAILED — WebSocket connection error from {}\n"
                "  error type : {}\n"
                "  error value: {}\n"
                "  Possible causes: wrong URL, network blocked, or server down\n"
                "  Full traceback:\n{}",
                self._socket_url,
                type(data).__name__,
                data,
                traceback.format_exc(),
            )
            # Unblock _ready_event immediately so the pipeline is not frozen
            self._ready_event.set()
            await self.push_error(ErrorFrame(error=f"Connection error: {data}"))

        @self._sio.on("ready")
        async def on_ready():
            logger.info("STT [CONNECT] Step 4/5 — Server sent 'ready' — STT is now active ✅")
            self._is_ready = True
            self._ready_event.set()

        @self._sio.on("response")
        async def on_response(response, streaming_status):
            logger.debug("STT [RESPONSE] Raw response received, streaming_status={}", streaming_status)
            await self._handle_response(response, streaming_status)

        @self._sio.on("abort")
        async def on_abort(message):
            logger.error(
                "STT [CONNECT] ❌ Step 3/5 FAILED — Server at {} sent 'abort'\n"
                "  abort message : {}\n"
                "  message type  : {}\n"
                "  Most likely cause: BHASHINI_API_KEY is invalid, expired, or inactive.\n"
                "  Fix: Regenerate your API key at https://bhashini.gov.in and update BHASHINI_API_KEY in .env\n"
                "  Full traceback at point of abort:\n{}",
                self._socket_url,
                message,
                type(message).__name__,
                traceback.format_exc(),
            )
            # Unblock _ready_event so pipeline StartFrame is not frozen for 10 seconds
            self._ready_event.set()
            await self.push_error(ErrorFrame(error=f"Aborted: {message}"))

        @self._sio.on("terminate")
        async def on_terminate():
            logger.warning("STT [CONNECT] Server sent 'terminate' — connection closed by server")
            self._is_ready = False
            self._is_connected = False

        @self._sio.event
        async def disconnect():
            logger.info("STT [CONNECT] WebSocket disconnected (is_ready={}, is_connected={})",
                        self._is_ready, self._is_connected)
            self._is_connected = False
            self._is_ready = False

        try:
            logger.debug(
                "STT [CONNECT] Calling sio.connect() | url={} transports=['websocket','polling'] path=/socket.io",
                self._socket_url,
            )
            await self._sio.connect(
                url=self._socket_url,
                transports=["websocket", "polling"],
                socketio_path="/socket.io",
                # auth={"authorization": self._api_key}
                auth={"authorization": "oMowmlxsAdR3TqmLaY7mG_SBYTjGmg_p124n8G6FKA67dkmoDLWBtNElIpQQqxHk"}
            )
            logger.debug("STT [CONNECT] sio.connect() returned — waiting for 'ready' event (timeout=10s)")
            await asyncio.wait_for(self._ready_event.wait(), timeout=10.0)

            if self._is_ready:
                logger.info("STT [CONNECT] Step 5/5 — Connection fully established ✅")
            else:
                logger.error(
                    "STT [CONNECT] Step 5/5 — ready_event was set but is_ready=False "
                    "(abort/connect_error fired — check errors above)"
                )
        except asyncio.TimeoutError:
            tb = traceback.format_exc()
            logger.error(
                "STT [CONNECT] ❌ Timed out after 10s waiting for 'ready' from {}\n"
                "  The server connected but never sent 'ready'. Possible causes:\n"
                "  1. Server received 'start' but rejected it silently\n"
                "  2. Network delay/packet loss between your server and dhruva-api.bhashini.gov.in\n"
                "  3. Wrong service_id configured: '{}'\n"
                "  Full traceback:\n{}",
                self._socket_url,
                self._service_id,
                tb,
            )
            await self.push_error(ErrorFrame(error="Connection timeout"))
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(
                "STT [CONNECT] ❌ Unexpected exception during sio.connect() to {}\n"
                "  exception type : {}\n"
                "  exception value: {}\n"
                "  Full traceback:\n{}",
                self._socket_url,
                type(e).__name__,
                e,
                tb,
            )
            await self.push_error(ErrorFrame(error=str(e)))

    async def _disconnect(self):
        if self._sio:
            logger.info("STT [DISCONNECT] Disconnecting from Bhashini (is_ready={}, is_connected={})",
                        self._is_ready, self._is_connected)
            self._is_ready = False
            self._is_connected = False
            try:
                await self._sio.disconnect()
                logger.debug("STT [DISCONNECT] sio.disconnect() completed")
            except Exception as e:
                logger.warning("STT [DISCONNECT] Error during disconnect: {}", e)
            self._sio = None
        else:
            logger.debug("STT [DISCONNECT] _disconnect called but sio is already None — skipping")

    async def _send_end_of_stream(self):
        """Signal end of speech to server - triggers final transcription."""
        if not self._sio or not self._is_connected:
            logger.debug("STT [EOS] Skipping end-of-stream signal (sio={}, is_connected={})",
                         self._sio is not None, self._is_connected)
            return
        try:
            logger.debug("STT [EOS] Sending end-of-stream finalize signals to Bhashini")
            await self._sio.emit("data", (None, None, True, False))
            await self._sio.emit("data", (None, None, True, True))
            logger.debug("STT [EOS] End-of-stream signals sent")
        except Exception as e:
            logger.warning("STT [EOS] Failed to send end-of-stream signal: {}", e)

    async def _handle_response(self, response: dict, streaming_status: dict):
        """Process transcription response from Bhashini."""
        try:
            is_interim = streaming_status.get("isIntermediateResult", True)
            result_type = "INTERIM" if is_interim else "FINAL"

            pipeline_response = response.get("pipelineResponse", [])
            if not pipeline_response:
                logger.debug("STT [RESPONSE] {} — empty pipelineResponse, skipping", result_type)
                return

            outputs = pipeline_response[0].get("output", [])
            if not outputs:
                logger.debug("STT [RESPONSE] {} — empty outputs array, skipping", result_type)
                return

            if is_interim:
                transcript = outputs[0].get("source", "")
            else:
                transcript = ". ".join(
                    chunk.get("source", "")
                    for chunk in outputs
                    if chunk.get("source", "").strip()
                )

            if not transcript.strip():
                logger.debug("STT [RESPONSE] {} — blank transcript, skipping", result_type)
                return

            if is_interim:
                logger.debug("STT [RESPONSE] INTERIM transcript: '{}'", transcript)
                await self.push_frame(InterimTranscriptionFrame(
                    text=transcript,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                ))
            else:
                logger.info("STT [RESPONSE] ✅ FINAL transcript: '{}'", transcript)
                await self.push_frame(TranscriptionFrame(
                    text=transcript,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                ))

        except Exception as e:
            logger.error("STT [RESPONSE] ❌ Exception while handling response: {} — {}", type(e).__name__, e)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio to Bhashini for transcription."""
        if not self._is_ready:
            logger.debug("STT [AUDIO] Dropping audio chunk ({} bytes) — STT not ready (is_ready=False)", len(audio) if audio else 0)
            yield None
            return
        if not audio:
            logger.debug("STT [AUDIO] Empty audio chunk received — skipping")
            yield None
            return

        BhashiniSTTService._stt_audio_chunk_count += 1
        c = BhashiniSTTService._stt_audio_chunk_count
        if c <= 5 or c % 100 == 0:
            logger.info(
                "🎤 Bhashini STT: sending audio to API chunk_count=%d audio_len=%d bytes",
                c,
                len(audio),
            )
        try:
            # logger.debug("STT [AUDIO] Sending {} bytes of audio to Bhashini", len(audio))
            await self._sio.emit("data", (
                {"audio": [{"audioContent": audio}]},
                {},
                False,  # clear_server_state
                False   # is_stream_inactive
            ))
            # logger.debug("STT [AUDIO] Audio chunk emitted successfully")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(
                "STT [AUDIO] ❌ Failed to send audio chunk to {}\n"
                "  exception type : {}\n"
                "  exception value: {}\n"
                "  Full traceback:\n{}",
                self._socket_url, type(e).__name__, e, tb,
            )
            yield ErrorFrame(error=str(e))
            return

        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handle speaking frames like Deepgram does."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.info("STT [VAD] User started speaking (is_ready={})", self._is_ready)
            self._is_speaking = True

        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info("STT [VAD] User stopped speaking — sending finalize signal (is_ready={})", self._is_ready)
            self._is_speaking = False
            if self._sio and self._is_ready:
                try:
                    await self._sio.emit("data", (None, None, True, False))
                    logger.debug("STT [VAD] Finalize signal sent")
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(
                        "STT [VAD] ❌ Failed to send finalize signal to {}\n"
                        "  exception type : {}\n"
                        "  exception value: {}\n"
                        "  Full traceback:\n{}",
                        self._socket_url, type(e).__name__, e, tb,
                    )
            else:
                logger.warning(
                    "STT [VAD] Could not send finalize signal — sio={}, is_ready={}",
                    self._sio is not None, self._is_ready,
                )

    async def set_language(self, language: str):
        logger.info("STT [CONFIG] Switching language: {} → {}", self._language, language)
        self._language = language
        await self._disconnect()
        await self._connect()

    async def set_model(self, service_id: str):
        logger.info("STT [CONFIG] Switching service_id: {} → {}", self._service_id, service_id)
        self._service_id = service_id
        await self._disconnect()
        await self._connect()

    def can_generate_metrics(self) -> bool:
        return True