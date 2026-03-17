"""Bhashini Socket.IO STT Service for Pipecat"""

import asyncio
import os
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

    def _build_task_sequence(self) -> list:
        return [{
            "taskType": "asr",
            "config": {
                "serviceId": self._service_id,
                "language": {"sourceLanguage": self._language},
                "samplingRate": self.sample_rate,
                "audioFormat": "wav"
            }
        }]

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await self._send_end_of_stream()
        await self._disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await self._disconnect()
        await super().cancel(frame)

    async def _connect(self):
        task_seq = self._build_task_sequence()
        logger.info(
            "🎤 Bhashini STT: connecting url=%s service_id=%s language=%s sample_rate=%s",
            self._socket_url,
            self._service_id,
            self._language,
            self.sample_rate,
        )
        logger.info("🎤 Bhashini STT: start task_sequence=%s", task_seq)
        self._ready_event = asyncio.Event()
        self._sio = socketio.AsyncClient(reconnection_attempts=5)

        @self._sio.event
        async def connect():
            sid = self._sio.get_sid()
            logger.info("🎤 Bhashini STT: socket connected sid=%s", sid)
            self._is_connected = True
            await self._sio.emit("start", (
                task_seq,
                {"responseFrequencyInSecs": self._response_frequency_secs}
            ))
            logger.info("🎤 Bhashini STT: emitted 'start' (waiting for 'ready')")

        @self._sio.event
        async def connect_error(data):
            logger.error("🎤 Bhashini STT: connection_error: %s", data)
            await self.push_error(ErrorFrame(error=f"Connection error: {data}"))

        @self._sio.on("ready")
        async def on_ready():
            logger.info("🎤 Bhashini STT: server 'ready' received - STT ready for audio")
            self._is_ready = True
            self._ready_event.set()

        @self._sio.on("response")
        async def on_response(response, streaming_status):
            is_interim = streaming_status.get("isIntermediateResult", True)
            logger.info(
                "🎤 Bhashini STT: 'response' received is_interim=%s streaming_status_keys=%s response_keys=%s",
                is_interim,
                list(streaming_status.keys()) if isinstance(streaming_status, dict) else type(streaming_status).__name__,
                list(response.keys()) if isinstance(response, dict) else type(response).__name__,
            )
            await self._handle_response(response, streaming_status)

        @self._sio.on("abort")
        async def on_abort(message):
            logger.warning("🎤 Bhashini STT: abort received: %s", message)
            await self.push_error(ErrorFrame(error=f"Aborted: {message}"))

        @self._sio.on("terminate")
        async def on_terminate():
            logger.info("Bhashini connection terminated by server")
            self._is_ready = False
            self._is_connected = False

        @self._sio.event
        async def disconnect():
            logger.debug("Bhashini disconnected")
            self._is_connected = False
            self._is_ready = False

        try:
            logger.info("🎤 Bhashini STT: connecting (auth key present=%s)", bool(self._api_key))
            await self._sio.connect(
                url=self._socket_url,
                transports=["websocket", "polling"],
                socketio_path="/socket.io",
                auth={"authorization": self._api_key}
            )
            await asyncio.wait_for(self._ready_event.wait(), timeout=10.0)
            logger.info("🎤 Bhashini STT: service ready - connected and received 'ready'")
        except asyncio.TimeoutError:
            logger.error("🎤 Bhashini STT: connection timeout (no 'ready' within 10s)")
            await self.push_error(ErrorFrame(error="Connection timeout"))
        except Exception as e:
            logger.error("🎤 Bhashini STT: connection failed: %s", e)
            await self.push_error(ErrorFrame(error=str(e)))

    async def _disconnect(self):
        if self._sio:
            logger.debug("Disconnecting from Bhashini")
            self._is_ready = False
            self._is_connected = False
            try:
                await self._sio.disconnect()
            except Exception as e:
                logger.warning(f"Disconnect error: {e}")
            self._sio = None

    async def _send_end_of_stream(self):
        """Signal end of speech to server - triggers final transcription."""
        if not self._sio or not self._is_connected:
            logger.debug("🎤 Bhashini STT: _send_end_of_stream skipped - not connected")
            return
        try:
            await self._sio.emit("data", (None, None, True, False))
            await self._sio.emit("data", (None, None, True, True))
            logger.info("🎤 Bhashini STT: sent end-of-stream signal (stop)")
        except Exception as e:
            logger.warning("🎤 Bhashini STT: end of stream error: %s", e)

    async def _handle_response(self, response: dict, streaming_status: dict):
        """Process transcription response from Bhashini."""
        try:
            is_interim = streaming_status.get("isIntermediateResult", True)

            pipeline_response = response.get("pipelineResponse", [])
            if not pipeline_response:
                logger.debug("🎤 Bhashini STT: response has no pipelineResponse")
                return

            outputs = pipeline_response[0].get("output", [])
            if not outputs:
                logger.debug("🎤 Bhashini STT: pipelineResponse has no output")
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
                logger.debug("🎤 Bhashini STT: empty transcript in response")
                return

            if is_interim:
                logger.info("🎤 Bhashini STT: interim transcript len=%d text=%s", len(transcript), transcript[:80])
                await self.push_frame(InterimTranscriptionFrame(
                    text=transcript,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                ))
            else:
                logger.info("🎤 Bhashini STT: final transcript len=%d text=%s", len(transcript), transcript)
                await self.push_frame(TranscriptionFrame(
                    text=transcript,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                ))

        except Exception as e:
            logger.error("🎤 Bhashini STT: response handling error: %s", e)

    _stt_audio_chunk_count = 0  # class-level for trace

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio to Bhashini for transcription."""
        if not self._is_ready or not audio:
            if not self._is_ready:
                logger.debug("🎤 Bhashini STT: run_stt skipped - not ready")
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
            await self._sio.emit("data", (
                {"audio": [{"audioContent": audio}]},
                {},
                False,  # clear_server_state
                False   # is_stream_inactive
            ))
        except Exception as e:
            logger.error("🎤 Bhashini STT: audio send error: %s", e)
            yield ErrorFrame(error=str(e))
            return

        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handle speaking frames like Deepgram does."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.info("🎤 Bhashini STT: UserStartedSpeakingFrame - user started speaking")
            self._is_speaking = True

        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info("🎤 Bhashini STT: UserStoppedSpeakingFrame - sending finalize signal to Bhashini")
            self._is_speaking = False
            # Like Deepgram's finalize() - tell server to flush and send final result
            if self._sio and self._is_ready:
                try:
                    await self._sio.emit("data", (None, None, True, False))
                    logger.info("🎤 Bhashini STT: emitted end-of-stream (clear_server_state=True)")
                except Exception as e:
                    logger.error("🎤 Bhashini STT: finalize signal error: %s", e)

    async def set_language(self, language: str):
        logger.info(f"Switching language to: {language}")
        self._language = language
        await self._disconnect()
        await self._connect()

    async def set_model(self, service_id: str):
        logger.info(f"Switching service to: {service_id}")
        self._service_id = service_id
        await self._disconnect()
        await self._connect()

    def can_generate_metrics(self) -> bool:
        return True