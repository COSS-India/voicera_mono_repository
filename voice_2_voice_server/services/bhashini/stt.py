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
        logger.debug(f"Connecting to Bhashini: {self._socket_url}")
        
        self._ready_event = asyncio.Event()
        self._sio = socketio.AsyncClient(reconnection_attempts=5)

        @self._sio.event
        async def connect():
            logger.debug(f"Bhashini socket connected: {self._sio.get_sid()}")
            self._is_connected = True
            await self._sio.emit("start", (
                self._build_task_sequence(),
                {"responseFrequencyInSecs": self._response_frequency_secs}
            ))

        @self._sio.event
        async def connect_error(data):
            logger.error(f"Bhashini connection error: {data}")
            await self.push_error(ErrorFrame(error=f"Connection error: {data}"))

        @self._sio.on("ready")
        async def on_ready():
            logger.debug("Bhashini server ready")
            self._is_ready = True
            self._ready_event.set()

        @self._sio.on("response")
        async def on_response(response, streaming_status):
            await self._handle_response(response, streaming_status)

        @self._sio.on("abort")
        async def on_abort(message):
            logger.warning(f"Bhashini aborted: {message}")
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
            await self._sio.connect(
                url=self._socket_url,
                transports=["websocket", "polling"],
                socketio_path="/socket.io",
                auth={"authorization": self._api_key}
            )
            await asyncio.wait_for(self._ready_event.wait(), timeout=10.0)
            logger.info("Bhashini STT service ready")
        except asyncio.TimeoutError:
            logger.error("Bhashini connection timeout")
            await self.push_error(ErrorFrame(error="Connection timeout"))
        except Exception as e:
            logger.error(f"Bhashini connection failed: {e}")
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
            return
        try:
            # clear_server_state=True tells server speaking stopped
            await self._sio.emit("data", (None, None, True, False))
            await self._sio.emit("data", (None, None, True, True))
            logger.debug("Sent end of stream signal")
        except Exception as e:
            logger.warning(f"End of stream error: {e}")

    async def _handle_response(self, response: dict, streaming_status: dict):
        """Process transcription response from Bhashini."""
        try:
            is_interim = streaming_status.get("isIntermediateResult", True)
            
            pipeline_response = response.get("pipelineResponse", [])
            if not pipeline_response:
                return
            
            outputs = pipeline_response[0].get("output", [])
            if not outputs:
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
                return

            if is_interim:
                logger.debug(f"Bhashini interim: {transcript}")
                await self.push_frame(InterimTranscriptionFrame(
                    text=transcript,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                ))
            else:
                logger.info(f"Bhashini final: {transcript}")
                await self.push_frame(TranscriptionFrame(
                    text=transcript,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                ))

        except Exception as e:
            logger.error(f"Response handling error: {e}")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio to Bhashini for transcription."""
        if not self._is_ready or not audio:
            yield None
            return

        try:
            await self._sio.emit("data", (
                {"audio": [{"audioContent": audio}]},
                {},
                False,  # clear_server_state
                False   # is_stream_inactive
            ))
        except Exception as e:
            logger.error(f"Audio send error: {e}")
            yield ErrorFrame(error=str(e))
            return
        
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handle speaking frames like Deepgram does."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("User started speaking")
            self._is_speaking = True
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("User stopped speaking - sending finalize signal")
            self._is_speaking = False
            # Like Deepgram's finalize() - tell server to flush and send final result
            if self._sio and self._is_ready:
                try:
                    await self._sio.emit("data", (None, None, True, False))
                except Exception as e:
                    logger.error(f"Finalize signal error: {e}")

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