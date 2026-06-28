"""Bhashini Socket.IO pipeline STT service for English (Whisper) via Dhruva."""

from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncGenerator, Awaitable, Callable, Optional

from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

from services.bhashini.stt import VADProcessor

try:
    import socketio
except ModuleNotFoundError as e:
    logger.error("Exception: {}", e)
    logger.error("Install with: pip install python-socketio[asyncio_client]")
    raise Exception(f"Missing module: {e}") from e


DEFAULT_SOCKET_URL = "wss://dhruva-api.bhashini.gov.in"
DEFAULT_SERVICE_ID = "ai4bharat/whisper-medium-en--gpu--t4"
DEFAULT_RESPONSE_FREQUENCY_SECS = 2.0


class BhashiniSocketIOSTTService(STTService):
    """Socket.IO pipeline ASR client for Dhruva Bhashini English streaming transcription."""

    def __init__(
        self,
        *,
        api_key: str = "",
        socket_url: str = "",
        service_id: str = DEFAULT_SERVICE_ID,
        language: str = "en",
        sample_rate: int = 16000,
        input_sample_rate: Optional[int] = None,
        audio_channels: int = 1,
        chunk_ms: int = 200,
        response_frequency_in_secs: float = DEFAULT_RESPONSE_FREQUENCY_SECS,
        suppress_vad_frames: bool = False,
        telemetry_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = (
            api_key.strip() or os.getenv("BHASHINI_API_KEY", "").strip()
        )
        if not self._api_key:
            raise ValueError(
                "BhashiniSocketIOSTTService requires BHASHINI_API_KEY in .env "
                "or an org Integrations key."
            )

        self._socket_url = (
            socket_url.strip()
            or os.getenv("BHASHINI_SOCKET_URL", DEFAULT_SOCKET_URL).strip()
        ).rstrip("/")
        self._service_id = service_id
        self._language = language
        self._sample_rate = sample_rate
        self._input_sample_rate = input_sample_rate or sample_rate
        self._audio_channels = audio_channels
        self._chunk_ms = chunk_ms
        self._response_frequency_in_secs = response_frequency_in_secs
        self._telemetry_callback = telemetry_callback
        self._pre_roll_ms = int(os.getenv("BHASHINI_PREROLL_MS", "400"))
        self._chunk_samples = int(self._input_sample_rate * self._chunk_ms / 1000)
        self._chunk_bytes = self._chunk_samples * self._audio_channels * 2
        self._pre_roll_bytes = max(
            0,
            int(self._input_sample_rate * self._pre_roll_ms / 1000)
            * self._audio_channels
            * 2,
        )
        self._target_sample_rate = 16000

        self._suppress_vad_frames = suppress_vad_frames
        self._resampler = create_stream_resampler()
        self._vad = VADProcessor(chunk_ms=self._chunk_ms)
        self._audio_buffer = bytearray()
        self._pre_roll_buffer = bytearray()
        self._disabled = False

        self._sio: Optional[socketio.AsyncClient] = None
        self._ready_event: Optional[asyncio.Event] = None
        self._send_lock = asyncio.Lock()
        self._is_stream_inactive = True
        self._is_speaking = False
        self._segment_active = False
        self._connected = False
        self._closed = False
        self._latest_transcript_text = ""
        self._speech_started_at: Optional[float] = None
        self._first_transcript_at: Optional[float] = None
        self._segment_started_at: Optional[float] = None

        logger.info(
            "Bhashini Socket.IO STT initialized | socket_url={} service_id={} language={} "
            "input_rate={} target_rate={} chunk_ms={} pre_roll_ms={} suppress_vad_frames={}",
            self._socket_url,
            self._service_id,
            self._language,
            self._input_sample_rate,
            self._target_sample_rate,
            self._chunk_ms,
            self._pre_roll_ms,
            self._suppress_vad_frames,
        )

    def _build_task_sequence(self) -> list:
        return [
            {
                "taskType": "asr",
                "config": {
                    "serviceId": self._service_id,
                    "language": {"sourceLanguage": self._language},
                    "samplingRate": self._target_sample_rate,
                    "audioFormat": "wav",
                    "encoding": None,
                },
            }
        ]

    def _build_streaming_config(self) -> dict:
        return {
            "responseFrequencyInSecs": self._response_frequency_in_secs,
            "responseTaskSequenceDepth": 1,
        }

    def _register_handlers(self, sio: socketio.AsyncClient) -> None:
        task_sequence = self._build_task_sequence()
        streaming_config = self._build_streaming_config()

        @sio.event
        async def connect():
            logger.info("Bhashini Socket.IO connected | sid={}", sio.get_sid())
            await sio.emit("start", data=(task_sequence, streaming_config))

        @sio.event
        async def connect_error(data):
            logger.error("Bhashini Socket.IO connection failed: {}", data)
            self._disabled = True
            if self._ready_event and not self._ready_event.is_set():
                self._ready_event.set()

        @sio.on("ready")
        async def ready():
            self._is_stream_inactive = False
            logger.info("Bhashini Socket.IO server ready to receive audio")
            if self._ready_event:
                self._ready_event.set()

        @sio.on("response")
        async def response(response_data, streaming_status):
            await self._handle_response(response_data, streaming_status or {})

        @sio.on("abort")
        async def abort(message):
            logger.error("Bhashini Socket.IO connection aborted: {}", message)
            self._disabled = True

        @sio.on("terminate")
        async def terminate():
            logger.info("Bhashini Socket.IO server terminated stream")
            await sio.disconnect()

        @sio.event
        async def disconnect():
            self._connected = False
            logger.info("Bhashini Socket.IO disconnected")

    async def _emit_latency_metric(
        self,
        metric: str,
        value_ms: float,
        stage: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        if not self._telemetry_callback:
            return
        payload = {
            "service": "stt",
            "metric": metric,
            "value_ms": round(float(value_ms), 1),
            "stage": stage,
            "details": details or {},
            "timestamp_monotonic": time.monotonic(),
        }
        try:
            await self._telemetry_callback(payload)
        except Exception as exc:
            logger.debug("Bhashini Socket.IO STT telemetry callback failed: {}", exc)

    async def _connect(self) -> bool:
        if self._connected or self._disabled:
            return self._connected

        self._sio = socketio.AsyncClient(reconnection_attempts=5)
        self._register_handlers(self._sio)
        self._ready_event = asyncio.Event()

        logger.info("Connecting to Bhashini Socket.IO at {}", self._socket_url)
        try:
            await self._sio.connect(
                url=self._socket_url,
                transports=["websocket", "polling"],
                socketio_path="/socket.io",
                auth={"authorization": self._api_key},
                wait_timeout=10,
            )
            self._connected = True
            await asyncio.wait_for(self._ready_event.wait(), timeout=15.0)
            if self._disabled:
                return False
            return True
        except Exception as e:
            self._disabled = True
            self._connected = False
            logger.error(
                "Bhashini Socket.IO setup failed; disabling STT for this call: {}",
                e,
            )
            return False

    async def _disconnect(self) -> None:
        self._closed = True
        if self._sio and self._connected:
            try:
                await self._sio.disconnect()
            except Exception:
                pass
        self._sio = None
        self._connected = False
        self._ready_event = None

    async def _handle_response(self, response_data, streaming_status: dict) -> None:
        try:
            if not isinstance(response_data, dict):
                raise TypeError("Expected response to be a dictionary.")

            pipeline_response = response_data.get("pipelineResponse", [])
            if not pipeline_response:
                raise IndexError("pipelineResponse is empty.")

            output_list = pipeline_response[0].get("output", [])
            if not isinstance(output_list, list) or not output_list:
                raise ValueError("Output is missing or malformed.")

            if streaming_status.get("isIntermediateResult"):
                text = str(output_list[0].get("target", "")).strip()
            else:
                text = ". ".join(
                    chunk["target"]
                    for chunk in output_list
                    if chunk.get("target", "").strip()
                ).strip()

            if not text:
                return

            self._latest_transcript_text = text
            now = time.monotonic()
            if self._first_transcript_at is None:
                self._first_transcript_at = now
                if self._speech_started_at is not None:
                    await self._emit_latency_metric(
                        "first_transcript_ms",
                        (now - self._speech_started_at) * 1000.0,
                        stage="first_transcript",
                        details={"text_preview": text[:80]},
                    )

            is_interim = bool(streaming_status.get("isIntermediateResult"))
            if is_interim:
                logger.debug("Bhashini Socket.IO interim transcript: {}", text)
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text=text,
                        user_id=getattr(self, "_user_id", ""),
                        timestamp=time_now_iso8601(),
                    )
                )
            else:
                logger.info("Bhashini Socket.IO final transcript: {}", text)
                if self._speech_started_at is not None:
                    await self._emit_latency_metric(
                        "final_transcript_ms",
                        (now - self._speech_started_at) * 1000.0,
                        stage="final_transcript",
                        details={"text_preview": text[:80]},
                    )
                await self.stop_processing_metrics()
                await self.push_frame(
                    TranscriptionFrame(
                        text=text,
                        user_id=getattr(self, "_user_id", ""),
                        timestamp=time_now_iso8601(),
                    )
                )
        except Exception as e:
            logger.error("Error while processing Bhashini Socket.IO response: {}", e)

    async def _resample_audio(self, audio_chunk: bytes) -> bytes:
        if self._input_sample_rate == self._target_sample_rate:
            return audio_chunk
        return await self._resampler.resample(
            audio_chunk,
            self._input_sample_rate,
            self._target_sample_rate,
        )

    async def _emit_audio(self, audio_chunk: bytes) -> None:
        if not self._sio or not self._connected or self._disabled:
            return

        outgoing = await self._resample_audio(audio_chunk)
        if not outgoing:
            return

        input_data = {"audio": [{"audioContent": outgoing}]}
        clear_server_state = not self._is_speaking
        async with self._send_lock:
            await self._sio.emit(
                "data",
                data=(input_data, {}, clear_server_state, self._is_stream_inactive),
            )

    async def _transmit_end_of_segment(self) -> None:
        if not self._sio or not self._connected:
            return

        clear_server_state = not self._is_speaking
        self._is_stream_inactive = True
        async with self._send_lock:
            await self._sio.emit(
                "data",
                data=(None, None, clear_server_state, self._is_stream_inactive),
            )
            await self._sio.emit(
                "data",
                data=(None, None, clear_server_state, self._is_stream_inactive),
            )

    async def _handle_audio_chunk(self, audio_chunk: bytes, pre_roll_bytes: bytes = b"") -> str:
        state = self._vad.process_chunk(audio_chunk)

        if state == "START":
            if not self._connected:
                if not await self._connect():
                    return "START_FAILED"
            self._is_stream_inactive = False
            self._is_speaking = True
            self._segment_active = True
            self._segment_started_at = time.monotonic()
            self._speech_started_at = time.monotonic()
            self._first_transcript_at = None
            self._latest_transcript_text = ""
            await self.start_processing_metrics()
            if pre_roll_bytes:
                await self._emit_audio(pre_roll_bytes)
            await self._emit_audio(audio_chunk)
            return "START"

        if state == "CONTINUE" and self._segment_active and self._is_speaking:
            await self._emit_audio(audio_chunk)
            return "CONTINUE"

        if state == "STOP":
            self._is_speaking = False
            await self._transmit_end_of_segment()
            self._segment_active = False
            return "STOP"

        return state

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._closed = False
        self._disabled = False
        self._audio_buffer.clear()
        self._pre_roll_buffer.clear()
        self._vad = VADProcessor(chunk_ms=self._chunk_ms)
        self._segment_active = False
        self._is_speaking = False
        self._is_stream_inactive = True
        self._speech_started_at = None
        self._first_transcript_at = None
        self._segment_started_at = None
        self._latest_transcript_text = ""
        logger.info("Bhashini Socket.IO STT service started")

    async def stop(self, frame: EndFrame):
        try:
            if self._segment_active:
                self._is_speaking = False
                await self._transmit_end_of_segment()
        finally:
            await self._disconnect()
            self._audio_buffer.clear()
            self._pre_roll_buffer.clear()
            self._vad = VADProcessor(chunk_ms=self._chunk_ms)
            self._segment_active = False
            self._is_speaking = False
            self._disabled = False
            self._speech_started_at = None
            self._first_transcript_at = None
            self._segment_started_at = None
            await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        try:
            if self._segment_active:
                self._is_speaking = False
                await self._transmit_end_of_segment()
        finally:
            await self._disconnect()
            self._audio_buffer.clear()
            self._pre_roll_buffer.clear()
            self._vad = VADProcessor(chunk_ms=self._chunk_ms)
            self._segment_active = False
            self._is_speaking = False
            self._disabled = False
            self._speech_started_at = None
            self._first_transcript_at = None
            self._segment_started_at = None
            await super().cancel(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not audio or self._disabled:
            return

        self._audio_buffer.extend(audio)

        while len(self._audio_buffer) >= self._chunk_bytes:
            pre_roll_snapshot = bytes(self._pre_roll_buffer)
            chunk = bytes(self._audio_buffer[: self._chunk_bytes])
            del self._audio_buffer[: self._chunk_bytes]
            try:
                vad_state = await self._handle_audio_chunk(chunk, pre_roll_snapshot)
                if not self._suppress_vad_frames:
                    if vad_state == "START":
                        yield UserStartedSpeakingFrame()
                    elif vad_state == "STOP":
                        yield UserStoppedSpeakingFrame()
            except Exception as e:
                logger.error("Bhashini Socket.IO STT processing error: {}", e)
                yield ErrorFrame(f"Bhashini Socket.IO STT processing failed: {e}")
            finally:
                if self._pre_roll_bytes > 0:
                    self._pre_roll_buffer.extend(chunk)
                    if len(self._pre_roll_buffer) > self._pre_roll_bytes:
                        overflow = len(self._pre_roll_buffer) - self._pre_roll_bytes
                        if overflow > 0:
                            del self._pre_roll_buffer[:overflow]
                else:
                    self._pre_roll_buffer.clear()

    async def set_language(self, language: str):
        logger.info("Switching Bhashini Socket.IO language to: {}", language)
        self._language = language

    async def set_model(self, service_id: str):
        logger.info("Switching Bhashini Socket.IO service to: {}", service_id)
        self._service_id = service_id

    def can_generate_metrics(self) -> bool:
        return True
