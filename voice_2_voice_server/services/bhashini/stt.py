"""Bhashini websocket STT service for Pipecat."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Awaitable, Callable, Optional
from urllib.parse import quote

import numpy as np
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.services.openai.llm import OpenAIUserContextAggregator
from pipecat.utils.time import time_now_iso8601

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("Install with: pip install websockets")
    raise Exception(f"Missing module: {e}")


@dataclass
class VADProcessor:
    """Simple energy-based VAD matching the reference Bhashini client.

    Threshold notes
    ---------------
    speech_start_rms  – RMS level above which sound is considered speech onset.
                        Raised from 0.024 → 0.035 so that faint background
                        noise and quiet cough overtones don't arm the detector.
    min_speech_ms     – How long sustained energy must last before we call it
                        speech START.  Raised from 250 → 350 ms: typical coughs
                        and barks are impulsive (< 200 ms onset), so a slightly
                        longer gate filters most of them out before they ever
                        reach the Bhashini WebSocket.
    min_pause_ms      – Kept at 950 ms so genuine speech segments with natural
                        short pauses are not chopped up.

    NOTE: Even when these thresholds are crossed by a non-speech sound, the
    BargeInInterruptionProcessor in bot.py will NOT interrupt the bot unless
    Bhashini actually returns a transcript — providing a second line of defence.
    """

    speech_start_rms: float = 0.035   # raised from 0.024
    speech_end_rms: float = 0.012
    min_speech_ms: int = 350           # raised from 250
    min_pause_ms: int = 400
    chunk_ms: int = 200

    is_speaking: bool = False
    speech_run_ms: int = 0
    silence_run_ms: int = 0

    def process_chunk(self, audio_data: bytes) -> str:
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return "IDLE"

        rms = float(np.sqrt(np.mean(samples**2)))
        meter = int(min(rms / 0.2, 1.0) * 20)
        

        if not self.is_speaking:
            if rms > self.speech_start_rms:
                self.speech_run_ms += self.chunk_ms
                if self.speech_run_ms >= self.min_speech_ms:
                    self.is_speaking = True
                    self.speech_run_ms = 0
                    self.silence_run_ms = 0
                    return "START"
            else:
                self.speech_run_ms = 0
        else:
            if rms < self.speech_end_rms:
                self.silence_run_ms += self.chunk_ms
                if self.silence_run_ms >= self.min_pause_ms:
                    self.is_speaking = False
                    self.silence_run_ms = 0
                    self.speech_run_ms = 0
                    return "STOP"
            else:
                self.silence_run_ms = 0

        return "CONTINUE" if self.is_speaking else "IDLE"


class BhashiniKenpathUserContextAggregator(OpenAIUserContextAggregator):
    """User aggregator for Bhashini STT + Kenpath LLM.

    Pushes the user turn to the LLM as soon as a final Bhashini
    ``TranscriptionFrame`` is received, without waiting for Silero
    ``UserStoppedSpeakingFrame`` or Pipecat's ``aggregation_timeout``.

    Uses the same guards as ``BargeInInterruptionProcessor`` in ``bot.py``:
    Silero must have seen real speech, and the final text must be long enough
    to avoid cough/noise one-character ASR artefacts triggering Vistaar.
    """

    # Keep in sync with BargeInInterruptionProcessor.MIN_TEXT_CHARS in bot.py
    MIN_TEXT_CHARS = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._silero_armed = False

    async def _handle_user_started_speaking(self, frame: UserStartedSpeakingFrame):
        await super()._handle_user_started_speaking(frame)
        self._silero_armed = True

    async def _handle_user_stopped_speaking(self, frame: UserStoppedSpeakingFrame):
        await super()._handle_user_stopped_speaking(frame)

    async def _handle_transcription(self, frame: TranscriptionFrame):
        text = frame.text.strip()
        if not text:
            return

        if len(text) < self.MIN_TEXT_CHARS:
            logger.debug(
                "Bhashini final too short for LLM ({} chars) — skipping: '{}'",
                len(text),
                text,
            )
            await self.reset()
            self._silero_armed = False
            return

        if not self._silero_armed:
            logger.debug(
                "Bhashini final ignored for LLM — Silero did not detect speech: '{}'",
                text[:80],
            )
            await self.reset()
            return

        await super()._handle_transcription(frame)
        if len(self._aggregation) > 0:
            logger.debug(
                "Bhashini final transcript — pushing LLM immediately | text='{}'",
                self._aggregation[:80],
            )
            await self.push_aggregation()
        self._silero_armed = False


class BhashiniSTTService(STTService):
    """WebSocket ASR client for Dhruva Bhashini streaming transcription."""

    def __init__(
        self,
        *,
        api_key: str = "",
        ws_url: str = "wss://dhruva-api.bhashini.gov.in/ws/v1/asr/stream",
        service_id: str = "bhashini/ai4b/indic-conformer/grpc",
        language: str = "hi",
        sample_rate: int = 16000,
        input_sample_rate: Optional[int] = None,
        audio_channels: int = 1,
        chunk_ms: int = 200,
        suppress_vad_frames: bool = False,
        telemetry_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = (
            api_key.strip()
            or os.getenv("BHASHINI_API_KEY", "").strip()

        )
        if not self._api_key:
            raise ValueError(
                "BhashiniSTTService requires BHASHINI_API_KEY in .env "
                "(or NVCF_API_KEY for backward compatibility)."
            )

        self._ws_url = ws_url.rstrip("/")
        self._service_id = service_id
        self._language = language
        self._sample_rate = sample_rate
        self._input_sample_rate = input_sample_rate or sample_rate
        self._audio_channels = audio_channels
        self._chunk_ms = chunk_ms
        self._telemetry_callback = telemetry_callback
        self._pre_roll_ms = int(os.getenv("BHASHINI_PREROLL_MS", "400"))
        self._chunk_samples = int(self._input_sample_rate * self._chunk_ms / 1000)
        self._chunk_bytes = self._chunk_samples * self._audio_channels * 2
        self._pre_roll_bytes = max(0, int(self._input_sample_rate * self._pre_roll_ms / 1000) * self._audio_channels * 2)
        self._target_sample_rate = 16000

        self._suppress_vad_frames = suppress_vad_frames
        self._resampler = create_stream_resampler()
        self._vad = VADProcessor(chunk_ms=self._chunk_ms)
        self._audio_buffer = bytearray()
        self._pre_roll_buffer = bytearray()
        self._disabled = False

        self._websocket = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._send_lock = asyncio.Lock()
        self._final_transcript_event: Optional[asyncio.Event] = None
        self._latest_transcript_text: str = ""
        self._stream_started = False
        self._segment_active = False
        self._closed = False
        self._segment_started_at: Optional[float] = None
        self._segment_ws_opened_at: Optional[float] = None
        self._speech_started_at: Optional[float] = None
        self._first_transcript_at: Optional[float] = None
        self._segment_finalized_at: Optional[float] = None

        logger.info(
            "Bhashini STT initialized | ws_url={} service_id={} language={} input_rate={} target_rate={} chunk_ms={} pre_roll_ms={} suppress_vad_frames={}",
            self._ws_url,
            self._service_id,
            self._language,
            self._input_sample_rate,
            self._target_sample_rate,
            self._chunk_ms,
            self._pre_roll_ms,
            self._suppress_vad_frames,
        )

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
            logger.debug("Bhashini STT telemetry callback failed: {}", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ws_url(self) -> str:
        separator = "&" if "?" in self._ws_url else "?"
        return f"{self._ws_url}{separator}api_key={quote(self._api_key)}"

    def _get_start_config(self) -> dict:
        return {
            "type": "start",
            "controlConfig": {
                "dataTracking": False,
            },
            "language": {"sourceLanguage": self._language},
            "config": {
                "serviceId": self._service_id,
                "language": {"sourceLanguage": self._language},
                "audioFormat": "pcm",
                "encoding": "raw",
                "samplingRate": self._target_sample_rate,
                "transcriptionFormat": {"value": "transcript"},
            },
            "streamingConfig": {
                "chunkDurationMs": self._chunk_ms,
                "interimResults": True,
                "endOfStreamPolicy": "client_signal",
            },
        }

    async def _send_json(self, payload: dict) -> None:
        if not self._websocket:
            raise RuntimeError("Bhashini websocket is not connected")
        async with self._send_lock:
            await self._websocket.send(json.dumps(payload))

    def _pcm16_to_float32_bytes(self, audio_chunk: bytes) -> bytes:
        """Convert PCM16 bytes into Float32 bytes in the range [-1, 1]."""
        if not audio_chunk:
            return b""
        pcm16 = np.frombuffer(audio_chunk, dtype=np.int16)
        float32 = (pcm16.astype(np.float32) / 32768.0).astype(np.float32)
        return float32.tobytes()

    async def _send_audio(self, audio_chunk: bytes) -> None:
        if not self._websocket:
            raise RuntimeError("Bhashini websocket is not connected")

        outgoing = audio_chunk
        if self._input_sample_rate != self._target_sample_rate:
            outgoing = await self._resampler.resample(
                audio_chunk,
                self._input_sample_rate,
                self._target_sample_rate,
            )

        if not outgoing:
            return

        outgoing = self._pcm16_to_float32_bytes(outgoing)
        if not outgoing:
            return

        async with self._send_lock:
            await self._websocket.send(outgoing)

    async def _open_websocket(self) -> bool:
        """Open a websocket connection for the current speech segment."""
        if self._websocket or self._disabled:
            return self._websocket is not None

        ws_uri = self._build_ws_url()
        logger.info("Connecting to Bhashini websocket ASR at {}", ws_uri)
        try:
            self._closed = False
            self._final_transcript_event = asyncio.Event()
            self._latest_transcript_text = ""
            self._first_transcript_at = None
            self._segment_started_at = time.monotonic()
            self._segment_ws_opened_at = None
            self._segment_finalized_at = None
            self._speech_started_at = None
            self._websocket = await websockets.connect(ws_uri, ping_interval=None)
            self._segment_ws_opened_at = time.monotonic()
            await self._emit_latency_metric(
                "ws_open_ms",
                (self._segment_ws_opened_at - self._segment_started_at) * 1000.0,
                stage="ws_open",
            )
            await self._send_json(self._get_start_config())
            self._stream_started = True
            self._receiver_task = asyncio.create_task(self._receive_handler())
            logger.info(
                "Bhashini STT segment stream started | ws_open_latency_ms={:.1f}",
                ((self._segment_ws_opened_at - self._segment_started_at) * 1000.0)
                if self._segment_started_at and self._segment_ws_opened_at
                else -1.0,
            )
            return True
        except Exception as e:
            self._disabled = True
            self._stream_started = False
            self._websocket = None
            logger.error(
                "Bhashini websocket setup failed; disabling STT for this call so the call can proceed: {}",
                e,
            )
            return False

    async def _receive_handler(self) -> None:
        try:
            async for message in self._websocket:
                try:
                    resp = json.loads(message)
                except Exception:
                    logger.debug("Bhashini websocket sent non-JSON message: {}", message)
                    continue

                if resp.get("type") != "transcript":
                    continue

                output = resp.get("output") or []
                if not output:
                    continue

                text = str(output[0].get("source", "")).strip()
                if not text:
                    continue
                self._latest_transcript_text = text
                now = time.monotonic()
                if self._first_transcript_at is None:
                    self._first_transcript_at = now
                    if self._segment_started_at is not None:
                        logger.info(
                            "Bhashini first transcript latency | {:.1f} ms | text='{}'",
                            (now - self._segment_started_at) * 1000.0,
                            text,
                        )
                    if self._speech_started_at is not None:
                        await self._emit_latency_metric(
                            "first_transcript_ms",
                            (now - self._speech_started_at) * 1000.0,
                            stage="first_transcript",
                            details={"text_preview": text[:80]},
                        )

                is_final = bool(resp.get("isFinal", False))
                if is_final:
                    logger.info(
                        "Bhashini final transcript: {} | segment_latency_ms={:.1f}",
                        text,
                        ((now - self._segment_started_at) * 1000.0)
                        if self._segment_started_at
                        else -1.0,
                    )
                    if self._speech_started_at is not None:
                        await self._emit_latency_metric(
                            "final_transcript_ms",
                            (now - self._speech_started_at) * 1000.0,
                            stage="final_transcript",
                            details={"text_preview": text[:80]},
                        )
                    if self._speech_started_at is not None:
                        await self._emit_latency_metric(
                            "segment_duration_ms",
                            (now - self._speech_started_at) * 1000.0,
                            stage="segment_complete",
                        )
                    if self._final_transcript_event:
                        self._final_transcript_event.set()
                    await self.stop_processing_metrics()
                    await self.push_frame(
                        TranscriptionFrame(
                            text=text,
                            user_id=getattr(self, "_user_id", ""),
                            timestamp=time_now_iso8601(),
                        )
                    )
                else:
                    logger.debug("Bhashini interim transcript: {}", text)
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            text=text,
                            user_id=getattr(self, "_user_id", ""),
                            timestamp=time_now_iso8601(),
                        )
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._closed:
                logger.error("Bhashini websocket receive loop failed: {}", e)
                await self.push_frame(ErrorFrame(f"Bhashini receive loop failed: {e}"))

    async def _close_websocket(self) -> None:
        self._closed = True
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except BaseException:
                pass
            self._receiver_task = None

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None

    async def _finalize_segment(self) -> None:
        if self._segment_active and self._websocket:
            self._segment_finalized_at = time.monotonic()
            await self._send_json({"type": "end"})
            self._segment_active = False
            try:
                if self._final_transcript_event:
                    await asyncio.wait_for(self._final_transcript_event.wait(), timeout=0.8)
            except asyncio.TimeoutError:
                if self._latest_transcript_text:
                    word_count = len(self._latest_transcript_text.split())
                    char_count = len(self._latest_transcript_text)
                    if word_count < 2 and char_count < 8:
                        logger.debug(
                            "Bhashini interim too short to promote safely (words={}, chars={}): {}",
                            word_count,
                            char_count,
                            self._latest_transcript_text,
                        )
                        return
                    logger.debug(
                        "Bhashini final transcript timeout; promoting latest interim text: {}",
                        self._latest_transcript_text,
                    )
                    if self._segment_started_at is not None:
                        logger.info(
                            "Bhashini fallback transcript latency | {:.1f} ms | text='{}'",
                            (time.monotonic() - self._segment_started_at) * 1000.0,
                            self._latest_transcript_text,
                        )
                    if self._speech_started_at is not None:
                        await self._emit_latency_metric(
                            "final_transcript_ms",
                            (time.monotonic() - self._speech_started_at) * 1000.0,
                            stage="final_transcript_fallback",
                            details={"text_preview": self._latest_transcript_text[:80]},
                        )
                    await self.stop_processing_metrics()
                    await self.push_frame(
                        TranscriptionFrame(
                            text=self._latest_transcript_text,
                            user_id=getattr(self, "_user_id", ""),
                            timestamp=time_now_iso8601(),
                        )
                    )

    async def _handle_audio_chunk(self, audio_chunk: bytes, pre_roll_bytes: bytes = b"") -> str:
        state = self._vad.process_chunk(audio_chunk)

        if state == "START":
            logger.debug("Bhashini VAD detected speech start")
            if not await self._open_websocket():
                return "START_FAILED"
            self._segment_active = True
            self._speech_started_at = time.monotonic()
            await self.start_processing_metrics()
            if pre_roll_bytes:
                logger.debug("Sending pre-roll buffer to Bhashini | bytes={}", len(pre_roll_bytes))
                await self._send_audio(pre_roll_bytes)
            await self._send_audio(audio_chunk)
            return "START"

        if state == "CONTINUE" and self._segment_active:
            await self._send_audio(audio_chunk)
            return "CONTINUE"

        if state == "STOP":
            logger.debug("Bhashini VAD detected speech stop")
            await self._finalize_segment()
            await self._close_websocket()
            return "STOP"

        return state

    # ------------------------------------------------------------------
    # STTService implementation
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._closed = False
        self._stream_started = False
        self._disabled = False
        self._audio_buffer.clear()
        self._pre_roll_buffer.clear()
        self._vad = VADProcessor(chunk_ms=self._chunk_ms)
        self._segment_active = False
        self._segment_started_at = None
        self._segment_ws_opened_at = None
        self._first_transcript_at = None
        self._segment_finalized_at = None
        logger.info("Bhashini STT service started")

    async def stop(self, frame: EndFrame):
        try:
            if self._websocket and self._stream_started:
                if self._segment_active:
                    await self._finalize_segment()
                else:
                    await self._send_json({"type": "end"})
        finally:
            await self._close_websocket()
            self._audio_buffer.clear()
            self._pre_roll_buffer.clear()
            self._vad = VADProcessor(chunk_ms=self._chunk_ms)
            self._segment_active = False
            self._stream_started = False
            self._disabled = False
            self._segment_started_at = None
            self._segment_ws_opened_at = None
            self._speech_started_at = None
            self._first_transcript_at = None
            self._segment_finalized_at = None
            await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        try:
            if self._websocket and self._stream_started:
                if self._segment_active:
                    await self._finalize_segment()
                else:
                    await self._send_json({"type": "end"})
        finally:
            await self._close_websocket()
            self._audio_buffer.clear()
            self._pre_roll_buffer.clear()
            self._vad = VADProcessor(chunk_ms=self._chunk_ms)
            self._segment_active = False
            self._stream_started = False
            self._disabled = False
            self._segment_started_at = None
            self._segment_ws_opened_at = None
            self._speech_started_at = None
            self._first_transcript_at = None
            self._segment_finalized_at = None
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
                # When suppress_vad_frames=True the transport's SileroVADAnalyzer
                # emits UserStarted/StoppedSpeakingFrame — we must not duplicate them.
                # Bhashini's internal VAD still controls WS open/close timing.
                if not self._suppress_vad_frames:
                    if vad_state == "START":
                        yield UserStartedSpeakingFrame()
                    elif vad_state == "STOP":
                        yield UserStoppedSpeakingFrame()
            except Exception as e:
                if "received 1000 (OK); then sent 1000 (OK)" in str(e):
                    logger.debug("Bhashini websocket closed normally between audio chunks")
                    continue
                logger.error("Bhashini STT processing error: {}", e)
                yield ErrorFrame(f"Bhashini STT processing failed: {e}")
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
        logger.info("Switching Bhashini language to: {}", language)
        self._language = language

    async def set_model(self, service_id: str):
        logger.info("Switching Bhashini service to: {}", service_id)
        self._service_id = service_id

    def can_generate_metrics(self) -> bool:
        return True