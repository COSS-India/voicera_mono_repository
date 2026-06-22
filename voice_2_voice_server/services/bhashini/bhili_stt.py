"""Bhashini Bhili NVCF gRPC streaming STT for Pipecat."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import AsyncGenerator, Awaitable, Callable, Optional

import numpy as np
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
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException, np_to_triton_dtype
    _TRITON_AVAILABLE = True
except ModuleNotFoundError:
    grpcclient = None  # type: ignore[assignment]
    InferenceServerException = Exception  # type: ignore[misc,assignment]
    np_to_triton_dtype = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


DEFAULT_GRPC_HOST = "grpc.nvcf.nvidia.com:443"
DEFAULT_MODEL = "asr_streaming"
DEFAULT_LANG = "bhb"


def _str_tensor(values, name: str):
    arr = np.array(values, dtype=object)
    tensor = grpcclient.InferInput(name, arr.shape, np_to_triton_dtype(arr.dtype))
    tensor.set_data_from_numpy(arr)
    return tensor


def build_triton_inputs(
    chunk: np.ndarray,
    *,
    lang_id: str,
    session_id: str,
    is_final: bool,
    hotwords: list[str] | None = None,
    hotword_weight: float = 20.0,
    alpha: float = 1.0,
    beta: float = 0.0,
):
    audio_t = grpcclient.InferInput("AUDIO_CHUNK", chunk.shape, "FP32")
    audio_t.set_data_from_numpy(chunk.reshape(-1).astype("float32"))

    final_t = grpcclient.InferInput("IS_FINAL", [1], "BOOL")
    final_t.set_data_from_numpy(np.array([is_final], dtype=bool))

    hw_t = grpcclient.InferInput("HOTWORD_WEIGHT", [1], "FP32")
    hw_t.set_data_from_numpy(np.array([hotword_weight], dtype=np.float32))

    alpha_t = grpcclient.InferInput("ALPHA", [1], "FP32")
    alpha_t.set_data_from_numpy(np.array([alpha], dtype=np.float32))

    beta_t = grpcclient.InferInput("BETA", [1], "FP32")
    beta_t.set_data_from_numpy(np.array([beta], dtype=np.float32))

    return [
        audio_t,
        _str_tensor([lang_id], "LANG_ID"),
        _str_tensor([session_id], "SESSION_ID"),
        final_t,
        _str_tensor(hotwords if hotwords else [""], "HOTWORD_LIST"),
        hw_t,
        alpha_t,
        beta_t,
    ]


class BhashiniBhiliSTTService(STTService):
    """NVCF Triton gRPC streaming ASR for Bhili (language code bhb)."""

    def __init__(
        self,
        *,
        api_key: str = "",
        function_id: str = "",
        function_version_id: str = "",
        grpc_host: str = DEFAULT_GRPC_HOST,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANG,
        sample_rate: int = 16000,
        input_sample_rate: Optional[int] = None,
        audio_channels: int = 1,
        chunk_ms: int = 200,
        suppress_vad_frames: bool = False,
        telemetry_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        if not _TRITON_AVAILABLE:
            raise ImportError(
                "tritonclient[grpc] is required for Bhashini Bhili STT. "
                "Install with: pip install tritonclient[grpc]"
            )

        self._api_key = (
            api_key.strip()
            or os.getenv("BHASHINI_BHILI_STT_AUTH_TOKEN", "").strip()
            or os.getenv("BHASHINI_STT_AUTH_TOKEN", "").strip()
            or os.getenv("BHASHINI_API_KEY", "").strip()
            or os.getenv("NVCF_API_KEY", "").strip()
        )
        if not self._api_key:
            raise ValueError(
                "BhashiniBhiliSTTService requires BHASHINI_BHILI_STT_AUTH_TOKEN "
                "(or BHASHINI_API_KEY / NVCF_API_KEY) in .env."
            )

        self._function_id = (
            function_id.strip()
            or os.getenv("BHASHINI_BHILI_STT_FUNCTION_ID", "").strip()
            or os.getenv("BHASHINI_STT_FUNCTION_ID", "").strip()
        )
        if not self._function_id:
            raise ValueError(
                "BhashiniBhiliSTTService requires BHASHINI_BHILI_STT_FUNCTION_ID "
                "(or BHASHINI_STT_FUNCTION_ID) in .env."
            )

        self._function_version_id = (
            function_version_id.strip()
            or os.getenv("BHASHINI_BHILI_STT_FUNCTION_VERSION_ID", "").strip()
            or os.getenv("BHASHINI_STT_FUNCTION_VERSION_ID", "").strip()
        )
        self._grpc_host = (
            grpc_host.strip()
            or os.getenv("BHASHINI_BHILI_STT_GRPC_URL", DEFAULT_GRPC_HOST).strip()
            or os.getenv("BHASHINI_STT_GRPC_URL", DEFAULT_GRPC_HOST).strip()
        )
        self._model = model
        self._language = language
        self._sample_rate = sample_rate
        self._input_sample_rate = input_sample_rate or sample_rate
        self._audio_channels = audio_channels
        self._chunk_ms = chunk_ms
        self._telemetry_callback = telemetry_callback
        self._pre_roll_ms = int(os.getenv("BHASHINI_PREROLL_MS", "400"))
        self._chunk_samples = int(self._input_sample_rate * self._chunk_ms / 1000)
        self._chunk_bytes = self._chunk_samples * self._audio_channels * 2
        self._pre_roll_bytes = max(
            0,
            int(self._input_sample_rate * self._pre_roll_ms / 1000) * self._audio_channels * 2,
        )
        self._target_sample_rate = 16000

        self._suppress_vad_frames = suppress_vad_frames
        self._resampler = create_stream_resampler()
        self._vad = VADProcessor(chunk_ms=self._chunk_ms)
        self._audio_buffer = bytearray()
        self._pre_roll_buffer = bytearray()
        self._disabled = False

        self._client: Optional[grpcclient.InferenceServerClient] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._result_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session_id = ""
        self._chunk_seq = 0
        self._pending_responses = 0
        self._final_transcript_event: Optional[asyncio.Event] = None
        self._latest_transcript_text = ""
        self._segment_active = False
        self._closed = False
        self._segment_started_at: Optional[float] = None
        self._speech_started_at: Optional[float] = None
        self._first_transcript_at: Optional[float] = None

        logger.info(
            "Bhashini Bhili STT initialized | grpc_host={} function_id={} model={} language={} "
            "input_rate={} target_rate={} chunk_ms={} pre_roll_ms={} suppress_vad_frames={}",
            self._grpc_host,
            self._function_id,
            self._model,
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
            logger.debug("Bhashini Bhili STT telemetry callback failed: {}", exc)

    def _grpc_headers(self) -> dict[str, str]:
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "function-id": self._function_id,
        }
        if self._function_version_id:
            headers["function-version-id"] = self._function_version_id
        return headers

    def _pcm16_to_float32(self, audio_chunk: bytes) -> np.ndarray:
        if not audio_chunk:
            return np.array([], dtype=np.float32)
        pcm16 = np.frombuffer(audio_chunk, dtype=np.int16)
        return (pcm16.astype(np.float32) / 32768.0).astype(np.float32)

    async def _prepare_audio(self, audio_chunk: bytes) -> np.ndarray:
        outgoing = audio_chunk
        if self._input_sample_rate != self._target_sample_rate:
            outgoing = await self._resampler.resample(
                audio_chunk,
                self._input_sample_rate,
                self._target_sample_rate,
            )
        return self._pcm16_to_float32(outgoing)

    def _on_stream_result(self, result, error) -> None:
        if self._loop and self._result_queue is not None:
            self._loop.call_soon_threadsafe(self._result_queue.put_nowait, (result, error))

    async def _open_stream(self) -> bool:
        if self._client or self._disabled:
            return self._client is not None

        self._loop = asyncio.get_running_loop()
        self._result_queue = asyncio.Queue()
        self._session_id = str(uuid.uuid4())
        self._chunk_seq = 0
        self._pending_responses = 0
        self._final_transcript_event = asyncio.Event()
        self._latest_transcript_text = ""
        self._first_transcript_at = None
        self._segment_started_at = time.monotonic()
        self._speech_started_at = None
        self._closed = False

        try:
            self._client = grpcclient.InferenceServerClient(url=self._grpc_host, ssl=True)
            await asyncio.to_thread(
                self._client.start_stream,
                callback=self._on_stream_result,
                headers=self._grpc_headers(),
            )
            self._receiver_task = asyncio.create_task(self._receive_handler())
            logger.info(
                "Bhashini Bhili STT segment stream started | session_id={}",
                self._session_id,
            )
            return True
        except Exception as exc:
            self._disabled = True
            self._client = None
            logger.error(
                "Bhashini Bhili gRPC setup failed; disabling STT for this call: {}",
                exc,
            )
            return False

    async def _receive_handler(self) -> None:
        try:
            while not self._closed:
                try:
                    result, error = await asyncio.wait_for(self._result_queue.get(), timeout=60)
                except asyncio.TimeoutError:
                    logger.warning("Bhashini Bhili STT response timeout")
                    break

                self._pending_responses = max(0, self._pending_responses - 1)

                if error:
                    raise InferenceServerException(str(error))

                transcript = result.as_numpy("PARTIAL_TRANSCRIPT")[0].decode("utf-8").strip()
                closed = bool(result.as_numpy("SESSION_CLOSED")[0])
                if not transcript:
                    if closed and self._final_transcript_event:
                        self._final_transcript_event.set()
                    continue

                self._latest_transcript_text = transcript
                now = time.monotonic()
                if self._first_transcript_at is None:
                    self._first_transcript_at = now
                    if self._segment_started_at is not None:
                        logger.info(
                            "Bhashini Bhili first transcript latency | {:.1f} ms | text='{}'",
                            (now - self._segment_started_at) * 1000.0,
                            transcript,
                        )
                    if self._speech_started_at is not None:
                        await self._emit_latency_metric(
                            "first_transcript_ms",
                            (now - self._speech_started_at) * 1000.0,
                            stage="first_transcript",
                            details={"text_preview": transcript[:80]},
                        )

                if closed:
                    logger.info("Bhashini Bhili final transcript: {}", transcript)
                    if self._speech_started_at is not None:
                        await self._emit_latency_metric(
                            "final_transcript_ms",
                            (now - self._speech_started_at) * 1000.0,
                            stage="final_transcript",
                            details={"text_preview": transcript[:80]},
                        )
                    if self._final_transcript_event:
                        self._final_transcript_event.set()
                    await self.stop_processing_metrics()
                    await self.push_frame(
                        TranscriptionFrame(
                            text=transcript,
                            user_id=getattr(self, "_user_id", ""),
                            timestamp=time_now_iso8601(),
                        )
                    )
                else:
                    logger.debug("Bhashini Bhili interim transcript: {}", transcript)
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            text=transcript,
                            user_id=getattr(self, "_user_id", ""),
                            timestamp=time_now_iso8601(),
                        )
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                logger.error("Bhashini Bhili gRPC receive loop failed: {}", exc)
                await self.push_frame(ErrorFrame(f"Bhashini Bhili receive loop failed: {exc}"))

    async def _send_chunk(self, audio_chunk: bytes, *, is_final: bool) -> None:
        if not self._client:
            raise RuntimeError("Bhashini Bhili gRPC stream is not connected")

        chunk = await self._prepare_audio(audio_chunk)
        if chunk.size == 0:
            return

        self._chunk_seq += 1
        inputs = build_triton_inputs(
            chunk,
            lang_id=self._language,
            session_id=self._session_id,
            is_final=is_final,
            hotwords=[],
        )
        self._pending_responses += 1
        await asyncio.to_thread(
            self._client.async_stream_infer,
            model_name=self._model,
            model_version="1",
            inputs=inputs,
            request_id=f"{self._session_id}-c{self._chunk_seq}",
        )

    async def _close_stream(self) -> None:
        self._closed = True
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except BaseException:
                pass
            self._receiver_task = None

        if self._client:
            try:
                await asyncio.to_thread(self._client.stop_stream)
            except Exception:
                pass
            self._client = None

        self._result_queue = None
        self._loop = None

    async def _finalize_segment(self) -> None:
        if not self._segment_active or not self._client:
            return

        self._segment_active = False
        try:
            if self._final_transcript_event:
                await asyncio.wait_for(self._final_transcript_event.wait(), timeout=1.5)
        except asyncio.TimeoutError:
            if self._latest_transcript_text:
                word_count = len(self._latest_transcript_text.split())
                char_count = len(self._latest_transcript_text)
                if word_count < 2 and char_count < 8:
                    logger.debug(
                        "Bhashini Bhili interim too short to promote safely (words={}, chars={}): {}",
                        word_count,
                        char_count,
                        self._latest_transcript_text,
                    )
                    return
                logger.debug(
                    "Bhashini Bhili final transcript timeout; promoting latest interim text: {}",
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
            logger.debug("Bhashini Bhili VAD detected speech start")
            if not await self._open_stream():
                return "START_FAILED"
            self._segment_active = True
            self._speech_started_at = time.monotonic()
            await self.start_processing_metrics()
            if pre_roll_bytes:
                logger.debug("Sending pre-roll buffer to Bhashini Bhili | bytes={}", len(pre_roll_bytes))
                await self._send_chunk(pre_roll_bytes, is_final=False)
            await self._send_chunk(audio_chunk, is_final=False)
            return "START"

        if state == "CONTINUE" and self._segment_active:
            await self._send_chunk(audio_chunk, is_final=False)
            return "CONTINUE"

        if state == "STOP":
            logger.debug("Bhashini Bhili VAD detected speech stop")
            if self._segment_active:
                await self._send_chunk(audio_chunk, is_final=True)
            await self._finalize_segment()
            await self._close_stream()
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
        self._segment_started_at = None
        self._speech_started_at = None
        self._first_transcript_at = None
        logger.info("Bhashini Bhili STT service started")

    async def stop(self, frame: EndFrame):
        try:
            if self._segment_active:
                await self._finalize_segment()
        finally:
            await self._close_stream()
            self._audio_buffer.clear()
            self._pre_roll_buffer.clear()
            self._vad = VADProcessor(chunk_ms=self._chunk_ms)
            self._segment_active = False
            self._disabled = False
            self._segment_started_at = None
            self._speech_started_at = None
            self._first_transcript_at = None
            await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        try:
            if self._segment_active:
                await self._finalize_segment()
        finally:
            await self._close_stream()
            self._audio_buffer.clear()
            self._pre_roll_buffer.clear()
            self._vad = VADProcessor(chunk_ms=self._chunk_ms)
            self._segment_active = False
            self._disabled = False
            self._segment_started_at = None
            self._speech_started_at = None
            self._first_transcript_at = None
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
            except Exception as exc:
                logger.error("Bhashini Bhili STT processing error: {}", exc)
                yield ErrorFrame(f"Bhashini Bhili STT processing failed: {exc}")
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
        logger.info("Switching Bhashini Bhili language to: {}", language)
        self._language = language

    async def set_model(self, model: str):
        logger.info("Switching Bhashini Bhili model to: {}", model)
        self._model = model

    def can_generate_metrics(self) -> bool:
        return True
