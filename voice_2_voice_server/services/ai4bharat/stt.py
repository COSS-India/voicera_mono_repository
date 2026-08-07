"""IndicConformer REST STT Service for Pipecat.

VAD, segment buffering, pre-roll, and barge-in behaviour mirror
``BhashiniSTTService``; energy-VAD thresholds are defined in this file
and can be tuned independently of Bhashini.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import aiohttp
import numpy as np
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAIUserContextAggregator
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

from utils.bot_utils import BotSpeakingLatch

try:
    import aiohttp as _aiohttp_check

    AIOHTTP_AVAILABLE = True
    del _aiohttp_check
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp package not installed. Install with: pip install aiohttp")


_PRE_ROLL_MS = 800


@dataclass
class VADProcessor:
    """Energy-based VAD for AI4Bharat REST STT segment boundaries.

    ``min_speech_ms`` (350) while the bot is talking — original barge-in gate.
    ``min_speech_ms_idle`` (200) only when the bot is silent — short user turns.
    """

    speech_start_rms: float = 0.035
    speech_end_rms: float = 0.012
    min_speech_ms: int = 350
    min_speech_ms_idle: int = 200
    min_pause_ms: int = 400
    chunk_ms: int = 200

    is_speaking: bool = False
    bot_speaking: bool = False
    speech_run_ms: int = 0
    silence_run_ms: int = 0

    def _active_min_speech_ms(self) -> int:
        return self.min_speech_ms if self.bot_speaking else self.min_speech_ms_idle

    def process_chunk(self, audio_data: bytes) -> str:
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return "IDLE"

        rms = float(np.sqrt(np.mean(samples**2)))

        if not self.is_speaking:
            if rms > self.speech_start_rms:
                self.speech_run_ms += self.chunk_ms
                if self.speech_run_ms >= self._active_min_speech_ms():
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


class IndicConformerRESTSTTService(STTService):
    """REST client for ai4bharat_stt_server. language_id \"bhb\" uses POST /transcribe/bhili."""

    def __init__(
        self,
        *,
        language_id: str = "hi",
        sample_rate: int = 16000,
        input_sample_rate: Optional[int] = None,
        audio_channels: int = 1,
        chunk_ms: int = 200,
        suppress_vad_frames: bool = False,
        **kwargs,
    ):
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp package required. Install with: pip install aiohttp")

        super().__init__(sample_rate=sample_rate, **kwargs)

        server_url = os.getenv("INDIC_STT_SERVER_URL")
        if not server_url:
            raise ValueError("INDIC_STT_SERVER_URL environment variable not set")

        base = server_url.rstrip("/")
        self._language_id = language_id
        self._bhili_endpoint = language_id == "bhb"
        self._transcribe_url = (
            f"{base}/transcribe/bhili" if self._bhili_endpoint else f"{base}/transcribe"
        )
        self._sample_rate = sample_rate
        self._input_sample_rate = input_sample_rate or sample_rate
        self._audio_channels = audio_channels
        self._chunk_ms = chunk_ms
        self._suppress_vad_frames = suppress_vad_frames
        self._pre_roll_ms = _PRE_ROLL_MS
        self._chunk_samples = int(self._input_sample_rate * self._chunk_ms / 1000)
        self._chunk_bytes = self._chunk_samples * self._audio_channels * 2
        self._pre_roll_bytes = max(
            0,
            int(self._input_sample_rate * self._pre_roll_ms / 1000) * self._audio_channels * 2,
        )
        self._target_sample_rate = 16000
        self._interim_interval_ms = int(os.getenv("AI4BHARAT_INTERIM_MS", "600"))
        # Interims re-transcribe the whole growing segment, so a fixed interval
        # makes STT cost O(n^2) in utterance length: a 12 s utterance costs
        # ~126 audio-seconds of GPU work. Requiring the buffer to grow by
        # AI4BHARAT_INTERIM_GROWTH before the next interim cuts that to ~50 s.
        # AI4BHARAT_INTERIM_MAX_S caps the resulting gap so long utterances keep
        # getting fresh partials and the last interim still covers nearly the
        # whole segment -- the "promote latest interim when the final comes back
        # empty" fallback below depends on that coverage.
        # Set AI4BHARAT_INTERIM_GROWTH=0 for the old fixed-interval behaviour.
        self._interim_growth = float(os.getenv("AI4BHARAT_INTERIM_GROWTH", "0.5"))
        self._interim_max_s = float(os.getenv("AI4BHARAT_INTERIM_MAX_S", "2.0"))

        self._session: Optional[aiohttp.ClientSession] = None
        self._resampler = create_stream_resampler()
        self._vad = VADProcessor(chunk_ms=self._chunk_ms)
        self._bot_latch = BotSpeakingLatch()
        self._audio_buffer = bytearray()
        self._pre_roll_buffer = bytearray()
        self._segment_buffer = bytearray()
        self._transcribe_lock = asyncio.Lock()
        self._disabled = False

        self._segment_active = False
        self._latest_transcript_text = ""
        self._bytes_since_last_interim = 0
        self._speech_started_at: Optional[float] = None

        logger.info(
            "AI4Bharat REST STT initialized | url={} language={} input_rate={} target_rate={} "
            "chunk_ms={} pre_roll_ms={} suppress_vad_frames={}",
            self._transcribe_url,
            self._language_id,
            self._input_sample_rate,
            self._target_sample_rate,
            self._chunk_ms,
            self._pre_roll_ms,
            self._suppress_vad_frames,
        )

    async def _resample_chunk(self, audio_chunk: bytes) -> bytes:
        if not audio_chunk:
            return b""
        if self._input_sample_rate == self._target_sample_rate:
            return audio_chunk
        return await self._resampler.resample(
            audio_chunk,
            self._input_sample_rate,
            self._target_sample_rate,
        )

    async def _transcribe_buffer(self, audio_buffer: bytes) -> str:
        if not audio_buffer or len(audio_buffer) < 3200:
            return ""

        # Every failure path below returns "", which is indistinguishable from
        # "the user said nothing" downstream -- so each one must log its own
        # cause, or STT overload looks exactly like silence.
        try:
            audio_b64 = base64.b64encode(audio_buffer).decode("utf-8")
            async with self._session.post(
                self._transcribe_url,
                json={"audio_b64": audio_b64, "language_id": self._language_id},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return str(data.get("text", "")).strip()
                if response.status == 503:
                    logger.warning(
                        "AI4Bharat STT shed the request (503: queue full / not ready / "
                        "worker unavailable) | audio={:.1f}s",
                        len(audio_buffer) / (self._target_sample_rate * 2),
                    )
                elif response.status == 429:
                    logger.warning("AI4Bharat STT rate-limited the request (429)")
                elif response.status == 504:
                    logger.error("AI4Bharat STT inference timed out server-side (504)")
                elif response.status in (400, 413):
                    logger.error(
                        "AI4Bharat STT rejected the payload ({}): {}",
                        response.status,
                        (await response.text())[:200],
                    )
                else:
                    logger.error(
                        "AI4Bharat transcription request failed: {}", response.status
                    )
                return ""
        except asyncio.TimeoutError:
            logger.warning(
                "AI4Bharat transcription timed out client-side after 10s | audio={:.1f}s",
                len(audio_buffer) / (self._target_sample_rate * 2),
            )
            return ""
        except aiohttp.ClientError as exc:
            logger.error("AI4Bharat transcription transport error ({}): {}", type(exc).__name__, exc)
            return ""
        except Exception as exc:
            logger.error("AI4Bharat transcription error ({}): {}", type(exc).__name__, exc)
            return ""

    def _interim_min_bytes(self) -> int:
        """New audio required before the next interim.

        Scales with the segment already buffered, because each interim
        re-transcribes the whole segment from the start.
        """
        base = int(self._target_sample_rate * self._interim_interval_ms / 1000) * 2
        if self._interim_growth <= 0:
            return base
        scaled = max(base, int(len(self._segment_buffer) * self._interim_growth))
        if self._interim_max_s > 0:
            ceiling = int(self._target_sample_rate * self._interim_max_s) * 2
            scaled = min(scaled, max(base, ceiling))
        return scaled

    async def _maybe_emit_interim(self) -> None:
        min_bytes = self._interim_min_bytes()
        if self._bytes_since_last_interim < min_bytes:
            return
        if self._transcribe_lock.locked():
            return

        async with self._transcribe_lock:
            text = await self._transcribe_buffer(bytes(self._segment_buffer))
            if text and text != self._latest_transcript_text:
                self._latest_transcript_text = text
                logger.debug("AI4Bharat interim transcript: {}", text)
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text=text,
                        user_id=getattr(self, "_user_id", ""),
                        timestamp=time_now_iso8601(),
                    )
                )
        self._bytes_since_last_interim = 0

    async def _finalize_segment(self) -> None:
        if not self._segment_active:
            return

        self._segment_active = False
        try:
            async with self._transcribe_lock:
                text = await self._transcribe_buffer(bytes(self._segment_buffer))
            if text:
                logger.info("AI4Bharat final transcript: {}", text)
                await self.push_frame(
                    TranscriptionFrame(
                        text=text,
                        user_id=getattr(self, "_user_id", ""),
                        timestamp=time_now_iso8601(),
                    )
                )
            elif self._latest_transcript_text:
                word_count = len(self._latest_transcript_text.split())
                char_count = len(self._latest_transcript_text)
                if word_count >= 2 or char_count >= 8:
                    logger.debug(
                        "AI4Bharat final empty; promoting latest interim: {}",
                        self._latest_transcript_text,
                    )
                    await self.push_frame(
                        TranscriptionFrame(
                            text=self._latest_transcript_text,
                            user_id=getattr(self, "_user_id", ""),
                            timestamp=time_now_iso8601(),
                        )
                    )
        finally:
            await self.stop_processing_metrics()
            self._segment_buffer.clear()
            self._latest_transcript_text = ""
            self._bytes_since_last_interim = 0
            self._speech_started_at = None

    async def _handle_audio_chunk(self, audio_chunk: bytes, pre_roll_bytes: bytes = b"") -> str:
        self._vad.bot_speaking = self._bot_latch.speaking
        state = self._vad.process_chunk(audio_chunk)

        if state == "START":
            logger.debug(
                "AI4Bharat VAD detected speech start (min_speech_ms={} bot_speaking={})",
                self._vad._active_min_speech_ms(),
                self._vad.bot_speaking,
            )
            self._segment_active = True
            self._segment_buffer.clear()
            self._latest_transcript_text = ""
            self._bytes_since_last_interim = 0
            self._speech_started_at = time.monotonic()
            await self.start_processing_metrics()
            if pre_roll_bytes:
                resampled_pre_roll = await self._resample_chunk(pre_roll_bytes)
                if resampled_pre_roll:
                    self._segment_buffer.extend(resampled_pre_roll)
            resampled_chunk = await self._resample_chunk(audio_chunk)
            if resampled_chunk:
                self._segment_buffer.extend(resampled_chunk)
                self._bytes_since_last_interim += len(resampled_chunk)
            return "START"

        if state == "CONTINUE" and self._segment_active:
            resampled_chunk = await self._resample_chunk(audio_chunk)
            if resampled_chunk:
                self._segment_buffer.extend(resampled_chunk)
                self._bytes_since_last_interim += len(resampled_chunk)
                await self._maybe_emit_interim()
            return "CONTINUE"

        if state == "STOP":
            logger.debug("AI4Bharat VAD detected speech stop")
            await self._finalize_segment()
            return "STOP"

        return state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, (BotStartedSpeakingFrame, TTSStartedFrame)):
            self._bot_latch.on_started()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_latch.on_stopped()
        await super().process_frame(frame, direction)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._session = aiohttp.ClientSession()
        self._disabled = False
        self._audio_buffer.clear()
        self._pre_roll_buffer.clear()
        self._segment_buffer.clear()
        self._vad = VADProcessor(chunk_ms=self._chunk_ms)
        self._bot_latch.reset()
        self._segment_active = False
        self._latest_transcript_text = ""
        self._bytes_since_last_interim = 0
        self._speech_started_at = None
        logger.info("AI4Bharat REST STT service started")

    async def stop(self, frame: EndFrame):
        try:
            if self._segment_active:
                await self._finalize_segment()
        finally:
            if self._session:
                await self._session.close()
                self._session = None
            self._audio_buffer.clear()
            self._pre_roll_buffer.clear()
            self._segment_buffer.clear()
            self._vad = VADProcessor(chunk_ms=self._chunk_ms)
            self._bot_latch.reset()
            self._segment_active = False
            self._latest_transcript_text = ""
            self._bytes_since_last_interim = 0
            self._speech_started_at = None
            self._disabled = False
            await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        try:
            if self._segment_active:
                await self._finalize_segment()
        finally:
            if self._session:
                await self._session.close()
                self._session = None
            self._audio_buffer.clear()
            self._pre_roll_buffer.clear()
            self._segment_buffer.clear()
            self._vad = VADProcessor(chunk_ms=self._chunk_ms)
            self._bot_latch.reset()
            self._segment_active = False
            self._latest_transcript_text = ""
            self._bytes_since_last_interim = 0
            self._speech_started_at = None
            self._disabled = False
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
                logger.error("AI4Bharat STT processing error: {}", exc)
                yield ErrorFrame(f"AI4Bharat STT processing failed: {exc}")
            finally:
                if self._pre_roll_bytes > 0:
                    self._pre_roll_buffer.extend(chunk)
                    if len(self._pre_roll_buffer) > self._pre_roll_bytes:
                        overflow = len(self._pre_roll_buffer) - self._pre_roll_bytes
                        if overflow > 0:
                            del self._pre_roll_buffer[:overflow]
                else:
                    self._pre_roll_buffer.clear()

    async def set_language(self, language_id: str) -> None:
        if self._bhili_endpoint:
            self._language_id = "bhb"
            logger.info("Bhili STT endpoint: language_id remains bhb")
        else:
            self._language_id = language_id
            logger.info("AI4Bharat language changed to: {}", language_id)

    def can_generate_metrics(self) -> bool:
        return True


class Ai4BharatKenpathUserContextAggregator(OpenAIUserContextAggregator):
    """User aggregator for AI4Bharat STT + Kenpath LLM.

    Pushes the user turn to the LLM as soon as a final AI4Bharat
    ``TranscriptionFrame`` is received, without waiting for Silero
    ``UserStoppedSpeakingFrame`` or Pipecat's ``aggregation_timeout``.

    While the bot is speaking, Silero must have armed (noise guard).
    While the bot is silent (user's turn), short replies like "hello" /
    "nahi" are accepted without requiring Silero.
    """

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
                "AI4Bharat final too short for LLM ({} chars) — skipping: '{}'",
                len(text),
                text,
            )
            await self.reset()
            self._silero_armed = False
            return

        # Bot speaking: keep Silero gate. Bot silent: accept single-word turns.
        if self._bot_speaking and not self._silero_armed:
            logger.debug(
                "AI4Bharat final ignored for LLM — Silero did not detect speech: '{}'",
                text[:80],
            )
            await self.reset()
            return

        await super()._handle_transcription(frame)
        if len(self._aggregation) > 0:
            logger.debug(
                "AI4Bharat final transcript — pushing LLM immediately | text='{}' | bot_speaking={}",
                self._aggregation[:80],
                self._bot_speaking,
            )
            await self.push_aggregation()
        self._silero_armed = False
