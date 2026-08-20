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
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

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
from services.ai4bharat.shadow import (
    AutoLanguageShadowConfig,
    AutoLanguageShadowObserver,
)
from services.ai4bharat.auto_language import (
    AUTO_LANGUAGE_UNKNOWN,
    AutoLanguageConfig,
    AutoLanguageController,
    AutoLanguageUnavailableError,
    is_language_unresolved,
)

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
        session_id: Optional[str] = None,
        enable_auto_language_shadow: object = None,
        auto_language_min_duration_ms: object = None,
        auto_language_max_duration_ms: object = None,
        auto_language_margin_threshold: object = None,
        auto_language_candidate_languages: object = None,
        enable_auto_language: object = None,
        auto_language_device: object = None,
        auto_language_confirmation_count: object = None,
        auto_language_reprobe_cooldown_ms: object = None,
        auto_language_event_log_path: object = None,
        bootstrap_fallback_language_id: object = None,
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
        self._shadow_url = f"{base}/shadow/language-probe"
        self._shadow_config = AutoLanguageShadowConfig.resolve(
            enabled=enable_auto_language_shadow,
            min_duration_ms=auto_language_min_duration_ms,
            max_duration_ms=auto_language_max_duration_ms,
            margin_threshold=auto_language_margin_threshold,
            candidate_languages=auto_language_candidate_languages,
        )
        self._auto_language_config = AutoLanguageConfig.resolve(
            enabled=enable_auto_language,
            device=auto_language_device,
            min_duration_ms=auto_language_min_duration_ms,
            max_duration_ms=auto_language_max_duration_ms,
            margin_threshold=auto_language_margin_threshold,
            candidate_languages=auto_language_candidate_languages,
            confirmation_count=auto_language_confirmation_count,
            reprobe_cooldown_ms=auto_language_reprobe_cooldown_ms,
            event_log_path=auto_language_event_log_path,
        )

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
        self._segment_language_id = self._language_id
        fallback = bootstrap_fallback_language_id
        if fallback is not None and not is_language_unresolved(str(fallback)):
            self._fallback_language_id = str(fallback)
        else:
            self._fallback_language_id = None
        self._auto_switch_callback: Optional[
            Callable[[str, str], Awaitable[None]]
        ] = None
        self._shadow = AutoLanguageShadowObserver(
            session_id=session_id or getattr(self, "_user_id", "") or "unknown",
            sample_rate=self._target_sample_rate,
            config=self._shadow_config,
            request_probe=self._request_shadow_probe,
        )
        self._auto_language = AutoLanguageController(
            session_id=session_id or getattr(self, "_user_id", "") or "unknown",
            sample_rate=self._target_sample_rate,
            config=self._auto_language_config,
            request_probe=self._request_auto_language_probe,
            current_language=lambda: self._language_id,
            switch_language=self._apply_auto_language_switch,
        )

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
        if self._shadow_config.enabled:
            logger.info(
                "AI4Bharat automatic-language shadow enabled | session={} "
                "min_duration_ms={} max_duration_ms={} preliminary_margin_threshold={} "
                "candidate_languages={}",
                self._shadow.session_id,
                self._shadow_config.min_duration_ms,
                self._shadow_config.max_duration_ms,
                self._shadow_config.margin_threshold,
                self._shadow_config.candidate_languages,
            )
        if self._auto_language_config.enabled:
            logger.info(
                "AI4Bharat LIVE automatic language enabled | session={} device={} "
                "min_duration_ms={} max_duration_ms={} margin_threshold={} "
                "confirmation_count={} cooldown_ms={} candidates={} event_log={}",
                self._auto_language.session_id,
                self._auto_language_config.device,
                self._auto_language_config.min_duration_ms,
                self._auto_language_config.max_duration_ms,
                self._auto_language_config.margin_threshold,
                self._auto_language_config.confirmation_count,
                self._auto_language_config.reprobe_cooldown_ms,
                self._auto_language_config.candidate_languages,
                self._auto_language_config.event_log_path,
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

    async def _transcribe_buffer(
        self,
        audio_buffer: bytes,
        language_id: Optional[str] = None,
    ) -> str:
        if not audio_buffer or len(audio_buffer) < 3200:
            return ""

        try:
            audio_b64 = base64.b64encode(audio_buffer).decode("utf-8")
            async with self._session.post(
                self._transcribe_url,
                json={
                    "audio_b64": audio_b64,
                    "language_id": language_id or self._language_id,
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return str(data.get("text", "")).strip()
                logger.error("AI4Bharat transcription request failed: {}", response.status)
                return ""
        except Exception as exc:
            logger.error("AI4Bharat transcription error: {}", exc)
            return ""

    async def _request_shadow_probe(
        self,
        audio: bytes,
        current_language: str,
        session_id: str,
        candidate_languages: tuple[str, ...],
    ) -> dict:
        if self._session is None:
            raise RuntimeError("AI4Bharat STT HTTP session is not available")
        audio_b64 = base64.b64encode(audio).decode("utf-8")
        async with self._session.post(
            self._shadow_url,
            json={
                "audio_b64": audio_b64,
                "session_id": session_id,
                "current_language": current_language,
                "candidate_languages": list(candidate_languages),
                "min_duration_ms": self._shadow_config.min_duration_ms,
            },
            timeout=aiohttp.ClientTimeout(total=20),
        ) as response:
            if response.status != 200:
                detail = await response.text()
                raise RuntimeError(
                    f"shadow endpoint returned HTTP {response.status}: {detail[:200]}"
                )
            return await response.json()

    async def _request_auto_language_probe(
        self,
        audio: bytes,
        current_language: str,
        session_id: str,
        candidate_languages: tuple[str, ...],
    ) -> dict:
        if self._session is None:
            raise RuntimeError("AI4Bharat STT HTTP session is not available")
        audio_b64 = base64.b64encode(audio).decode("utf-8")
        async with self._session.post(
            self._shadow_url.replace("/shadow/language-probe", "/language-probe"),
            json={
                "audio_b64": audio_b64,
                "session_id": session_id,
                "current_language": current_language,
                "candidate_languages": list(candidate_languages),
                "min_duration_ms": self._auto_language_config.min_duration_ms,
                "require_cuda": True,
            },
            timeout=aiohttp.ClientTimeout(total=20),
        ) as response:
            if response.status in {404, 503}:
                detail = await response.text()
                raise AutoLanguageUnavailableError(
                    f"live GPU probe unavailable (HTTP {response.status}): {detail[:200]}"
                )
            if response.status != 200:
                detail = await response.text()
                raise RuntimeError(
                    f"live probe returned HTTP {response.status}: {detail[:200]}"
                )
            return await response.json()

    def _probe_audio_window(self, audio: bytes) -> bytes:
        max_bytes = int(self._target_sample_rate * self._auto_language_config.max_duration_ms / 1000) * 2
        if max_bytes and len(audio) > max_bytes:
            return audio[-max_bytes:]
        return audio

    def _audio_duration_ms(self, audio: bytes) -> float:
        return len(audio) * 1000.0 / (self._target_sample_rate * 2)

    async def _probe_accepted_language(self, audio: bytes) -> tuple[str | None, dict | None]:
        duration_ms = self._audio_duration_ms(audio)
        if duration_ms < self._auto_language_config.min_duration_ms:
            return None, None
        probe_audio = self._probe_audio_window(audio)
        try:
            result = await self._request_auto_language_probe(
                probe_audio,
                self._language_id,
                self._auto_language.session_id,
                self._auto_language_config.candidate_languages,
            )
        except Exception as exc:
            logger.warning(
                "Auto-language bootstrap probe failed | session={} error={}",
                self._auto_language.session_id,
                exc,
            )
            return None, None

        predicted = str(result.get("predicted_language") or "")
        margin_raw = result.get("margin")
        margin = float(margin_raw) if margin_raw is not None else None
        accepted = (
            margin is not None
            and margin >= self._auto_language_config.margin_threshold
            and predicted in self._auto_language_config.candidate_languages
        )
        if not accepted:
            return None, result
        return predicted, result

    async def _establish_detected_language(
        self,
        language: str,
        *,
        source: str,
        result: dict | None = None,
        audio_duration_ms: float = 0.0,
    ) -> None:
        previous = self._language_id
        if self._auto_switch_callback is not None:
            await self._auto_switch_callback(previous, language)
        self._language_id = language
        self._auto_language.explicit_language_changed(language, source)
        logger.info(
            "AI4Bharat language established | from={} to={} source={} session={}",
            previous,
            language,
            source,
            self._auto_language.session_id,
        )
        if result is not None and source == "auto_probe_bootstrap":
            self._auto_language.record_bootstrap_probe(
                audio_duration_ms=audio_duration_ms,
                predicted_language=language,
                result=result,
            )

    async def _maybe_emit_interim(self) -> None:
        if self._auto_language_config.enabled:
            return
        if is_language_unresolved(self._segment_language_id):
            return
        min_bytes = int(self._target_sample_rate * self._interim_interval_ms / 1000) * 2
        if self._bytes_since_last_interim < min_bytes:
            return
        if self._transcribe_lock.locked():
            return

        async with self._transcribe_lock:
            text = await self._transcribe_buffer(
                bytes(self._segment_buffer),
                self._segment_language_id,
            )
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
        shadow_audio = bytes(self._segment_buffer)
        utterance_language = self._segment_language_id
        auto_enabled = self._auto_language_config.enabled
        try:
            if auto_enabled:
                detected_language, probe_result = await self._probe_accepted_language(
                    shadow_audio
                )
                unresolved = is_language_unresolved(self._language_id)
                if detected_language is not None:
                    decode_language = detected_language
                elif unresolved and self._fallback_language_id:
                    decode_language = self._fallback_language_id
                    logger.info(
                        "Auto-language probe uncertain; using configured fallback={}",
                        decode_language,
                    )
                elif not unresolved:
                    decode_language = utterance_language
                else:
                    decode_language = None

                if decode_language is None:
                    logger.info(
                        "Auto-language finalize skipped | session={} reason=no_confident_language",
                        self._auto_language.session_id,
                    )
                    text = ""
                else:
                    if detected_language is not None and (
                        unresolved or detected_language != self._language_id
                    ):
                        await self._establish_detected_language(
                            detected_language,
                            source=(
                                "auto_probe_bootstrap"
                                if unresolved
                                else "auto_probe"
                            ),
                            result=probe_result,
                            audio_duration_ms=self._audio_duration_ms(
                                self._probe_audio_window(shadow_audio)
                            ),
                        )
                    async with self._transcribe_lock:
                        text = await self._transcribe_buffer(
                            shadow_audio,
                            decode_language,
                        )
            else:
                async with self._transcribe_lock:
                    text = await self._transcribe_buffer(
                        shadow_audio,
                        utterance_language,
                    )

            if text:
                logger.info("AI4Bharat final transcript: {}", text)
                await self.push_frame(
                    TranscriptionFrame(
                        text=text,
                        user_id=getattr(self, "_user_id", ""),
                        timestamp=time_now_iso8601(),
                    )
                )
            elif not auto_enabled and self._latest_transcript_text:
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
            if not auto_enabled:
                self._shadow.observe(shadow_audio, utterance_language)
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
            # Freeze the explicit language for this entire utterance. Any auto
            # decision completed concurrently can only affect a later START.
            self._segment_language_id = self._language_id
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
        self._segment_language_id = self._language_id
        self._shadow.reset()
        self._auto_language.reset()
        logger.info("AI4Bharat REST STT service started")

    async def stop(self, frame: EndFrame):
        try:
            if self._segment_active:
                await self._finalize_segment()
            await self._auto_language.drain()
            self._auto_language.log_summary()
            await self._shadow.drain()
            if not self._auto_language_config.enabled:
                self._shadow.log_summary(self._language_id)
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
            await self._auto_language.drain()
            self._auto_language.log_summary()
            await self._shadow.drain()
            if not self._auto_language_config.enabled:
                self._shadow.log_summary(self._language_id)
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

    def set_auto_language_switch_callback(
        self,
        callback: Callable[[str, str], Awaitable[None]],
    ) -> None:
        self._auto_switch_callback = callback

    async def _apply_auto_language_switch(
        self,
        from_language: str,
        to_language: str,
        event: dict[str, Any],
    ) -> None:
        if self._language_id != from_language:
            raise RuntimeError(
                f"stale auto-language switch {from_language}->{to_language}; "
                f"current language is {self._language_id}"
            )
        if self._auto_switch_callback is not None:
            await self._auto_switch_callback(from_language, to_language)
        self._language_id = to_language
        logger.info(
            "AI4Bharat language changed | from={} to={} source=auto_probe session={}",
            from_language,
            to_language,
            self._auto_language.session_id,
        )

    async def set_language(self, language_id: str, source: str = "manual") -> None:
        if self._bhili_endpoint:
            self._language_id = "bhb"
            logger.info("Bhili STT endpoint: language_id remains bhb")
        else:
            previous = self._language_id
            self._language_id = language_id
            self._auto_language.explicit_language_changed(language_id, source)
            logger.info(
                "AI4Bharat language changed | from={} to={} source={}",
                previous,
                language_id,
                source,
            )

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
