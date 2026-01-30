"""
IndicConformer REST STT Service for Pipecat
Client-side buffering with REST calls for transcription
"""

import os
import asyncio
import base64
import time
from typing import AsyncGenerator, Optional, Dict, Any
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    ErrorFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.audio.utils import create_stream_resampler
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp package not installed. Install with: pip install aiohttp")


class IndicConformerRESTSTTService(STTService):
    
    def __init__(
        self,
        *,
        language_id: str = "hi",
        sample_rate: int = 16000,
        input_sample_rate: int = 8000,
        vad_analyzer: Optional[VADAnalyzer] = None,
        **kwargs
    ):
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp package required. Install with: pip install aiohttp")
        
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        server_url = os.getenv("INDIC_STT_SERVER_URL")
        if not server_url:
            raise ValueError("INDIC_STT_SERVER_URL environment variable not set")
        
        self._server_url = server_url.rstrip('/') + "/transcribe"
        self._language_id = language_id
        self._sample_rate = sample_rate
        self._input_sample_rate = input_sample_rate
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._vad_analyzer: Optional[VADAnalyzer] = vad_analyzer
        
        self._audio_buffer = b""
        self._text_chunks = []
        self._is_speaking = False
        
        self._stopping_start_time: Optional[float] = None
        self._stopping_triggered = False
        self._STOPPING_DURATION_MS = 10
        
        self._resampler = create_stream_resampler()
        
        logger.info(f"IndicConformerRESTSTTService initialized - Server: {self._server_url}")

    async def _transcribe_buffer(self) -> str:
        if not self._audio_buffer or len(self._audio_buffer) < 3200:
            return ""
        
        try:
            audio_b64 = base64.b64encode(self._audio_buffer).decode('utf-8')
            
            async with self._session.post(
                self._server_url,
                json={"audio_b64": audio_b64, "language_id": self._language_id},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("text", "")
                else:
                    logger.error(f"Transcription request failed: {response.status}")
                    return ""
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def _check_stopping_state(self) -> bool:
        if self._vad_analyzer is None:
            return False
        
        try:
            vad_state = self._vad_analyzer._vad_state
            
            if vad_state == VADState.STOPPING:
                current_time = time.time() * 1000
                
                if self._stopping_start_time is None:
                    self._stopping_start_time = current_time
                    return False
                
                duration_ms = current_time - self._stopping_start_time
                
                if duration_ms >= self._STOPPING_DURATION_MS and not self._stopping_triggered:
                    self._stopping_triggered = True
                    return True
                
                return False
            else:
                self._stopping_start_time = None
                self._stopping_triggered = False
                return False
                
        except AttributeError:
            return False

    async def process_frame(self, frame: Frame, direction):
        # Handle UserStoppedSpeakingFrame BEFORE calling super() to ensure
        # TranscriptionFrame is pushed downstream BEFORE UserStoppedSpeakingFrame
        # This matches the WebSocket behavior where the server sends final transcription
        # immediately when is_speaking=false, and the frame reaches aggregator first
        if isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("VAD: User stopped speaking")
            self._is_speaking = False
            self._stopping_start_time = None
            self._stopping_triggered = False
            
            # Push final transcription BEFORE the UserStoppedSpeakingFrame
            # so aggregator receives transcription first, then stop frame
            if self._text_chunks:
                accumulated = " ".join(self._text_chunks)
                logger.info(f"Final: {accumulated}")
                await self.push_frame(TranscriptionFrame(
                    text=accumulated,
                    user_id=self._user_id,
                    timestamp=str(int(time.time() * 1000))
                ))
                # Clear everything after sending
                self._text_chunks = []
                self._audio_buffer = b""
        
        # Now call parent's process_frame which will push UserStoppedSpeakingFrame downstream
        await super().process_frame(frame, direction)
        
        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("VAD: User started speaking")
            self._is_speaking = True
            self._stopping_start_time = None
            self._stopping_triggered = False
            self._audio_buffer = b""
            self._text_chunks = []

    async def start(self, frame: Frame) -> None:
        logger.info("Starting IndicConformer REST STT service")
        self._session = aiohttp.ClientSession()
        self._is_speaking = False
        self._audio_buffer = b""
        self._text_chunks = []
        self._stopping_start_time = None
        self._stopping_triggered = False
        await super().start(frame)

    async def stop(self, frame: Frame) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        await super().stop(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not audio:
            return
        
        try:
            resampled_audio = audio
            if self._input_sample_rate != self._sample_rate:
                resampled_audio = await self._resampler.resample(
                    audio,
                    self._input_sample_rate,
                    self._sample_rate
                )
            
            self._audio_buffer += resampled_audio            
            if self._check_stopping_state():
                logger.info("STOPPING state triggered, transcribing buffer")
                text = await self._transcribe_buffer()
                if text:
                    self._text_chunks.append(text)
                    accumulated = " ".join(self._text_chunks)
                    logger.info(f"Interim: {accumulated}")
                    yield InterimTranscriptionFrame(
                        text=accumulated,
                        user_id=self._user_id,
                        timestamp=str(int(time.time() * 1000))
                    )
                self._audio_buffer = b""
            
            # Note: Final transcription is now sent immediately in process_frame()
            # when UserStoppedSpeakingFrame is received, matching WebSocket behavior
                
        except Exception as e:
            logger.error(f"STT processing error: {e}")
            yield ErrorFrame(f"STT processing failed: {str(e)}")

    async def set_language(self, language_id: str) -> None:
        self._language_id = language_id
        logger.info(f"Language changed to: {language_id}")

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "server_url": self._server_url,
            "language_id": self._language_id,
            "sample_rate": self._sample_rate,
            "input_sample_rate": self._input_sample_rate,
            "is_speaking": self._is_speaking,
            "buffer_size": len(self._audio_buffer),
            "text_chunks": len(self._text_chunks),
        }

    def get_supported_languages(self) -> list:
        return [
            "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai",
            "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"
        ]

    def can_generate_metrics(self) -> bool:
        return True