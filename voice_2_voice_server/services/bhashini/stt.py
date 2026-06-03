"""Bhashini HTTP REST STT Service for Pipecat"""

import asyncio
import base64
import io
import os
from dotenv import load_dotenv
load_dotenv()
import wave
from typing import AsyncGenerator, Optional
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
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
    import aiohttp
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("Install with: pip install aiohttp")
    raise Exception(f"Missing module: {e}")


class BhashiniSTTService(STTService):
    """NVCF-hosted AI4Bharat STT, drop-in replacement for Bhashini STT."""

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "https://c1bba2d4-7cdc-439c-8e6b-77bb87a191fa.invocation.api.nvcf.nvidia.com",
        service_id: str = "bhashini/ai4bharat/conformer-multilingual-asr",
        language: str = "hi",
        sample_rate: int = 16000,
        audio_channels: int = 1,
        audio_format: str = "wav",
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = os.getenv("NVCF_API_KEY", "")
        self._endpoint = f"{base_url.rstrip('/')}/transcribe"
        self._service_id = service_id
        self._language = language
        self._sample_rate = sample_rate
        self._audio_channels = audio_channels
        self._audio_format = audio_format

        self._session: Optional[aiohttp.ClientSession] = None

        self._audio_buffer: list[bytes] = []
        self._is_speaking = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "*/*",
            }
        )
        logger.info("Bhashini STT service started")

    async def stop(self, frame: EndFrame):
        await self._close_session()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._audio_buffer.clear()
        await self._close_session()
        await super().cancel(frame)

    async def _close_session(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("Bhashini HTTP session closed")

    # ------------------------------------------------------------------
    # Audio handling
    # ------------------------------------------------------------------

    def _pcm_to_wav_b64(self, pcm_chunks: list[bytes]) -> str:
        """Combine raw PCM chunks into a WAV file and return base64 string."""
        raw = b"".join(pcm_chunks)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self._audio_channels)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(raw)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _build_payload(self, audio_b64: str) -> dict:
        return {
            "audio_b64": audio_b64,
            "language_id": self._language,
        }

    async def _transcribe(self, audio_b64: str) -> Optional[str]:
        """POST to NVCF endpoint and return the transcript string."""
        if not self._session:
            logger.error("No active HTTP session")
            return None

        payload = self._build_payload(audio_b64)
        try:
            async with self._session.post(self._endpoint, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Bhashini API error {resp.status}: {text}")
                    return None

                data = await resp.json()

            transcript = data.get("text", "").strip()
            return transcript or None

        except aiohttp.ClientError as e:
            logger.error(f"Bhashini HTTP request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Bhashini transcription error: {e}")
            return None

    # ------------------------------------------------------------------
    # STTService interface
    # ------------------------------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if audio:
            self._audio_buffer.append(audio)
        yield None

    async def _flush_buffer(self):
        if not self._audio_buffer:
            return

        chunks = self._audio_buffer.copy()
        self._audio_buffer.clear()

        audio_b64 = self._pcm_to_wav_b64(chunks)
        await self.start_processing_metrics()
        try:
            transcript = await self._transcribe(audio_b64)
        finally:
            await self.stop_processing_metrics()

        if transcript:
            logger.info(f"Bhashini transcript: {transcript}")
            await self.push_frame(TranscriptionFrame(
                text=transcript,
                user_id=getattr(self, "_user_id", ""),
                timestamp=time_now_iso8601(),
            ))
        else:
            logger.debug("Bhashini returned empty transcript")

    # ------------------------------------------------------------------
    # Speaking detection
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("User started speaking — buffering audio")
            self._is_speaking = True
            self._audio_buffer.clear()

        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("User stopped speaking — flushing buffer to Bhashini")
            self._is_speaking = False
            await self._flush_buffer()

    # ------------------------------------------------------------------
    # Runtime config changes
    # ------------------------------------------------------------------

    async def set_language(self, language: str):
        logger.info(f"Switching language to: {language}")
        self._language = language

    async def set_model(self, service_id: str):
        logger.info(f"Switching service to: {service_id}")
        self._service_id = service_id

    def can_generate_metrics(self) -> bool:
        return True