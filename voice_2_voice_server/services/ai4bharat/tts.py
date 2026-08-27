import asyncio
import os
from typing import AsyncGenerator

import aiohttp
import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


def _parse_gain() -> float:
    raw = os.getenv("INDIC_TTS_GAIN", "1.0").strip()
    try:
        gain = float(raw)
    except ValueError:
        logger.warning("Invalid INDIC_TTS_GAIN={!r}, using 1.0", raw)
        return 1.0
    if gain <= 0:
        logger.warning("INDIC_TTS_GAIN must be > 0, got {}, using 1.0", gain)
        return 1.0
    if gain > 4.0:
        logger.warning(
            "INDIC_TTS_GAIN={} is very high; audio clipping may occur", gain
        )
    return gain


class IndicParlerRESTTTSService(TTSService):
    def __init__(
        self,
        *,
        speaker: str | None = "Divya",
        description: str = "A clear, natural voice with good audio quality.",
        language_id: str = "hi",
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        server_url = os.getenv("MODEL_SERVER_URL") or os.getenv("INDIC_TTS_SERVER_URL")
        if not server_url:
            raise ValueError("MODEL_SERVER_URL environment variable not set")
        self._speech_url = server_url.rstrip("/") + "/v1/audio/speech"
        self._speaker = speaker
        self._description = description
        self._language_id = language_id
        self._gain = _parse_gain()
        self._session: aiohttp.ClientSession | None = None

    def _description_for_server(self) -> str:
        if self._speaker:
            return f"{self._speaker}. {self._description}"
        return self._description

    async def start(self, frame: Frame):
        if self._gain != 1.0:
            logger.info("Starting IndicParler TTS service (output gain={})", self._gain)
        else:
            logger.info("Starting IndicParler TTS service")
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        await super().start(frame)

    def can_generate_metrics(self) -> bool:
        return True

    async def stop(self, frame: Frame):
        logger.info("Stopping IndicParler TTS service")
        if self._session:
            await self._session.close()
            self._session = None
        await super().stop(frame)

    def set_language(self, language_id: str) -> None:
        self._language_id = language_id
        logger.info(f"TTS language changed to: {language_id}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            return

        session = self._session
        should_close = False
        if not session or session.closed:
            logger.warning("TTS session not available, creating temporary session")
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)
            )
            should_close = True

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()
        first_audio = True

        try:
            payload = {
                "input": text,
                "voice": self._speaker,
                "instructions": self._description,
                "language": self._language_id,
                # float32 is what the engine produces; asking for anything else
                # would resample on the server and change the audio.
                "response_format": "pcm_f32le",
            }
            async with session.post(self._speech_url, json=payload) as resp:
                if resp.status != 200:
                    detail = (await resp.text())[:200]
                    yield ErrorFrame(f"TTS request failed: {resp.status} {detail}")
                    return

                out_rate = int(resp.headers.get("X-Sample-Rate", self.sample_rate))

                # Chunked HTTP can split a 4-byte float across reads, which a
                # WebSocket message never did. Carry the remainder forward or the
                # samples desynchronise and the audio turns to noise.
                remainder = b""
                async for chunk in resp.content.iter_any():
                    if not chunk:
                        continue
                    if first_audio:
                        first_audio = False
                        await self.stop_ttfb_metrics()
                    buf = remainder + chunk
                    usable = len(buf) - (len(buf) % 4)
                    remainder = buf[usable:]
                    if not usable:
                        continue
                    f32 = np.frombuffer(buf[:usable], dtype=np.float32)
                    pcm = (
                        np.clip(f32 * self._gain, -1.0, 1.0) * 32767.0
                    ).astype(np.int16).tobytes()
                    yield TTSAudioRawFrame(
                        audio=pcm,
                        sample_rate=out_rate,
                        num_channels=1,
                    )

            yield TTSStoppedFrame()

        except aiohttp.ClientError as e:
            yield ErrorFrame(f"Connection error: {e}")
        except asyncio.TimeoutError:
            yield ErrorFrame("Request timeout")
        except Exception as e:
            yield ErrorFrame(f"TTS error: {e}")
        finally:
            if should_close and session:
                await session.close()