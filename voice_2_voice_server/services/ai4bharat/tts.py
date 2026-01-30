import asyncio
import base64
import json
import os
from typing import AsyncGenerator

import aiohttp
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

class IndicParlerRESTTTSService(TTSService):

    def __init__(
        self,
        *,
        speaker: str = "Divya",
        description: str = "A clear, natural voice with good audio quality.",
        sample_rate: int = 44100,
        play_steps_in_s: float = 0.15,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        server_url = os.getenv("INDIC_TTS_SERVER_URL")
        if not server_url:
            raise ValueError("INDIC_TTS_SERVER_URL environment variable not set")

        self._server_url = f"{server_url.rstrip('/')}/tts/stream"
        self._speaker = speaker
        self._description = description
        self._play_steps_in_s = play_steps_in_s
        self._session = None

    async def start(self, frame: Frame):
        logger.info("Starting IndicParler TTS service")
        # Optimization: Use TCPConnector with keepalive to ensure connection reuse
        # limit=0 means no limit on concurrent connections
        # ttl_dns_cache=300 caches DNS lookups for 5 minutes
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=120, connect=10)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        await super().start(frame)

    async def stop(self, frame: Frame):
        logger.info("Stopping IndicParler TTS service")
        if self._session:
            await self._session.close()
            self._session = None
        await super().stop(frame)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            return

        # Use persistent session if available, otherwise fall back to temporary (safety)
        session = self._session
        should_close = False
        if not session or session.closed:
            logger.warning("TTS session not available, creating temporary session")
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
            should_close = True

        try:
            payload = {
                "text": text,
                "description": self._description,
                "speaker": self._speaker,
                "play_steps_in_s": self._play_steps_in_s,
            }

            yield TTSStartedFrame()

            async with session.post(
                self._server_url,
                json=payload,
                headers={"Accept": "application/x-ndjson"},
            ) as response:
                if response.status != 200:
                    yield ErrorFrame(f"Server error: {response.status}")
                    return

                buffer = ""
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue

                    buffer += chunk.decode("utf-8")
                    
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.strip():
                            continue

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if "error" in data:
                            yield ErrorFrame(data["error"])
                            return

                        if data.get("done"):
                            break

                        if "audio" in data:
                            audio_bytes = base64.b64decode(data["audio"])
                            logger.info(f"Audio chunk sent to Telephony: {len(audio_bytes)} bytes")
                            yield TTSAudioRawFrame(
                                audio=audio_bytes,
                                sample_rate=data.get("sample_rate", self.sample_rate),
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
