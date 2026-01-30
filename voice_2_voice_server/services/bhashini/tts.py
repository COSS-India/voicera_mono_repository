import asyncio
import base64
import json
import os
from typing import AsyncGenerator

import aiohttp
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

from loguru import logger

class BhashiniTTSService(TTSService):

    def __init__(
        self,
        *,
        speaker: str = "Divya",
        description: str = "A clear, natural voice with good audio quality.",
        sample_rate: int = 44100,
        play_steps_in_s: float = 0.5,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        server_url = os.getenv("BHASHINI_TTS_SERVER_URL")
        if not server_url:
            raise ValueError("BHASHINI_TTS_SERVER_URL environment variable not set")

        self._server_url = f"{server_url.rstrip('/')}/tts/stream"
        self._auth_token = os.getenv("BHASHINI_TTS_AUTH_TOKEN")
        if not self._auth_token:
            raise ValueError("BHASHINI_TTS_AUTH_TOKEN environment variable not set")
        self._speaker = speaker
        self._description = description
        self._play_steps_in_s = play_steps_in_s
        self._play_steps_in_s = play_steps_in_s

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            return

        timeout = aiohttp.ClientTimeout(total=120)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
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
                    headers={
                        "Accept": "application/x-ndjson",
                        "Authorization": f"Bearer {self._auth_token}",
                    },
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
