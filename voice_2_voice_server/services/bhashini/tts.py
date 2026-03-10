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
            logger.debug("🔊 Bhashini TTS: run_tts skipped - empty text")
            return

        timeout = aiohttp.ClientTimeout(total=120)
        payload = {
            "text": text,
            "description": self._description,
            "speaker": self._speaker,
            "play_steps_in_s": self._play_steps_in_s,
        }
        logger.info(
            "🔊 Bhashini TTS: calling API url=%s text_len=%d text_preview=%s speaker=%s",
            self._server_url,
            len(text),
            text[:80].replace("\n", " "),
            self._speaker,
        )

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                yield TTSStartedFrame()

                async with session.post(
                    self._server_url,
                    json=payload,
                    headers={
                        "Accept": "application/x-ndjson",
                        "Authorization": f"Bearer {self._auth_token}",
                    },
                ) as response:
                    status = response.status
                    logger.info(
                        "🔊 Bhashini TTS: response status=%s reason=%s content_type=%s",
                        status,
                        getattr(response, "reason", ""),
                        response.headers.get("Content-Type", ""),
                    )
                    if status != 200:
                        body = ""
                        try:
                            body = (await response.read()).decode("utf-8", errors="replace")[:500]
                        except Exception:
                            pass
                        logger.error(
                            "🔊 Bhashini TTS: API error status=%s body=%s",
                            status,
                            body,
                        )
                        yield ErrorFrame(f"Server error: {status}")
                        return

                    buffer = ""
                    chunk_count = 0
                    total_audio_bytes = 0
                    stream_done = False
                    async for chunk in response.content.iter_any():
                        if stream_done:
                            break
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
                                logger.error("🔊 Bhashini TTS: API returned error in payload: %s", data.get("error"))
                                yield ErrorFrame(data["error"])
                                return

                            if data.get("done"):
                                logger.info(
                                    "🔊 Bhashini TTS: stream done chunk_count=%d total_audio_bytes=%d",
                                    chunk_count,
                                    total_audio_bytes,
                                )
                                stream_done = True
                                break

                            if "audio" in data:
                                audio_bytes = base64.b64decode(data["audio"])
                                total_audio_bytes += len(audio_bytes)
                                chunk_count += 1
                                if chunk_count <= 3 or chunk_count % 20 == 0:
                                    logger.info(
                                        "🔊 Bhashini TTS: audio chunk %d len=%d total_so_far=%d",
                                        chunk_count,
                                        len(audio_bytes),
                                        total_audio_bytes,
                                    )
                                yield TTSAudioRawFrame(
                                    audio=audio_bytes,
                                    sample_rate=data.get("sample_rate", self.sample_rate),
                                    num_channels=1,
                                )

                    if chunk_count > 0 and not stream_done:
                        logger.info(
                            "🔊 Bhashini TTS: completed chunk_count=%d total_audio_bytes=%d",
                            chunk_count,
                            total_audio_bytes,
                        )

                yield TTSStoppedFrame()

        except aiohttp.ClientError as e:
            logger.error("🔊 Bhashini TTS: ClientError: %s", e)
            yield ErrorFrame(f"Connection error: {e}")
        except asyncio.TimeoutError:
            logger.error("🔊 Bhashini TTS: Request timeout")
            yield ErrorFrame("Request timeout")
        except Exception as e:
            logger.error("🔊 Bhashini TTS: error: %s", e)
            yield ErrorFrame(f"TTS error: {e}")
