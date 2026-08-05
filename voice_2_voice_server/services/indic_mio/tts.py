"""Pipecat TTS adapter for the on-prem Indic-Mio server.

Mirrors services/ai4bharat/tts.py (IndicParlerRESTTTSService): same WebSocket
contract (meta / float32 PCM / done), same float32->int16 + gain conversion. The
`speaker` field carries the preset voice id and is forwarded to the server as
"voice" (selects the speaker embedding); unknown/empty -> the server's default
voice. `description` is informational. An optional `emotion` maps to the model's
sentence-end tag (e.g. "<happy>").
"""
import asyncio
import json
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

# Emotion tags supported by Indic-Mio (Indian-language set; English adds a few more).
_VALID_EMOTIONS = {
    "happy",
    "sad",
    "angry",
    "disgust",
    "fear",
    "surprise",
    "enunciated",
    "confused",
    "whisper",
}


def _parse_gain() -> float:
    raw = os.getenv("INDIC_MIO_GAIN", "1.0").strip()
    try:
        gain = float(raw)
    except ValueError:
        logger.warning("Invalid INDIC_MIO_GAIN={!r}, using 1.0", raw)
        return 1.0
    if gain <= 0:
        logger.warning("INDIC_MIO_GAIN must be > 0, got {}, using 1.0", gain)
        return 1.0
    if gain > 4.0:
        logger.warning("INDIC_MIO_GAIN={} is very high; audio clipping may occur", gain)
    return gain


def _ws_url(raw: str) -> str:
    base = raw.strip().rstrip("/")
    for suffix in ("/tts/stream", "/tts"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    low = base.lower()
    if low.startswith("https://"):
        return "wss://" + base[8:]
    if low.startswith("http://"):
        return "ws://" + base[7:]
    return base


class IndicMioRESTTTSService(TTSService):
    def __init__(
        self,
        *,
        speaker: str | None = None,
        description: str = "A clear, natural voice with good audio quality.",
        language_id: str = "hi",
        emotion: str | None = None,
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        server_url = os.getenv("INDIC_MIO_SERVER_URL")
        if not server_url:
            raise ValueError("INDIC_MIO_SERVER_URL environment variable not set")
        self._ws_url = _ws_url(server_url)
        self._speaker = speaker
        self._description = description
        self._language_id = language_id
        self._emotion = self._normalize_emotion(emotion)
        self._gain = _parse_gain()
        self._session: aiohttp.ClientSession | None = None

    @staticmethod
    def _normalize_emotion(emotion: str | None) -> str | None:
        if not emotion:
            return None
        tag = emotion.strip().strip("<>").lower()
        if tag not in _VALID_EMOTIONS:
            logger.warning("Ignoring unsupported Indic-Mio emotion {!r}", emotion)
            return None
        return tag

    def _prompt_for_server(self, text: str) -> str:
        # Indic-Mio reads emotion from a sentence-end tag in the text itself.
        if self._emotion:
            return f"{text} <{self._emotion}>"
        return text

    async def start(self, frame: Frame):
        if self._gain != 1.0:
            logger.info("Starting Indic-Mio TTS service (output gain={})", self._gain)
        else:
            logger.info("Starting Indic-Mio TTS service")
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        await super().start(frame)

    def can_generate_metrics(self) -> bool:
        return True

    async def stop(self, frame: Frame):
        logger.info("Stopping Indic-Mio TTS service")
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
            async with session.ws_connect(self._ws_url, autoping=True) as ws:
                await ws.send_str(
                    json.dumps(
                        {
                            "prompt": self._prompt_for_server(text),
                            "voice": self._speaker,
                            "description": self._description,
                            "language": self._language_id,
                        }
                    )
                )

                out_rate = self.sample_rate
                completed = False

                while True:
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                        except json.JSONDecodeError as e:
                            yield ErrorFrame(f"Invalid server JSON: {e}")
                            return
                        kind = data.get("type")
                        if kind == "error":
                            yield ErrorFrame(str(data.get("message", "TTS error")))
                            return
                        if kind == "meta":
                            out_rate = int(data.get("sample_rate", out_rate))
                        elif kind == "done":
                            completed = True
                            break
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        if not msg.data:
                            continue
                        if first_audio:
                            first_audio = False
                            await self.stop_ttfb_metrics()
                        f32 = np.frombuffer(msg.data, dtype=np.float32)
                        if f32.size == 0:
                            continue
                        pcm = (
                            np.clip(f32 * self._gain, -1.0, 1.0) * 32767.0
                        ).astype(np.int16).tobytes()
                        yield TTSAudioRawFrame(
                            audio=pcm,
                            sample_rate=out_rate,
                            num_channels=1,
                        )
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        yield ErrorFrame(str(ws.exception() or "WebSocket error"))
                        return
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        break

                if not completed:
                    yield ErrorFrame("TTS closed before completion")
                    return

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
