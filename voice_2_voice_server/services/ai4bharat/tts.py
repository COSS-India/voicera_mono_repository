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


class IndicParlerRESTTTSService(TTSService):
    def __init__(
        self,
        *,
        speaker: str | None = "Divya",
        description: str = "A clear, natural voice with good audio quality.",
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        server_url = os.getenv("INDIC_TTS_SERVER_URL")
        if not server_url:
            raise ValueError("INDIC_TTS_SERVER_URL environment variable not set")
        self._ws_url = _ws_url(server_url)
        self._speaker = speaker
        self._description = description
        self._session: aiohttp.ClientSession | None = None

    def _description_for_server(self) -> str:
        if self._speaker:
            return f"{self._speaker}. {self._description}"
        return self._description

    async def start(self, frame: Frame):
        logger.info("Starting IndicParler TTS service")
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)
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

        session = self._session
        should_close = False
        if not session or session.closed:
            logger.warning("TTS session not available, creating temporary session")
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)
            )
            should_close = True

        yield TTSStartedFrame()

        try:
            async with session.ws_connect(self._ws_url, autoping=True) as ws:
                await ws.send_str(
                    json.dumps(
                        {"prompt": text, "description": self._description_for_server()}
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
                            yield ErrorFrame(
                                str(data.get("message", "TTS error"))
                            )
                            return
                        if kind == "meta":
                            out_rate = int(data.get("sample_rate", out_rate))
                        elif kind == "done":
                            completed = True
                            break
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        if not msg.data:
                            continue
                        f32 = np.frombuffer(msg.data, dtype=np.float32)
                        if f32.size == 0:
                            continue
                        pcm = (np.clip(f32, -1.0, 1.0) * 32767.0).astype(
                            np.int16
                        ).tobytes()
                        yield TTSAudioRawFrame(
                            audio=pcm,
                            sample_rate=out_rate,
                            num_channels=1,
                        )
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        yield ErrorFrame(
                            str(ws.exception() or "WebSocket error")
                        )
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