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


# What the client can decode, keyed by the X-Audio-Format the server declares.
# `pcm` is OpenAI's own name for 16-bit little-endian mono; `pcm_f32le` is an
# extension Indic Parler serves because float32 is what its engine produces.
#
# The point of decoding by header rather than by model name: two TTS models here
# disagree on both sample width and rate -- Orpheus is 24 kHz s16le, Indic Parler
# 44.1 kHz f32le -- and the client must not have to know which one is loaded.
# A model that declares something not in this table gets a clear error naming the
# format, never silence or noise.
_DECODERS: dict[str, np.dtype] = {
    "pcm_f32le": np.dtype("<f4"),
    "pcm": np.dtype("<i2"),
    "pcm_s16le": np.dtype("<i2"),
}


class ModelServerTTSService(TTSService):
    """Any model-server TTS slot, over the OpenAI speech endpoint.

    Not tied to a model: `voice`, `instructions` and `language` are the OpenAI
    fields, and each model interprets them its own way -- Indic Parler recomposes
    them into a free-text style prompt, Orpheus reads `voice` as a speaker name
    that also picks the language. Neither needs client code of its own.
    """
    def __init__(
        self,
        *,
        speaker: str | None = "Divya",
        description: str = "A clear, natural voice with good audio quality.",
        language_id: str = "hi",
        sample_rate: int = 44100,
        audio_format: str = "pcm_f32le",
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
        # What to ask for. The response header decides how it is decoded, so a
        # server that answers in a different supported format still works.
        self._audio_format = audio_format
        self._session: aiohttp.ClientSession | None = None

    def _description_for_server(self) -> str:
        if self._speaker:
            return f"{self._speaker}. {self._description}"
        return self._description

    async def start(self, frame: Frame):
        if self._gain != 1.0:
            logger.info("Starting model-server TTS (format={}, output gain={})",
                        self._audio_format, self._gain)
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

    def _to_int16(self, samples: np.ndarray) -> bytes:
        """Whatever arrived, hand Pipecat 16-bit PCM.

        A server that already sent int16 and a gain of 1.0 is a pure passthrough:
        no arithmetic, so no chance of changing audio that was already correct.
        """
        if samples.dtype.kind == "i":
            if self._gain == 1.0:
                return samples.tobytes()
            scaled = samples.astype(np.float32) * self._gain
            return np.clip(scaled, -32768.0, 32767.0).astype(np.int16).tobytes()
        scaled = np.clip(samples.astype(np.float32) * self._gain, -1.0, 1.0) * 32767.0
        return scaled.astype(np.int16).tobytes()
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
                "response_format": self._audio_format,
            }
            async with session.post(self._speech_url, json=payload) as resp:
                if resp.status != 200:
                    detail = (await resp.text())[:200]
                    yield ErrorFrame(f"TTS request failed: {resp.status} {detail}")
                    return

                # The server says what it sent. Fall back to what we asked for
                # only if it stayed silent, and to our configured rate likewise.
                wire = resp.headers.get("X-Audio-Format", self._audio_format)
                out_rate = int(resp.headers.get("X-Sample-Rate", self.sample_rate))
                dtype = _DECODERS.get(wire)
                if dtype is None:
                    yield ErrorFrame(
                        f"TTS server sent {wire!r}, which this client cannot decode. "
                        f"Supported: {', '.join(sorted(_DECODERS))}."
                    )
                    return
                width = dtype.itemsize

                # Chunked HTTP can split a sample across reads, which a WebSocket
                # message never did. Carry the remainder forward or the samples
                # desynchronise and the audio turns to noise.
                remainder = b""
                async for chunk in resp.content.iter_any():
                    if not chunk:
                        continue
                    if first_audio:
                        first_audio = False
                        await self.stop_ttfb_metrics()
                    buf = remainder + chunk
                    usable = len(buf) - (len(buf) % width)
                    remainder = buf[usable:]
                    if not usable:
                        continue
                    yield TTSAudioRawFrame(
                        audio=self._to_int16(np.frombuffer(buf[:usable], dtype=dtype)),
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


# The name the voice server has always constructed. Kept so nothing downstream
# has to change; the class itself is no longer Parler-specific.
IndicParlerRESTTTSService = ModelServerTTSService
