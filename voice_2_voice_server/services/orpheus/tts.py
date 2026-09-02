"""Pipecat TTS adapter for the on-prem AI4Bharat Orpheus Indic server.

Mirrors services/ai4bharat/tts.py (IndicParlerRESTTTSService): same lifecycle,
same frame protocol, one utterance per WebSocket connection. Three things differ,
and all three come from the Orpheus wire format:

  * The request carries a strict `voice` (speaker name from GET /v1/voices) rather
    than a speaker prefixed into a free-text description. The speaker name also
    selects the language - every name in the roster belongs to exactly one - so
    `language` is sent only to have the server validate the pair.
  * The first JSON frame is `start`, not Parler's `meta`.
  * The binary frames are ALREADY 24 kHz mono s16le. Parler emits float32 and has
    to convert; here the bytes go straight to TTSAudioRawFrame untouched, so the
    default path does no numpy work at all.

Cancellation is handled by the `async with ws_connect` block: when Pipecat cancels
run_tts on a barge-in, CancelledError is raised at `await ws.receive()` and the
context manager closes the socket. CancelledError derives from BaseException, so
the `except Exception` clauses below do not swallow it.
"""
import asyncio
import json
import os
from typing import AsyncGenerator, Optional

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

# Native output rate of the Orpheus SNAC codec. The transport resamples down to
# the carrier rate (8 kHz telephony / 16 kHz browser), so this is declared, not
# converted, here. See ai4bharat_orpheustts_server/src/orpheus_server/codec.py.
ORPHEUS_SAMPLE_RATE = 24000

# Speaking style used when an agent config does not pick one. Matches
# `default_style` in the server's voices.json.
DEFAULT_STYLE = "CONV"

# Path the WebSocket endpoint is mounted at on the Orpheus server.
_WS_PATH = "/v1/tts/ws"


def _parse_gain() -> float:
    raw = os.getenv("ORPHEUS_TTS_GAIN", "1.0").strip()
    try:
        gain = float(raw)
    except ValueError:
        logger.warning("Invalid ORPHEUS_TTS_GAIN={!r}, using 1.0", raw)
        return 1.0
    if gain <= 0:
        logger.warning("ORPHEUS_TTS_GAIN must be > 0, got {}, using 1.0", gain)
        return 1.0
    if gain > 4.0:
        logger.warning(
            "ORPHEUS_TTS_GAIN={} is very high; audio clipping may occur", gain
        )
    return gain


def _ws_url(raw: str) -> str:
    """Normalize a configured base URL into the full ws:// endpoint.

    Accepts a bare base (`http://host:8004`), a base that already names the
    WebSocket path, or one that names a sibling HTTP synthesis path - so an
    operator who copies the URL out of the Orpheus Readme still gets a working
    value.
    """
    base = raw.strip().rstrip("/")
    for suffix in (_WS_PATH, "/v1/tts/stream", "/v1/tts", "/tts/stream", "/tts"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    low = base.lower()
    if low.startswith("https://"):
        base = "wss://" + base[8:]
    elif low.startswith("http://"):
        base = "ws://" + base[7:]
    return base.rstrip("/") + _WS_PATH


def _apply_gain(pcm: bytes, gain: float) -> bytes:
    """Scale s16le PCM in place of the identity path.

    Only reached when ORPHEUS_TTS_GAIN is set. numpy is imported lazily so the
    default path never pays for it.
    """
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * gain
    return np.clip(samples, -32768.0, 32767.0).astype(np.int16).tobytes()


class OrpheusTTSService(TTSService):
    def __init__(
        self,
        *,
        speaker: str = "Amit",
        language_id: str = "hi",
        style: Optional[str] = DEFAULT_STYLE,
        max_tokens: Optional[int] = None,
        sample_rate: int = ORPHEUS_SAMPLE_RATE,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        server_url = os.getenv("ORPHEUS_TTS_SERVER_URL")
        if not server_url:
            raise ValueError("ORPHEUS_TTS_SERVER_URL environment variable not set")
        self._ws_url = _ws_url(server_url)
        self._speaker = speaker
        self._language_id = language_id
        self._style = style or DEFAULT_STYLE
        self._max_tokens = max_tokens
        self._gain = _parse_gain()
        self._session: aiohttp.ClientSession | None = None

    async def start(self, frame: Frame):
        if self._gain != 1.0:
            logger.info("Starting Orpheus TTS service (output gain={})", self._gain)
        else:
            logger.info("Starting Orpheus TTS service")
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        await super().start(frame)

    def can_generate_metrics(self) -> bool:
        return True

    async def stop(self, frame: Frame):
        logger.info("Stopping Orpheus TTS service")
        if self._session:
            await self._session.close()
            self._session = None
        await super().stop(frame)

    def set_language(self, language_id: str) -> None:
        # Synchronous on purpose: utils/language_switching.py calls this un-awaited.
        self._language_id = language_id
        logger.info(f"TTS language changed to: {language_id}")

    def set_voice(self, speaker: str) -> None:
        self._speaker = speaker
        logger.info(f"TTS speaker changed to: {speaker}")

    def _request_payload(self, text: str) -> dict:
        payload = {
            "text": text,
            "voice": self._speaker,
            "language": self._language_id,
            "style": self._style,
        }
        if self._max_tokens:
            payload["max_tokens"] = self._max_tokens
        return payload

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
                await ws.send_str(json.dumps(self._request_payload(text)))

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
                            # Unknown voice, unknown style, or text that cannot fit
                            # the context all arrive here.
                            yield ErrorFrame(str(data.get("message", "TTS error")))
                            return
                        if kind == "start":
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
                        # Already s16le - no dtype conversion on the default path.
                        pcm = (
                            msg.data
                            if self._gain == 1.0
                            else _apply_gain(msg.data, self._gain)
                        )
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
