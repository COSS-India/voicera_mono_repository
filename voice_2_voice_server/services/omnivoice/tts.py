"""
OmniVoice TTS Service for Pipecat voice pipelines.

Architecture (optimal, minimal):
  - ref audio is downloaded from MinIO ONCE at start() → bytes in memory
  - POST /voice-prompt: encodes ref audio → VoiceClonePrompt (.pt) on server disk
    → returns prompt_id (a short string).  This happens ONCE per call session.
  - every TTS call: WS /ws/tts with { prompt_id }  — zero audio bytes transferred
  - at stop(): DELETE /voice-prompt/{prompt_id} to free server disk
  - no ref audio → WS /ws/tts with optional instruct for voice design (fastest path)

Environment variables:
  OMNIVOICE_SERVER_URL   Base URL of OmniVoice server  (default: http://localhost:8005)
  MINIO_ENDPOINT         MinIO host:port               (default: localhost:9000)
  MINIO_ACCESS_KEY       MinIO access key              (default: minioadmin)
  MINIO_SECRET_KEY       MinIO secret key              (default: minioadmin)
  MINIO_SECURE           "true" / "false"              (default: false)
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import wave
from typing import AsyncGenerator, Optional

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

# ---------------------------------------------------------------------------
# Language mapping
# ---------------------------------------------------------------------------
OMNIVOICE_LANG_MAP: dict[str, str] = {
    "English": "en",            "English (India)": "en",
    "English (United States)": "en", "Hindi": "hi",
    "Bengali": "bn",            "Tamil": "ta",
    "Telugu": "te",             "Kannada": "kn",
    "Malayalam": "ml",          "Marathi": "mr",
    "Gujarati": "gu",           "Punjabi": "pa",
    "Odia": "or",               "Assamese": "as",
    "Urdu": "ur",               "Nepali": "ne",
    "Sanskrit": "sa",           "Bodo": "brx",
    "Dogri": "doi",             "Konkani": "kok",
    "Kashmiri": "ks",           "Maithili": "mai",
    "Manipuri": "mni",          "Santali": "sat",
    "Sindhi": "sd",             "Chinese": "zh",
    "Japanese": "ja",           "Korean": "ko",
    "French": "fr",             "German": "de",
    "Spanish": "es",            "Portuguese": "pt",
    "Arabic": "ar",             "Russian": "ru",
    "Italian": "it",            "Dutch": "nl",
    "Turkish": "tr",            "Polish": "pl",
    "Vietnamese": "vi",         "Thai": "th",
    "Indonesian": "id",         "Malay": "ms",
}

_LANG_CODE_CACHE: dict[str, str] = {k.lower(): v for k, v in OMNIVOICE_LANG_MAP.items()}


def _resolve_lang_code(language: Optional[str]) -> str:
    if not language:
        return "en"
    code = _LANG_CODE_CACHE.get(language.lower())
    if code:
        return code
    if len(language) <= 3:
        return language.lower()
    logger.warning(f"OmniVoice: unknown language '{language}', defaulting to 'en'")
    return "en"


def _derive_ws_url(raw: str) -> str:
    base = raw.strip().rstrip("/")
    low = base.lower()
    if low.startswith("https://"):
        return "wss://" + base[8:] + "/ws/tts"
    if low.startswith("http://"):
        return "ws://" + base[7:] + "/ws/tts"
    if not base.endswith("/ws/tts"):
        return base + "/ws/tts"
    return base


def _wav_bytes_to_s16le(wav_bytes: bytes, target_rate: int) -> bytes:
    """Decode OmniVoice WAV (typically 24 kHz) → mono s16le PCM at target_rate.

    CRITICAL: OmniVoice always synthesises at 24 kHz. The pipeline (browser /
    telephony) usually runs at 16 kHz or 8 kHz. Labelling 24 kHz PCM as 16 kHz
    without resampling produces the classic slow / deep / "ghostly" voice.
    """
    import soxr

    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            src_rate = wf.getframerate()
            raw_pcm = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
    elif sampwidth == 4:
        samples = (np.frombuffer(raw_pcm, dtype=np.int32) >> 16).astype(np.float32)
    elif sampwidth == 3:
        arr = np.frombuffer(raw_pcm, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((arr.shape[0], 4), dtype=np.uint8)
        padded[:, 1:] = arr
        samples = (np.frombuffer(padded.tobytes(), dtype=np.int32) >> 8).astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if src_rate != target_rate and len(samples) > 0:
        samples = soxr.resample(samples, src_rate, target_rate)

    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# MinIO download helper (sync, run in thread)
# ---------------------------------------------------------------------------

def _download_from_minio(ref_audio_key: str) -> bytes:
    """Download ref audio bytes from MinIO. key format: 'org_id/uuid_filename'."""
    from minio import Minio

    client = Minio(
        os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )
    bucket = "tts-ref-audio"
    response = client.get_object(bucket, ref_audio_key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


# ---------------------------------------------------------------------------
# OmniVoiceTTSService
# ---------------------------------------------------------------------------

class OmniVoiceTTSService(TTSService):
    """
    Pipecat TTSService for OmniVoice.

    On start():
      - If ref_audio_key is set, downloads from MinIO once → writes to /tmp/
      - The temp file path is reused for every TTS call (voice reuse, zero per-call I/O)

    On run_tts():
      - If ref audio is available: uses WS with ref_audio_path (voice clone)
      - If instruct is set: uses WS with instruct (voice design)
      - Otherwise: uses WS with no voice params (model default)

    Args:
        server_url:     OmniVoice server base URL (env: OMNIVOICE_SERVER_URL)
        language:       Language display name or ISO code
        ref_audio_key:  MinIO object key "org_id/uuid_filename.wav" (voice clone)
        ref_text:       Optional transcript of reference audio
        instruct:       Voice design tags e.g. "male, young adult, indian accent"
        speed:          Speed multiplier 0.5–2.0 (default 1.0)
        sample_rate:    Pipeline sample rate (default 24000)
    """

    def __init__(
        self,
        *,
        server_url: Optional[str] = None,
        language: str = "English",
        ref_audio_key: Optional[str] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: int = 24_000,
        **kwargs,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)

        raw_url = server_url or os.getenv("OMNIVOICE_SERVER_URL", "http://localhost:8005")
        self._ws_url = _derive_ws_url(raw_url)
        self._language = _resolve_lang_code(language)
        self._ref_audio_key = ref_audio_key   # MinIO key — downloaded once in start()
        self._ref_text = ref_text
        self._instruct = instruct
        self._speed = float(speed)
        self._session: Optional[aiohttp.ClientSession] = None
        self._ref_audio_bytes: Optional[bytes] = None  # downloaded from MinIO once
        self._prompt_id: Optional[str] = None           # returned by POST /voice-prompt

        logger.info(
            f"OmniVoiceTTSService: url={self._ws_url} lang={self._language} "
            f"ref_key={ref_audio_key} instruct={instruct} speed={speed}"
        )

    def can_generate_metrics(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: Frame) -> None:
        logger.info("Starting OmniVoice TTS service")
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        # Download ref audio from MinIO once, then register as a VoiceClonePrompt
        # on the OmniVoice server so every TTS call sends only a prompt_id (no bytes).
        if self._ref_audio_key:
            try:
                self._ref_audio_bytes = await asyncio.to_thread(
                    _download_from_minio, self._ref_audio_key
                )
                logger.info(
                    f"OmniVoice: ref audio downloaded for key '{self._ref_audio_key}' "
                    f"({len(self._ref_audio_bytes)} bytes)"
                )
            except Exception as exc:
                logger.error(f"OmniVoice: failed to download ref audio '{self._ref_audio_key}': {exc}")

        if self._ref_audio_bytes and self._session:
            try:
                self._prompt_id = await self._register_voice_prompt(
                    self._session, self._ref_audio_bytes, self._ref_text
                )
                logger.info(f"OmniVoice: voice prompt registered: {self._prompt_id}")
            except Exception as exc:
                logger.error(f"OmniVoice: failed to register voice prompt: {exc}")
                # Fall through — will use REST /tts with raw bytes as fallback

        await super().start(frame)

    async def stop(self, frame: Frame) -> None:
        logger.info("Stopping OmniVoice TTS service")
        # Delete the cached voice prompt from the server to free disk space
        if self._prompt_id and self._session and not self._session.closed:
            try:
                rest_url = self._ws_url.replace("wss://", "https://").replace("ws://", "http://")
                rest_url = rest_url.replace("/ws/tts", "")
                async with self._session.delete(
                    f"{rest_url}/voice-prompt/{self._prompt_id}"
                ) as resp:
                    logger.info(f"OmniVoice: voice prompt {self._prompt_id} deleted (status={resp.status})")
            except Exception as exc:
                logger.warning(f"OmniVoice: failed to delete voice prompt: {exc}")
            self._prompt_id = None

        if self._session:
            await self._session.close()
            self._session = None
        self._ref_audio_bytes = None
        await super().stop(frame)

    # ------------------------------------------------------------------
    # Voice prompt registration
    # ------------------------------------------------------------------

    async def _register_voice_prompt(
        self,
        session: aiohttp.ClientSession,
        audio_bytes: bytes,
        ref_text: Optional[str],
    ) -> str:
        """POST audio bytes to /voice-prompt → returns prompt_id."""
        rest_url = self._ws_url.replace("wss://", "https://").replace("ws://", "http://")
        rest_url = rest_url.replace("/ws/tts", "")

        # Preserve original extension so OmniVoice load_audio can decode correctly
        # (browser recordings are often .webm; labeling them .wav breaks soundfile).
        suffix = ".wav"
        content_type = "audio/wav"
        if self._ref_audio_key:
            ext = os.path.splitext(self._ref_audio_key)[-1].lower()
            if ext in (".webm", ".ogg", ".mp3", ".m4a", ".flac", ".wav"):
                suffix = ext
                content_type = {
                    ".webm": "audio/webm",
                    ".ogg": "audio/ogg",
                    ".mp3": "audio/mpeg",
                    ".m4a": "audio/mp4",
                    ".flac": "audio/flac",
                    ".wav": "audio/wav",
                }[ext]

        form = aiohttp.FormData()
        form.add_field(
            "ref_audio",
            audio_bytes,
            filename=f"ref{suffix}",
            content_type=content_type,
        )
        if ref_text:
            form.add_field("ref_text", ref_text)

        async with session.post(
            f"{rest_url}/voice-prompt",
            data=form,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"POST /voice-prompt failed {resp.status}: {body[:200]}")
            data = await resp.json()
            return data["prompt_id"]

    # ------------------------------------------------------------------
    # Core TTS
    # ------------------------------------------------------------------

    def _get_session(self) -> tuple[aiohttp.ClientSession, bool]:
        """Return (session, should_close). should_close=True only for fallback sessions."""
        if self._session and not self._session.closed:
            return self._session, False
        logger.warning("OmniVoice: session not available, creating temporary session")
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)
        ), True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            return

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        try:
            if self._prompt_id:
                # Fastest: WS with pre-encoded voice prompt (no audio bytes transferred)
                async for frame in self._run_tts_ws(text, prompt_id=self._prompt_id):
                    yield frame
            elif self._ref_audio_bytes:
                # Fallback: REST with raw audio bytes (prompt registration failed)
                async for frame in self._run_tts_rest(text):
                    yield frame
            else:
                # Voice design or plain TTS — WS with optional instruct
                async for frame in self._run_tts_ws(text):
                    yield frame
        except Exception as e:
            logger.exception(f"OmniVoice TTS unexpected error: {e}")
            yield ErrorFrame(f"OmniVoice TTS error: {e}")

        yield TTSStoppedFrame()

    async def _run_tts_rest(self, text: str) -> AsyncGenerator[Frame, None]:
        """POST /tts with multipart form — used when we have ref audio bytes."""
        rest_url = self._ws_url.replace("wss://", "https://").replace("ws://", "http://")
        rest_url = rest_url.replace("/ws/tts", "/tts")

        form = aiohttp.FormData()
        form.add_field("text", text)
        form.add_field("language_id", self._language)
        form.add_field(
            "ref_audio",
            self._ref_audio_bytes,
            filename="ref.wav",
            content_type="audio/wav",
        )
        if self._ref_text:
            form.add_field("ref_text", self._ref_text)
        if self._instruct:
            form.add_field("instruct", self._instruct)
        if self._speed != 1.0:
            form.add_field("speed", str(self._speed))

        session, should_close = self._get_session()
        try:
            async with session.post(rest_url, data=form) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    yield ErrorFrame(f"OmniVoice REST error {resp.status}: {body[:200]}")
                    return
                wav_bytes = await resp.read()

            pcm = _wav_bytes_to_s16le(wav_bytes, self.sample_rate)
            await self.stop_ttfb_metrics()

            chunk_size = int(self.sample_rate * 0.020) * 2
            for i in range(0, len(pcm), chunk_size):
                yield TTSAudioRawFrame(
                    audio=pcm[i: i + chunk_size],
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
            logger.debug(
                f"OmniVoice REST done: text='{text[:60]}' pcm_bytes={len(pcm)}"
            )
        except aiohttp.ClientConnectorError as e:
            yield ErrorFrame(f"OmniVoice: cannot connect to server: {e}")
        except aiohttp.ClientError as e:
            yield ErrorFrame(f"OmniVoice: REST connection error: {e}")
        except asyncio.TimeoutError:
            yield ErrorFrame("OmniVoice: REST request timeout")
        finally:
            if should_close:
                await session.close()

    async def _run_tts_ws(
        self, text: str, prompt_id: Optional[str] = None
    ) -> AsyncGenerator[Frame, None]:
        """WS /ws/tts — used for voice cloning via prompt_id or plain/design TTS."""
        import json

        session, should_close = self._get_session()
        try:
            # One automatic retry if the server returns empty/near-empty audio
            # (known OmniVoice issue with short greetings + quiet clone prompts).
            for attempt in range(2):
                async with session.ws_connect(self._ws_url, autoping=True) as ws:
                    payload: dict = {
                        "text": text,
                        "language_id": self._language,
                    }
                    if prompt_id:
                        payload["prompt_id"] = prompt_id
                    elif self._instruct:
                        payload["instruct"] = self._instruct
                    if self._speed != 1.0:
                        payload["speed"] = self._speed

                    await ws.send_json(payload)
                    logger.debug(
                        f"OmniVoice WS sent: text='{text[:60]}' lang={self._language} "
                        f"prompt_id={prompt_id} attempt={attempt + 1}"
                    )

                    got_audio = False
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                            except Exception as e:
                                yield ErrorFrame(f"OmniVoice: invalid JSON: {e}")
                                return

                            status = data.get("status")
                            if status in ("queued", "processing"):
                                continue

                            if status == "done":
                                audio_b64: str = data.get("audio_b64", "")
                                audio_dur = float(data.get("audio_duration") or 0)
                                if not audio_b64 or audio_dur < 0.05:
                                    logger.warning(
                                        f"OmniVoice returned empty audio "
                                        f"(dur={audio_dur}s) for '{text[:40]}' "
                                        f"attempt={attempt + 1}"
                                    )
                                    break  # retry or give up

                                wav_bytes = base64.b64decode(audio_b64)
                                pcm = _wav_bytes_to_s16le(wav_bytes, self.sample_rate)
                                if len(pcm) < int(self.sample_rate * 0.05) * 2:
                                    logger.warning(
                                        f"OmniVoice PCM too short ({len(pcm)} bytes) "
                                        f"for '{text[:40]}' attempt={attempt + 1}"
                                    )
                                    break

                                await self.stop_ttfb_metrics()
                                chunk_size = int(self.sample_rate * 0.020) * 2
                                for i in range(0, len(pcm), chunk_size):
                                    yield TTSAudioRawFrame(
                                        audio=pcm[i: i + chunk_size],
                                        sample_rate=self.sample_rate,
                                        num_channels=1,
                                    )
                                logger.debug(
                                    f"OmniVoice WS done: audio_dur={audio_dur}s "
                                    f"RTF={data.get('rtf', '?')}"
                                )
                                got_audio = True
                                break

                            if status == "error":
                                detail = data.get("detail", "unknown error")
                                logger.error(f"OmniVoice server error: {detail}")
                                yield ErrorFrame(f"OmniVoice TTS error: {detail}")
                                return

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            yield ErrorFrame(str(ws.exception() or "OmniVoice WS error"))
                            return
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                            break

                if got_audio:
                    return

            yield ErrorFrame(f"OmniVoice: empty audio after retries for '{text[:40]}'")

        except aiohttp.ClientConnectorError as e:
            yield ErrorFrame(f"OmniVoice: cannot connect to server: {e}")
        except aiohttp.ClientError as e:
            yield ErrorFrame(f"OmniVoice: WS connection error: {e}")
        except asyncio.TimeoutError:
            yield ErrorFrame("OmniVoice: WS request timeout")
        finally:
            if should_close:
                await session.close()
