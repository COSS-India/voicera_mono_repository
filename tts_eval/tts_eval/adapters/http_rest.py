"""Adapter for HTTP/REST TTS APIs (streaming or single-shot).

One adapter covers the whole commercial-API family — Sarvam, ElevenLabs,
Cartesia, OpenAI, Google, or any in-house REST endpoint — because the parts that
differ between them are all data: URL, headers, request field names, and how the
audio comes back. A new REST provider is a model card, not code.

Supported response shapes:

*   ``audio_stream``  — chunked ``audio/*`` body. Timings are real per-chunk
    arrival times, so TTFB is meaningful.
*   ``json_base64``   — a JSON body with base64 audio at a configured JSON path
    (e.g. Sarvam's ``audios[0]``). Non-streaming by definition: TTFB equals total
    time, and the harness marks streaming-jitter metrics ``not_applicable``
    rather than inventing a number.

Container handling is explicit: ``raw`` PCM and ``wav`` are decoded in-process;
``mp3``/``opus`` are refused with a clear message instead of being half-decoded,
because a wrong decode would corrupt every audio-quality metric downstream.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import wave
from typing import Any, Mapping

import numpy as np

from ..errors import ConfigError, SynthesisFailed
from ..types import Capabilities, SynthesisRequest
from .base import TTSAdapter, _Capture, register_adapter

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@register_adapter
class HTTPRestAdapter(TTSAdapter):
    name = "http_rest"
    requires = ("http",)

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self._aiohttp = self._require("aiohttp", "http")

        self._url = str(self.config.get("url") or "").strip()
        if not self._url:
            raise ConfigError("http_rest adapter requires 'url' in adapter_config")

        self._method = str(self.config.get("method") or "POST").upper()
        # Header values may reference env vars as ${NAME} so a model card can be
        # committed to git without embedding credentials.
        self._headers = {k: _expand_env(str(v)) for k, v in (self.config.get("headers") or {}).items()}
        self._field_map = {
            "text": "text",
            "voice": "voice",
            "language": "language",
            "seed": "seed",
            **(self.config.get("field_map") or {}),
        }
        self._static_fields = dict(self.config.get("static_fields") or {})
        self._response_format = str(self.config.get("response_format") or "audio_stream")
        if self._response_format not in ("audio_stream", "json_base64"):
            raise ConfigError(
                f"unsupported response_format {self._response_format!r}; "
                "expected 'audio_stream' or 'json_base64'"
            )
        self._json_audio_path = str(self.config.get("json_audio_path") or "audio")
        self._container = str(self.config.get("container") or "raw").lower()
        self._encoding = str(self.config.get("encoding") or "int16")
        self._timeout_s = float(self.config.get("timeout_s") or 120.0)
        self._text_template = str(self.config.get("text_template") or "{text}")
        self._session = None

    def _build_capabilities(self, config: Mapping[str, Any]) -> Capabilities:
        base = super()._build_capabilities(config)
        streaming = str(config.get("response_format") or "audio_stream") == "audio_stream"
        return Capabilities(
            streaming=streaming,
            voices=base.voices,
            languages=base.languages,
            supports_seed=base.supports_seed,
            supports_emotion=base.supports_emotion,
            native_sample_rate=base.native_sample_rate,
            determinism=base.determinism,
        )

    # ------------------------------------------------------------------
    async def aopen(self) -> None:
        timeout = self._aiohttp.ClientTimeout(total=None, connect=15, sock_read=self._timeout_s)
        # limit=0 (unbounded) so the harness's own concurrency setting is the only
        # throttle; an aiohttp pool cap would silently serialise requests and turn
        # a throughput measurement into a queueing measurement.
        connector = self._aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        self._session = self._aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _payload(self, request: SynthesisRequest) -> dict[str, Any]:
        text = self._text_template.format(
            text=request.text,
            emotion=request.params.get("emotion", ""),
            language=request.language,
        ).strip()
        payload: dict[str, Any] = dict(self._static_fields)
        payload[self._field_map["text"]] = text
        if request.voice is not None and self._field_map.get("voice"):
            payload[self._field_map["voice"]] = request.voice
        if self._field_map.get("language"):
            payload[self._field_map["language"]] = request.language
        if request.seed is not None and self.capabilities.supports_seed:
            payload[self._field_map["seed"]] = request.seed
        for k, v in request.params.items():
            if k != "emotion":
                payload[k] = v
        return payload

    async def _synthesise(self, request: SynthesisRequest, capture: _Capture) -> None:
        session = self._session
        if session is None or session.closed:
            raise SynthesisFailed("HTTP session not open; aopen() was not called")

        if self.capabilities.native_sample_rate:
            capture.sample_rate = self.capabilities.native_sample_rate

        url = self._url.format(
            voice=request.voice or "", language=request.language, model=self.config.get("model", "")
        )
        try:
            async with session.request(
                self._method, url, json=self._payload(request), headers=self._headers
            ) as resp:
                if resp.status >= 400:
                    body = (await resp.text())[:400]
                    raise SynthesisFailed(f"HTTP {resp.status}: {body}")

                if self._response_format == "json_base64":
                    body = await resp.json(content_type=None)
                    raw = self._extract_base64(body)
                    capture.chunk(self._decode(raw, capture))
                    return

                # Streaming body: chunk boundaries are the provider's, and each
                # is timestamped on arrival by capture.chunk().
                pending = b""
                header_stripped = self._container != "wav"
                # Sample encoding for post-header chunks. For a streamed WAV we
                # take it from the header rather than the card, because a card
                # that disagrees with the actual header would silently halve or
                # double every amplitude-based metric.
                stream_encoding = self._encoding
                async for chunk in resp.content.iter_any():
                    if not chunk:
                        continue
                    if not header_stripped:
                        # A streamed WAV puts its header in the first chunk(s);
                        # buffer until it parses, then discard it.
                        pending += chunk
                        if len(pending) < 44:
                            continue
                        samples, rate, width = _parse_wav_bytes(pending, allow_partial=True)
                        if rate:
                            capture.sample_rate = rate
                        if width:
                            stream_encoding = _WIDTH_TO_ENCODING[width]
                        header_stripped = True
                        pending = b""
                        if samples.size:
                            capture.chunk(samples)
                        continue
                    capture.chunk(self.to_float32(chunk, stream_encoding))
        except SynthesisFailed:
            raise
        except Exception as e:  # noqa: BLE001
            raise SynthesisFailed(f"{type(e).__name__}: {e}") from e

    # ------------------------------------------------------------------
    def _extract_base64(self, body: Any) -> bytes:
        """Pull base64 audio out of a JSON body at ``json_audio_path``.

        Path syntax is ``a.b[0].c`` — enough for every REST TTS response seen in
        practice (Sarvam's ``audios[0]``, OpenAI-style ``data.audio``), and small
        enough to read at a glance.
        """
        node: Any = body
        for token in _parse_json_path(self._json_audio_path):
            if isinstance(token, int):
                if not isinstance(node, (list, tuple)) or token >= len(node):
                    raise SynthesisFailed(
                        f"json_audio_path {self._json_audio_path!r}: index [{token}] out of range"
                    )
                node = node[token]
            else:
                if not isinstance(node, Mapping) or token not in node:
                    available = (
                        ", ".join(sorted(map(str, node))) if isinstance(node, Mapping) else "-"
                    )
                    raise SynthesisFailed(
                        f"json_audio_path {self._json_audio_path!r}: key {token!r} not in response "
                        f"(available: {available})"
                    )
                node = node[token]
        if not isinstance(node, str):
            raise SynthesisFailed(
                f"json_audio_path {self._json_audio_path!r} resolved to {type(node).__name__}, "
                "expected a base64 string"
            )
        try:
            return base64.b64decode(node)
        except Exception as e:  # noqa: BLE001
            raise SynthesisFailed(f"response audio is not valid base64: {e}") from e

    def _decode(self, raw: bytes, capture: _Capture) -> np.ndarray:
        if self._container == "wav":
            samples, rate, _ = _parse_wav_bytes(raw, allow_partial=False)
            if rate:
                capture.sample_rate = rate
            return samples
        if self._container in ("mp3", "opus", "ogg", "flac", "aac"):
            raise SynthesisFailed(
                f"container {self._container!r} needs an external decoder; configure the "
                "provider to return wav or raw PCM, or pre-decode and use the replay adapter"
            )
        return self.to_float32(raw, self._encoding)


_WIDTH_TO_ENCODING = {1: "uint8", 2: "int16", 4: "int32"}


def _parse_json_path(path: str) -> list[str | int]:
    """``"audios[0].data"`` -> ``["audios", 0, "data"]``."""
    tokens: list[str | int] = []
    for part in path.split("."):
        m = re.fullmatch(r"([^\[\]]*)((?:\[\d+\])*)", part)
        if m is None:
            raise ConfigError(f"malformed json_audio_path segment {part!r} in {path!r}")
        key, indices = m.group(1), m.group(2)
        if key:
            tokens.append(key)
        tokens.extend(int(i) for i in re.findall(r"\[(\d+)\]", indices))
    if not tokens:
        raise ConfigError(f"json_audio_path {path!r} is empty")
    return tokens


def _parse_wav_bytes(
    raw: bytes, *, allow_partial: bool
) -> tuple[np.ndarray, int | None, int | None]:
    """Decode WAV bytes to (float32 mono, sample_rate, sample_width).

    ``allow_partial`` tolerates a truncated body — the first chunk of a streamed
    WAV — by reading whatever frames are actually present rather than trusting
    the frame count the header declares.
    """
    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            rate = wf.getframerate()
            width = wf.getsampwidth()
            channels = wf.getnchannels()
            declared = wf.getnframes()
            if allow_partial:
                # Header frame counts are unreliable mid-stream (often 0 or the
                # full expected length), so bound the read by the bytes we hold.
                available = max(0, len(raw) - 44) // max(1, width * channels)
                declared = min(declared, available) if declared else available
            frames = wf.readframes(declared)
    except (wave.Error, EOFError) as e:
        if not allow_partial:
            raise SynthesisFailed(f"invalid WAV payload: {e}") from e
        return np.zeros(0, dtype=np.float32), None, None

    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise SynthesisFailed(f"unsupported WAV sample width {width} bytes")

    if channels > 1:
        usable = (data.size // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), rate, width


def _expand_env(value: str) -> str:
    """Substitute ${VAR} from the environment, leaving unknown names untouched.

    Left untouched (rather than blanked) so an auth failure points at the missing
    variable name instead of producing an empty Bearer token.
    """
    return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), m.group(0)), value)


__all__ = ["HTTPRestAdapter"]
