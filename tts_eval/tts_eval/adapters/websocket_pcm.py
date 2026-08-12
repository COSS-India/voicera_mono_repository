"""Adapter for streaming WebSocket TTS servers that emit raw PCM frames.

Covers the wire contract used by the on-prem servers in this monorepo — one JSON
request, a ``meta`` reply, N binary PCM frames, a ``done`` reply — which is what
both ``indic_mio_tts_server`` and ``ai4bharat_tts_server`` speak. It is written
against the *shape* of that protocol rather than either server, so the field
names, message-type names and sample encoding are all config:

    adapter: websocket_pcm
    adapter_config:
      url: ws://localhost:8003
      encoding: float32
      field_map: {text: prompt, voice: voice, language: language}
      static_fields: {description: "A clear, natural voice."}
      message_types: {meta: meta, done: done, error: error}

That is why Indic-Mio and AI4Bharat Parler need zero adapter code between them:
they differ only in URL and static fields.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from ..errors import ConfigError, SynthesisFailed
from ..types import Capabilities, SynthesisRequest
from .base import TTSAdapter, _Capture, register_adapter

# Defaults matching the servers in this repo, so a card only overrides what differs.
_DEFAULT_FIELD_MAP = {
    "text": "prompt",
    "voice": "voice",
    "language": "language",
    "seed": "seed",
}
_DEFAULT_MESSAGE_TYPES = {"meta": "meta", "done": "done", "error": "error"}


@register_adapter
class WebSocketPCMAdapter(TTSAdapter):
    name = "websocket_pcm"
    requires = ("ws",)

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self._ws_mod = self._require("websockets", "ws")

        url = str(self.config.get("url") or "").strip()
        if not url:
            raise ConfigError("websocket_pcm adapter requires 'url' in adapter_config")
        self._url = _normalise_ws_url(url)

        self._encoding = str(self.config.get("encoding") or "float32")
        self._field_map = {**_DEFAULT_FIELD_MAP, **(self.config.get("field_map") or {})}
        self._static_fields = dict(self.config.get("static_fields") or {})
        self._message_types = {**_DEFAULT_MESSAGE_TYPES, **(self.config.get("message_types") or {})}
        self._type_field = str(self.config.get("type_field") or "type")
        self._sample_rate_field = str(self.config.get("sample_rate_field") or "sample_rate")
        self._open_timeout = float(self.config.get("open_timeout_s") or 15.0)
        self._recv_timeout = float(self.config.get("recv_timeout_s") or 120.0)
        # Indic-Mio reads emotion from a sentence-end tag inside the text itself
        # rather than a protocol field, so styling is expressed as a text template.
        self._text_template = str(self.config.get("text_template") or "{text}")
        # Parler-style servers carry voice AND language inside a free-text
        # `description` prompt rather than structured fields, so the description is a
        # template too. When set it is formatted with {voice}/{language}/{emotion}
        # and written to `description_field`, letting a card make its declared voices
        # actually select something. Absent -> the static `description` (if any) is
        # sent unchanged.
        self._description_template = self.config.get("description_template")
        self._description_field = str(self.config.get("description_field") or "description")

    def _build_capabilities(self, config: Mapping[str, Any]) -> Capabilities:
        base = super()._build_capabilities(config)
        # A WS PCM server streams by definition; the card cannot claim otherwise.
        return Capabilities(
            streaming=True,
            voices=base.voices,
            languages=base.languages,
            supports_seed=base.supports_seed,
            supports_emotion=base.supports_emotion,
            native_sample_rate=base.native_sample_rate,
            determinism=base.determinism,
        )

    # ------------------------------------------------------------------
    async def probe(self) -> None:
        """Open and immediately close a connection so an unreachable server fails
        the run in one second rather than N synthesis timeouts."""
        try:
            async with self._ws_mod.connect(
                self._url, open_timeout=self._open_timeout, max_size=None
            ):
                pass
        except Exception as e:  # noqa: BLE001 - surfaced as a fatal config problem
            from ..errors import AdapterUnavailable

            raise AdapterUnavailable(f"cannot reach {self._url}: {type(e).__name__}: {e}") from e

    def _payload(self, request: SynthesisRequest) -> dict[str, Any]:
        text = self._text_template.format(
            text=request.text,
            emotion=request.params.get("emotion", ""),
            language=request.language,
        ).strip()

        payload: dict[str, Any] = dict(self._static_fields)
        payload[self._field_map["text"]] = text
        if self._description_template:
            payload[self._description_field] = self._description_template.format(
                voice=request.voice or "",
                language=request.language,
                emotion=request.params.get("emotion", ""),
            ).strip()
        if request.voice is not None and self._field_map.get("voice"):
            payload[self._field_map["voice"]] = request.voice
        if self._field_map.get("language"):
            payload[self._field_map["language"]] = request.language
        if request.seed is not None and self.capabilities.supports_seed:
            payload[self._field_map["seed"]] = request.seed
        # Generation overrides are passed through under their own names so a card
        # can expose temperature/top_p without an adapter change. `emotion` is
        # consumed by the text template above, not sent as a field.
        for k, v in request.params.items():
            if k != "emotion":
                payload[k] = v
        return payload

    async def _synthesise(self, request: SynthesisRequest, capture: _Capture) -> None:
        import asyncio

        # Seed the expected rate before connecting so the first-audible
        # interpolation in _Capture is accurate even if `meta` is slow.
        if self.capabilities.native_sample_rate:
            capture.sample_rate = self.capabilities.native_sample_rate

        try:
            ws_cm = self._ws_mod.connect(
                self._url, open_timeout=self._open_timeout, max_size=None
            )
        except Exception as e:  # noqa: BLE001
            raise SynthesisFailed(f"connect failed: {type(e).__name__}: {e}") from e

        async with ws_cm as ws:
            await ws.send(json.dumps(self._payload(request), ensure_ascii=False))

            saw_done = False
            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=self._recv_timeout)
                except asyncio.TimeoutError as e:
                    raise SynthesisFailed(
                        f"no frame for {self._recv_timeout:.0f}s (server stalled)"
                    ) from e
                except Exception as e:  # noqa: BLE001 - includes ConnectionClosed
                    if type(e).__name__.startswith("ConnectionClosed"):
                        break
                    raise SynthesisFailed(f"recv failed: {type(e).__name__}: {e}") from e

                if isinstance(message, (bytes, bytearray)):
                    if not message:
                        continue
                    capture.chunk(self.to_float32(bytes(message), self._encoding))
                    continue

                # Text frame: control message.
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as e:
                    raise SynthesisFailed(f"server sent non-JSON text frame: {e}") from e

                kind = data.get(self._type_field)
                if kind == self._message_types["error"]:
                    raise SynthesisFailed(str(data.get("message") or "server reported error"))
                if kind == self._message_types["meta"]:
                    capture.meta(
                        sample_rate=data.get(self._sample_rate_field),
                        **{
                            k: v
                            for k, v in data.items()
                            if k not in (self._type_field, self._sample_rate_field)
                        },
                    )
                    continue
                if kind == self._message_types["done"]:
                    saw_done = True
                    break
                # Unknown control frames are kept, not dropped: they are often
                # the provider's own token/latency telemetry, which is worth
                # having in the record.
                capture.meta(**{f"extra_{kind or 'unknown'}": data})

            if not saw_done:
                raise SynthesisFailed("connection closed before 'done' (truncated response)")


def _normalise_ws_url(raw: str) -> str:
    """Accept http(s):// or ws(s):// and strip a trailing REST path.

    Operators paste whatever is in their .env — usually the http URL used by the
    voice pipeline — so silently accepting it avoids a class of config mistake
    that would otherwise look like an unreachable server.
    """
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


__all__ = ["WebSocketPCMAdapter"]
