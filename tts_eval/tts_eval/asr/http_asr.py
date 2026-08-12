"""ASR over HTTP — works against any transcription service, config-only.

Preferred backend for Indic evaluation, for two reasons:

*   **No new dependency.** It uses stdlib ``urllib`` and encodes multipart by
    hand, so round-trip CER is available in the light install. Adding a local
    Whisper would pull in torch and a multi-gigabyte model just to score text.
*   **The right model for Indic.** Whisper is comparatively weak on Indian
    languages, so a Whisper-measured CER for Santali or Maithili mostly reports
    Whisper's own error rate. Pointing this backend at an IndicConformer-class
    server gives a far tighter measurement floor. (For English, or for numbers
    comparable to published work, use the Whisper backend instead — the run
    record names whichever was used, and the comparison engine refuses to treat
    CER from different ASR backends as equivalent.)

Request shape is configuration, so the same class serves a multipart file upload,
a JSON base64 body, or a raw audio POST.
"""
from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Mapping

from ..audio import AudioBuffer, resample
from ..errors import MetricUnavailable
from .base import ASRBackend

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class HTTPASRBackend(ASRBackend):
    name = "http_asr"

    def __init__(self, options: Mapping[str, Any] | None = None):
        super().__init__(options)
        self.url = str(self.options.get("url") or "").strip()
        self.method = str(self.options.get("method") or "POST").upper()
        # "multipart" | "json_base64" | "raw"
        self.body_format = str(self.options.get("body_format") or "multipart")
        self.file_field = str(self.options.get("file_field") or "file")
        self.language_field = str(self.options.get("language_field") or "language")
        # Provider-specific language codes, e.g. {"hi": "hi-IN"}. Absent keys pass
        # through unchanged so a new language needs no config edit.
        self.language_map: dict[str, str] = dict(self.options.get("language_map") or {})
        self.extra_fields: dict[str, Any] = dict(self.options.get("extra_fields") or {})
        self.transcript_path = str(self.options.get("transcript_path") or "text")
        # Wire container for the audio bytes: "wav" (default, self-describing header)
        # or "pcm_s16le" (headerless raw 16-bit LE samples). Some servers decode the
        # base64/body straight into an int16 buffer and a WAV header would corrupt the
        # leading samples, so the container is config rather than hard-coded.
        self.audio_format = str(self.options.get("audio_format") or "wav").lower()
        # Most ASR models want 16 kHz mono; resampling here rather than server-side
        # keeps the request small and the rate explicit in the record.
        self.target_sample_rate = int(self.options.get("target_sample_rate") or 16000)
        self.timeout_s = float(self.options.get("timeout_s") or 120.0)
        self.headers = {
            k: _expand_env(str(v)) for k, v in (self.options.get("headers") or {}).items()
        }

    def available(self) -> tuple[bool, str]:
        if not self.url:
            return False, "no 'url' configured for the http_asr backend"
        return True, f"http_asr -> {self.url}"

    def _describe_extra(self) -> dict[str, Any]:
        # The endpoint identity is part of the measurement, so it is recorded.
        return {
            "endpoint": self.url,
            "target_sample_rate": self.target_sample_rate,
            "audio_format": self.audio_format,
        }

    # ------------------------------------------------------------------
    def transcribe(self, audio: AudioBuffer, language: str) -> str:
        resampled = resample(audio, self.target_sample_rate)
        if self.audio_format in ("pcm_s16le", "pcm16", "s16le", "raw_pcm"):
            audio_bytes, filename, audio_mime = _to_pcm16_bytes(resampled), "audio.pcm", "audio/L16"
        elif self.audio_format == "wav":
            audio_bytes, filename, audio_mime = _to_wav_bytes(resampled), "audio.wav", "audio/wav"
        else:
            raise MetricUnavailable(
                f"unsupported http_asr audio_format {self.audio_format!r}; "
                "expected wav or pcm_s16le"
            )
        lang = self.language_map.get(language, language)

        if self.body_format == "multipart":
            body, content_type = _encode_multipart(
                {self.language_field: lang, **{k: str(v) for k, v in self.extra_fields.items()}},
                {self.file_field: (filename, audio_bytes, audio_mime)},
            )
        elif self.body_format == "json_base64":
            payload = {
                self.file_field: base64.b64encode(audio_bytes).decode("ascii"),
                self.language_field: lang,
                **self.extra_fields,
            }
            body = json.dumps(payload).encode("utf-8")
            content_type = "application/json"
        elif self.body_format == "raw":
            body, content_type = audio_bytes, audio_mime
        else:
            raise MetricUnavailable(
                f"unsupported http_asr body_format {self.body_format!r}; "
                "expected multipart, json_base64 or raw"
            )

        url = self.url
        if self.body_format == "raw" and lang:
            # A raw body has nowhere to carry the language, so it goes in the query.
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{self.language_field}={urllib.parse.quote(lang)}"

        request = urllib.request.Request(
            url, data=body, method=self.method, headers={"Content-Type": content_type, **self.headers}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            raise MetricUnavailable(f"ASR HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, OSError) as e:
            raise MetricUnavailable(f"ASR request failed: {type(e).__name__}: {e}") from e

        text = raw.decode("utf-8", "replace")
        # A plain-text response is a legitimate shape; try JSON first and fall back.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text.strip()
        return _extract_path(parsed, self.transcript_path)


def _extract_path(node: Any, path: str) -> str:
    """Walk ``a.b[0].c`` and return the value as a string."""
    for part in path.split("."):
        m = re.fullmatch(r"([^\[\]]*)((?:\[\d+\])*)", part)
        if m is None:
            raise MetricUnavailable(f"malformed transcript_path segment {part!r}")
        key, indices = m.group(1), m.group(2)
        if key:
            if not isinstance(node, Mapping) or key not in node:
                available = ", ".join(sorted(map(str, node))) if isinstance(node, Mapping) else "-"
                raise MetricUnavailable(
                    f"transcript_path {path!r}: key {key!r} not in ASR response (available: {available})"
                )
            node = node[key]
        for i in re.findall(r"\[(\d+)\]", indices):
            try:
                node = node[int(i)]
            except (TypeError, IndexError, KeyError) as e:
                raise MetricUnavailable(f"transcript_path {path!r}: index [{i}] out of range") from e
    if isinstance(node, (list, tuple)):
        # Segment-list responses (Whisper-style) are joined rather than rejected.
        node = " ".join(
            str(seg.get("text", "")) if isinstance(seg, Mapping) else str(seg) for seg in node
        )
    return str(node).strip()


def _to_wav_bytes(audio: AudioBuffer) -> bytes:
    """In-memory 16-bit mono WAV. Avoids a temp file per utterance."""
    import io
    import wave

    import numpy as np

    buf = io.BytesIO()
    pcm = (np.clip(audio.samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(audio.sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _to_pcm16_bytes(audio: AudioBuffer) -> bytes:
    """Headerless raw 16-bit signed little-endian mono PCM.

    For servers that base64-decode straight into an int16 buffer (no WAV parse);
    a WAV header would be read as ~22 leading garbage samples.
    """
    import numpy as np

    pcm = (np.clip(audio.samples, -1.0, 1.0) * 32767.0).astype("<i2")
    return pcm.tobytes()


def _encode_multipart(
    fields: Mapping[str, str], files: Mapping[str, tuple[str, bytes, str]]
) -> tuple[bytes, str]:
    """Hand-rolled multipart/form-data so no HTTP library is required."""
    boundary = f"----ttsEvalBoundary{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        if value is None or value == "":
            continue
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    for name, (filename, payload, mime) in files.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {mime}\r\n\r\n"
            ).encode()
        )
        parts.append(payload)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _expand_env(value: str) -> str:
    import os

    return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), m.group(0)), value)


__all__ = ["HTTPASRBackend"]
