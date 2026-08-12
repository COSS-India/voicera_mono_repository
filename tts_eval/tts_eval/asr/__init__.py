"""Round-trip ASR backends and the text-scoring primitives they feed.

Selection is by name in the suite config (``asr: {backend: http_asr, ...}``), so
swapping IndicConformer for Whisper is a config change, not a code change.
"""
from __future__ import annotations

from typing import Any, Mapping

from .base import (  # noqa: F401
    ASRBackend,
    ErrorRate,
    character_error_rate,
    normalise_text,
    slot_hits,
    word_error_rate,
)
from .http_asr import HTTPASRBackend
from .whisper_asr import WhisperASRBackend

_ASR_REGISTRY: dict[str, type[ASRBackend]] = {
    HTTPASRBackend.name: HTTPASRBackend,
    WhisperASRBackend.name: WhisperASRBackend,
}


def available_asr_backends() -> list[str]:
    return sorted(_ASR_REGISTRY)


def build_asr(config: Mapping[str, Any] | None) -> ASRBackend | None:
    """Construct an ASR backend from a config block, or None if none is configured.

    Returning None (rather than raising) is deliberate: intelligibility is an
    optional tier, and a run without ASR is valid — it simply reports CER/WER as
    ``not_computed`` and marks per-language intelligibility unverified.
    """
    if not config:
        return None
    backend = str(config.get("backend") or "").strip()
    if not backend or backend == "none":
        return None
    if backend not in _ASR_REGISTRY:
        known = ", ".join(available_asr_backends())
        raise KeyError(f"unknown ASR backend {backend!r}; available: {known}")
    options = {k: v for k, v in config.items() if k != "backend"}
    return _ASR_REGISTRY[backend](options)


__all__ = [
    "ASRBackend",
    "ErrorRate",
    "HTTPASRBackend",
    "WhisperASRBackend",
    "available_asr_backends",
    "build_asr",
    "character_error_rate",
    "normalise_text",
    "slot_hits",
    "word_error_rate",
]
