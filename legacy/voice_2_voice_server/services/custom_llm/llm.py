"""OpenAI Chat Completions v1-compatible custom LLM for Pipecat voice agents."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from pipecat.frames.frames import Frame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService

VOICE_LLM_PARAMS = BaseOpenAILLMService.InputParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)


def normalize_base_url(url: str) -> str:
    """Normalize user URL to OpenAI-compatible base URL ending in /v1."""
    raw = (url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("Endpoint URL is required")

    if raw.endswith("/chat/completions"):
        raw = raw[: -len("/chat/completions")].rstrip("/")

    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Endpoint URL must be a valid http(s) URL")

    if not raw.endswith("/v1"):
        raw = f"{raw}/v1"

    return raw


class CustomLLMService(OpenAILLMService):
    """OpenAILLMService pointed at a user-provided OpenAI-compatible endpoint."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._strip_voice_prefix = False

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMFullResponseStartFrame):
            self._strip_voice_prefix = True
        elif isinstance(frame, LLMTextFrame) and self._strip_voice_prefix:
            raw = frame.text or ""
            stripped = raw.lstrip("\n")
            if not stripped:
                return
            self._strip_voice_prefix = False
            frame = LLMTextFrame(text=stripped)
        await super().push_frame(frame, direction)


def create_custom_llm(
    *,
    model: str,
    api_key: str,
    base_url: str,
    params: BaseOpenAILLMService.InputParams | None = None,
    **kwargs: Any,
) -> CustomLLMService:
    """Build a voice-oriented custom LLM service."""
    return CustomLLMService(
        model=model,
        api_key=api_key,
        base_url=normalize_base_url(base_url),
        params=params or VOICE_LLM_PARAMS,
        **kwargs,
    )


__all__ = [
    "CustomLLMService",
    "VOICE_LLM_PARAMS",
    "create_custom_llm",
    "normalize_base_url",
]
