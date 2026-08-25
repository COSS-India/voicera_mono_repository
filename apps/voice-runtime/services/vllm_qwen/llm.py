"""Local vLLM (OpenAI-compatible) LLM setup for Pipecat telephony/voice agents.

Uses Qwen3 with /no_think in the system prompt to avoid silent chain-of-thought
(100–300 tokens) that causes dead air on phone calls.
"""

from __future__ import annotations

import os
from typing import Any

from openai.types.chat import ChatCompletionChunk
from pipecat.frames.frames import Frame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService

# OpenAI-compatible base URL. Must include /v1 — the AsyncOpenAI client appends
# /chat/completions under this. Local vLLM often uses another port (e.g. 8003);
# set VLLM_BASE_URL in the environment to override without code changes.
VLLM_BASE_URL = "http://100.64.1.16:8003/v1"

# vLLM does not validate keys; a placeholder satisfies the OpenAI client.
VLLM_API_KEY = "EMPTY"

# Must match the served model id on vLLM (e.g. Qwen/Qwen3-8B).
VLLM_MODEL = "Qwen/Qwen3-8B"

_NO_THINK_SUFFIX = "/no_think"

# Pipecat merges InputParams.extra into create() kwargs; the OpenAI client only accepts
# standard fields there. vLLM extensions must be sent via extra_body instead.
_VLLM_EXTRA_BODY_KEYS = frozenset({"chat_template_kwargs", "top_k", "repetition_penalty"})


def ensure_no_think_suffix(prompt: str) -> str:
    """Ensure the system prompt ends with /no_think for Qwen3 voice use.

    Without this, Qwen3 may spend 100–300 hidden "thinking" tokens before the
    first spoken token, adding 1–3s latency (bad for telephony).

    Pipecat streams ``delta.content`` to TTS. When vLLM has thinking disabled,
    the answer may arrive on ``delta.reasoning`` instead; see
    ``VllmQwenVoiceLLMService._normalize_qwen_chunk``.
    """
    text = (prompt or "").rstrip()
    if text.endswith(_NO_THINK_SUFFIX):
        return text
    if not text:
        return _NO_THINK_SUFFIX
    return f"{text} {_NO_THINK_SUFFIX}"


# Replace or build your own with ensure_no_think_suffix(custom_prompt).
DEFAULT_VOICE_SYSTEM_PROMPT = ensure_no_think_suffix(
    "You are a concise voice assistant. Reply in short, natural sentences suitable "
    "for phone calls. Do not use markdown or bullet lists unless the user asks."
)

# Pipecat's OpenAI path always requests streaming (see BaseOpenAILLMService
# ``build_chat_completion_params``: stream=True). There is no service-level flag;
# first LLM tokens reach TTS immediately, which is required for low time-to-speech.
VOICE_LLM_PARAMS = BaseOpenAILLMService.InputParams(
    # Non-thinking Qwen3 setups are commonly run around 0.7 for natural but stable speech.
    temperature=0.7,
    # Qwen3 non-thinking defaults for conversational voice responses.
    top_p=0.8,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    # Keeps answers short enough for voice; avoids long monologues and reduces latency.
    max_tokens=200,
    # vLLM-specific fields; VllmQwenVoiceLLMService moves these into extra_body.
    extra={
        "top_k": 20,
        "repetition_penalty": 1.0,
        # Telephony default: no thinking tokens. Stream normalizer maps delta.reasoning
        # into delta.content when this is False (vLLM quirk).
        "chat_template_kwargs": {"enable_thinking": False},
    },
)


def _enable_thinking_from_extra(extra: dict[str, Any] | None) -> bool:
    chat_template_kwargs = (extra or {}).get("chat_template_kwargs") or {}
    return bool(chat_template_kwargs.get("enable_thinking", False))


class VllmQwenVoiceLLMService(OpenAILLMService):
    """OpenAILLMService pointed at vLLM, with leading newlines stripped from TTS text.

    Qwen/vLLM sometimes prefixes assistant content with blank lines; stripping avoids
    awkward pauses or empty audio at the start of playback.

    When ``enable_thinking`` is False, vLLM often streams the spoken answer on
    ``delta.reasoning`` with an empty ``delta.content``. Pipecat only forwards
    ``content`` to TTS, so this service copies ``reasoning`` into ``content`` for
    that mode only (never while thinking is enabled, to avoid TTS of chain-of-thought).
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._strip_voice_prefix = False
        self._enable_thinking = _enable_thinking_from_extra(self._settings.get("extra"))

    def _refresh_enable_thinking(self) -> None:
        self._enable_thinking = _enable_thinking_from_extra(self._settings.get("extra"))

    def build_chat_completion_params(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> dict:
        params = super().build_chat_completion_params(params_from_context)
        vllm_extra: dict[str, Any] = {}
        for key in _VLLM_EXTRA_BODY_KEYS:
            if key in params:
                vllm_extra[key] = params.pop(key)
        if not vllm_extra:
            return params
        existing = params.get("extra_body")
        if isinstance(existing, dict):
            merged = {**existing, **vllm_extra}
        else:
            merged = vllm_extra
        params["extra_body"] = merged
        return params

    @staticmethod
    def _reasoning_delta_text(delta: Any) -> str | None:
        for attr in ("reasoning", "reasoning_content"):
            value = getattr(delta, attr, None)
            if value:
                return value
        return None

    def _normalize_qwen_chunk(self, chunk: ChatCompletionChunk) -> ChatCompletionChunk:
        """When thinking is off, map vLLM ``delta.reasoning`` into ``delta.content`` for TTS."""
        if self._enable_thinking or not chunk.choices:
            return chunk
        choice = chunk.choices[0]
        delta = choice.delta
        if not delta:
            return chunk
        content = delta.content or ""
        if content.strip():
            return chunk
        reasoning = self._reasoning_delta_text(delta)
        if not reasoning or not reasoning.strip():
            return chunk
        new_delta = delta.model_copy(update={"content": reasoning})
        new_choice = choice.model_copy(update={"delta": new_delta})
        return chunk.model_copy(update={"choices": [new_choice, *chunk.choices[1:]]})

    async def _normalize_qwen_stream(self, stream: Any):
        self._refresh_enable_thinking()
        async for chunk in stream:
            yield self._normalize_qwen_chunk(chunk)

    async def _stream_chat_completions_specific_context(
        self, context: OpenAILLMContext
    ) -> Any:
        chunks = await super()._stream_chat_completions_specific_context(context)
        return self._normalize_qwen_stream(chunks)

    async def _stream_chat_completions_universal_context(self, context: LLMContext) -> Any:
        chunks = await super()._stream_chat_completions_universal_context(context)
        return self._normalize_qwen_stream(chunks)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMFullResponseStartFrame):
            self._strip_voice_prefix = True
        elif isinstance(frame, LLMTextFrame) and self._strip_voice_prefix:
            raw = frame.text or ""
            stripped = raw.lstrip("\n")
            if not stripped:
                # Leading chunk was only newlines; hold off TTS until real text arrives.
                return
            self._strip_voice_prefix = False
            frame = LLMTextFrame(text=stripped)
        await super().push_frame(frame, direction)


def create_voice_llm(
    *,
    model: str = VLLM_MODEL,
    api_key: str = VLLM_API_KEY,
    base_url: str | None = None,
    params: BaseOpenAILLMService.InputParams | None = None,
    **kwargs: Any,
) -> VllmQwenVoiceLLMService:
    """Build a voice-oriented vLLM LLM service with defaults; pass kwargs through to OpenAILLMService.

    Optional kwargs (see BaseOpenAILLMService / OpenAILLMService), for example:
    - ``retry_timeout_secs``: HTTP timeout; local GPUs can spike past the default 5s.
    - ``retry_on_timeout``: set True if you want one automatic retry on timeout.
    """
    return VllmQwenVoiceLLMService(
        model=model,
        api_key=api_key,
        base_url=base_url or os.environ.get("VLLM_BASE_URL", VLLM_BASE_URL),
        params=params or VOICE_LLM_PARAMS,
        **kwargs,
    )


llm = create_voice_llm()

__all__ = [
    "DEFAULT_VOICE_SYSTEM_PROMPT",
    "VOICE_LLM_PARAMS",
    "VLLM_API_KEY",
    "VLLM_BASE_URL",
    "VLLM_MODEL",
    "VllmQwenVoiceLLMService",
    "create_voice_llm",
    "ensure_no_think_suffix",
    "llm",
]
