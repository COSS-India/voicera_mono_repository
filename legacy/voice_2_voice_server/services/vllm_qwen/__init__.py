"""Local vLLM (Qwen3) OpenAI-compatible LLM helpers for Pipecat voice pipelines."""

from .llm import (
    DEFAULT_VOICE_SYSTEM_PROMPT,
    VOICE_LLM_PARAMS,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
    VllmQwenVoiceLLMService,
    create_voice_llm,
    ensure_no_think_suffix,
    llm,
)

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
