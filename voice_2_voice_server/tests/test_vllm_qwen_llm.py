"""Tests for services/vllm_qwen/llm.py."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipecat.frames.frames import Frame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection

from services.vllm_qwen.llm import (
    ensure_no_think_suffix,
    _enable_thinking_from_extra,
    VllmQwenVoiceLLMService,
    _NO_THINK_SUFFIX,
    VLLM_BASE_URL,
    VLLM_API_KEY,
    VLLM_MODEL,
)


# ---------------------------------------------------------------------------
# ensure_no_think_suffix
# ---------------------------------------------------------------------------

def test_ensure_no_think_suffix_plain():
    result = ensure_no_think_suffix("You are an assistant.")
    assert result == "You are an assistant. /no_think"


def test_ensure_no_think_suffix_already_present():
    result = ensure_no_think_suffix("You are an assistant. /no_think")
    assert result == "You are an assistant. /no_think"


def test_ensure_no_think_suffix_empty_string():
    result = ensure_no_think_suffix("")
    assert result == "/no_think"


def test_ensure_no_think_suffix_none():
    result = ensure_no_think_suffix(None)
    assert result == "/no_think"


def test_ensure_no_think_suffix_whitespace_only():
    result = ensure_no_think_suffix("   ")
    assert result == "/no_think"


def test_ensure_no_think_suffix_trailing_whitespace():
    result = ensure_no_think_suffix("Prompt text   ")
    assert result == "Prompt text /no_think"


# ---------------------------------------------------------------------------
# _enable_thinking_from_extra
# ---------------------------------------------------------------------------

def test_enable_thinking_true():
    assert _enable_thinking_from_extra({"chat_template_kwargs": {"enable_thinking": True}}) is True


def test_enable_thinking_false():
    assert _enable_thinking_from_extra({"chat_template_kwargs": {"enable_thinking": False}}) is False


def test_enable_thinking_missing_key():
    assert _enable_thinking_from_extra({}) is False


def test_enable_thinking_none_extra():
    assert _enable_thinking_from_extra(None) is False


def test_enable_thinking_empty_chat_kwargs():
    assert _enable_thinking_from_extra({"chat_template_kwargs": {}}) is False


# ---------------------------------------------------------------------------
# VllmQwenVoiceLLMService._reasoning_delta_text
# ---------------------------------------------------------------------------

def test_reasoning_delta_text_from_reasoning():
    delta = MagicMock()
    delta.reasoning = "Think text"
    delta.reasoning_content = None
    result = VllmQwenVoiceLLMService._reasoning_delta_text(delta)
    assert result == "Think text"


def test_reasoning_delta_text_from_reasoning_content():
    delta = MagicMock(spec=[])
    delta.reasoning_content = "Think text 2"
    # No 'reasoning' attr
    result = VllmQwenVoiceLLMService._reasoning_delta_text(delta)
    assert result == "Think text 2"


def test_reasoning_delta_text_none():
    delta = MagicMock()
    delta.reasoning = None
    delta.reasoning_content = None
    result = VllmQwenVoiceLLMService._reasoning_delta_text(delta)
    assert result is None


# ---------------------------------------------------------------------------
# VllmQwenVoiceLLMService._normalize_qwen_chunk
# ---------------------------------------------------------------------------

def _make_service_instance(enable_thinking=False):
    with patch("pipecat.services.openai.llm.OpenAILLMService.__init__", return_value=None):
        svc = VllmQwenVoiceLLMService.__new__(VllmQwenVoiceLLMService)
        svc._strip_voice_prefix = False
        svc._enable_thinking = enable_thinking
        svc._settings = {}
        return svc


def _make_chunk(content=None, reasoning=None):
    delta = MagicMock()
    delta.content = content
    delta.reasoning = reasoning
    delta.reasoning_content = None

    choice = MagicMock()
    choice.delta = delta

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.model_copy = lambda update=None, **kw: _apply_update(chunk, update or {})
    choice.model_copy = lambda update=None, **kw: _apply_update(choice, update or {})
    delta.model_copy = lambda update=None, **kw: _apply_delta_update(delta, update or {})

    return chunk, choice, delta


def _apply_update(obj, update):
    new = MagicMock()
    for k, v in update.items():
        setattr(new, k, v)
    return new


def _apply_delta_update(delta, update):
    new = MagicMock()
    new.content = update.get("content", delta.content)
    new.reasoning = getattr(delta, "reasoning", None)
    new.reasoning_content = getattr(delta, "reasoning_content", None)
    return new


def test_normalize_chunk_thinking_enabled_passthrough():
    svc = _make_service_instance(enable_thinking=True)
    chunk, _, _ = _make_chunk(content=None, reasoning="Think...")
    result = svc._normalize_qwen_chunk(chunk)
    assert result is chunk  # passthrough unchanged


def test_normalize_chunk_no_choices():
    svc = _make_service_instance(enable_thinking=False)
    chunk = MagicMock()
    chunk.choices = []
    result = svc._normalize_qwen_chunk(chunk)
    assert result is chunk


def test_normalize_chunk_content_present_passthrough():
    svc = _make_service_instance(enable_thinking=False)
    chunk, choice, delta = _make_chunk(content="Real answer", reasoning=None)
    result = svc._normalize_qwen_chunk(chunk)
    # content has text → passthrough
    assert result is chunk


def test_normalize_chunk_no_delta():
    svc = _make_service_instance(enable_thinking=False)
    chunk = MagicMock()
    choice = MagicMock()
    choice.delta = None
    chunk.choices = [choice]
    result = svc._normalize_qwen_chunk(chunk)
    assert result is chunk


def test_normalize_chunk_reasoning_mapped_to_content():
    svc = _make_service_instance(enable_thinking=False)

    # Build a realistic chunk where content is empty and reasoning has text
    delta = MagicMock()
    delta.content = ""
    delta.reasoning = "spoken answer"
    delta.reasoning_content = None

    new_delta = MagicMock()
    new_delta.content = "spoken answer"
    delta.model_copy = MagicMock(return_value=new_delta)

    new_choice = MagicMock()
    new_choice.delta = new_delta

    choice = MagicMock()
    choice.delta = delta
    choice.model_copy = MagicMock(return_value=new_choice)

    new_chunk = MagicMock()
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.model_copy = MagicMock(return_value=new_chunk)

    result = svc._normalize_qwen_chunk(chunk)
    assert result is new_chunk
    delta.model_copy.assert_called_once_with(update={"content": "spoken answer"})


def test_normalize_chunk_reasoning_whitespace_only_passthrough():
    svc = _make_service_instance(enable_thinking=False)
    delta = MagicMock()
    delta.content = ""
    delta.reasoning = "   "  # whitespace only → no mapping
    delta.reasoning_content = None

    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]

    result = svc._normalize_qwen_chunk(chunk)
    assert result is chunk  # unchanged


# ---------------------------------------------------------------------------
# VllmQwenVoiceLLMService.push_frame
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_push_frame_start_sets_flag():
    svc = _make_service_instance()
    pushed = []

    async def fake_super(self_arg, frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    with patch("pipecat.services.openai.llm.OpenAILLMService.push_frame", new=fake_super):
        await svc.push_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

    assert svc._strip_voice_prefix is True
    assert len(pushed) == 1


@pytest.mark.asyncio
async def test_push_frame_strips_leading_newlines():
    svc = _make_service_instance()
    svc._strip_voice_prefix = True
    pushed = []

    async def fake_super(self_arg, frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    with patch("pipecat.services.openai.llm.OpenAILLMService.push_frame", new=fake_super):
        await svc.push_frame(LLMTextFrame(text="\n\nHello"), FrameDirection.DOWNSTREAM)

    assert pushed[0].text == "Hello"
    assert svc._strip_voice_prefix is False


@pytest.mark.asyncio
async def test_push_frame_empty_after_strip_skipped():
    svc = _make_service_instance()
    svc._strip_voice_prefix = True
    pushed = []

    async def fake_super(self_arg, frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    with patch("pipecat.services.openai.llm.OpenAILLMService.push_frame", new=fake_super):
        await svc.push_frame(LLMTextFrame(text="\n\n"), FrameDirection.DOWNSTREAM)

    assert len(pushed) == 0  # skipped


# ---------------------------------------------------------------------------
# build_chat_completion_params — extra body extraction
# ---------------------------------------------------------------------------

def test_build_chat_completion_params_extracts_vllm_keys():
    svc = _make_service_instance()

    base_params = {
        "model": "Qwen/Qwen3-8B",
        "messages": [],
        "stream": True,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.7,
    }

    with patch(
        "pipecat.services.openai.llm.OpenAILLMService.build_chat_completion_params",
        return_value=base_params.copy(),
    ):
        params_from_context = MagicMock()
        result = svc.build_chat_completion_params(params_from_context)

    assert "top_k" not in result
    assert "chat_template_kwargs" not in result
    assert "extra_body" in result
    assert result["extra_body"]["top_k"] == 20


def test_build_chat_completion_params_no_vllm_keys():
    svc = _make_service_instance()

    base_params = {"model": "Qwen/Qwen3-8B", "messages": [], "stream": True}

    with patch(
        "pipecat.services.openai.llm.OpenAILLMService.build_chat_completion_params",
        return_value=base_params.copy(),
    ):
        params_from_context = MagicMock()
        result = svc.build_chat_completion_params(params_from_context)

    assert "extra_body" not in result


def test_build_chat_completion_params_merges_existing_extra_body():
    svc = _make_service_instance()

    base_params = {
        "model": "Qwen/Qwen3-8B",
        "messages": [],
        "stream": True,
        "top_k": 20,
        "extra_body": {"existing_key": "value"},
    }

    with patch(
        "pipecat.services.openai.llm.OpenAILLMService.build_chat_completion_params",
        return_value=base_params.copy(),
    ):
        params_from_context = MagicMock()
        result = svc.build_chat_completion_params(params_from_context)

    assert result["extra_body"]["existing_key"] == "value"
    assert result["extra_body"]["top_k"] == 20


# ---------------------------------------------------------------------------
# _refresh_enable_thinking
# ---------------------------------------------------------------------------

def test_refresh_enable_thinking_updates():
    svc = _make_service_instance(enable_thinking=False)
    svc._settings = {"extra": {"chat_template_kwargs": {"enable_thinking": True}}}
    svc._refresh_enable_thinking()
    assert svc._enable_thinking is True
