"""Tests for services/custom_llm/llm.py."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.custom_llm.llm import normalize_base_url, CustomLLMService, VOICE_LLM_PARAMS, create_custom_llm
from pipecat.frames.frames import LLMFullResponseStartFrame, LLMTextFrame, Frame
from pipecat.processors.frame_processor import FrameDirection


# ---------------------------------------------------------------------------
# normalize_base_url
# ---------------------------------------------------------------------------

def test_normalize_base_url_plain():
    assert normalize_base_url("http://example.com") == "http://example.com/v1"


def test_normalize_base_url_already_v1():
    assert normalize_base_url("http://example.com/v1") == "http://example.com/v1"


def test_normalize_base_url_trailing_slash():
    assert normalize_base_url("http://example.com/") == "http://example.com/v1"


def test_normalize_base_url_strips_chat_completions():
    url = "http://example.com/v1/chat/completions"
    assert normalize_base_url(url) == "http://example.com/v1"


def test_normalize_base_url_strips_chat_completions_no_v1():
    url = "http://example.com/chat/completions"
    assert normalize_base_url(url) == "http://example.com/v1"


def test_normalize_base_url_https():
    assert normalize_base_url("https://api.example.com") == "https://api.example.com/v1"


def test_normalize_base_url_empty_raises():
    with pytest.raises(ValueError, match="required"):
        normalize_base_url("")


def test_normalize_base_url_whitespace_raises():
    with pytest.raises(ValueError, match="required"):
        normalize_base_url("   ")


def test_normalize_base_url_no_scheme_raises():
    with pytest.raises(ValueError, match="valid http"):
        normalize_base_url("example.com/v1")


def test_normalize_base_url_invalid_scheme_raises():
    with pytest.raises(ValueError, match="valid http"):
        normalize_base_url("ftp://example.com/v1")


# ---------------------------------------------------------------------------
# CustomLLMService.push_frame
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_service():
    """Create a CustomLLMService with mocked parent push_frame."""
    with patch("services.custom_llm.llm.OpenAILLMService.__init__", return_value=None):
        svc = CustomLLMService.__new__(CustomLLMService)
        svc._strip_voice_prefix = False
        svc._super_push_frame = AsyncMock()
        # Patch super().push_frame
        with patch.object(
            CustomLLMService, "push_frame", wraps=svc.push_frame
        ):
            pass
        return svc


@pytest.mark.asyncio
async def test_push_frame_start_sets_strip_flag():
    with patch("services.custom_llm.llm.OpenAILLMService.__init__", return_value=None):
        svc = CustomLLMService.__new__(CustomLLMService)
        svc._strip_voice_prefix = False

        calls = []

        async def fake_super(self_arg, frame, direction=FrameDirection.DOWNSTREAM):
            calls.append(frame)

        with patch(
            "pipecat.services.openai.llm.OpenAILLMService.push_frame",
            new=fake_super,
        ):
            frame = LLMFullResponseStartFrame()
            await svc.push_frame(frame, FrameDirection.DOWNSTREAM)
            assert svc._strip_voice_prefix is True
            assert len(calls) == 1


@pytest.mark.asyncio
async def test_push_frame_text_strips_leading_newlines():
    with patch("services.custom_llm.llm.OpenAILLMService.__init__", return_value=None):
        svc = CustomLLMService.__new__(CustomLLMService)
        svc._strip_voice_prefix = True

        pushed = []

        async def fake_super(self_arg, frame, direction=FrameDirection.DOWNSTREAM):
            pushed.append(frame)

        with patch(
            "pipecat.services.openai.llm.OpenAILLMService.push_frame",
            new=fake_super,
        ):
            frame = LLMTextFrame(text="\n\nhello")
            await svc.push_frame(frame, FrameDirection.DOWNSTREAM)
            assert svc._strip_voice_prefix is False
            assert len(pushed) == 1
            assert pushed[0].text == "hello"


@pytest.mark.asyncio
async def test_push_frame_text_empty_after_strip_skipped():
    with patch("services.custom_llm.llm.OpenAILLMService.__init__", return_value=None):
        svc = CustomLLMService.__new__(CustomLLMService)
        svc._strip_voice_prefix = True

        pushed = []

        async def fake_super(self_arg, frame, direction=FrameDirection.DOWNSTREAM):
            pushed.append(frame)

        with patch(
            "pipecat.services.openai.llm.OpenAILLMService.push_frame",
            new=fake_super,
        ):
            frame = LLMTextFrame(text="\n\n")
            await svc.push_frame(frame, FrameDirection.DOWNSTREAM)
            # Should be skipped — nothing pushed
            assert len(pushed) == 0


@pytest.mark.asyncio
async def test_push_frame_text_no_strip_when_flag_false():
    with patch("services.custom_llm.llm.OpenAILLMService.__init__", return_value=None):
        svc = CustomLLMService.__new__(CustomLLMService)
        svc._strip_voice_prefix = False

        pushed = []

        async def fake_super(self_arg, frame, direction=FrameDirection.DOWNSTREAM):
            pushed.append(frame)

        with patch(
            "pipecat.services.openai.llm.OpenAILLMService.push_frame",
            new=fake_super,
        ):
            frame = LLMTextFrame(text="\n\nhello")
            await svc.push_frame(frame, FrameDirection.DOWNSTREAM)
            # flag is False → pass through unchanged
            assert len(pushed) == 1
            assert pushed[0].text == "\n\nhello"
