import pytest
import re
from unittest.mock import AsyncMock, MagicMock
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from utils.call_goodbye import GoodbyeHangupProcessor, _END_CALL


def test_goodbye_regex():
    # Matches
    assert _END_CALL.search("goodbye") is not None
    assert _END_CALL.search("good bye") is not None
    assert _END_CALL.search("bye bye") is not None
    assert _END_CALL.search("end of the conversation") is not None
    assert _END_CALL.search("end of call") is not None
    assert _END_CALL.search("take care") is not None
    assert _END_CALL.search("see you later") is not None
    assert _END_CALL.search("have a nice day") is not None
    assert _END_CALL.search("that's all for now") is not None
    assert _END_CALL.search("signing off") is not None

    # Non-matches
    assert _END_CALL.search("hello") is None
    assert _END_CALL.search("how can I help you") is None
    assert _END_CALL.search("good day to you") is None


class TestGoodbyeHangupProcessor:
    @pytest.mark.asyncio
    async def test_no_hangup_on_regular_response(self):
        mock_end_call = AsyncMock()
        processor = GoodbyeHangupProcessor(schedule_call_end=mock_end_call)
        processor.push_frame = AsyncMock()

        # Simulate regular flow
        await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        await processor.process_frame(LLMTextFrame("Hello, this is a test."), FrameDirection.DOWNSTREAM)
        await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        assert processor._ending is False
        assert processor.should_suppress_idle() is False
        mock_end_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_hangup_on_goodbye_response(self):
        mock_end_call = AsyncMock()
        processor = GoodbyeHangupProcessor(schedule_call_end=mock_end_call)
        processor.push_frame = AsyncMock()

        # Simulate goodbye flow
        await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        await processor.process_frame(LLMTextFrame("Thank you for calling. "), FrameDirection.DOWNSTREAM)
        await processor.process_frame(LLMTextFrame("Goodbye!"), FrameDirection.DOWNSTREAM)
        await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        assert processor._ending is True
        assert processor.should_suppress_idle() is True
        mock_end_call.assert_not_called()  # Should not be called until bot stops speaking

        await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        mock_end_call.assert_called_once()
        assert processor._end_scheduled is True

    @pytest.mark.asyncio
    async def test_only_calls_end_once(self):
        mock_end_call = AsyncMock()
        processor = GoodbyeHangupProcessor(schedule_call_end=mock_end_call)
        processor.push_frame = AsyncMock()

        processor._ending = True
        processor._end_scheduled = False

        await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        mock_end_call.assert_called_once()
