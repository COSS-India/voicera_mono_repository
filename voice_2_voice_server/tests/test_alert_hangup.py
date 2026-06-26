import pytest
from unittest.mock import AsyncMock
from pipecat.frames.frames import BotStoppedSpeakingFrame, AudioRawFrame
from pipecat.processors.frame_processor import FrameDirection

from utils.call_management.alert_hangup import AlertHangupProcessor


class TestAlertHangupProcessor:
    @pytest.mark.asyncio
    async def test_initialization(self):
        mock_end = AsyncMock()
        processor = AlertHangupProcessor(schedule_call_end=mock_end)
        assert processor._schedule_call_end == mock_end
        assert processor._end_scheduled is False

    @pytest.mark.asyncio
    async def test_hangup_on_bot_stopped_speaking(self):
        mock_end = AsyncMock()
        processor = AlertHangupProcessor(schedule_call_end=mock_end)
        processor.push_frame = AsyncMock()

        # Regular frame passes through
        audio_frame = AudioRawFrame(audio=b"", sample_rate=8000, num_channels=1)
        await processor.process_frame(audio_frame, FrameDirection.DOWNSTREAM)
        mock_end.assert_not_called()
        processor.push_frame.assert_called_once_with(audio_frame, FrameDirection.DOWNSTREAM)

        # Bot stopped speaking triggers hangup
        await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        mock_end.assert_called_once()
        assert processor._end_scheduled is True

        # Second bot stopped speaking frame does not trigger again
        mock_end.reset_mock()
        await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        mock_end.assert_not_called()
