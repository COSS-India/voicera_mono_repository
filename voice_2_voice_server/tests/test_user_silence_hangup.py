import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    StartFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from utils.call_management.user_silence_hangup import UserSilenceHangupProcessor
from conftest import run_task_with_blocking_sleep


class TestUserSilenceHangupProcessor:
    @pytest.mark.asyncio
    async def test_initialization(self):
        mock_end = AsyncMock()
        processor = UserSilenceHangupProcessor(
            timeout_secs=5.0,
            schedule_call_end=mock_end,
        )
        assert processor._timeout_secs == 5.0
        assert processor._schedule_call_end == mock_end
        assert processor._idle_task is None
        assert processor._hangup_scheduled is False

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_timer_fires_triggers_hangup(self, mock_sleep):
        mock_end = AsyncMock()
        processor = UserSilenceHangupProcessor(
            timeout_secs=3.0,
            schedule_call_end=mock_end,
        )
        processor.push_frame = AsyncMock()

        # Bot stops speaking -> schedules timer
        await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        assert processor._idle_task is not None

        # Await the task to run to completion (since sleep is mocked and returns immediately)
        await processor._idle_task

        mock_sleep.assert_called_once_with(3.0)
        assert processor._hangup_scheduled is True
        mock_end.assert_called_once()

    @pytest.mark.asyncio
    async def test_timer_cancelled_by_user_speaking(self):
        block = asyncio.Event()

        async def blocking_sleep(_):
            await block.wait()  # Waits forever until cancelled

        mock_end = AsyncMock()
        processor = UserSilenceHangupProcessor(
            timeout_secs=3.0,
            schedule_call_end=mock_end,
        )
        processor.push_frame = AsyncMock()
        with patch("asyncio.sleep", side_effect=blocking_sleep):
            task = await run_task_with_blocking_sleep(
                processor,
                BotStoppedSpeakingFrame(),
                UserStartedSpeakingFrame(),
                FrameDirection.DOWNSTREAM,
            )
        assert processor._idle_task is None
        assert task.done()
        mock_end.assert_not_called()

    @pytest.mark.asyncio
    async def test_timer_cancelled_by_bot_speaking(self):
        block = asyncio.Event()

        async def blocking_sleep(_):
            await block.wait()

        mock_end = AsyncMock()
        processor = UserSilenceHangupProcessor(
            timeout_secs=3.0,
            schedule_call_end=mock_end,
        )
        processor.push_frame = AsyncMock()
        with patch("asyncio.sleep", side_effect=blocking_sleep):
            task = await run_task_with_blocking_sleep(
                processor,
                BotStoppedSpeakingFrame(),
                BotStartedSpeakingFrame(),
                FrameDirection.DOWNSTREAM,
            )
        assert processor._idle_task is None
        assert task.done()
        mock_end.assert_not_called()

    @pytest.mark.asyncio
    async def test_timer_cancelled_by_end_frame(self):
        block = asyncio.Event()

        async def blocking_sleep(_):
            await block.wait()

        mock_end = AsyncMock()
        processor = UserSilenceHangupProcessor(timeout_secs=3.0, schedule_call_end=mock_end)
        processor.push_frame = AsyncMock()
        with patch("asyncio.sleep", side_effect=blocking_sleep):
            task = await run_task_with_blocking_sleep(
                processor,
                BotStoppedSpeakingFrame(),
                EndFrame(),
                FrameDirection.DOWNSTREAM,
            )
        assert processor._idle_task is None
        assert task.done()
        mock_end.assert_not_called()

    @pytest.mark.asyncio
    async def test_timer_cancelled_by_cancel_frame(self):
        block = asyncio.Event()

        async def blocking_sleep(_):
            await block.wait()

        mock_end = AsyncMock()
        processor = UserSilenceHangupProcessor(timeout_secs=3.0, schedule_call_end=mock_end)
        processor.push_frame = AsyncMock()
        with patch("asyncio.sleep", side_effect=blocking_sleep):
            task = await run_task_with_blocking_sleep(
                processor,
                BotStoppedSpeakingFrame(),
                CancelFrame(),
                FrameDirection.DOWNSTREAM,
            )
        assert processor._idle_task is None
        assert task.done()
        mock_end.assert_not_called()

    @pytest.mark.asyncio
    async def test_suppress_idle_when(self):
        suppress_mock = MagicMock(return_value=True)
        mock_end = AsyncMock()
        processor = UserSilenceHangupProcessor(
            timeout_secs=3.0,
            schedule_call_end=mock_end,
            suppress_idle_when=suppress_mock,
        )
        processor._schedule_idle_timer = MagicMock()
        processor.push_frame = AsyncMock()

        await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        suppress_mock.assert_called_once()
        processor._schedule_idle_timer.assert_not_called()
        assert processor._idle_task is None
