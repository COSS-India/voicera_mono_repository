import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from utils.audio.user_online_detection_filter import UserOnlineDetectionFilter
from conftest import run_task_with_blocking_sleep


class TestUserOnlineDetectionFilter:
    @pytest.mark.asyncio
    async def test_initialization(self):
        processor = UserOnlineDetectionFilter(
            prompt_text="Are you there?",
            timeout_secs=5.0,
        )
        assert processor._prompt_text == "Are you there?"
        assert processor._timeout_secs == 5.0
        assert processor._idle_task is None
        assert processor._ignore_next_bot_stop is False

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_timer_fires_sends_prompt(self, mock_sleep):
        processor = UserOnlineDetectionFilter(
            prompt_text="Hello?",
            timeout_secs=2.0,
        )
        processor.push_frame = AsyncMock()

        # Bot stops speaking -> schedules timer
        await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        assert processor._idle_task is not None

        # Await the task to run to completion (since sleep is mocked and returns immediately)
        await processor._idle_task

        # Verify sleep was called with timeout
        mock_sleep.assert_called_once_with(2.0)
        assert processor._ignore_next_bot_stop is True

        # Verify TTSSpeakFrame was pushed upstream - check a call was made with correct frame type
        call_args_list = processor.push_frame.call_args_list
        found = any(
            isinstance(args[0], TTSSpeakFrame) and args[0].text == "Hello?"
            for args, kwargs in [c for c in call_args_list]
        )
        assert found, "Expected TTSSpeakFrame with 'Hello?' to be pushed upstream"

    @pytest.mark.asyncio
    async def test_timer_cancelled_by_user_speaking(self):
        block = asyncio.Event()

        async def blocking_sleep(_):
            await block.wait()

        processor = UserOnlineDetectionFilter(prompt_text="Hello?", timeout_secs=2.0)
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

    @pytest.mark.asyncio
    async def test_timer_cancelled_by_bot_speaking(self):
        block = asyncio.Event()

        async def blocking_sleep(_):
            await block.wait()

        processor = UserOnlineDetectionFilter(prompt_text="Hello?", timeout_secs=2.0)
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

    @pytest.mark.asyncio
    async def test_timer_cancelled_by_end_frame(self):
        block = asyncio.Event()

        async def blocking_sleep(_):
            await block.wait()

        processor = UserOnlineDetectionFilter(prompt_text="Hello?")
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

    @pytest.mark.asyncio
    async def test_timer_cancelled_by_cancel_frame(self):
        block = asyncio.Event()

        async def blocking_sleep(_):
            await block.wait()

        processor = UserOnlineDetectionFilter(prompt_text="Hello?")
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

    @pytest.mark.asyncio
    async def test_ignore_next_bot_stop(self):
        processor = UserOnlineDetectionFilter(prompt_text="Hello?")
        processor._ignore_next_bot_stop = True
        processor._schedule_idle_timer = MagicMock()
        processor.push_frame = AsyncMock()

        await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        # Should NOT schedule a timer and should reset the flag
        assert processor._ignore_next_bot_stop is False
        processor._schedule_idle_timer.assert_not_called()

    @pytest.mark.asyncio
    async def test_suppress_idle_when(self):
        suppress_mock = MagicMock(return_value=True)
        processor = UserOnlineDetectionFilter(
            prompt_text="Hello?",
            suppress_idle_when=suppress_mock,
        )
        processor._schedule_idle_timer = MagicMock()
        processor.push_frame = AsyncMock()

        await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        suppress_mock.assert_called_once()
        processor._schedule_idle_timer.assert_not_called()
        assert processor._idle_task is None
