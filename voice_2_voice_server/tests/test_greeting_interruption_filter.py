import pytest
from unittest.mock import AsyncMock, patch
from pipecat.frames.frames import (
    StartInterruptionFrame,
    InterruptionFrame,
    UserStartedSpeakingFrame,
    TTSStoppedFrame,
    AudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from utils.audio.greeting_interruption_filter import (
    GreetingGuard,
    GreetingInterruptionFilter,
    create_greeting_filters,
)


class TestGreetingGuard:
    def test_initial_state(self):
        guard = GreetingGuard()
        assert guard.in_progress is False

    def test_start_greeting(self):
        guard = GreetingGuard()
        guard.start_greeting()
        assert guard.in_progress is True

    def test_complete_greeting(self):
        guard = GreetingGuard()
        guard.complete_greeting()  # No-op when not in progress
        assert guard.in_progress is False

        guard.start_greeting()
        guard.complete_greeting()
        assert guard.in_progress is False


class TestGreetingInterruptionFilter:
    @pytest.mark.asyncio
    async def test_blocker_blocks_when_greeting_in_progress(self):
        guard = GreetingGuard()
        filter_proc = GreetingInterruptionFilter(guard, completes_on_tts_stop=False)
        filter_proc.push_frame = AsyncMock()

        # Bypass pipecat's internal lifecycle (TaskManager) by patching
        # FrameProcessor.process_frame so super() calls are no-ops.
        with patch.object(
            FrameProcessor,
            "process_frame",
            new_callable=lambda: lambda *a, **kw: __import__("asyncio").sleep(0),
        ):
            # Start protection
            filter_proc.start_greeting()
            assert guard.in_progress is True

            # Frames that should be blocked
            for frame in [InterruptionFrame(), UserStartedSpeakingFrame()]:
                filter_proc.push_frame.reset_mock()
                await filter_proc.process_frame(frame, FrameDirection.DOWNSTREAM)
                filter_proc.push_frame.assert_not_called()

            # Regular frame that should NOT be blocked
            audio_frame = AudioRawFrame(audio=b"123", sample_rate=8000, num_channels=1)
            filter_proc.push_frame.reset_mock()
            await filter_proc.process_frame(audio_frame, FrameDirection.DOWNSTREAM)
            filter_proc.push_frame.assert_called_once_with(audio_frame, FrameDirection.DOWNSTREAM)

    @pytest.mark.asyncio
    async def test_blocker_allows_when_greeting_completed(self):
        guard = GreetingGuard()
        filter_proc = GreetingInterruptionFilter(guard, completes_on_tts_stop=False)
        filter_proc.push_frame = AsyncMock()

        with patch.object(
            FrameProcessor,
            "process_frame",
            new_callable=lambda: lambda *a, **kw: __import__("asyncio").sleep(0),
        ):
            # Greeting not in progress — frames pass through
            assert guard.in_progress is False

            for frame in [InterruptionFrame(), UserStartedSpeakingFrame()]:
                filter_proc.push_frame.reset_mock()
                await filter_proc.process_frame(frame, FrameDirection.DOWNSTREAM)
                filter_proc.push_frame.assert_called_once_with(frame, FrameDirection.DOWNSTREAM)

    @pytest.mark.asyncio
    async def test_completer_triggers_completion_on_tts_stopped(self):
        guard = GreetingGuard()
        filter_proc = GreetingInterruptionFilter(guard, completes_on_tts_stop=True)
        filter_proc.push_frame = AsyncMock()

        with patch.object(
            FrameProcessor,
            "process_frame",
            new_callable=lambda: lambda *a, **kw: __import__("asyncio").sleep(0),
        ):
            guard.start_greeting()
            assert guard.in_progress is True

            # Regular frame should not affect guard
            await filter_proc.process_frame(
                AudioRawFrame(audio=b"", sample_rate=8000, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )
            assert guard.in_progress is True

            # TTSStoppedFrame should complete the greeting
            await filter_proc.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
            assert guard.in_progress is False


def test_create_greeting_filters():
    guard, blocker, completer = create_greeting_filters()
    assert isinstance(guard, GreetingGuard)
    assert isinstance(blocker, GreetingInterruptionFilter)
    assert blocker._guard == guard
    assert blocker._completes_on_tts_stop is False

    assert isinstance(completer, GreetingInterruptionFilter)
    assert completer._guard == guard
    assert completer._completes_on_tts_stop is True
