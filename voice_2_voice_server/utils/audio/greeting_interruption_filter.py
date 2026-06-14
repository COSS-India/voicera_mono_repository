
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    StartInterruptionFrame,
    InterruptionFrame,
    UserStartedSpeakingFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class GreetingGuard:
    """Shared greeting state for block + complete filters in the pipeline."""

    def __init__(self) -> None:
        self.in_progress = False

    def start_greeting(self) -> None:
        self.in_progress = True
        logger.debug("Greeting started - interruptions blocked")

    def complete_greeting(self) -> None:
        if not self.in_progress:
            return
        self.in_progress = False
        logger.debug("Greeting completed - interruptions enabled")


class GreetingInterruptionFilter(FrameProcessor):
    """Blocks user interruption frames while the greeting TTS is in progress.

    Use two instances sharing one :class:`GreetingGuard`:
    - Before barge-in: blocks ``UserStartedSpeakingFrame`` / interruption frames.
    - Immediately after TTS: ends protection on ``TTSStoppedFrame`` (not
      ``BotStoppedSpeakingFrame``, which fires between streaming audio chunks).
    """

    def __init__(
        self,
        guard: GreetingGuard,
        *,
        completes_on_tts_stop: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._guard = guard
        self._completes_on_tts_stop = completes_on_tts_stop

    def start_greeting(self) -> None:
        self._guard.start_greeting()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._completes_on_tts_stop:
            if isinstance(frame, TTSStoppedFrame) and self._guard.in_progress:
                self._guard.complete_greeting()
        elif self._guard.in_progress and isinstance(
            frame, (StartInterruptionFrame, InterruptionFrame, UserStartedSpeakingFrame)
        ):
            logger.debug(f"Blocked {frame.__class__.__name__} during greeting")
            return

        await self.push_frame(frame, direction)


def create_greeting_filters() -> tuple[GreetingGuard, GreetingInterruptionFilter, GreetingInterruptionFilter]:
    """Return (guard, blocker, completer) for the voice pipeline."""
    guard = GreetingGuard()
    blocker = GreetingInterruptionFilter(guard)
    completer = GreetingInterruptionFilter(guard, completes_on_tts_stop=True)
    return guard, blocker, completer
