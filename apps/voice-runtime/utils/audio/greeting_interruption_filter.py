
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    StartInterruptionFrame,
    InterruptionFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class GreetingGuard:
    """Shared greeting state for block + complete filters in the pipeline."""

    def __init__(self) -> None:
        self.in_progress = False
        self._tts_stopped = False
        self._bot_stopped = False

    def start_greeting(self) -> None:
        self.in_progress = True
        self._tts_stopped = False
        self._bot_stopped = False
        logger.debug("Greeting started - interruptions blocked")

    def note_tts_stopped(self) -> None:
        if not self.in_progress:
            return
        self._tts_stopped = True
        self._try_complete()

    def note_bot_stopped(self) -> None:
        if not self.in_progress:
            return
        self._bot_stopped = True
        self._try_complete()

    def note_bot_started(self) -> None:
        if not self.in_progress or self._tts_stopped:
            return
        self._bot_stopped = False

    def _try_complete(self) -> None:
        if not (self._tts_stopped and self._bot_stopped):
            return
        self.in_progress = False
        self._tts_stopped = False
        self._bot_stopped = False
        logger.debug("Greeting completed - interruptions enabled")


class GreetingInterruptionFilter(FrameProcessor):
    """Blocks user interruption frames while the greeting TTS is in progress.

    Use two instances sharing one :class:`GreetingGuard`:
    - Before barge-in: blocks ``UserStartedSpeakingFrame`` / interruption frames.
    - Downstream of TTS: ends protection only after both ``TTSStoppedFrame`` (synthesis
      done) and ``BotStoppedSpeakingFrame`` (playback done). Mid-greeting playback gaps
      reset the bot-stop latch via ``BotStartedSpeakingFrame``.
    """

    def __init__(
        self,
        guard: GreetingGuard,
        *,
        completes_greeting: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._guard = guard
        self._completes_greeting = completes_greeting

    def start_greeting(self) -> None:
        self._guard.start_greeting()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._completes_greeting and self._guard.in_progress:
            if isinstance(frame, TTSStoppedFrame):
                self._guard.note_tts_stopped()
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._guard.note_bot_stopped()
            elif isinstance(frame, BotStartedSpeakingFrame):
                self._guard.note_bot_started()
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
    completer = GreetingInterruptionFilter(guard, completes_greeting=True)
    return guard, blocker, completer
