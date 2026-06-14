import asyncio
from typing import Callable, Optional

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class UserOnlineDetectionFilter(FrameProcessor):
    """Plays a prompt if the user stays silent after the bot finishes speaking."""

    def __init__(
        self,
        prompt_text: str,
        timeout_secs: float = 10.0,
        suppress_idle_when: Optional[Callable[[], bool]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._prompt_text = str(prompt_text).strip()
        self._timeout_secs = max(1.0, float(timeout_secs))
        self._idle_task: Optional[asyncio.Task] = None
        self._ignore_next_bot_stop = False
        self._suppress_idle_when = suppress_idle_when

    def _cancel_idle_timer(self) -> None:
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None

    def _schedule_idle_timer(self) -> None:
        self._cancel_idle_timer()

        async def _timer() -> None:
            try:
                await asyncio.sleep(self._timeout_secs)
                logger.info(
                    "User silence detected for {:.0f}s, sending online detection prompt",
                    self._timeout_secs,
                )
                self._ignore_next_bot_stop = True
                # This processor is placed after TTS in the pipeline, so send
                # the speak frame upstream to ensure it flows through TTS.
                await self.push_frame(
                    TTSSpeakFrame(self._prompt_text),
                    FrameDirection.UPSTREAM,
                )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error(f"Failed to send user online detection prompt: {exc}")

        self._idle_task = asyncio.create_task(_timer())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._cancel_idle_timer()

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._cancel_idle_timer()

        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._ignore_next_bot_stop:
                self._ignore_next_bot_stop = False
            elif self._suppress_idle_when and self._suppress_idle_when():
                self._cancel_idle_timer()
            else:
                self._schedule_idle_timer()

        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._cancel_idle_timer()

        await self.push_frame(frame, direction)
