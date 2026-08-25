"""Hang up when the user stays silent after the bot finishes speaking."""

import asyncio
from typing import Awaitable, Callable, Optional

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class UserSilenceHangupProcessor(FrameProcessor):
    """End the call if the user does not speak within ``timeout_secs`` after bot speech."""

    def __init__(
        self,
        timeout_secs: float,
        schedule_call_end: Callable[[], Awaitable[None]],
        suppress_idle_when: Optional[Callable[[], bool]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._timeout_secs = max(1.0, float(timeout_secs))
        self._schedule_call_end = schedule_call_end
        self._suppress_idle_when = suppress_idle_when
        self._idle_task: Optional[asyncio.Task] = None
        self._hangup_scheduled = False

    def _cancel_idle_timer(self) -> None:
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None

    def _schedule_idle_timer(self) -> None:
        self._cancel_idle_timer()

        async def _timer() -> None:
            try:
                await asyncio.sleep(self._timeout_secs)
                if self._hangup_scheduled:
                    return
                logger.info(
                    "User silence hangup after {:.0f}s — ending call",
                    self._timeout_secs,
                )
                self._hangup_scheduled = True
                await self._schedule_call_end()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error(f"User silence hangup timer failed: {exc}")

        self._idle_task = asyncio.create_task(_timer())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._cancel_idle_timer()

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._cancel_idle_timer()

        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._suppress_idle_when and self._suppress_idle_when():
                self._cancel_idle_timer()
            else:
                self._schedule_idle_timer()

        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._cancel_idle_timer()

        await self.push_frame(frame, direction)
