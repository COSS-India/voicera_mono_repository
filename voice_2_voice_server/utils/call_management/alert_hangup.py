"""Hang up after a one-way alert message finishes playing."""

from typing import Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import BotStoppedSpeakingFrame, Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AlertHangupProcessor(FrameProcessor):
    """End the call after alert TTS finishes (BotStoppedSpeakingFrame)."""

    def __init__(
        self,
        schedule_call_end: Callable[[], Awaitable[None]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._schedule_call_end = schedule_call_end
        self._end_scheduled = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStoppedSpeakingFrame) and not self._end_scheduled:
            self._end_scheduled = True
            logger.info("Alert message finished — ending call")
            await self._schedule_call_end()

        await self.push_frame(frame, direction)
