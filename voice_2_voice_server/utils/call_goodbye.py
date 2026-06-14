"""End calls when the LLM signals goodbye or end of conversation."""

import re
from typing import Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_END_CALL = re.compile(
    r"\b(?:"
    r"goodbye|good\s+bye|bye(?:\s+bye)?|"
    r"end\s+of\s+(?:the\s+)?(?:conversation|call)|"
    r"see\s+you(?:\s+(?:later|soon))?|take\s+care|farewell|"
    r"talk\s+(?:to\s+you\s+)?later|until\s+next\s+time|signing\s+off|"
    r"have\s+a\s+(?:good|nice|great)\s+day|that(?:'s| is)\s+all\s+for\s+now"
    r")\b",
    re.IGNORECASE,
)


class GoodbyeHangupProcessor(FrameProcessor):
    """Hang up after TTS when the LLM response includes goodbye."""

    def __init__(
        self,
        schedule_call_end: Callable[[], Awaitable[None]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._schedule_call_end = schedule_call_end
        self._buffer = ""
        self._ending = False
        self._suppress_idle = False
        self._end_scheduled = False

    def should_suppress_idle(self) -> bool:
        return self._suppress_idle

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = ""
        elif isinstance(frame, LLMTextFrame):
            self._buffer += frame.text or ""
        elif isinstance(frame, LLMFullResponseEndFrame):
            if _END_CALL.search(self._buffer):
                logger.info("End-of-call phrase detected in LLM response — will hang up after speech")
                self._ending = True
                self._suppress_idle = True
            self._buffer = ""
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._ending and not self._end_scheduled:
                self._end_scheduled = True
                await self._schedule_call_end()

        await self.push_frame(frame, direction)
