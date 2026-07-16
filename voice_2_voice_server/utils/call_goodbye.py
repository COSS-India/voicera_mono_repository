"""End calls when the LLM signals goodbye or end of conversation."""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSStoppedFrame,
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

# Brief debounce after the last TTSStoppedFrame (multi-sentence responses).
_HANGUP_DEBOUNCE_SECS = 0.35


@dataclass
class _GoodbyeState:
    ending: bool = False
    suppress_idle: bool = False
    end_scheduled: bool = False
    buffer: str = ""
    _hangup_task: Optional[asyncio.Task] = field(default=None, repr=False)


def _arm_hangup_after_tts(state: _GoodbyeState, schedule_call_end: Callable[[], Awaitable[None]]) -> None:
    if state.end_scheduled:
        return
    if state._hangup_task and not state._hangup_task.done():
        state._hangup_task.cancel()

    async def _hangup() -> None:
        await asyncio.sleep(_HANGUP_DEBOUNCE_SECS)
        if state.ending and not state.end_scheduled:
            state.end_scheduled = True
            logger.info("Goodbye TTS finished — ending call")
            await schedule_call_end()

    state._hangup_task = asyncio.create_task(_hangup())


class _GoodbyeDetectProcessor(FrameProcessor):
    def __init__(
        self,
        state: _GoodbyeState,
        schedule_call_end: Callable[[], Awaitable[None]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._state = state
        self._schedule_call_end = schedule_call_end

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._state.buffer = ""
            if not self._state.end_scheduled:
                self._state.ending = False
                self._state.suppress_idle = False
        elif isinstance(frame, LLMTextFrame):
            self._state.buffer += frame.text or ""
        elif isinstance(frame, LLMFullResponseEndFrame):
            if _END_CALL.search(self._state.buffer):
                logger.info(
                    "End-of-call phrase detected in LLM response — will hang up after TTS"
                )
                self._state.ending = True
                self._state.suppress_idle = True
                _arm_hangup_after_tts(self._state, self._schedule_call_end)
            self._state.buffer = ""

        await self.push_frame(frame, direction)


class _GoodbyeHangupProcessor(FrameProcessor):
    def __init__(
        self,
        schedule_call_end: Callable[[], Awaitable[None]],
        state: _GoodbyeState,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._schedule_call_end = schedule_call_end
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStoppedFrame) and self._state.ending:
            _arm_hangup_after_tts(self._state, self._schedule_call_end)

        await self.push_frame(frame, direction)


class GoodbyeHandlers:
    """Detect goodbye after LLM; hang up after TTS finishes."""

    def __init__(self, schedule_call_end: Callable[[], Awaitable[None]]):
        state = _GoodbyeState()
        self.detect = _GoodbyeDetectProcessor(state, schedule_call_end)
        self.hangup = _GoodbyeHangupProcessor(schedule_call_end, state)
        self._state = state

    def should_suppress_idle(self) -> bool:
        return self._state.suppress_idle
