import asyncio
from typing import Awaitable, Callable, Optional

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartInterruptionFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_HANGUP_DELAY_SECS = 1.0
_BLOCKED_FRAMES = (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    StartInterruptionFrame,
    InterruptionFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)


class UserOnlineDetectionInterruptionBlocker(FrameProcessor):
    """Drops user speech / interruption frames while online detection prompts play."""

    def __init__(self, should_block: Callable[[], bool], **kwargs):
        super().__init__(**kwargs)
        self._should_block = should_block

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._should_block() and isinstance(frame, _BLOCKED_FRAMES):
            logger.debug("Blocked {} during user online detection prompt", type(frame).__name__)
            return
        await self.push_frame(frame, direction)


class UserOnlineDetectionFilter(FrameProcessor):
    """Silence prompts after greeting/LLM TTS; repeats, closing message, then hangup."""

    def __init__(
        self,
        prompt_text: str,
        timeout_secs: float = 10.0,
        max_repeats: int = 1,
        closing_message: str = "",
        schedule_call_end: Optional[Callable[[], Awaitable[None]]] = None,
        suppress_idle_when: Optional[Callable[[], bool]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._prompt_text = str(prompt_text).strip()
        self._timeout_secs = max(1.0, float(timeout_secs))
        self._max_repeats = max(1, int(max_repeats))
        self._closing_message = str(closing_message or "").strip()
        self._schedule_call_end = schedule_call_end
        self._suppress_idle_when = suppress_idle_when
        self._idle_task: Optional[asyncio.Task] = None
        self._hangup_task: Optional[asyncio.Task] = None
        self._armed = False
        self._llm_turn_pending = False
        self._pending_tts_stop = False  # greeting (or non-LLM) TTS finished synthesizing
        self._turn_interrupted = False
        self._user_speaking = False
        self._user_speech_confirmed = False
        self._prompt_count = 0
        self._awaiting_prompt_stop = False
        self._awaiting_closing_stop = False
        self._own_tts_stopped = False
        self._own_bot_stopped = False
        self._hangup_scheduled = False

    def is_playing_detection_audio(self) -> bool:
        return self._awaiting_prompt_stop

    def _watching_own_speech(self) -> bool:
        return self._awaiting_prompt_stop or self._awaiting_closing_stop

    def _reset_cycle(self) -> None:
        self._prompt_count = 0
        self._awaiting_prompt_stop = False
        self._awaiting_closing_stop = False
        self._own_tts_stopped = False
        self._own_bot_stopped = False

    def _cancel_task(self, attr: str) -> None:
        task: Optional[asyncio.Task] = getattr(self, attr)
        if task and not task.done():
            task.cancel()
        setattr(self, attr, None)

    def _cancel_idle_timer(self) -> None:
        self._cancel_task("_idle_task")

    def _cancel_hangup(self) -> None:
        self._cancel_task("_hangup_task")
        self._hangup_scheduled = False

    def _clear_own_speech_flags(self) -> None:
        self._own_tts_stopped = False
        self._own_bot_stopped = False

    def _can_schedule(self) -> bool:
        if (
            not self._armed
            or self._user_speaking
            or self._hangup_scheduled
            or self._watching_own_speech()
            or self._prompt_count >= self._max_repeats
        ):
            return False
        if self._suppress_idle_when and self._suppress_idle_when():
            self._cancel_idle_timer()
            return False
        return True

    async def _speak_upstream(self, text: str, *, closing: bool = False) -> None:
        self._awaiting_closing_stop = closing
        self._awaiting_prompt_stop = not closing
        self._clear_own_speech_flags()
        await self.push_frame(TTSSpeakFrame(text), FrameDirection.UPSTREAM)

    def _hangup(self) -> None:
        if self._hangup_scheduled or self._schedule_call_end is None:
            return
        self._hangup_scheduled = True
        self._cancel_idle_timer()

        async def _do_hangup() -> None:
            try:
                logger.info(
                    "Last online-detection message finished — hanging up in {:.1f}s",
                    _HANGUP_DELAY_SECS,
                )
                await asyncio.sleep(_HANGUP_DELAY_SECS)
                logger.info("User online detection cycle finished — ending call")
                await self._schedule_call_end()
            except asyncio.CancelledError:
                logger.info("Online detection hangup cancelled — user returned")

        self._hangup_task = asyncio.create_task(_do_hangup())

    def _schedule_idle_timer(self) -> None:
        self._cancel_idle_timer()

        async def _timer() -> None:
            try:
                await asyncio.sleep(self._timeout_secs)
                if self._hangup_scheduled or self._prompt_count >= self._max_repeats:
                    return
                if self._user_speaking:
                    logger.info(
                        "Online detection timer elapsed while provisional speech active"
                    )
                    return
                self._prompt_count += 1
                logger.info(
                    "User silence detected for {:.0f}s, sending online detection "
                    "prompt ({}/{})",
                    self._timeout_secs,
                    self._prompt_count,
                    self._max_repeats,
                )
                await self._speak_upstream(self._prompt_text)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error(f"Failed to send user online detection prompt: {exc}")

        self._idle_task = asyncio.create_task(_timer())

    def _maybe_schedule_idle_timer(self) -> None:
        if not self._can_schedule():
            return
        if self._idle_task and not self._idle_task.done():
            return
        self._schedule_idle_timer()

    def _pause_for_provisional_speech(self) -> None:
        self._user_speaking = True
        self._cancel_idle_timer()

    def _resume_after_unconfirmed_speech(self) -> None:
        self._user_speaking = False
        if not self._can_schedule():
            return
        logger.info(
            "No confirmed user speech — resuming online detection ({}/{})",
            self._prompt_count,
            self._max_repeats,
        )
        self._schedule_idle_timer()

    def _confirm_user_activity(self) -> None:
        self._user_speech_confirmed = True
        self._cancel_idle_timer()
        self._cancel_hangup()
        self._reset_cycle()
        logger.info("Confirmed user speech — online detection cycle reset")

    def _schedule_next_after_prompt(self) -> None:
        if self._hangup_scheduled:
            return
        if self._suppress_idle_when and self._suppress_idle_when():
            self._cancel_idle_timer()
            return
        if self._prompt_count >= self._max_repeats:
            return
        self._user_speaking = False
        self._user_speech_confirmed = False
        logger.info(
            "Scheduling next online detection prompt ({}/{} remaining after this wait)",
            self._prompt_count + 1,
            self._max_repeats,
        )
        self._schedule_idle_timer()

    async def _on_own_speech_complete(self) -> None:
        if self._awaiting_closing_stop:
            self._awaiting_closing_stop = False
            self._clear_own_speech_flags()
            if self._user_speech_confirmed:
                logger.info("Closing message interrupted by user speech — skipping hangup")
                return
            self._hangup()
            return
        if not self._awaiting_prompt_stop:
            return
        self._awaiting_prompt_stop = False
        self._clear_own_speech_flags()
        if self._prompt_count < self._max_repeats:
            self._schedule_next_after_prompt()
        elif self._closing_message:
            logger.info("Speaking user online detection closing message")
            await self._speak_upstream(self._closing_message, closing=True)
        else:
            self._hangup()

    async def _try_complete_own_speech(self) -> None:
        if self._watching_own_speech() and self._own_tts_stopped and self._own_bot_stopped:
            await self._on_own_speech_complete()

    def _arm_and_schedule(self) -> None:
        """Arm (if needed) and start a fresh silence cycle after bot speech ends."""
        self._llm_turn_pending = False
        self._pending_tts_stop = False
        self._reset_cycle()
        self._armed = True
        self._maybe_schedule_idle_timer()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (UserStartedSpeakingFrame, StartInterruptionFrame)):
            if self._watching_own_speech():
                logger.debug("Ignoring provisional speech during online detection audio")
            else:
                self._pause_for_provisional_speech()

        elif isinstance(frame, UserStoppedSpeakingFrame):
            was_provisional = self._user_speaking and not self._user_speech_confirmed
            self._user_speaking = False
            if was_provisional:
                self._resume_after_unconfirmed_speech()
            else:
                self._user_speech_confirmed = False

        elif isinstance(frame, InterruptionFrame):
            self._turn_interrupted = True
            if self._awaiting_closing_stop:
                self._cancel_hangup()
                self._reset_cycle()
                logger.info("Closing message interrupted — hangup aborted pending user turn")
            elif not self._watching_own_speech():
                self._pause_for_provisional_speech()
            else:
                logger.debug("Ignoring interruption during online detection audio")

        elif isinstance(frame, TranscriptionFrame) and not isinstance(
            frame, InterimTranscriptionFrame
        ):
            if frame.text.strip():
                self._confirm_user_activity()

        elif isinstance(frame, LLMFullResponseStartFrame):
            self._cancel_hangup()
            self._reset_cycle()
            self._llm_turn_pending = False
            self._pending_tts_stop = False
            self._turn_interrupted = False
            self._user_speech_confirmed = False
            logger.info("LLM response started — online detection hangup cancelled, cycle reset")

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._llm_turn_pending = True

        elif isinstance(frame, TTSStoppedFrame):
            if self._watching_own_speech():
                self._own_tts_stopped = True
                await self._try_complete_own_speech()
            elif not self._armed:
                # Greeting / first non-LLM TTS finished synthesizing.
                self._pending_tts_stop = True

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._cancel_idle_timer()
            if self._watching_own_speech() and not self._own_tts_stopped:
                self._own_bot_stopped = False

        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._watching_own_speech():
                self._own_bot_stopped = True
                await self._try_complete_own_speech()
            elif self._turn_interrupted:
                self._turn_interrupted = False
            elif self._llm_turn_pending or (not self._armed and self._pending_tts_stop):
                # LLM turn end, or greeting/first TTS fully played out.
                self._arm_and_schedule()

        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._cancel_idle_timer()

        await self.push_frame(frame, direction)
