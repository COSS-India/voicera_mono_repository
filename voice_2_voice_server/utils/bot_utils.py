"""Shared helpers, config parsers, and pipeline utilities for the voice bot."""

import os
import time

from loguru import logger
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.text.base_text_aggregator import (
    Aggregation,
    AggregationType,
    BaseTextAggregator,
)


def get_sample_rate() -> int:
    """Get the audio sample rate from environment."""
    return int(os.getenv("SAMPLE_RATE", "8000"))


def is_non_conversational(agent_config: dict) -> bool:
    """True for one-way alert agents (TTS only, no STT/LLM)."""
    return agent_config.get("interaction_mode") == "non_conversational"


def get_alert_call_timeout_seconds(agent_config: dict) -> int:
    """Safety timeout for non-conversational alert calls (default 120s)."""
    raw = agent_config.get("call_timeout_seconds")
    if raw is not None:
        try:
            return max(60, int(raw))
        except (TypeError, ValueError):
            pass
    return 120


def get_ignore_user_speech_before_greeting(agent_config: dict) -> bool:
    """Default True for agents created before this setting existed."""
    raw = agent_config.get("ignore_user_speech_before_greeting")
    if raw is None:
        return True
    return bool(raw)


def get_interruption_min_words(agent_config: dict) -> int:
    """Default 1 word for agents created before this setting existed."""
    raw = agent_config.get("interruption_min_words")
    if raw is None:
        return 1
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def get_call_timeout_seconds(agent_config: dict) -> int:
    """Default 600s (10 min), falling back to legacy session_timeout_minutes."""
    raw = agent_config.get("call_timeout_seconds")
    if raw is not None:
        try:
            return max(60, int(raw))
        except (TypeError, ValueError):
            pass
    try:
        minutes = int(agent_config.get("session_timeout_minutes", 10))
        return max(60, minutes * 60)
    except (TypeError, ValueError):
        return 600


def get_user_silence_hangup_seconds(agent_config: dict) -> int:
    """Default 0 (disabled) for agents created before this setting existed."""
    raw = agent_config.get("user_silence_hangup_seconds")
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def get_hold_messages(agent_config: dict) -> list[str]:
    """Hold messages from agent config; empty list disables hold audio."""
    raw = agent_config.get("hold_messages")
    if not isinstance(raw, list):
        return []
    return [str(msg).strip() for msg in raw if str(msg).strip()]


def get_hold_message_timeout_seconds(agent_config: dict) -> float:
    """Seconds to wait for first LLM chunk before playing a hold message."""
    raw = agent_config.get("hold_message_timeout_seconds")
    if raw is None:
        return 0.3
    try:
        return max(0.05, float(raw))
    except (TypeError, ValueError):
        return 0.3


def get_user_online_detection_enabled(agent_config: dict) -> bool:
    """Whether to prompt the caller after silence following bot speech."""
    return bool(agent_config.get("user_online_detection_enabled"))


def get_user_online_detection_message(agent_config: dict) -> str:
    """Prompt played when the user stays silent after bot speech."""
    raw = agent_config.get("user_online_detection_message")
    if raw is None:
        return ""
    return str(raw).strip()


def get_user_online_detection_seconds(agent_config: dict) -> float:
    """Seconds of user silence after bot speech before playing the prompt."""
    raw = agent_config.get("user_online_detection_seconds")
    if raw is None:
        return 10.0
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 10.0


class FastPunctuationAggregator(BaseTextAggregator):
    """Fast aggregator that sends text immediately on punctuation - no lookahead/NLTK."""

    def __init__(self):
        self._text = ""

    @property
    def text(self):
        return Aggregation(text=self._text.strip(), type=AggregationType.SENTENCE)

    async def aggregate(self, text: str):
        for char in text:
            self._text += char
            if char in ".!?,":
                if self._text.strip():
                    yield Aggregation(self._text.strip(), AggregationType.SENTENCE)
                    self._text = ""

    async def flush(self):
        if self._text.strip():
            result = self._text.strip()
            self._text = ""
            return Aggregation(result, AggregationType.SENTENCE)
        return None

    async def handle_interruption(self):
        self._text = ""

    async def reset(self):
        self._text = ""


class BargeInInterruptionProcessor(FrameProcessor):
    """Smart barge-in: interrupt bot only on real human speech, never on noise.

    Three-layer noise filter
    ────────────────────────
    Layer 1 — SileroVAD (neural, transport level)
        Runs a trained speech/non-speech classifier on every 30 ms audio window.
        confidence=0.7 means the audio must be 70%+ speech-like before
        UserStartedSpeakingFrame is emitted.
        • Cough   → typically scores 0.1–0.3  → SILENT, no frame ✓
        • Dog bark → typically scores 0.1–0.25 → SILENT, no frame ✓
        • Background noise → scores ~0.0       → SILENT, no frame ✓
        • Human speech     → scores 0.8–1.0    → UserStartedSpeakingFrame ✓

    Layer 2 — Speaking guard (_user_speaking flag)
        Only armed when UserStartedSpeakingFrame is seen (Silero approved).

    Layer 3 — Transcript gate + minimum word count
        Requires at least ``min_words`` spoken words before emitting InterruptionFrame.
        A loud cough slipping Silero typically produces garbage with too few words.
    """

    def __init__(self, min_words: int = 1, **kwargs):
        super().__init__(**kwargs)
        self._min_words = max(1, min_words)
        self._user_speaking: bool = False
        self._interrupted: bool = False

    def _word_count(self, text: str) -> int:
        return len(text.split()) if text else 0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            self._interrupted = False
            logger.debug("Silero: speech detected — armed, waiting for transcript")

        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._user_speaking and not self._interrupted:
                logger.debug("Silero: speech ended with no valid transcript — no barge-in")
            self._user_speaking = False
            self._interrupted = False

        elif isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            text = frame.text.strip()
            if (
                self._user_speaking
                and not self._interrupted
                and self._word_count(text) >= self._min_words
            ):
                self._interrupted = True
                logger.debug(
                    "Barge-in confirmed (Silero + {} words in '{}') — interrupting bot",
                    self._word_count(text),
                    text[:80],
                )
                await self.push_frame(InterruptionFrame(), direction)

        await self.push_frame(frame, direction)


def patch_immediate_first_chunk(transport):
    """Patch transport to send first audio chunk immediately with zero delay."""
    output = transport.output()
    output._send_interval = 0
    output._first_chunk_sent = False

    _orig_write = output.write_audio_frame

    async def _write_immediate(frame):
        if not output._first_chunk_sent:
            output._first_chunk_sent = True
            output._next_send_time = time.monotonic() - 0.001
            logger.info(
                f"🚀 Sending first chunk immediately: {len(frame.audio)} bytes (bypassing queue)"
            )
        await _orig_write(frame)

    output.write_audio_frame = _write_immediate

    _orig_process = output.process_frame

    async def _reset_on_tts(frame, direction):
        if isinstance(frame, TTSStartedFrame):
            output._first_chunk_sent = False
            logger.debug("🔄 Reset first_chunk_sent flag for new TTS response")
        await _orig_process(frame, direction)

    output.process_frame = _reset_on_tts
