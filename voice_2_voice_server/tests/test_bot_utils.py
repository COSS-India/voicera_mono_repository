import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    AudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType

from utils.bot_utils import (
    get_sample_rate,
    is_non_conversational,
    get_alert_call_timeout_seconds,
    get_ignore_user_speech_before_greeting,
    get_interruption_min_words,
    get_call_timeout_seconds,
    get_user_silence_hangup_seconds,
    get_hold_messages,
    get_hold_message_timeout_seconds,
    get_user_online_detection_enabled,
    get_user_online_detection_message,
    get_user_online_detection_seconds,
    FastPunctuationAggregator,
    BargeInInterruptionProcessor,
    patch_immediate_first_chunk,
)


class TestBotUtilsConfigParsers:
    @patch.dict(os.environ, {}, clear=True)
    def test_get_sample_rate_default(self):
        assert get_sample_rate() == 8000

    @patch.dict(os.environ, {"SAMPLE_RATE": "16000"}, clear=True)
    def test_get_sample_rate_custom(self):
        assert get_sample_rate() == 16000

    def test_is_non_conversational(self):
        assert is_non_conversational({"interaction_mode": "non_conversational"}) is True
        assert is_non_conversational({"interaction_mode": "conversational"}) is False
        assert is_non_conversational({}) is False

    def test_get_alert_call_timeout_seconds(self):
        assert get_alert_call_timeout_seconds({"call_timeout_seconds": 150}) == 150
        assert get_alert_call_timeout_seconds({"call_timeout_seconds": "200"}) == 200
        assert get_alert_call_timeout_seconds({"call_timeout_seconds": 30}) == 60  # max(60, ...)
        assert get_alert_call_timeout_seconds({"call_timeout_seconds": "invalid"}) == 120
        assert get_alert_call_timeout_seconds({}) == 120

    def test_get_ignore_user_speech_before_greeting(self):
        assert get_ignore_user_speech_before_greeting({"ignore_user_speech_before_greeting": False}) is False
        assert get_ignore_user_speech_before_greeting({"ignore_user_speech_before_greeting": True}) is True
        assert get_ignore_user_speech_before_greeting({}) is True

    def test_get_interruption_min_words(self):
        assert get_interruption_min_words({"interruption_min_words": 3}) == 3
        assert get_interruption_min_words({"interruption_min_words": "4"}) == 4
        assert get_interruption_min_words({"interruption_min_words": 0}) == 1  # max(1, ...)
        assert get_interruption_min_words({"interruption_min_words": "invalid"}) == 1
        assert get_interruption_min_words({}) == 1

    def test_get_call_timeout_seconds(self):
        assert get_call_timeout_seconds({"call_timeout_seconds": 150}) == 150
        assert get_call_timeout_seconds({"call_timeout_seconds": 30}) == 60  # max(60, ...)
        assert get_call_timeout_seconds({"session_timeout_minutes": 5}) == 300
        assert get_call_timeout_seconds({"session_timeout_minutes": "invalid"}) == 600
        assert get_call_timeout_seconds({}) == 600

    def test_get_user_silence_hangup_seconds(self):
        assert get_user_silence_hangup_seconds({"user_silence_hangup_seconds": 15}) == 15
        assert get_user_silence_hangup_seconds({"user_silence_hangup_seconds": -5}) == 0
        assert get_user_silence_hangup_seconds({"user_silence_hangup_seconds": "invalid"}) == 0
        assert get_user_silence_hangup_seconds({}) == 0

    def test_get_hold_messages(self):
        assert get_hold_messages({"hold_messages": ["hold on ", "", " wait "]}) == ["hold on", "wait"]
        assert get_hold_messages({"hold_messages": "not a list"}) == []
        assert get_hold_messages({}) == []

    def test_get_hold_message_timeout_seconds(self):
        assert get_hold_message_timeout_seconds({"hold_message_timeout_seconds": 1.5}) == 1.5
        assert get_hold_message_timeout_seconds({"hold_message_timeout_seconds": 0.01}) == 0.05
        assert get_hold_message_timeout_seconds({"hold_message_timeout_seconds": "invalid"}) == 0.3
        assert get_hold_message_timeout_seconds({}) == 0.3

    def test_get_user_online_detection_enabled(self):
        assert get_user_online_detection_enabled({"user_online_detection_enabled": True}) is True
        assert get_user_online_detection_enabled({"user_online_detection_enabled": False}) is False
        assert get_user_online_detection_enabled({}) is False

    def test_get_user_online_detection_message(self):
        assert get_user_online_detection_message({"user_online_detection_message": " Hello? "}) == "Hello?"
        assert get_user_online_detection_message({}) == ""

    def test_get_user_online_detection_seconds(self):
        assert get_user_online_detection_seconds({"user_online_detection_seconds": 5.0}) == 5.0
        assert get_user_online_detection_seconds({"user_online_detection_seconds": 0.5}) == 1.0
        assert get_user_online_detection_seconds({"user_online_detection_seconds": "invalid"}) == 10.0
        assert get_user_online_detection_seconds({}) == 10.0


class TestFastPunctuationAggregator:
    @pytest.mark.asyncio
    async def test_aggregate_and_flush(self):
        aggregator = FastPunctuationAggregator()
        
        # Test aggregate with punctuation
        results = []
        async for item in aggregator.aggregate("Hello"):
            results.append(item)
        assert len(results) == 0

        async for item in aggregator.aggregate(", world!"):
            results.append(item)
        
        assert len(results) == 2
        assert results[0].text == "Hello,"
        assert results[1].text == "world!"
        assert aggregator.text.text == ""

        # Test aggregate followed by flush
        async for _ in aggregator.aggregate("Remaining text"):
            pass
        
        assert aggregator.text.text == "Remaining text"
        flush_result = await aggregator.flush()
        assert flush_result.text == "Remaining text"
        assert aggregator.text.text == ""

    @pytest.mark.asyncio
    async def test_handle_interruption_and_reset(self):
        aggregator = FastPunctuationAggregator()
        async for _ in aggregator.aggregate("Hello"):
            pass
        await aggregator.handle_interruption()
        assert aggregator.text.text == ""

        async for _ in aggregator.aggregate("World"):
            pass
        await aggregator.reset()
        assert aggregator.text.text == ""


class TestBargeInInterruptionProcessor:
    @pytest.mark.asyncio
    async def test_no_interruption_before_speech_frame(self):
        processor = BargeInInterruptionProcessor(min_words=2)
        processor.push_frame = AsyncMock()

        frame = InterimTranscriptionFrame(text="Hello world test", user_id="user", timestamp=123)
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
        
        # No InterruptionFrame should be pushed because speech frame not received yet
        processor.push_frame.assert_called_once_with(frame, FrameDirection.DOWNSTREAM)

    @pytest.mark.asyncio
    async def test_interruption_with_speech_frame_and_enough_words(self):
        processor = BargeInInterruptionProcessor(min_words=2)
        processor.push_frame = AsyncMock()

        # 1. User starts speaking
        await processor.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        assert processor._user_speaking is True
        assert processor._interrupted is False

        # 2. Interim transcription with 1 word
        frame1 = InterimTranscriptionFrame(text="Hello", user_id="user", timestamp=123)
        await processor.process_frame(frame1, FrameDirection.DOWNSTREAM)
        assert processor._interrupted is False

        # 3. Interim transcription with 2 words
        frame2 = InterimTranscriptionFrame(text="Hello world", user_id="user", timestamp=123)
        await processor.process_frame(frame2, FrameDirection.DOWNSTREAM)
        assert processor._interrupted is True

        # Check pushed frames:
        # UserStartedSpeakingFrame -> frame1 -> InterruptionFrame -> frame2
        pushed_frames = [call[0][0] for call in processor.push_frame.call_args_list]
        assert isinstance(pushed_frames[0], UserStartedSpeakingFrame)
        assert pushed_frames[1] == frame1
        assert isinstance(pushed_frames[2], InterruptionFrame)
        assert pushed_frames[3] == frame2

    @pytest.mark.asyncio
    async def test_user_stops_speaking_resets_flags(self):
        processor = BargeInInterruptionProcessor(min_words=1)
        processor.push_frame = AsyncMock()

        await processor.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        assert processor._user_speaking is True

        await processor.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        assert processor._user_speaking is False
        assert processor._interrupted is False


class TestPatchImmediateFirstChunk:
    @pytest.mark.asyncio
    async def test_patch_behavior(self):
        transport = MagicMock()
        output = MagicMock()
        transport.output.return_value = output
        
        # Define original methods
        orig_write = AsyncMock()
        orig_process = AsyncMock()
        output.write_audio_frame = orig_write
        output.process_frame = orig_process

        patch_immediate_first_chunk(transport)

        assert output._send_interval == 0
        assert output._first_chunk_sent is False

        # Test write_audio_frame hook
        frame = AudioRawFrame(audio=b"123", sample_rate=8000, num_channels=1)
        await output.write_audio_frame(frame)
        
        assert output._first_chunk_sent is True
        orig_write.assert_called_once_with(frame)

        # Test process_frame hook resetting on TTSStartedFrame
        tts_frame = TTSStartedFrame()
        await output.process_frame(tts_frame, FrameDirection.DOWNSTREAM)
        assert output._first_chunk_sent is False
        orig_process.assert_called_once_with(tts_frame, FrameDirection.DOWNSTREAM)
