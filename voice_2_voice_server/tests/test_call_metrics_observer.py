import pytest
from unittest.mock import MagicMock
from pipecat.frames.frames import TranscriptionFrame, MetricsFrame
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import FramePushed

from utils.metrics.call_metrics_observer import (
    _ms,
    _normalize_text,
    _is_same_utterance,
    _pick_metric,
    _merge_turn,
    CallMetricsObserver,
)


def test_ms_helper():
    assert _ms(1.234) == 1234.0
    assert _ms(0.0) == 0.0


def test_normalize_text_helper():
    assert _normalize_text("  hello   world  ") == "hello world"
    assert _normalize_text("") == ""
    assert _normalize_text(None) == ""


def test_is_same_utterance_helper():
    assert _is_same_utterance("hello", "hello world") is True
    assert _is_same_utterance("hello world", "hello") is True
    assert _is_same_utterance("hello", "world") is False
    assert _is_same_utterance("", "hello") is False
    assert _is_same_utterance(None, "hello") is False


def test_pick_metric_helper():
    assert _pick_metric(None, 10.0) == 10.0
    assert _pick_metric(5.0, None) == 5.0
    assert _pick_metric(None, None) is None
    assert _pick_metric(5.0, 10.0) == 10.0
    assert _pick_metric(10.0, 5.0) == 10.0

    # Test prefer_nonzero
    assert _pick_metric(0.0, 5.0, prefer_nonzero=True) == 5.0
    assert _pick_metric(5.0, 0.0, prefer_nonzero=True) == 5.0
    assert _pick_metric(0.0, 0.0, prefer_nonzero=True) == 0.0


def test_merge_turn_helper():
    turn_a = {
        "stt_ms": 100.0,
        "llm_ttfb_ms": 200.0,
        "tts_first_chunk_ms": None,
        "_full_text": "hello",
        "user_text_preview": "hello",
    }
    turn_b = {
        "stt_ms": 120.0,
        "llm_ttfb_ms": None,
        "tts_first_chunk_ms": 300.0,
        "_full_text": "hello world",
        "user_text_preview": "hello world",
    }
    merged = _merge_turn(turn_a, turn_b)
    assert merged["stt_ms"] == 120.0
    assert merged["llm_ttfb_ms"] == 200.0
    assert merged["tts_first_chunk_ms"] == 300.0
    assert merged["_full_text"] == "hello world"
    assert merged["user_text_preview"] == "hello world"


class TestCallMetricsObserver:
    def test_observer_initialization(self):
        observer = CallMetricsObserver(
            stt_processor_name="deepgram_stt",
            llm_processor_name="openai_llm",
            tts_processor_name="elevenlabs_tts",
        )
        assert observer._stt_name == "deepgram_stt"
        assert observer._llm_name == "openai_llm"
        assert observer._tts_name == "elevenlabs_tts"
        assert len(observer._turns) == 0
        assert observer._current is None

    def test_processor_matching(self):
        observer = CallMetricsObserver(
            stt_processor_name="deepgram_stt",
            llm_processor_name="openai_llm",
            tts_processor_name="elevenlabs_tts",
        )
        assert observer._is_stt("deepgram_stt") is True
        assert observer._is_stt("deepgram") is True
        assert observer._is_stt("stt") is True
        assert observer._is_stt("llm") is False

        assert observer._is_llm("openai_llm") is True
        assert observer._is_tts("elevenlabs_tts") is True

    @pytest.mark.asyncio
    async def test_on_push_frame_transcription(self):
        observer = CallMetricsObserver(
            stt_processor_name="stt",
            llm_processor_name="llm",
            tts_processor_name="tts",
        )

        frame = TranscriptionFrame(text="Hello", user_id="user", timestamp=123)
        data = FramePushed(source=MagicMock(), destination=MagicMock(), frame=frame, direction=MagicMock(), timestamp=123)
        await observer.on_push_frame(data)

        assert observer._current is not None
        assert observer._current["_full_text"] == "Hello"
        assert observer._current["turn_index"] == 1

        # Same utterance should merge
        frame2 = TranscriptionFrame(text="Hello world", user_id="user", timestamp=124)
        data2 = FramePushed(source=MagicMock(), destination=MagicMock(), frame=frame2, direction=MagicMock(), timestamp=124)
        await observer.on_push_frame(data2)
        assert observer._current["_full_text"] == "Hello world"
        assert len(observer._turns) == 0

        # Different utterance should close current and start new
        frame3 = TranscriptionFrame(text="Goodbye", user_id="user", timestamp=125)
        data3 = FramePushed(source=MagicMock(), destination=MagicMock(), frame=frame3, direction=MagicMock(), timestamp=125)
        await observer.on_push_frame(data3)
        assert len(observer._turns) == 1
        assert observer._turns[0]["_full_text"] == "Hello world"
        assert observer._current["_full_text"] == "Goodbye"

    @pytest.mark.asyncio
    async def test_on_push_frame_metrics(self):
        observer = CallMetricsObserver(
            stt_processor_name="stt",
            llm_processor_name="llm",
            tts_processor_name="tts",
        )

        # 1. Processing metric arrives for STT before transcription starts
        stt_metric = ProcessingMetricsData(processor="stt", value=0.15)
        frame1 = MetricsFrame(data=[stt_metric])
        await observer.on_push_frame(FramePushed(source=MagicMock(), destination=MagicMock(), frame=frame1, direction=MagicMock(), timestamp=123))
        assert observer._pending["stt_ms"] == 150.0

        # 2. Transcription arrives
        tx_frame = TranscriptionFrame(text="test query", user_id="user", timestamp=123)
        await observer.on_push_frame(FramePushed(source=MagicMock(), destination=MagicMock(), frame=tx_frame, direction=MagicMock(), timestamp=123))
        assert observer._current["stt_ms"] == 150.0

        # 3. LLM TTFB metric arrives
        llm_metric = TTFBMetricsData(processor="llm", value=0.45)
        frame2 = MetricsFrame(data=[llm_metric])
        await observer.on_push_frame(FramePushed(source=MagicMock(), destination=MagicMock(), frame=frame2, direction=MagicMock(), timestamp=123))
        assert observer._current["llm_ttfb_ms"] == 450.0

        # 4. TTS TTFB metric arrives
        tts_metric = TTFBMetricsData(processor="tts", value=0.6)
        frame3 = MetricsFrame(data=[tts_metric])
        await observer.on_push_frame(FramePushed(source=MagicMock(), destination=MagicMock(), frame=frame3, direction=MagicMock(), timestamp=123))
        assert observer._current["tts_first_chunk_ms"] == 600.0

    def test_to_dict_empty(self):
        observer = CallMetricsObserver(
            stt_processor_name="stt",
            llm_processor_name="llm",
            tts_processor_name="tts",
        )
        res = observer.to_dict()
        assert res["turns"] == []
        assert "turn_count" in res["summary"]
        assert res["summary"]["turn_count"] == 0

    def test_to_dict_with_data(self):
        observer = CallMetricsObserver(
            stt_processor_name="stt",
            llm_processor_name="llm",
            tts_processor_name="tts",
        )
        # Mock turns directly to test duduplication and aggregation in to_dict()
        observer._turns = [
            {
                "turn_index": 1,
                "_full_text": "hello",
                "user_text_preview": "hello",
                "stt_ms": 100.0,
                "llm_ttfb_ms": 200.0,
                "tts_first_chunk_ms": 300.0,
            },
            {
                "turn_index": 2,
                "_full_text": "hello world",
                "user_text_preview": "hello world",
                "stt_ms": 120.0,
                "llm_ttfb_ms": None,
                "tts_first_chunk_ms": 350.0,
            },
            {
                "turn_index": 3,
                "_full_text": "different",
                "user_text_preview": "different",
                "stt_ms": 200.0,
                "llm_ttfb_ms": 400.0,
                "tts_first_chunk_ms": 500.0,
            }
        ]

        res = observer.to_dict()
        turns = res["turns"]
        # Turn 1 and Turn 2 should be merged because "hello" is prefix of "hello world"
        assert len(turns) == 2
        
        # Check first turn (merged)
        assert turns[0]["turn_index"] == 1
        assert turns[0]["user_text_preview"] == "hello world"
        assert turns[0]["stt_ms"] == 120.0  # preferred nonzero / max
        assert turns[0]["llm_ttfb_ms"] == 200.0
        assert turns[0]["tts_first_chunk_ms"] == 350.0

        # Check second turn
        assert turns[1]["turn_index"] == 2
        assert turns[1]["user_text_preview"] == "different"
        assert turns[1]["stt_ms"] == 200.0

        # Summary averages
        summary = res["summary"]
        assert summary["turn_count"] == 2
        # average stt: (120 + 200) / 2 = 160
        assert summary["avg_stt_ms"] == 160.0
        assert summary["max_stt_ms"] == 200.0
        # average llm: (200 + 400) / 2 = 300
        assert summary["avg_llm_ttfb_ms"] == 300.0
        assert summary["max_llm_ttfb_ms"] == 400.0
        # average tts: (350 + 500) / 2 = 425
        assert summary["avg_tts_first_chunk_ms"] == 425.0
        assert summary["max_tts_first_chunk_ms"] == 500.0
