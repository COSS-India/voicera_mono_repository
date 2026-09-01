"""Shared pytest fixtures and helpers for voice_2_voice_server tests."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Shared bot config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_bot_config():
    """Typical bot configuration dict used across test_bot.py, test_services.py,
    and test_bot_utils.py."""
    return {
        "llm_model": {"name": "openai", "args": {"model": "gpt-4o"}},
        "stt_model": {"name": "deepgram", "language": "English"},
        "tts_model": {"name": "elevenlabs", "voice_id": "test_voice"},
        "greeting_message": "Hello! How can I help you?",
        "system_prompt": "You are a helpful assistant.",
        "language": "English",
        "org_id": "test_org_123",
        "interaction_mode": "conversational",
        "call_timeout_seconds": 300,
        "user_silence_hangup_seconds": 10,
        "user_online_detection_enabled": True,
        "user_online_detection_message": "Are you there?",
        "user_online_detection_seconds": 5.0,
        "hold_messages": ["Please hold."],
        "hold_message_timeout_seconds": 0.3,
        "interruption_min_words": 1,
        "ignore_user_speech_before_greeting": True,
        "knowledge_base_enabled": False,
        "knowledge_document_ids": [],
        "knowledge_top_k": 10,
    }


# ---------------------------------------------------------------------------
# Async task runner helper
# ---------------------------------------------------------------------------

async def run_task_with_blocking_sleep(processor, trigger_frame, cancel_frame, direction):
    """Helper that:
    1. Sends *trigger_frame* to *processor* to start an async idle timer task.
    2. Captures and asserts the started task is not None.
    3. Sends *cancel_frame* to cancel the task.
    4. Awaits the task (swallowing CancelledError).
    5. Returns the task so the caller can assert task.done().

    This avoids the repeated boilerplate in test_user_online_detection_filter.py
    and test_user_silence_hangup.py.
    """
    await processor.process_frame(trigger_frame, direction)
    task = processor._idle_task
    assert task is not None, "Expected an idle task to be scheduled after trigger_frame"
    await processor.process_frame(cancel_frame, direction)
    try:
        await task
    except asyncio.CancelledError:
        pass
    return task
