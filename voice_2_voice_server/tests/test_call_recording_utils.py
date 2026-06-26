"""Tests for utils/call_recording_utils.py."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from utils.call_recording_utils import submit_call_recording


def _make_storage(transcript_bytes=None, raise_on_get=False):
    storage = MagicMock()
    if raise_on_get:
        storage.get_object = AsyncMock(side_effect=Exception("storage error"))
    else:
        mock_resp = MagicMock()
        mock_resp.read.return_value = transcript_bytes or b"Agent: Hello\nUser: Hi\n"
        mock_resp.close = MagicMock()
        mock_resp.release_conn = MagicMock()
        storage.get_object = AsyncMock(return_value=mock_resp)
    return storage


def _base_config():
    return {"org_id": "org_123", "agent_type": "inbound"}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_call_recording_success():
    storage = _make_storage()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch("utils.call_recording_utils.requests.post", return_value=mock_response) as mock_post, \
         patch("utils.call_recording_utils.time.monotonic", return_value=100.0):
        await submit_call_recording(
            call_sid="call_abc",
            agent_type="inbound",
            agent_config=_base_config(),
            storage=storage,
            call_start_time=90.0,
        )

    mock_post.assert_called_once()
    payload = mock_post.call_args.kwargs["json"]
    assert payload["call_sid"] == "call_abc"
    assert payload["call_duration"] == pytest.approx(10.0)
    assert "recording_url" in payload


@pytest.mark.asyncio
async def test_submit_call_recording_omit_recording_url():
    storage = _make_storage()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch("utils.call_recording_utils.requests.post", return_value=mock_response) as mock_post, \
         patch("utils.call_recording_utils.time.monotonic", return_value=100.0):
        await submit_call_recording(
            call_sid="call_abc",
            agent_type="inbound",
            agent_config=_base_config(),
            storage=storage,
            call_start_time=90.0,
            omit_recording_url=True,
        )

    payload = mock_post.call_args.kwargs["json"]
    assert "recording_url" not in payload


@pytest.mark.asyncio
async def test_submit_call_recording_explicit_recording_url():
    storage = _make_storage()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch("utils.call_recording_utils.requests.post", return_value=mock_response) as mock_post, \
         patch("utils.call_recording_utils.time.monotonic", return_value=100.0):
        await submit_call_recording(
            call_sid="call_abc",
            agent_type="inbound",
            agent_config=_base_config(),
            storage=storage,
            call_start_time=90.0,
            recording_url="https://cdn.example.com/recording.wav",
        )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["recording_url"] == "https://cdn.example.com/recording.wav"


@pytest.mark.asyncio
async def test_submit_call_recording_with_latency_metrics():
    storage = _make_storage()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    metrics = {"turns": [{"ttfb": 0.3}]}

    with patch("utils.call_recording_utils.requests.post", return_value=mock_response) as mock_post, \
         patch("utils.call_recording_utils.time.monotonic", return_value=100.0):
        await submit_call_recording(
            call_sid="call_abc",
            agent_type="inbound",
            agent_config=_base_config(),
            storage=storage,
            call_start_time=90.0,
            latency_metrics=metrics,
        )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["latency_metrics"] == metrics


@pytest.mark.asyncio
async def test_submit_call_recording_no_turns_latency_skipped():
    storage = _make_storage()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch("utils.call_recording_utils.requests.post", return_value=mock_response) as mock_post, \
         patch("utils.call_recording_utils.time.monotonic", return_value=100.0):
        await submit_call_recording(
            call_sid="call_abc",
            agent_type="inbound",
            agent_config=_base_config(),
            storage=storage,
            call_start_time=90.0,
            latency_metrics={"turns": []},  # empty turns → not included
        )

    payload = mock_post.call_args.kwargs["json"]
    assert "latency_metrics" not in payload


# ---------------------------------------------------------------------------
# Transcript read failure — should not raise
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_call_recording_transcript_read_fails():
    storage = _make_storage(raise_on_get=True)
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch("utils.call_recording_utils.requests.post", return_value=mock_response), \
         patch("utils.call_recording_utils.time.monotonic", return_value=100.0):
        # Should not raise — warning logged
        await submit_call_recording(
            call_sid="call_abc",
            agent_type="inbound",
            agent_config=_base_config(),
            storage=storage,
            call_start_time=90.0,
        )


# ---------------------------------------------------------------------------
# HTTP errors — should not raise
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_call_recording_request_exception():
    import requests as req
    storage = _make_storage()

    with patch("utils.call_recording_utils.requests.post", side_effect=req.exceptions.RequestException("timeout")), \
         patch("utils.call_recording_utils.time.monotonic", return_value=100.0):
        # Should not raise
        await submit_call_recording(
            call_sid="call_abc",
            agent_type="inbound",
            agent_config=_base_config(),
            storage=storage,
            call_start_time=90.0,
        )


@pytest.mark.asyncio
async def test_submit_call_recording_generic_exception():
    storage = _make_storage()

    with patch("utils.call_recording_utils.requests.post", side_effect=Exception("boom")), \
         patch("utils.call_recording_utils.time.monotonic", return_value=100.0):
        # Should not raise
        await submit_call_recording(
            call_sid="call_abc",
            agent_type="inbound",
            agent_config=_base_config(),
            storage=storage,
            call_start_time=90.0,
        )


# ---------------------------------------------------------------------------
# Backend URL from env
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_uses_backend_url_env(monkeypatch):
    monkeypatch.setenv("VOICERA_BACKEND_URL", "http://custom-backend:9000")
    storage = _make_storage()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch("utils.call_recording_utils.requests.post", return_value=mock_response) as mock_post, \
         patch("utils.call_recording_utils.time.monotonic", return_value=100.0):
        await submit_call_recording(
            call_sid="call_abc",
            agent_type="inbound",
            agent_config=_base_config(),
            storage=storage,
            call_start_time=90.0,
        )

    url = mock_post.call_args.args[0]
    assert url.startswith("http://custom-backend:9000")
