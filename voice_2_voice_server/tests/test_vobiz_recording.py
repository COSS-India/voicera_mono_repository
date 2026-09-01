"""Tests for utils/vobiz_recording.py."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from utils.vobiz_recording import (
    _get_vobiz_api_base,
    _get_vobiz_auth,
    _vobiz_headers,
    start_vobiz_call_recording,
    fetch_vobiz_recording_metadata,
    download_vobiz_recording,
    wait_and_download_vobiz_recording,
)


# ---------------------------------------------------------------------------
# _get_vobiz_api_base
# ---------------------------------------------------------------------------

def test_get_vobiz_api_base_present(monkeypatch):
    monkeypatch.setenv("VOBIZ_API_BASE", "https://api.vobiz.com/")
    assert _get_vobiz_api_base() == "https://api.vobiz.com"


def test_get_vobiz_api_base_missing(monkeypatch):
    monkeypatch.delenv("VOBIZ_API_BASE", raising=False)
    with pytest.raises(ValueError, match="VOBIZ_API_BASE"):
        _get_vobiz_api_base()


# ---------------------------------------------------------------------------
# _get_vobiz_auth
# ---------------------------------------------------------------------------

def test_get_vobiz_auth_success():
    with patch("utils.vobiz_recording.fetch_integration_key", side_effect=["auth_id_val", "auth_token_val"]):
        result = _get_vobiz_auth("org_123")
    assert result == ("auth_id_val", "auth_token_val")


def test_get_vobiz_auth_missing_id():
    with patch("utils.vobiz_recording.fetch_integration_key", side_effect=[None, "token"]):
        result = _get_vobiz_auth("org_123")
    assert result is None


def test_get_vobiz_auth_missing_token():
    with patch("utils.vobiz_recording.fetch_integration_key", side_effect=["auth_id", None]):
        result = _get_vobiz_auth("org_123")
    assert result is None


# ---------------------------------------------------------------------------
# _vobiz_headers
# ---------------------------------------------------------------------------

def test_vobiz_headers_success():
    with patch("utils.vobiz_recording._get_vobiz_auth", return_value=("aid", "atok")):
        headers = _vobiz_headers("org_123")
    assert headers["X-Auth-ID"] == "aid"
    assert headers["X-Auth-Token"] == "atok"
    assert headers["Content-Type"] == "application/json"


def test_vobiz_headers_no_auth():
    with patch("utils.vobiz_recording._get_vobiz_auth", return_value=None):
        assert _vobiz_headers("org_123") is None


# ---------------------------------------------------------------------------
# start_vobiz_call_recording
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_vobiz_call_recording_success(monkeypatch):
    monkeypatch.setenv("VOBIZ_API_BASE", "https://api.vobiz.com")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"recording_id": "rec_abc"}

    with patch("utils.vobiz_recording._vobiz_headers", return_value={"X-Auth-ID": "aid", "X-Auth-Token": "tok", "Content-Type": "application/json"}), \
         patch("utils.vobiz_recording.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
        result = await start_vobiz_call_recording("call_abc", "org_123", 3600)

    assert result == "rec_abc"


@pytest.mark.asyncio
async def test_start_vobiz_call_recording_no_headers():
    with patch("utils.vobiz_recording._vobiz_headers", return_value=None):
        result = await start_vobiz_call_recording("call_abc", "org_123", 3600)
    assert result is None


@pytest.mark.asyncio
async def test_start_vobiz_call_recording_exception(monkeypatch):
    monkeypatch.setenv("VOBIZ_API_BASE", "https://api.vobiz.com")
    with patch("utils.vobiz_recording._vobiz_headers", return_value={"X-Auth-ID": "aid", "X-Auth-Token": "tok", "Content-Type": "application/json"}), \
         patch("utils.vobiz_recording.asyncio.to_thread", new_callable=AsyncMock, side_effect=Exception("network error")):
        result = await start_vobiz_call_recording("call_abc", "org_123", 3600)
    assert result is None


@pytest.mark.asyncio
async def test_start_vobiz_recording_uuid_fallback(monkeypatch):
    monkeypatch.setenv("VOBIZ_API_BASE", "https://api.vobiz.com")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"uuid": "uuid_xyz"}  # fallback field

    with patch("utils.vobiz_recording._vobiz_headers", return_value={"X-Auth-ID": "aid", "X-Auth-Token": "tok", "Content-Type": "application/json"}), \
         patch("utils.vobiz_recording.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
        result = await start_vobiz_call_recording("call_abc", "org_123", 3600)

    assert result == "uuid_xyz"


# ---------------------------------------------------------------------------
# fetch_vobiz_recording_metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_vobiz_recording_metadata_success(monkeypatch):
    monkeypatch.setenv("VOBIZ_API_BASE", "https://api.vobiz.com")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"recording_url": "https://cdn.vobiz.com/rec.mp3"}

    with patch("utils.vobiz_recording._vobiz_headers", return_value={"X-Auth-ID": "aid", "X-Auth-Token": "tok", "Content-Type": "application/json"}), \
         patch("utils.vobiz_recording.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
        result = await fetch_vobiz_recording_metadata("rec_abc", "org_123")

    assert result["recording_url"] == "https://cdn.vobiz.com/rec.mp3"


@pytest.mark.asyncio
async def test_fetch_vobiz_recording_metadata_no_headers():
    with patch("utils.vobiz_recording._vobiz_headers", return_value=None):
        result = await fetch_vobiz_recording_metadata("rec_abc", "org_123")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_vobiz_recording_metadata_exception(monkeypatch):
    monkeypatch.setenv("VOBIZ_API_BASE", "https://api.vobiz.com")
    with patch("utils.vobiz_recording._vobiz_headers", return_value={"X-Auth-ID": "aid", "X-Auth-Token": "tok", "Content-Type": "application/json"}), \
         patch("utils.vobiz_recording.asyncio.to_thread", new_callable=AsyncMock, side_effect=Exception("timeout")):
        result = await fetch_vobiz_recording_metadata("rec_abc", "org_123")
    assert result is None


# ---------------------------------------------------------------------------
# download_vobiz_recording
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_download_vobiz_recording_success():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.content = b"audio_bytes"

    with patch("utils.vobiz_recording._vobiz_headers", return_value={"X-Auth-ID": "aid", "X-Auth-Token": "tok", "Content-Type": "application/json"}), \
         patch("utils.vobiz_recording.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
        result = await download_vobiz_recording("https://cdn.vobiz.com/rec.mp3", "org_123")

    assert result == b"audio_bytes"


@pytest.mark.asyncio
async def test_download_vobiz_recording_no_headers():
    with patch("utils.vobiz_recording._vobiz_headers", return_value=None):
        result = await download_vobiz_recording("https://cdn.vobiz.com/rec.mp3", "org_123")
    assert result is None


@pytest.mark.asyncio
async def test_download_vobiz_recording_exception():
    with patch("utils.vobiz_recording._vobiz_headers", return_value={"X-Auth-ID": "aid", "X-Auth-Token": "tok", "Content-Type": "application/json"}), \
         patch("utils.vobiz_recording.asyncio.to_thread", new_callable=AsyncMock, side_effect=Exception("connection error")):
        result = await download_vobiz_recording("https://cdn.vobiz.com/rec.mp3", "org_123")
    assert result is None


# ---------------------------------------------------------------------------
# wait_and_download_vobiz_recording
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wait_and_download_ready_first_attempt():
    with patch("utils.vobiz_recording.fetch_vobiz_recording_metadata", new_callable=AsyncMock,
               return_value={"recording_url": "https://cdn.vobiz.com/rec.mp3"}), \
         patch("utils.vobiz_recording.download_vobiz_recording", new_callable=AsyncMock,
               return_value=b"audio_bytes"):
        result = await wait_and_download_vobiz_recording("rec_abc", "org_123", max_attempts=3, interval_secs=0)

    assert result == b"audio_bytes"


@pytest.mark.asyncio
async def test_wait_and_download_retries_then_succeeds():
    metadata_responses = [None, None, {"recording_url": "https://cdn.vobiz.com/rec.mp3"}]

    with patch("utils.vobiz_recording.fetch_vobiz_recording_metadata", new_callable=AsyncMock,
               side_effect=metadata_responses), \
         patch("utils.vobiz_recording.download_vobiz_recording", new_callable=AsyncMock,
               return_value=b"audio_bytes"), \
         patch("utils.vobiz_recording.asyncio.sleep", new_callable=AsyncMock):
        result = await wait_and_download_vobiz_recording("rec_abc", "org_123", max_attempts=3, interval_secs=0)

    assert result == b"audio_bytes"


@pytest.mark.asyncio
async def test_wait_and_download_max_attempts_exhausted():
    with patch("utils.vobiz_recording.fetch_vobiz_recording_metadata", new_callable=AsyncMock,
               return_value=None), \
         patch("utils.vobiz_recording.asyncio.sleep", new_callable=AsyncMock):
        result = await wait_and_download_vobiz_recording("rec_abc", "org_123", max_attempts=3, interval_secs=0)

    assert result is None


@pytest.mark.asyncio
async def test_wait_and_download_metadata_ready_but_download_fails():
    with patch("utils.vobiz_recording.fetch_vobiz_recording_metadata", new_callable=AsyncMock,
               return_value={"recording_url": "https://cdn.vobiz.com/rec.mp3"}), \
         patch("utils.vobiz_recording.download_vobiz_recording", new_callable=AsyncMock,
               return_value=None), \
         patch("utils.vobiz_recording.asyncio.sleep", new_callable=AsyncMock):
        result = await wait_and_download_vobiz_recording("rec_abc", "org_123", max_attempts=2, interval_secs=0)

    assert result is None


@pytest.mark.asyncio
async def test_wait_and_download_no_url_in_metadata():
    with patch("utils.vobiz_recording.fetch_vobiz_recording_metadata", new_callable=AsyncMock,
               return_value={"some_other_key": "value"}), \
         patch("utils.vobiz_recording.asyncio.sleep", new_callable=AsyncMock):
        result = await wait_and_download_vobiz_recording("rec_abc", "org_123", max_attempts=2, interval_secs=0)

    assert result is None
