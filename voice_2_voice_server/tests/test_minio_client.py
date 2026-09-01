"""Tests for storage/minio_client.py."""

import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from minio.error import S3Error

from storage.minio_client import MinIOStorage, _get_env_or_raise


# ---------------------------------------------------------------------------
# _get_env_or_raise
# ---------------------------------------------------------------------------

def test_get_env_or_raise_present(monkeypatch):
    monkeypatch.setenv("SOME_KEY", "value123")
    assert _get_env_or_raise("SOME_KEY") == "value123"


def test_get_env_or_raise_missing(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    with pytest.raises(ValueError, match="SOME_KEY"):
        _get_env_or_raise("SOME_KEY")


# ---------------------------------------------------------------------------
# MinIOStorage.__init__ and from_env
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_minio_cls():
    with patch("storage.minio_client.Minio") as m:
        client = MagicMock()
        client.bucket_exists.return_value = True
        m.return_value = client
        yield m, client


def make_storage(mock_minio_cls):
    _, client = mock_minio_cls
    return MinIOStorage(endpoint="localhost:9000", access_key="key", secret_key="secret")


def test_init_creates_minio_client(mock_minio_cls):
    cls, client = mock_minio_cls
    storage = MinIOStorage("localhost:9000", "key", "secret", secure=False)
    cls.assert_called_once_with("localhost:9000", access_key="key", secret_key="secret", secure=False)


def test_init_creates_missing_buckets(mock_minio_cls):
    _, client = mock_minio_cls
    client.bucket_exists.return_value = False
    MinIOStorage("localhost:9000", "key", "secret")
    assert client.make_bucket.call_count == 2


def test_init_skips_existing_buckets(mock_minio_cls):
    _, client = mock_minio_cls
    client.bucket_exists.return_value = True
    MinIOStorage("localhost:9000", "key", "secret")
    client.make_bucket.assert_not_called()


def test_from_env(monkeypatch, mock_minio_cls):
    monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "mykey")
    monkeypatch.setenv("MINIO_SECRET_KEY", "mysecret")
    monkeypatch.setenv("MINIO_SECURE", "false")
    storage = MinIOStorage.from_env()
    assert storage is not None


def test_from_env_secure_true(monkeypatch, mock_minio_cls):
    monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "mykey")
    monkeypatch.setenv("MINIO_SECRET_KEY", "mysecret")
    monkeypatch.setenv("MINIO_SECURE", "true")
    cls, _ = mock_minio_cls
    MinIOStorage.from_env()
    cls.assert_called_with("minio:9000", access_key="mykey", secret_key="mysecret", secure=True)


def test_from_env_missing_key_raises(monkeypatch, mock_minio_cls):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
    with pytest.raises(ValueError):
        MinIOStorage.from_env()


# ---------------------------------------------------------------------------
# save_recording
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_recording(mock_minio_cls):
    _, client = mock_minio_cls
    storage = MinIOStorage("localhost:9000", "key", "secret")

    with patch("storage.minio_client.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = None
        result = await storage.save_recording("call_abc", b"\x00\x01" * 100, 8000, 1)

    assert result == "call_abc.wav"
    mock_thread.assert_called_once()


# ---------------------------------------------------------------------------
# save_recording_bytes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_recording_bytes_mp3(mock_minio_cls):
    _, client = mock_minio_cls
    storage = MinIOStorage("localhost:9000", "key", "secret")

    with patch("storage.minio_client.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = None
        result = await storage.save_recording_bytes("call_abc", b"mp3data", extension="mp3")

    assert result == "call_abc.mp3"


@pytest.mark.asyncio
async def test_save_recording_bytes_unknown_ext(mock_minio_cls):
    _, client = mock_minio_cls
    storage = MinIOStorage("localhost:9000", "key", "secret")

    with patch("storage.minio_client.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = None
        result = await storage.save_recording_bytes("call_abc", b"data", extension="ogg")

    assert result == "call_abc.ogg"


# ---------------------------------------------------------------------------
# append_transcript
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_transcript_no_existing(mock_minio_cls):
    _, client = mock_minio_cls
    storage = MinIOStorage("localhost:9000", "key", "secret")

    mock_http_response = MagicMock()
    s3_err = S3Error(mock_http_response, "NoSuchKey", "no key", None, None, None)

    call_count = 0

    async def fake_to_thread(fn, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise s3_err
        return None

    with patch("storage.minio_client.asyncio.to_thread", side_effect=fake_to_thread):
        result = await storage.append_transcript("call_abc", "Agent: Hello")

    assert result == "call_abc.txt"


@pytest.mark.asyncio
async def test_append_transcript_other_s3_error_raises(mock_minio_cls):
    _, client = mock_minio_cls
    storage = MinIOStorage("localhost:9000", "key", "secret")

    mock_http_response = MagicMock()
    s3_err = S3Error(mock_http_response, "AccessDenied", "denied", None, None, None)

    async def fake_to_thread(fn, *args, **kwargs):
        raise s3_err

    with patch("storage.minio_client.asyncio.to_thread", side_effect=fake_to_thread):
        with pytest.raises(S3Error):
            await storage.append_transcript("call_abc", "line")


@pytest.mark.asyncio
async def test_append_transcript_existing_content(mock_minio_cls):
    _, client = mock_minio_cls
    storage = MinIOStorage("localhost:9000", "key", "secret")

    existing_response = MagicMock()
    existing_response.read.return_value = b"previous line\n"
    existing_response.close = MagicMock()
    existing_response.release_conn = MagicMock()

    call_count = 0

    async def fake_to_thread(fn, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return existing_response
        return None

    with patch("storage.minio_client.asyncio.to_thread", side_effect=fake_to_thread):
        result = await storage.append_transcript("call_abc", "new line")

    assert result == "call_abc.txt"


# ---------------------------------------------------------------------------
# save_recording_from_chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_recording_from_chunks_empty(mock_minio_cls):
    _, client = mock_minio_cls
    storage = MinIOStorage("localhost:9000", "key", "secret")
    result = await storage.save_recording_from_chunks("call_abc", [], 8000, 1)
    assert result is None


@pytest.mark.asyncio
async def test_save_recording_from_chunks_data(mock_minio_cls):
    _, client = mock_minio_cls
    storage = MinIOStorage("localhost:9000", "key", "secret")

    with patch("storage.minio_client.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = None
        result = await storage.save_recording_from_chunks("call_abc", [b"\x00\x01", b"\x02\x03"], 8000, 1)

    assert result == "call_abc.wav"


# ---------------------------------------------------------------------------
# save_transcript_from_lines
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_transcript_from_lines_empty(mock_minio_cls):
    _, client = mock_minio_cls
    storage = MinIOStorage("localhost:9000", "key", "secret")
    result = await storage.save_transcript_from_lines("call_abc", [])
    assert result is None


@pytest.mark.asyncio
async def test_save_transcript_from_lines_data(mock_minio_cls):
    _, client = mock_minio_cls
    storage = MinIOStorage("localhost:9000", "key", "secret")

    with patch("storage.minio_client.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = None
        result = await storage.save_transcript_from_lines("call_abc", ["line1", "line2"])

    assert result == "call_abc.txt"


# ---------------------------------------------------------------------------
# get_object
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_object(mock_minio_cls):
    _, client = mock_minio_cls
    storage = MinIOStorage("localhost:9000", "key", "secret")

    mock_response = MagicMock()

    with patch("storage.minio_client.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = mock_response
        result = await storage.get_object("transcripts", "call_abc.txt")

    assert result is mock_response
