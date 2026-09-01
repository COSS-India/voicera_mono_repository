"""Tests for app/storage/minio_client.py"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from minio.error import S3Error

from app.storage.minio_client import MinIOStorage


@pytest.fixture
def storage():
    with patch("app.storage.minio_client.Minio"):
        return MinIOStorage()


class TestGetObject:
    @pytest.mark.anyio
    async def test_returns_object(self, storage):
        mock_obj = MagicMock()
        with patch("app.storage.minio_client.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_obj):
            result = await storage.get_object("my-bucket", "my/object.wav")
        assert result is mock_obj


class TestObjectExists:
    def test_object_exists_returns_true(self, storage):
        storage.client.stat_object.return_value = MagicMock()
        assert storage.object_exists("bucket", "obj") is True

    def test_no_such_key_returns_false(self, storage):
        err = S3Error(
            code="NoSuchKey", message="not found",
            resource="obj", request_id="req1", host_id="h1",
            response=MagicMock(status=404, headers={}, read=lambda: b""),
        )
        storage.client.stat_object.side_effect = err
        assert storage.object_exists("bucket", "obj") is False

    def test_other_s3error_raises(self, storage):
        err = S3Error(
            code="AccessDenied", message="denied",
            resource="obj", request_id="req1", host_id="h1",
            response=MagicMock(status=403, headers={}, read=lambda: b""),
        )
        storage.client.stat_object.side_effect = err
        with pytest.raises(S3Error):
            storage.object_exists("bucket", "obj")


class TestParseMinioUrl:
    def test_valid_url(self, storage):
        result = storage.parse_minio_url("minio://my-bucket/path/to/file.wav")
        assert result == ("my-bucket", "path/to/file.wav")

    def test_invalid_prefix_returns_none(self, storage):
        assert storage.parse_minio_url("s3://bucket/obj") is None
        assert storage.parse_minio_url("") is None
        assert storage.parse_minio_url(None) is None

    def test_no_slash_after_bucket_returns_none(self, storage):
        assert storage.parse_minio_url("minio://bucketonly") is None

