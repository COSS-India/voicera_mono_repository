#!/usr/bin/env python3
"""Tests for VI recording download and ingest helpers."""

import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

sys.modules.setdefault("minio", MagicMock())
_minio_client = MagicMock()
_minio_client.MinIOStorage = MagicMock
sys.modules.setdefault("storage.minio_client", _minio_client)

from utils.vi_recording import guess_extension, ingest_vi_recording_from_url


class TestGuessExtension(unittest.TestCase):
    def test_from_content_type(self):
        self.assertEqual(guess_extension("https://vi.example/rec", "audio/mpeg"), "mp3")
        self.assertEqual(guess_extension("https://vi.example/rec", "audio/wav"), "wav")

    def test_from_url_path(self):
        self.assertEqual(guess_extension("https://vi.example/file.mp3", None), "mp3")


class TestIngestViRecording(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_success(self):
        storage_instance = MagicMock()
        storage_instance.save_recording_bytes = AsyncMock(return_value="23580729.wav")
        storage_cls = MagicMock()
        storage_cls.from_env.return_value = storage_instance

        with patch("utils.vi_recording.MinIOStorage", storage_cls), patch(
            "utils.vi_recording.wait_and_download_vi_recording",
            new=AsyncMock(return_value=(b"audio-bytes", "wav")),
        ), patch(
            "utils.vi_recording.patch_call_recording_url",
            new=AsyncMock(return_value=True),
        ):
            ok = await ingest_vi_recording_from_url(
                "https://vi.example/rec?callid=23580729",
            )

        self.assertTrue(ok)
        storage_instance.save_recording_bytes.assert_awaited_once_with(
            "23580729", b"audio-bytes", "wav"
        )

    async def test_ingest_missing_call_id(self):
        ok = await ingest_vi_recording_from_url("https://vi.example/rec-without-id")
        self.assertFalse(ok)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
