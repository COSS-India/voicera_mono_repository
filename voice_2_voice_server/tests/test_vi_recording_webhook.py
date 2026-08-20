#!/usr/bin/env python3
"""Tests for VI recording webhook parsing and route."""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# server import chain pulls MinIO; stub for route tests.
sys.modules.setdefault("minio", MagicMock())
_minio_client = MagicMock()
_minio_client.MinIOStorage = MagicMock
sys.modules.setdefault("storage.minio_client", _minio_client)

from utils.vi_recording_webhook import (
    extract_recording_url,
    format_raw_body_for_log,
    parse_webhook_body,
)


class TestExtractRecordingUrl(unittest.TestCase):
    def test_recording_url_key(self):
        self.assertEqual(
            extract_recording_url({"recording_url": "https://vi.example/rec.wav"}),
            "https://vi.example/rec.wav",
        )

    def test_record_voice_alias(self):
        self.assertEqual(
            extract_recording_url({"RecordVoice": "https://vi.example/a.mp3"}),
            "https://vi.example/a.mp3",
        )

    def test_record_voice_snake_case(self):
        self.assertEqual(
            extract_recording_url({"record_voice": "https://vi.example/b.wav"}),
            "https://vi.example/b.wav",
        )

    def test_nested_dict(self):
        payload = {"data": {"recording_url": "https://vi.example/nested.wav"}}
        self.assertEqual(extract_recording_url(payload), "https://vi.example/nested.wav")

    def test_missing_key(self):
        self.assertIsNone(extract_recording_url({"other": "value"}))
        self.assertIsNone(extract_recording_url(None))

    def test_empty_string_ignored(self):
        self.assertIsNone(extract_recording_url({"recording_url": "   "}))


class TestParseWebhookBody(unittest.TestCase):
    def test_valid_json(self):
        raw = b'{"recording_url":"https://vi.example/rec.wav"}'
        parsed, mode = parse_webhook_body(raw, "application/json")
        self.assertEqual(mode, "json")
        self.assertEqual(parsed["recording_url"], "https://vi.example/rec.wav")

    def test_invalid_json_form_fallback(self):
        raw = b"recording_url=https%3A%2F%2Fvi.example%2Frec.wav"
        parsed, mode = parse_webhook_body(raw, "application/x-www-form-urlencoded")
        self.assertEqual(mode, "form")
        self.assertEqual(parsed["recording_url"], "https://vi.example/rec.wav")

    def test_garbage_returns_raw(self):
        raw = b"\xff\xfe not json or form"
        parsed, mode = parse_webhook_body(raw, "application/octet-stream")
        self.assertEqual(mode, "raw")
        self.assertIsNone(parsed)


class TestFormatRawBodyForLog(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(format_raw_body_for_log(b""), "(empty body)")

    def test_text_body(self):
        self.assertIn("hello", format_raw_body_for_log(b"hello"))


class TestRecordingWebhookRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            from api.server import app
        except ImportError as exc:
            raise unittest.SkipTest(f"FastAPI TestClient unavailable: {exc}") from exc
        cls.client = TestClient(app)

    def test_json_payload_returns_200(self):
        response = self.client.post(
            "/vi/recording-webhook",
            json={"recording_url": "https://vi.example/rec.wav"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "received"})

    def test_garbage_body_still_returns_200(self):
        response = self.client.post(
            "/vi/recording-webhook",
            content=b"not-json-at-all",
            headers={"Content-Type": "text/plain"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "received"})


if __name__ == "__main__":
    raise SystemExit(unittest.main())
