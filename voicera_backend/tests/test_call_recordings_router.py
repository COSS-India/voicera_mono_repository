"""
Integration tests for /api/v1/call-recordings endpoint.

The endpoint is unauthenticated (service-to-service) so no auth fixtures needed.
"""
import pytest
from unittest.mock import patch

BASE = "/api/v1/call-recordings"

RECORDING_BODY = {
    "call_sid": "call-001",
    "transcript_url": "https://example.com/transcript.txt",
    "agent_type": "sales_bot",
    "recording_url": "minio://recordings/call-001.wav",
    "transcript_content": "user: Hello",
    "call_duration": 120.0,
    "end_time_utc": "2024-01-01T10:02:00",
    "org_id": "testorg1",
}

SAVED_DOC = {
    "meeting_id": "call-001",
    "agent_type": "sales_bot",
    "org_id": "testorg1",
    "transcript_url": "https://example.com/transcript.txt",
}


class TestSaveCallRecording:
    def test_success_returns_200(self, client):
        with patch("app.services.call_recording_service.save_call_recording",
                   return_value=SAVED_DOC):
            resp = client.post(BASE, json=RECORDING_BODY)
        assert resp.status_code == 200

    def test_service_fail_returns_400(self, client):
        fail = {"status": "fail", "message": "Failed to save"}
        with patch("app.services.call_recording_service.save_call_recording",
                   return_value=fail):
            resp = client.post(BASE, json=RECORDING_BODY)
        assert resp.status_code == 400

    def test_missing_call_sid_returns_422(self, client):
        body = {k: v for k, v in RECORDING_BODY.items() if k != "call_sid"}
        resp = client.post(BASE, json=body)
        assert resp.status_code == 422

    def test_missing_transcript_url_returns_422(self, client):
        body = {k: v for k, v in RECORDING_BODY.items() if k != "transcript_url"}
        resp = client.post(BASE, json=body)
        assert resp.status_code == 422

    def test_missing_agent_type_returns_422(self, client):
        body = {k: v for k, v in RECORDING_BODY.items() if k != "agent_type"}
        resp = client.post(BASE, json=body)
        assert resp.status_code == 422

    def test_optional_fields_can_be_omitted(self, client):
        minimal = {
            "call_sid": "call-002",
            "transcript_url": "https://example.com/t.txt",
            "agent_type": "sales_bot",
        }
        with patch("app.services.call_recording_service.save_call_recording",
                   return_value={"meeting_id": "call-002", "agent_type": "sales_bot"}):
            resp = client.post(BASE, json=minimal)
        assert resp.status_code == 200
