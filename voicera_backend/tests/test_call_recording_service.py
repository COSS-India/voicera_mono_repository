"""
Unit tests for app.services.call_recording_service.
"""
import pytest
from unittest.mock import MagicMock, patch

from app.services.call_recording_service import save_call_recording
from app.models.schemas import CallRecordingCreate

# ── Sample data ───────────────────────────────────────────────────────────────

RECORDING_CREATE = CallRecordingCreate(
    call_sid="call-sid-001",
    transcript_url="https://example.com/transcript.txt",
    agent_type="sales_bot",
    recording_url="minio://recordings/call-sid-001.wav",
    transcript_content="user: Hello\nagent: Hi there",
    call_duration=300.0,
    end_time_utc="2024-01-01T10:05:00",
    org_id="testorg1",
    latency_metrics={"tts_latency": 0.5},
)

MINIMAL_RECORDING = CallRecordingCreate(
    call_sid="call-sid-002",
    transcript_url="https://example.com/transcript.txt",
    agent_type="sales_bot",
)

SAVED_DOC = {
    "meeting_id": "call-sid-001",
    "agent_type": "sales_bot",
    "org_id": "testorg1",
    "recording_url": "minio://recordings/call-sid-001.wav",
    "transcript_url": "https://example.com/transcript.txt",
    "created_at": "2024-01-01T10:00:00",
}


def _make_db(updated_doc=None):
    coll = MagicMock()
    upsert_result = MagicMock()
    upsert_result.upserted_id = None
    coll.update_one.return_value = upsert_result
    coll.find_one.return_value = updated_doc
    db = MagicMock()
    db.__getitem__.side_effect = lambda k: coll if k == "CallLogs" else MagicMock()
    return db, coll


# ── TestSaveCallRecording ─────────────────────────────────────────────────

class TestSaveCallRecording:
    def test_success_upserts_and_returns_doc(self):
        db, coll = _make_db(updated_doc=SAVED_DOC)
        with patch("app.services.call_recording_service.get_database", return_value=db):
            result = save_call_recording(RECORDING_CREATE)
        coll.update_one.assert_called_once()
        # upsert filter uses call_sid as meeting_id
        update_filter = coll.update_one.call_args[0][0]
        assert update_filter.get("meeting_id") == "call-sid-001"

    def test_recording_url_stored_in_update_set(self):
        db, coll = _make_db(updated_doc=SAVED_DOC)
        with patch("app.services.call_recording_service.get_database", return_value=db):
            save_call_recording(RECORDING_CREATE)
        update_set = coll.update_one.call_args[0][1]["$set"]
        assert update_set.get("recording_url") == "minio://recordings/call-sid-001.wav"

    def test_transcript_content_stored(self):
        db, coll = _make_db(updated_doc=SAVED_DOC)
        with patch("app.services.call_recording_service.get_database", return_value=db):
            save_call_recording(RECORDING_CREATE)
        update_set = coll.update_one.call_args[0][1]["$set"]
        assert "transcript_content" in update_set

    def test_duration_stored_from_call_duration(self):
        db, coll = _make_db(updated_doc=SAVED_DOC)
        with patch("app.services.call_recording_service.get_database", return_value=db):
            save_call_recording(RECORDING_CREATE)
        update_set = coll.update_one.call_args[0][1]["$set"]
        assert update_set.get("duration") == 300.0

    def test_org_id_stored(self):
        db, coll = _make_db(updated_doc=SAVED_DOC)
        with patch("app.services.call_recording_service.get_database", return_value=db):
            save_call_recording(RECORDING_CREATE)
        update_set = coll.update_one.call_args[0][1]["$set"]
        assert update_set.get("org_id") == "testorg1"

    def test_minimal_recording_optional_fields_absent(self):
        db, coll = _make_db(updated_doc={"meeting_id": "call-sid-002", "agent_type": "sales_bot"})
        with patch("app.services.call_recording_service.get_database", return_value=db):
            result = save_call_recording(MINIMAL_RECORDING)
        update_set = coll.update_one.call_args[0][1]["$set"]
        # recording_url should not be set when not provided
        assert "recording_url" not in update_set

    def test_upsert_on_existing_call_sid(self):
        db, coll = _make_db(updated_doc=SAVED_DOC)
        with patch("app.services.call_recording_service.get_database", return_value=db):
            save_call_recording(RECORDING_CREATE)
        # upsert=True should be in kwargs
        call_kwargs = coll.update_one.call_args[1]
        assert call_kwargs.get("upsert") is True

    def test_document_not_found_after_upsert_returns_fail(self):
        db, coll = _make_db(updated_doc=None)
        with patch("app.services.call_recording_service.get_database", return_value=db):
            result = save_call_recording(RECORDING_CREATE)
        assert result.get("status") == "fail"

    def test_exception_returns_fail_dict(self):
        db, coll = _make_db()
        coll.update_one.side_effect = Exception("DB connection lost")
        with patch("app.services.call_recording_service.get_database", return_value=db):
            result = save_call_recording(RECORDING_CREATE)
        assert result.get("status") == "fail"
