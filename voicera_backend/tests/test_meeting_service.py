"""
Unit tests for app.services.meeting_service.

Strategy
--------
- get_database() is patched at "app.services.meeting_service.get_database".
- fetch_agent_config is patched at "app.services.meeting_service.fetch_agent_config".
- Pure functions (_serialize_doc, parse_transcript, transform_recording_url, etc.)
  are imported directly and tested without mocking.
"""
import pytest
from unittest.mock import MagicMock, patch
from bson import ObjectId

from app.services.meeting_service import (
    _serialize_doc,
    _build_meetings_query,
    setup_meeting_id,
    fetch_meeting_details,
    fetch_meetings_paginated,
    fetch_meeting_filter_options,
    update_meeting_end_time,
    parse_transcript,
    transform_recording_url,
    transform_meeting_for_frontend,
)
from app.models.schemas import MeetingCreate

# ── Sample data ───────────────────────────────────────────────────────────────

ORG_ID = "testorg1"
MEETING_ID = "call-abc-123"

MEETING_DOC = {
    "_id": ObjectId("507f1f77bcf86cd799439011"),
    "meeting_id": MEETING_ID,
    "agent_type": "sales_bot",
    "org_id": ORG_ID,
    "start_time_utc": "2024-01-01T10:00:00",
    "end_time_utc": "2024-01-01T10:05:00",
    "from_number": "+1234567890",
    "to_number": "+1987654321",
    "inbound": False,
    "call_busy": False,
    "recording_url": None,
    "transcript_content": None,
    "created_at": "2024-01-01T10:00:00",
}

AGENT_CONFIG = {
    "agent_id": "agent-001",
    "agent_type": "sales_bot",
    "org_id": ORG_ID,
    "agent_category": "sales",
    "agent_config": {"prompt": "You are a sales assistant"},
}


# ── TestSerializeDoc ──────────────────────────────────────────────────────

class TestSerializeDoc:
    def test_converts_objectid_to_string_under_id_key(self):
        oid = ObjectId()
        result = _serialize_doc({"_id": oid, "name": "test"})
        assert "id" in result
        assert result["id"] == str(oid)
        assert "_id" not in result

    def test_preserves_all_other_fields(self):
        oid = ObjectId()
        result = _serialize_doc({"_id": oid, "meeting_id": "m-001", "org_id": "org1"})
        assert result["meeting_id"] == "m-001"
        assert result["org_id"] == "org1"

    def test_none_input_returns_none(self):
        assert _serialize_doc(None) is None

    def test_doc_without_id_passes_through(self):
        result = _serialize_doc({"meeting_id": "m-001"})
        assert result == {"meeting_id": "m-001"}


# ── TestBuildMeetingsQuery ────────────────────────────────────────────────

class TestBuildMeetingsQuery:
    def test_org_only_returns_flat_condition(self):
        query = _build_meetings_query(org_id=ORG_ID)
        assert query == {"org_id": ORG_ID}

    def test_agent_type_filter_uses_and(self):
        query = _build_meetings_query(org_id=ORG_ID, agent_type="sales_bot")
        assert "$and" in query
        assert {"agent_type": "sales_bot"} in query["$and"]

    def test_from_number_filter_added(self):
        query = _build_meetings_query(org_id=ORG_ID, from_number="+123")
        assert {"from_number": "+123"} in query["$and"]

    def test_to_number_filter_added(self):
        query = _build_meetings_query(org_id=ORG_ID, to_number="+456")
        assert {"to_number": "+456"} in query["$and"]

    def test_inbound_true_filter_added(self):
        query = _build_meetings_query(org_id=ORG_ID, inbound=True)
        assert {"inbound": True} in query["$and"]

    def test_inbound_false_filter_added(self):
        query = _build_meetings_query(org_id=ORG_ID, inbound=False)
        assert {"inbound": False} in query["$and"]

    def test_busy_status_maps_to_call_busy_true(self):
        query = _build_meetings_query(org_id=ORG_ID, call_status="busy")
        assert {"call_busy": True} in query["$and"]

    def test_completed_status_contains_end_time_condition(self):
        query = _build_meetings_query(org_id=ORG_ID, call_status="completed")
        conditions = query["$and"]
        compound = [c for c in conditions if "$and" in c]
        assert len(compound) == 1
        inner = compound[0]["$and"]
        assert any("end_time_utc" in str(c) for c in inner)

    def test_in_progress_status_excludes_end_time(self):
        query = _build_meetings_query(org_id=ORG_ID, call_status="in progress")
        conditions = query["$and"]
        compound = [c for c in conditions if "$and" in c]
        assert len(compound) == 1
        inner = compound[0]["$and"]
        # Should have OR conditions for missing end_time
        assert any("$or" in str(c) for c in inner)

    def test_date_from_adds_expr_condition(self):
        query = _build_meetings_query(org_id=ORG_ID, date_from="2024-01-01")
        conditions = query["$and"]
        assert any("$expr" in c for c in conditions)

    def test_date_to_adds_expr_condition(self):
        query = _build_meetings_query(org_id=ORG_ID, date_to="2024-01-31")
        conditions = query["$and"]
        assert any("$expr" in c for c in conditions)

    def test_multiple_filters_all_included(self):
        query = _build_meetings_query(
            org_id=ORG_ID,
            agent_type="sales_bot",
            inbound=True,
            call_status="busy",
        )
        conditions = query["$and"]
        assert len(conditions) == 4  # org + agent_type + inbound + busy


# ── TestTranscriptParsing ─────────────────────────────────────────────────

class TestTranscriptParsing:
    def test_empty_string_returns_empty_list(self):
        assert parse_transcript("") == []

    def test_none_returns_empty_list(self):
        assert parse_transcript(None) == []

    def test_bracketed_timestamp_user_role(self):
        messages = parse_transcript("[10:00] user: Hello there")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello there"
        assert messages[0]["timestamp"] == "10:00"

    def test_bracketed_timestamp_agent_role(self):
        messages = parse_transcript("[10:01] agent: Hi, how can I help?")
        assert messages[0]["role"] == "agent"
        assert messages[0]["content"] == "Hi, how can I help?"
        assert messages[0]["timestamp"] == "10:01"

    def test_human_role_normalized_to_user(self):
        messages = parse_transcript("[10:00] human: Hello")
        assert messages[0]["role"] == "user"

    def test_assistant_role_normalized_to_agent(self):
        messages = parse_transcript("[10:00] assistant: Hello")
        assert messages[0]["role"] == "agent"

    def test_bot_role_normalized_to_agent(self):
        messages = parse_transcript("[10:00] bot: Hello")
        assert messages[0]["role"] == "agent"

    def test_prefix_without_timestamp_user(self):
        messages = parse_transcript("user: Hello there")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello there"
        assert "timestamp" not in messages[0]

    def test_prefix_without_timestamp_agent(self):
        messages = parse_transcript("agent: How can I help?")
        assert messages[0]["role"] == "agent"
        assert messages[0]["content"] == "How can I help?"

    def test_human_prefix_normalized_to_user(self):
        messages = parse_transcript("human: Hi")
        assert messages[0]["role"] == "user"

    def test_bot_prefix_normalized_to_agent(self):
        messages = parse_transcript("bot: Hello")
        assert messages[0]["role"] == "agent"

    def test_plain_first_line_defaults_to_agent(self):
        messages = parse_transcript("Some plain text")
        assert messages[0]["role"] == "agent"
        assert messages[0]["content"] == "Some plain text"

    def test_plain_lines_alternate_roles(self):
        messages = parse_transcript("First\nSecond")
        assert messages[0]["role"] == "agent"
        assert messages[1]["role"] == "user"

    def test_three_plain_lines_cycle(self):
        messages = parse_transcript("A\nB\nC")
        assert messages[0]["role"] == "agent"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "agent"

    def test_multiline_bracketed_conversation(self):
        transcript = "[10:00] user: Hi\n[10:01] agent: Hello\n[10:02] user: Bye"
        messages = parse_transcript(transcript)
        assert len(messages) == 3
        assert [m["role"] for m in messages] == ["user", "agent", "user"]

    def test_blank_lines_ignored(self):
        messages = parse_transcript("[10:00] user: Hi\n\n[10:01] agent: Hello")
        assert len(messages) == 2


# ── TestTransformRecordingUrl ─────────────────────────────────────────────

class TestTransformRecordingUrl:
    def test_minio_url_returns_proxy_path(self):
        result = transform_recording_url(f"minio://recordings/file.wav", MEETING_ID)
        assert result == f"/api/meetings/{MEETING_ID}/recording"

    def test_http_url_passes_through(self):
        url = "http://example.com/audio.wav"
        assert transform_recording_url(url, MEETING_ID) == url

    def test_https_url_passes_through(self):
        url = "https://cdn.example.com/recording.mp3"
        assert transform_recording_url(url, MEETING_ID) == url

    def test_empty_string_returns_none(self):
        assert transform_recording_url("", MEETING_ID) is None

    def test_none_returns_none(self):
        assert transform_recording_url(None, MEETING_ID) is None

    def test_unknown_scheme_returned_as_is(self):
        url = "ftp://storage/audio.wav"
        assert transform_recording_url(url, MEETING_ID) == url

    def test_meeting_id_included_in_proxy_path(self):
        result = transform_recording_url("minio://bucket/obj", "unique-meeting-99")
        assert "unique-meeting-99" in result


# ── TestTransformMeetingForFrontend ───────────────────────────────────────

class TestTransformMeetingForFrontend:
    def _plain_doc(self, **kwargs):
        doc = {k: v for k, v in MEETING_DOC.items() if k != "_id"}
        doc.update(kwargs)
        return doc

    def test_minio_recording_url_converted_to_proxy(self):
        doc = self._plain_doc(recording_url="minio://recordings/file.wav")
        result = transform_meeting_for_frontend(doc)
        assert result["recording_url"] == f"/api/meetings/{MEETING_ID}/recording"

    def test_http_recording_url_passes_through(self):
        url = "http://cdn.example.com/audio.wav"
        doc = self._plain_doc(recording_url=url)
        result = transform_meeting_for_frontend(doc)
        assert result["recording_url"] == url

    def test_transcript_content_parsed_to_list(self):
        doc = self._plain_doc(
            transcript_content="[10:00] user: Hello\n[10:01] agent: Hi"
        )
        result = transform_meeting_for_frontend(doc)
        assert isinstance(result["transcript"], list)
        assert len(result["transcript"]) == 2

    def test_missing_transcript_content_gives_empty_list(self):
        doc = self._plain_doc(transcript_content=None)
        result = transform_meeting_for_frontend(doc)
        assert result["transcript"] == []

    def test_no_recording_url_field_unchanged(self):
        doc = self._plain_doc(recording_url=None)
        result = transform_meeting_for_frontend(doc)
        assert result.get("recording_url") is None

    def test_none_input_returned_as_is(self):
        assert transform_meeting_for_frontend(None) is None

    def test_original_doc_not_mutated(self):
        original_url = "minio://bucket/file.wav"
        doc = self._plain_doc(recording_url=original_url)
        transform_meeting_for_frontend(doc)
        assert doc["recording_url"] == original_url


# ── TestSetupMeetingId ────────────────────────────────────────────────────

class TestSetupMeetingId:
    def _create(self, **kwargs):
        defaults = {
            "meeting_id": MEETING_ID,
            "agent_type": "sales_bot",
            "start_time_utc": "2024-01-01T10:00:00",
            "org_id": ORG_ID,
        }
        defaults.update(kwargs)
        return MeetingCreate(**defaults)

    def test_success_saves_and_returns_doc(self):
        mock_db = MagicMock()
        with patch("app.services.meeting_service.get_database", return_value=mock_db), \
             patch("app.services.meeting_service.fetch_agent_config", return_value=AGENT_CONFIG):
            result = setup_meeting_id(self._create())
        assert result["meeting_id"] == MEETING_ID
        mock_db["CallLogs"].update_one.assert_called_once()

    def test_org_id_from_request_takes_priority_over_agent_config(self):
        mock_db = MagicMock()
        agent_with_diff_org = {**AGENT_CONFIG, "org_id": "other-org"}
        with patch("app.services.meeting_service.get_database", return_value=mock_db), \
             patch("app.services.meeting_service.fetch_agent_config",
                   return_value=agent_with_diff_org):
            result = setup_meeting_id(self._create(org_id=ORG_ID))
        assert result["org_id"] == ORG_ID

    def test_org_id_falls_back_to_agent_config_when_not_in_request(self):
        mock_db = MagicMock()
        with patch("app.services.meeting_service.get_database", return_value=mock_db), \
             patch("app.services.meeting_service.fetch_agent_config", return_value=AGENT_CONFIG):
            result = setup_meeting_id(self._create(org_id=None))
        assert result.get("org_id") == ORG_ID

    def test_agent_fetch_error_still_saves_meeting(self):
        mock_db = MagicMock()
        with patch("app.services.meeting_service.get_database", return_value=mock_db), \
             patch("app.services.meeting_service.fetch_agent_config",
                   side_effect=Exception("DB error")):
            result = setup_meeting_id(self._create())
        assert result["meeting_id"] == MEETING_ID
        mock_db["CallLogs"].update_one.assert_called_once()

    def test_call_busy_stored_when_provided(self):
        mock_db = MagicMock()
        with patch("app.services.meeting_service.get_database", return_value=mock_db), \
             patch("app.services.meeting_service.fetch_agent_config", return_value=AGENT_CONFIG):
            result = setup_meeting_id(self._create(call_busy=True))
        assert result.get("call_busy") is True

    def test_agent_category_and_config_populated_from_agent(self):
        mock_db = MagicMock()
        with patch("app.services.meeting_service.get_database", return_value=mock_db), \
             patch("app.services.meeting_service.fetch_agent_config", return_value=AGENT_CONFIG):
            result = setup_meeting_id(self._create())
        assert result.get("agent_category") == "sales"
        assert result.get("agent_config") == {"prompt": "You are a sales assistant"}


# ── TestFetchMeetingDetails ───────────────────────────────────────────────

class TestFetchMeetingDetails:
    def test_returns_serialized_doc_with_id_key(self):
        mock_db = MagicMock()
        mock_db["CallLogs"].find_one.return_value = MEETING_DOC
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            result = fetch_meeting_details(MEETING_ID)
        assert result is not None
        assert result["meeting_id"] == MEETING_ID
        assert "id" in result
        assert "_id" not in result

    def test_returns_none_when_not_found(self):
        mock_db = MagicMock()
        mock_db["CallLogs"].find_one.return_value = None
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            result = fetch_meeting_details("ghost-id")
        assert result is None

    def test_returns_none_on_db_exception(self):
        mock_db = MagicMock()
        mock_db["CallLogs"].find_one.side_effect = Exception("DB error")
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            result = fetch_meeting_details(MEETING_ID)
        assert result is None


# ── TestFetchMeetingsPaginated ────────────────────────────────────────────

class TestFetchMeetingsPaginated:
    def _make_db(self, total=10, docs=None):
        mock_db = MagicMock()
        docs = docs or []
        mock_db["CallLogs"].count_documents.return_value = total
        (mock_db["CallLogs"].find.return_value
         .sort.return_value.skip.return_value.limit.return_value) = docs
        return mock_db

    def test_returns_paginated_structure(self):
        mock_db = self._make_db(total=25)
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            result = fetch_meetings_paginated(org_id=ORG_ID)
        assert "items" in result
        assert "total" in result
        assert "page" in result
        assert "limit" in result
        assert result["total"] == 25

    def test_skip_calculated_as_page_minus_one_times_limit(self):
        mock_db = self._make_db()
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            fetch_meetings_paginated(org_id=ORG_ID, page=3, limit=10)
        skip_val = (mock_db["CallLogs"].find.return_value
                    .sort.return_value.skip.call_args[0][0])
        assert skip_val == 20

    def test_first_page_skip_is_zero(self):
        mock_db = self._make_db()
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            fetch_meetings_paginated(org_id=ORG_ID, page=1, limit=50)
        skip_val = (mock_db["CallLogs"].find.return_value
                    .sort.return_value.skip.call_args[0][0])
        assert skip_val == 0

    def test_db_exception_returns_empty_result(self):
        mock_db = MagicMock()
        mock_db["CallLogs"].count_documents.side_effect = Exception("DB error")
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            result = fetch_meetings_paginated(org_id=ORG_ID)
        assert result["items"] == []
        assert result["total"] == 0

    def test_items_are_transformed_for_frontend(self):
        meeting = {k: v for k, v in MEETING_DOC.items()}
        meeting["recording_url"] = "minio://bucket/f.wav"
        mock_db = self._make_db(total=1, docs=[meeting])
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            result = fetch_meetings_paginated(org_id=ORG_ID)
        assert result["items"][0]["recording_url"] == f"/api/meetings/{MEETING_ID}/recording"


# ── TestFetchMeetingFilterOptions ─────────────────────────────────────────

class TestFetchMeetingFilterOptions:
    def test_returns_sorted_distinct_values(self):
        mock_db = MagicMock()
        mock_db["CallLogs"].distinct.side_effect = [
            ["sales_bot", "support_bot"],
            ["+1234567890"],
            ["+1111111111", "+2222222222"],
        ]
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            result = fetch_meeting_filter_options(org_id=ORG_ID)
        assert result["agent_types"] == ["sales_bot", "support_bot"]
        assert result["from_numbers"] == ["+1234567890"]
        assert len(result["to_numbers"]) == 2

    def test_none_and_blank_values_excluded(self):
        mock_db = MagicMock()
        mock_db["CallLogs"].distinct.side_effect = [
            ["sales_bot", None, ""],
            [],
            [],
        ]
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            result = fetch_meeting_filter_options(org_id=ORG_ID)
        assert None not in result["agent_types"]
        assert "" not in result["agent_types"]
        assert "sales_bot" in result["agent_types"]

    def test_db_exception_returns_empty_dict(self):
        mock_db = MagicMock()
        mock_db["CallLogs"].distinct.side_effect = Exception("DB error")
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            result = fetch_meeting_filter_options(org_id=ORG_ID)
        assert result == {"agent_types": [], "from_numbers": [], "to_numbers": []}


# ── TestUpdateMeetingEndTime ──────────────────────────────────────────────

class TestUpdateMeetingEndTime:
    def test_success_returns_serialized_updated_doc(self):
        updated = {**MEETING_DOC, "end_time_utc": "2024-01-01T10:10:00"}
        mock_db = MagicMock()
        mock_db["CallLogs"].find_one.side_effect = [MEETING_DOC, updated]
        mock_db["CallLogs"].update_one.return_value.modified_count = 1
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            result = update_meeting_end_time(MEETING_ID, "2024-01-01T10:10:00")
        assert result is not None
        assert "id" in result  # serialized
        assert result.get("end_time_utc") == "2024-01-01T10:10:00"

    def test_not_found_returns_fail_dict(self):
        mock_db = MagicMock()
        mock_db["CallLogs"].find_one.return_value = None
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            result = update_meeting_end_time("ghost-id", "2024-01-01T10:10:00")
        assert result["status"] == "fail"
        assert "ghost-id" in result["message"]

    def test_db_exception_returns_fail_dict(self):
        mock_db = MagicMock()
        mock_db["CallLogs"].find_one.side_effect = Exception("connection lost")
        with patch("app.services.meeting_service.get_database", return_value=mock_db):
            result = update_meeting_end_time(MEETING_ID, "2024-01-01T10:10:00")
        assert result["status"] == "fail"
