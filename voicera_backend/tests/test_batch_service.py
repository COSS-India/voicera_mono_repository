"""
Unit tests for app.services.batch_service.

Strategy
--------
- get_database() is patched at "app.services.batch_service.get_database" so every
  internal helper that calls it receives the same MagicMock DB.
- GridFS is patched at "app.services.batch_service.GridFS".
- Private helpers (_normalize_contact_number, etc.) are imported directly; they
  contain pure logic and need no mocking.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from bson import ObjectId
from tests.helpers import make_mock_db as _conftest_make_mock_db

from app.services.batch_service import (
    _normalize_contact_number,
    _is_valid_contact_number,
    _resolve_local_schedule_to_utc,
    create_batch_from_csv,
    list_batches,
    delete_batch,
    run_batch,
    schedule_batch,
    cancel_scheduled_batch,
    reschedule_batch,
    stop_batch,
    claim_next_contact_for_execution,
    report_contact_execution_result,
    finalize_batch_execution,
    BatchNotFoundError,
    BatchRunStateError,
    DEFAULT_BATCH_CONCURRENCY,
)

# ── Sample data ───────────────────────────────────────────────────────────────

BATCH_DOC = {
    "batch_id": "b-test-001",
    "org_id": "testorg1",
    "batch_name": "Test Batch",
    "agent_type": "sales_bot",
    "concurrency": 5,
    "original_filename": "contacts.csv",
    "status": "uploaded",
    "execution_status": "not_started",
    "total_contacts": 10,
    "valid_contacts": 8,
    "invalid_contacts": 2,
    "attempted_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "schedule_mode": "run_now",
    "scheduled_at_utc": None,
    "scheduled_timezone": None,
    "scheduled_status": "none",
    "scheduled_by": None,
    "source_file_id": str(ObjectId()),
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z",
}

RUNNING_BATCH_DOC = {**BATCH_DOC, "execution_status": "running"}

SCHEDULED_BATCH_DOC = {
    **BATCH_DOC,
    "execution_status": "scheduled",
    "scheduled_status": "scheduled",
    "scheduled_at_utc": "2099-01-01T10:00:00Z",
}

CONTACT_DOC = {
    "batch_id": "b-test-001",
    "org_id": "testorg1",
    "agent_type": "sales_bot",
    "row_number": 2,
    "contact_number": "+12345678901",
    "is_valid": True,
    "status": "queued",
    "dynamic_fields": {},
}

AGENT_DOC = {
    "agent_id": "agent-001",
    "phone_number": "+15550001234",
    "org_id": "testorg1",
    "agent_type": "sales_bot",
}

VALID_CSV = b"contact_number,name\n+12345678901,Alice\n+12345678902,Bob\n"
MISSING_COL_CSV = b"phone,name\n+12345678901,Alice\n"
EMPTY_ROWS_CSV = b"contact_number,name\n"


def _make_mock_db(
    *,
    batch_doc=None,
    contact_doc=None,
    agent_doc=None,
    queued_count=5,
):
    """Return a MagicMock DB that routes each collection to its own independent mock.

    Delegates to the shared conftest make_mock_db helper so the side_effect
    pattern is consistent across the test suite.
    """
    batches_coll = MagicMock()
    batches_coll.find_one.return_value = batch_doc

    contacts_coll = MagicMock()
    contacts_coll.find_one_and_update.return_value = contact_doc
    contacts_coll.count_documents.return_value = queued_count

    agents_coll = MagicMock()
    agents_coll.find_one.return_value = agent_doc

    return _conftest_make_mock_db(
        Batches=batches_coll,
        BatchContacts=contacts_coll,
        AgentConfig=agents_coll,
    )


# ── TestNormalizeContactNumber ─────────────────────────────────────────────

class TestNormalizeContactNumber:
    def test_strips_spaces_and_dashes(self):
        assert _normalize_contact_number("+1 234-567-8901") == "+12345678901"

    def test_removes_parens_and_dots(self):
        assert _normalize_contact_number("+1(234)567.8901") == "+12345678901"

    def test_double_plus_stripped_to_digits_only(self):
        assert _normalize_contact_number("++12345678901") == "12345678901"

    def test_empty_string_returns_empty(self):
        assert _normalize_contact_number("") == ""

    def test_whitespace_only_returns_empty(self):
        assert _normalize_contact_number("   ") == ""


# ── TestIsValidContactNumber ───────────────────────────────────────────────

class TestIsValidContactNumber:
    def test_valid_international_format(self):
        assert _is_valid_contact_number("+12345678901") is True

    def test_valid_without_plus(self):
        assert _is_valid_contact_number("12345678901") is True

    def test_too_short_is_invalid(self):
        assert _is_valid_contact_number("+1234567") is False  # 7 digits

    def test_too_long_is_invalid(self):
        assert _is_valid_contact_number("+1234567890123456") is False  # 16 digits

    def test_letters_are_invalid(self):
        assert _is_valid_contact_number("+1234abc901") is False

    def test_empty_is_invalid(self):
        assert _is_valid_contact_number("") is False

    def test_exactly_8_digits_valid(self):
        assert _is_valid_contact_number("12345678") is True

    def test_exactly_15_digits_valid(self):
        assert _is_valid_contact_number("+123456789012345") is True


# ── TestResolveLocalScheduleToUtc ─────────────────────────────────────────

class TestResolveLocalScheduleToUtc:
    def test_naive_datetime_with_tz_name(self):
        from datetime import timezone
        result = _resolve_local_schedule_to_utc(
            scheduled_at_local="2099-06-15T14:00:00",
            timezone_name="America/New_York",
        )
        assert result.tzinfo == timezone.utc

    def test_offset_aware_datetime_ignores_tz_name(self):
        from datetime import timezone
        result = _resolve_local_schedule_to_utc(
            scheduled_at_local="2099-06-15T14:00:00+05:30",
            timezone_name="",
        )
        assert result.tzinfo == timezone.utc

    def test_empty_local_raises_valueerror(self):
        with pytest.raises(ValueError, match="required"):
            _resolve_local_schedule_to_utc(scheduled_at_local="", timezone_name="UTC")

    def test_invalid_tz_name_raises_valueerror(self):
        with pytest.raises(ValueError):
            _resolve_local_schedule_to_utc(
                scheduled_at_local="2099-01-01T10:00:00",
                timezone_name="Not/A/Timezone",
            )

    def test_invalid_datetime_format_raises_valueerror(self):
        with pytest.raises(ValueError, match="Invalid"):
            _resolve_local_schedule_to_utc(
                scheduled_at_local="not-a-datetime",
                timezone_name="UTC",
            )

    def test_utc_tz_converts_correctly(self):
        from datetime import timezone, datetime
        result = _resolve_local_schedule_to_utc(
            scheduled_at_local="2099-01-01T10:00:00",
            timezone_name="UTC",
        )
        assert result == datetime(2099, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


# ── TestCreateBatchFromCsv ────────────────────────────────────────────────

class TestCreateBatchFromCsv:
    def _make_fs(self, file_id=None):
        fs = MagicMock()
        fs.put.return_value = file_id or ObjectId()
        fs.exists.return_value = True
        return fs

    def test_success_returns_batch_doc(self):
        mock_db = _make_mock_db()
        mock_db["Batches"].find_one.return_value = None  # no duplicate
        mock_fs = self._make_fs()
        with patch("app.services.batch_service.get_database", return_value=mock_db), \
             patch("app.services.batch_service.GridFS", return_value=mock_fs):
            result = create_batch_from_csv(
                org_id="testorg1",
                batch_name="New Batch",
                agent_type="sales_bot",
                original_filename="contacts.csv",
                csv_bytes=VALID_CSV,
            )
        assert result["batch_id"] is not None
        assert result["org_id"] == "testorg1"
        assert result["batch_name"] == "New Batch"
        assert result["agent_type"] == "sales_bot"
        assert result["total_contacts"] == 2
        assert result["valid_contacts"] == 2
        assert result["invalid_contacts"] == 0
        # Verify the document inserted into Batches has the correct fields
        mock_db["Batches"].insert_one.assert_called_once()
        inserted_doc = mock_db["Batches"].insert_one.call_args[0][0]
        assert inserted_doc["org_id"] == "testorg1"
        assert inserted_doc["agent_type"] == "sales_bot"
        assert inserted_doc["batch_name"] == "New Batch"

    def test_csv_too_large_raises_valueerror(self):
        with pytest.raises(ValueError, match="too large"):
            create_batch_from_csv(
                org_id="testorg1",
                batch_name="Big Batch",
                agent_type="sales_bot",
                original_filename="big.csv",
                csv_bytes=b"x" * (11 * 1024 * 1024),
            )

    def test_invalid_encoding_raises_valueerror(self):
        with pytest.raises(ValueError, match="UTF-8"):
            create_batch_from_csv(
                org_id="testorg1",
                batch_name="Bad Batch",
                agent_type="sales_bot",
                original_filename="bad.csv",
                csv_bytes=b"\x80\x81\x82invalid",
            )

    def test_empty_batch_name_raises_valueerror(self):
        mock_db = MagicMock()
        mock_db["Batches"].find_one.return_value = None
        with patch("app.services.batch_service.get_database", return_value=mock_db), \
             patch("app.services.batch_service.GridFS", return_value=self._make_fs()):
            with pytest.raises(ValueError, match="required"):
                create_batch_from_csv(
                    org_id="testorg1",
                    batch_name="",
                    agent_type="sales_bot",
                    original_filename="test.csv",
                    csv_bytes=VALID_CSV,
                )

    def test_whitespace_batch_name_raises_valueerror(self):
        mock_db = MagicMock()
        mock_db["Batches"].find_one.return_value = None
        with patch("app.services.batch_service.get_database", return_value=mock_db), \
             patch("app.services.batch_service.GridFS", return_value=self._make_fs()):
            with pytest.raises(ValueError, match="required"):
                create_batch_from_csv(
                    org_id="testorg1",
                    batch_name="   ",
                    agent_type="sales_bot",
                    original_filename="test.csv",
                    csv_bytes=VALID_CSV,
                )

    def test_duplicate_batch_name_raises_valueerror(self):
        mock_db = _make_mock_db()
        mock_db["Batches"].find_one.return_value = BATCH_DOC  # duplicate exists
        with patch("app.services.batch_service.get_database", return_value=mock_db), \
             patch("app.services.batch_service.GridFS", return_value=self._make_fs()):
            with pytest.raises(ValueError, match="already exists"):
                create_batch_from_csv(
                    org_id="testorg1",
                    batch_name="Test Batch",
                    agent_type="sales_bot",
                    original_filename="test.csv",
                    csv_bytes=VALID_CSV,
                )

    def test_missing_contact_column_cleans_up_gridfs(self):
        mock_db = _make_mock_db()
        mock_db["Batches"].find_one.return_value = None
        mock_fs = self._make_fs()
        with patch("app.services.batch_service.get_database", return_value=mock_db), \
             patch("app.services.batch_service.GridFS", return_value=mock_fs):
            with pytest.raises(ValueError, match="contact_number"):
                create_batch_from_csv(
                    org_id="testorg1",
                    batch_name="Bad Batch",
                    agent_type="sales_bot",
                    original_filename="bad.csv",
                    csv_bytes=MISSING_COL_CSV,
                )
        mock_fs.delete.assert_called_once()

    def test_empty_csv_raises_valueerror_and_cleans_up(self):
        mock_db = _make_mock_db()
        mock_db["Batches"].find_one.return_value = None
        mock_fs = self._make_fs()
        with patch("app.services.batch_service.get_database", return_value=mock_db), \
             patch("app.services.batch_service.GridFS", return_value=mock_fs):
            with pytest.raises(ValueError, match="no contact rows"):
                create_batch_from_csv(
                    org_id="testorg1",
                    batch_name="Empty Batch",
                    agent_type="sales_bot",
                    original_filename="empty.csv",
                    csv_bytes=EMPTY_ROWS_CSV,
                )
        mock_fs.delete.assert_called_once()

    def test_invalid_contact_numbers_counted_correctly(self):
        csv_with_bad = b"contact_number\n+12345678901\nbad-number\n12\n"
        mock_db = _make_mock_db()
        mock_db["Batches"].find_one.return_value = None
        mock_fs = self._make_fs()
        with patch("app.services.batch_service.get_database", return_value=mock_db), \
             patch("app.services.batch_service.GridFS", return_value=mock_fs):
            result = create_batch_from_csv(
                org_id="testorg1",
                batch_name="Mixed Batch",
                agent_type="sales_bot",
                original_filename="mixed.csv",
                csv_bytes=csv_with_bad,
            )
        assert result["total_contacts"] == 3
        assert result["valid_contacts"] == 1
        assert result["invalid_contacts"] == 2

    def test_gridfs_file_and_contacts_deleted_when_insert_one_fails(self):
        mock_db = _make_mock_db()
        mock_db["Batches"].find_one.return_value = None  # no duplicate
        mock_db["Batches"].insert_one.side_effect = RuntimeError("DB write failed")
        mock_fs = self._make_fs()
        with patch("app.services.batch_service.get_database", return_value=mock_db), \
             patch("app.services.batch_service.GridFS", return_value=mock_fs):
            with pytest.raises(RuntimeError, match="DB write failed"):
                create_batch_from_csv(
                    org_id="testorg1",
                    batch_name="Fail Batch",
                    agent_type="sales_bot",
                    original_filename="f.csv",
                    csv_bytes=VALID_CSV,
                )
        # GridFS file must not be orphaned
        mock_fs.delete.assert_called_once()
        # Already-inserted contact rows must be rolled back
        mock_db["BatchContacts"].delete_many.assert_called_once()


# ── TestListBatches ───────────────────────────────────────────────────────

class TestListBatches:
    def test_returns_docs_without_id_field(self):
        doc = {**BATCH_DOC, "_id": "some-mongo-id"}
        mock_db = MagicMock()
        mock_db["Batches"].find.return_value.sort.return_value = [doc]
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            result = list_batches(org_id="testorg1")
        assert len(result) == 1
        assert "_id" not in result[0]

    def test_agent_type_filter_added_to_query(self):
        mock_db = MagicMock()
        mock_db["Batches"].find.return_value.sort.return_value = []
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            list_batches(org_id="testorg1", agent_type="sales_bot")
        query = mock_db["Batches"].find.call_args[0][0]
        assert query.get("agent_type") == "sales_bot"

    def test_empty_batch_name_uses_filename_fallback(self):
        doc = {**BATCH_DOC, "batch_name": "", "original_filename": "my_file.csv"}
        mock_db = MagicMock()
        mock_db["Batches"].find.return_value.sort.return_value = [doc]
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            result = list_batches(org_id="testorg1")
        assert result[0]["batch_name"] == "my_file.csv"

    def test_invalid_concurrency_replaced_by_default(self):
        doc = {**BATCH_DOC, "concurrency": 99}
        mock_db = MagicMock()
        mock_db["Batches"].find.return_value.sort.return_value = [doc]
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            result = list_batches(org_id="testorg1")
        assert result[0]["concurrency"] == DEFAULT_BATCH_CONCURRENCY

    def test_no_agent_type_filter_queries_all(self):
        mock_db = MagicMock()
        mock_db["Batches"].find.return_value.sort.return_value = []
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            list_batches(org_id="testorg1")
        query = mock_db["Batches"].find.call_args[0][0]
        assert "agent_type" not in query


# ── TestDeleteBatch ───────────────────────────────────────────────────────

class TestDeleteBatch:
    def test_success_deletes_contacts_and_batch(self):
        file_id = ObjectId()
        doc = {**BATCH_DOC, "source_file_id": str(file_id)}
        mock_db = MagicMock()
        mock_db["Batches"].find_one.return_value = doc
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        with patch("app.services.batch_service.get_database", return_value=mock_db), \
             patch("app.services.batch_service.GridFS", return_value=mock_fs):
            delete_batch(org_id="testorg1", batch_id="b-test-001")
        mock_fs.delete.assert_called_once()
        mock_db["BatchContacts"].delete_many.assert_called_once()
        mock_db["Batches"].delete_one.assert_called_once()

    def test_not_found_raises_batchnotfounderror(self):
        mock_db = MagicMock()
        mock_db["Batches"].find_one.return_value = None
        with patch("app.services.batch_service.get_database", return_value=mock_db), \
             patch("app.services.batch_service.GridFS", return_value=MagicMock()):
            with pytest.raises(BatchNotFoundError):
                delete_batch(org_id="testorg1", batch_id="missing-id")

    def test_invalid_source_file_id_skips_gridfs_delete(self):
        doc = {**BATCH_DOC, "source_file_id": "not-a-valid-objectid"}
        mock_db = MagicMock()
        mock_db["Batches"].find_one.return_value = doc
        mock_fs = MagicMock()
        with patch("app.services.batch_service.get_database", return_value=mock_db), \
             patch("app.services.batch_service.GridFS", return_value=mock_fs):
            delete_batch(org_id="testorg1", batch_id="b-test-001")
        mock_fs.delete.assert_not_called()
        mock_db["Batches"].delete_one.assert_called_once()

    def test_gridfs_file_not_existing_skips_delete(self):
        file_id = ObjectId()
        doc = {**BATCH_DOC, "source_file_id": str(file_id)}
        mock_db = MagicMock()
        mock_db["Batches"].find_one.return_value = doc
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        with patch("app.services.batch_service.get_database", return_value=mock_db), \
             patch("app.services.batch_service.GridFS", return_value=mock_fs):
            delete_batch(org_id="testorg1", batch_id="b-test-001")
        mock_fs.delete.assert_not_called()
        mock_db["Batches"].delete_one.assert_called_once()


# ── TestRunBatch ──────────────────────────────────────────────────────────

class TestRunBatch:
    def _runnable_db(self):
        return _make_mock_db(
            batch_doc=BATCH_DOC,
            agent_doc=AGENT_DOC,
            queued_count=8,
        )

    def test_success_returns_run_info(self):
        mock_db = self._runnable_db()
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            result = run_batch(org_id="testorg1", batch_id="b-test-001")
        assert result["status"] == "success"
        assert result["agent_type"] == "sales_bot"
        assert result["concurrency"] == 5

    def test_not_found_raises_batchnotfounderror(self):
        mock_db = _make_mock_db(batch_doc=None)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchNotFoundError):
                run_batch(org_id="testorg1", batch_id="missing")

    def test_no_valid_contacts_raises_batchrunstateerror(self):
        no_contacts = {**BATCH_DOC, "valid_contacts": 0}
        mock_db = _make_mock_db(batch_doc=no_contacts)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchRunStateError, match="no valid contacts"):
                run_batch(org_id="testorg1", batch_id="b-test-001")

    def test_already_processed_raises_batchrunstateerror(self):
        processed = {**BATCH_DOC, "attempted_calls": 5}
        mock_db = _make_mock_db(batch_doc=processed, queued_count=0)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchRunStateError, match="already processed"):
                run_batch(org_id="testorg1", batch_id="b-test-001")

    def test_running_batch_raises_batchrunstateerror(self):
        mock_db = _make_mock_db(batch_doc=RUNNING_BATCH_DOC)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchRunStateError):
                run_batch(org_id="testorg1", batch_id="b-test-001")

    def test_invalid_agent_raises_batchrunstateerror(self):
        mock_db = self._runnable_db()
        mock_db["AgentConfig"].find_one.return_value = None
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchRunStateError, match="Invalid agent"):
                run_batch(org_id="testorg1", batch_id="b-test-001")

    def test_invalid_concurrency_raises_batchrunstateerror(self):
        mock_db = self._runnable_db()
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchRunStateError, match="Concurrency"):
                run_batch(org_id="testorg1", batch_id="b-test-001", concurrency=99)

    def test_preserve_schedule_sets_triggered_status(self):
        mock_db = self._runnable_db()
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            run_batch(
                org_id="testorg1",
                batch_id="b-test-001",
                preserve_schedule=True,
            )
        first_update = mock_db["Batches"].update_one.call_args_list[0]
        update_set = first_update[0][1]["$set"]
        assert update_set.get("scheduled_status") == "triggered"

    def test_explicit_concurrency_overrides_batch_default(self):
        mock_db = self._runnable_db()
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            result = run_batch(
                org_id="testorg1", batch_id="b-test-001", concurrency=3
            )
        assert result["concurrency"] == 3


# ── TestScheduleBatch ─────────────────────────────────────────────────────

class TestScheduleBatch:
    FUTURE_TIME = "2099-01-01T10:00:00+00:00"

    def test_success_returns_scheduled_info(self):
        mock_db = _make_mock_db(batch_doc=BATCH_DOC, agent_doc=AGENT_DOC)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            result = schedule_batch(
                org_id="testorg1",
                batch_id="b-test-001",
                scheduled_at_local=self.FUTURE_TIME,
                timezone_name="UTC",
            )
        assert result["status"] == "success"
        assert "scheduled_at_utc" in result

    def test_past_time_raises_batchrunstateerror(self):
        mock_db = _make_mock_db(batch_doc=BATCH_DOC, agent_doc=AGENT_DOC)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchRunStateError, match="future"):
                schedule_batch(
                    org_id="testorg1",
                    batch_id="b-test-001",
                    scheduled_at_local="2020-01-01T10:00:00+00:00",
                    timezone_name="UTC",
                )

    def test_not_found_raises_batchnotfounderror(self):
        mock_db = _make_mock_db(batch_doc=None)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchNotFoundError):
                schedule_batch(
                    org_id="testorg1",
                    batch_id="missing",
                    scheduled_at_local=self.FUTURE_TIME,
                    timezone_name="UTC",
                )

    def test_no_valid_contacts_raises_batchrunstateerror(self):
        no_contacts = {**BATCH_DOC, "valid_contacts": 0}
        mock_db = _make_mock_db(batch_doc=no_contacts)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchRunStateError, match="no valid contacts"):
                schedule_batch(
                    org_id="testorg1",
                    batch_id="b-test-001",
                    scheduled_at_local=self.FUTURE_TIME,
                    timezone_name="UTC",
                )

    def test_running_batch_not_schedulable(self):
        mock_db = _make_mock_db(batch_doc=RUNNING_BATCH_DOC, agent_doc=AGENT_DOC)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchRunStateError):
                schedule_batch(
                    org_id="testorg1",
                    batch_id="b-test-001",
                    scheduled_at_local=self.FUTURE_TIME,
                    timezone_name="UTC",
                )


# ── TestCancelScheduledBatch ──────────────────────────────────────────────

class TestCancelScheduledBatch:
    def test_success_returns_canceled_message(self):
        mock_db = _make_mock_db(batch_doc=SCHEDULED_BATCH_DOC)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            result = cancel_scheduled_batch(org_id="testorg1", batch_id="b-test-001")
        assert result["status"] == "success"
        assert "cancel" in result["message"].lower()

    def test_not_scheduled_raises_batchrunstateerror(self):
        mock_db = _make_mock_db(batch_doc=BATCH_DOC)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchRunStateError, match="pending scheduled"):
                cancel_scheduled_batch(org_id="testorg1", batch_id="b-test-001")

    def test_not_found_raises_batchnotfounderror(self):
        mock_db = _make_mock_db(batch_doc=None)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchNotFoundError):
                cancel_scheduled_batch(org_id="testorg1", batch_id="missing")


# ── TestRescheduleBatch ───────────────────────────────────────────────────

class TestRescheduleBatch:
    FUTURE_TIME = "2099-06-01T10:00:00+00:00"

    def test_non_editable_batch_raises_batchrunstateerror(self):
        mock_db = _make_mock_db(batch_doc=BATCH_DOC)  # not in scheduled state
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchRunStateError, match="pending scheduled"):
                reschedule_batch(
                    org_id="testorg1",
                    batch_id="b-test-001",
                    scheduled_at_local=self.FUTURE_TIME,
                    timezone_name="UTC",
                )

    def test_not_found_raises_batchnotfounderror(self):
        mock_db = _make_mock_db(batch_doc=None)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchNotFoundError):
                reschedule_batch(
                    org_id="testorg1",
                    batch_id="missing",
                    scheduled_at_local=self.FUTURE_TIME,
                    timezone_name="UTC",
                )


# ── TestStopBatch ─────────────────────────────────────────────────────────

class TestStopBatch:
    def test_success_marks_stopping(self):
        mock_db = _make_mock_db(batch_doc=RUNNING_BATCH_DOC)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            result = stop_batch(org_id="testorg1", batch_id="b-test-001")
        assert result["status"] == "success"

    def test_not_running_raises_batchrunstateerror(self):
        mock_db = _make_mock_db(batch_doc=BATCH_DOC)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchRunStateError, match="not running"):
                stop_batch(org_id="testorg1", batch_id="b-test-001")

    def test_not_found_raises_batchnotfounderror(self):
        mock_db = _make_mock_db(batch_doc=None)
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            with pytest.raises(BatchNotFoundError):
                stop_batch(org_id="testorg1", batch_id="missing")


# ── TestClaimNextContact ──────────────────────────────────────────────────

class TestClaimNextContact:
    def test_returns_contact_doc_without_id(self):
        contact_with_id = {**CONTACT_DOC, "_id": ObjectId()}
        mock_db = MagicMock()
        mock_db["BatchContacts"].find_one_and_update.return_value = contact_with_id
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            result = claim_next_contact_for_execution(org_id="testorg1", batch_id="b-test-001")
        assert result is not None
        assert result["contact_number"] == "+12345678901"
        assert "_id" not in result

    def test_returns_none_when_no_queued_contacts(self):
        mock_db = MagicMock()
        mock_db["BatchContacts"].find_one_and_update.return_value = None
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            result = claim_next_contact_for_execution(org_id="testorg1", batch_id="b-test-001")
        assert result is None

    def test_claim_filter_targets_valid_queued_contacts_for_correct_batch(self):
        mock_db = MagicMock()
        mock_db["BatchContacts"].find_one_and_update.return_value = None
        with patch("app.services.batch_service.get_database", return_value=mock_db):
            claim_next_contact_for_execution(org_id="testorg1", batch_id="b-test-001")
        filter_arg = mock_db["BatchContacts"].find_one_and_update.call_args[0][0]
        assert filter_arg["batch_id"] == "b-test-001"
        assert filter_arg["org_id"] == "testorg1"
        assert filter_arg["status"] == "queued"
        assert filter_arg["is_valid"] is True


# ── TestReportContactResult ───────────────────────────────────────────────

class TestReportContactResult:
    def _make_report_db(self):
        batches_coll = MagicMock()
        contacts_coll = MagicMock()
        db = MagicMock()
        db.__getitem__.side_effect = lambda k: {
            "Batches": batches_coll,
            "BatchContacts": contacts_coll,
        }.get(k, MagicMock())
        return db, batches_coll, contacts_coll

    def test_success_updates_contact_and_batch(self):
        db, batches_coll, contacts_coll = self._make_report_db()
        with patch("app.services.batch_service.get_database", return_value=db):
            report_contact_execution_result(
                org_id="testorg1",
                batch_id="b-test-001",
                row_number=2,
                ok=True,
                error=None,
            )
        contacts_coll.update_one.assert_called_once()
        batches_coll.update_one.assert_called_once()

    def test_success_increments_successful_calls(self):
        db, batches_coll, contacts_coll = self._make_report_db()
        with patch("app.services.batch_service.get_database", return_value=db):
            report_contact_execution_result(
                org_id="testorg1",
                batch_id="b-test-001",
                row_number=2,
                ok=True,
                error=None,
            )
        inc_args = batches_coll.update_one.call_args[0][1]["$inc"]
        assert "successful_calls" in inc_args
        assert "attempted_calls" in inc_args
        assert "failed_calls" not in inc_args

    def test_failed_contact_increments_failed_calls(self):
        db, batches_coll, contacts_coll = self._make_report_db()
        with patch("app.services.batch_service.get_database", return_value=db):
            report_contact_execution_result(
                org_id="testorg1",
                batch_id="b-test-001",
                row_number=2,
                ok=False,
                error="Call timed out",
            )
        inc_args = batches_coll.update_one.call_args[0][1]["$inc"]
        assert "failed_calls" in inc_args
        assert "attempted_calls" in inc_args
        assert "successful_calls" not in inc_args

    def test_contact_status_called_on_success(self):
        db, batches_coll, contacts_coll = self._make_report_db()
        with patch("app.services.batch_service.get_database", return_value=db):
            report_contact_execution_result(
                org_id="testorg1",
                batch_id="b-test-001",
                row_number=3,
                ok=True,
                error=None,
            )
        update_set = contacts_coll.update_one.call_args[0][1]["$set"]
        assert update_set["status"] == "called"

    def test_contact_status_failed_on_failure(self):
        db, batches_coll, contacts_coll = self._make_report_db()
        with patch("app.services.batch_service.get_database", return_value=db):
            report_contact_execution_result(
                org_id="testorg1",
                batch_id="b-test-001",
                row_number=3,
                ok=False,
                error="timeout",
            )
        update_set = contacts_coll.update_one.call_args[0][1]["$set"]
        assert update_set["status"] == "failed"


# ── TestFinalizeBatchExecution ────────────────────────────────────────────

class TestFinalizeBatchExecution:
    def _db(self, attempted=5, successful=5, failed=0):
        mock_db = MagicMock()
        mock_db["Batches"].find_one.return_value = {
            "attempted_calls": attempted,
            "successful_calls": successful,
            "failed_calls": failed,
        }
        return mock_db

    def test_all_successful_marks_completed(self):
        with patch("app.services.batch_service.get_database", return_value=self._db(5, 5, 0)):
            result = finalize_batch_execution(org_id="testorg1", batch_id="b-test-001")
        assert result["status"] == "completed"

    def test_partial_success_marks_completed(self):
        with patch("app.services.batch_service.get_database", return_value=self._db(5, 3, 2)):
            result = finalize_batch_execution(org_id="testorg1", batch_id="b-test-001")
        assert result["status"] == "completed"

    def test_all_failed_marks_failed(self):
        with patch("app.services.batch_service.get_database", return_value=self._db(5, 0, 5)):
            result = finalize_batch_execution(org_id="testorg1", batch_id="b-test-001")
        assert result["status"] == "failed"

    def test_stopped_flag_marks_stopped_regardless_of_counts(self):
        with patch("app.services.batch_service.get_database", return_value=self._db(5, 5, 0)):
            result = finalize_batch_execution(
                org_id="testorg1", batch_id="b-test-001", stopped=True
            )
        assert result["status"] == "stopped"

    def test_zero_attempts_marks_completed(self):
        with patch("app.services.batch_service.get_database", return_value=self._db(0, 0, 0)):
            result = finalize_batch_execution(org_id="testorg1", batch_id="b-test-001")
        assert result["status"] == "completed"
