"""
Integration tests for /api/v1/batches endpoints.

Strategy
--------
- Service layer is mocked via "app.services.batch_service.<function>".
- Voice server HTTP calls are mocked via "app.routers.batches.requests.post".
- Worker endpoints use API key auth (bypassed by the `client` fixture).
- CSV upload uses TestClient's `files` + `data` multipart parameters.
"""
import io
import pytest
from unittest.mock import patch, MagicMock
import requests as _requests

from app.services.batch_service import BatchNotFoundError, BatchRunStateError

BASE = "/api/v1/batches"

BATCH_DOC = {
    "batch_id": "b-001",
    "org_id": "testorg1",
    "batch_name": "Q1 Calls",
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
    "error_message": None,
    "schedule_mode": "run_now",
    "scheduled_at_utc": None,
    "scheduled_timezone": None,
    "scheduled_status": "none",
    "scheduled_by": None,
    "source_file_id": "507f1f77bcf86cd799439011",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z",
}

RUN_RESULT = {
    "status": "success",
    "message": "Batch execution prepared",
    "agent_type": "sales_bot",
    "concurrency": 5,
}

SCHEDULE_RESULT = {
    "status": "success",
    "message": "Batch scheduled successfully",
    "scheduled_at_utc": "2099-01-01T10:00:00Z",
    "concurrency": 5,
    "agent_type": "sales_bot",
}

CONTACT_DOC = {
    "batch_id": "b-001",
    "org_id": "testorg1",
    "row_number": 2,
    "contact_number": "+12345678901",
    "is_valid": True,
    "status": "queued",
    "dynamic_fields": {},
}

VALID_CSV = b"contact_number,name\n+12345678901,Alice\n+12345678902,Bob\n"
SCHEDULE_BODY = {"scheduled_at_local": "2099-01-01T10:00:00", "timezone": "UTC"}


def _voice_ok(payload=None):
    mock_resp = MagicMock()
    mock_resp.text = '{"status": "ok"}'
    mock_resp.json.return_value = payload or {"status": "ok"}
    return mock_resp


# ── GET /batches ──────────────────────────────────────────────────────────

class TestGetBatches:
    def test_success_returns_list(self, client):
        with patch("app.services.batch_service.list_batches", return_value=[BATCH_DOC]):
            resp = client.get(BASE)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["batch_name"] == "Q1 Calls"

    def test_agent_type_filter_forwarded_to_service(self, client):
        with patch("app.services.batch_service.list_batches", return_value=[]) as mock_svc:
            client.get(f"{BASE}?agent_type=sales_bot")
        _, kwargs = mock_svc.call_args
        assert kwargs.get("agent_type") == "sales_bot"

    def test_unauthenticated_returns_401(self, unauth_client):
        resp = unauth_client.get(BASE)
        assert resp.status_code == 401

    def test_empty_list_returns_200(self, client):
        with patch("app.services.batch_service.list_batches", return_value=[]):
            resp = client.get(BASE)
        assert resp.status_code == 200
        assert resp.json() == []


# ── POST /batches/upload ──────────────────────────────────────────────────

class TestUploadBatchCsv:
    def _upload(self, client, *, org_id="testorg1", batch_name="Test", agent_type="sales_bot",
                filename="test.csv", content=VALID_CSV):
        return client.post(
            f"{BASE}/upload",
            files={"file": (filename, io.BytesIO(content), "text/csv")},
            data={"org_id": org_id, "batch_name": batch_name, "agent_type": agent_type},
        )

    def test_success_returns_201(self, client):
        with patch("app.services.batch_service.validate_agent_for_org", return_value=True), \
             patch("app.services.batch_service.create_batch_from_csv", return_value=BATCH_DOC):
            resp = self._upload(client)
        assert resp.status_code == 201
        assert resp.json()["batch_id"] == "b-001"

    def test_wrong_org_returns_403(self, client):
        resp = self._upload(client, org_id="otherorg9")
        assert resp.status_code == 403

    def test_invalid_agent_type_returns_400(self, client):
        with patch("app.services.batch_service.validate_agent_for_org", return_value=False):
            resp = self._upload(client)
        assert resp.status_code == 400

    def test_non_csv_filename_returns_400(self, client):
        with patch("app.services.batch_service.validate_agent_for_org", return_value=True):
            resp = self._upload(client, filename="contacts.txt")
        assert resp.status_code == 400
        assert "CSV" in resp.json()["detail"]

    def test_empty_file_returns_400(self, client):
        with patch("app.services.batch_service.validate_agent_for_org", return_value=True):
            resp = self._upload(client, content=b"")
        assert resp.status_code == 400

    def test_service_value_error_returns_400(self, client):
        with patch("app.services.batch_service.validate_agent_for_org", return_value=True), \
             patch("app.services.batch_service.create_batch_from_csv",
                   side_effect=ValueError("already exists")):
            resp = self._upload(client)
        assert resp.status_code == 400
        assert "already exists" in resp.json()["detail"]

    def test_service_unexpected_exception_returns_500(self, client):
        with patch("app.services.batch_service.validate_agent_for_org", return_value=True), \
             patch("app.services.batch_service.create_batch_from_csv",
                   side_effect=RuntimeError("DB gone")):
            resp = self._upload(client)
        assert resp.status_code == 500


# ── DELETE /batches/{batch_id} ────────────────────────────────────────────

class TestDeleteBatch:
    def test_success_returns_deleted_true(self, client):
        with patch("app.services.batch_service.delete_batch"):
            resp = client.delete(f"{BASE}/b-001")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_not_found_returns_404(self, client):
        with patch("app.services.batch_service.delete_batch", side_effect=BatchNotFoundError()):
            resp = client.delete(f"{BASE}/missing")
        assert resp.status_code == 404


# ── POST /batches/{batch_id}/run ──────────────────────────────────────────

class TestRunBatch:
    def test_success_returns_voice_server_payload(self, client):
        with patch("app.services.batch_service.run_batch", return_value=RUN_RESULT), \
             patch("app.routers.batches.requests.post",
                   return_value=_voice_ok({"status": "running"})), \
             patch("app.services.batch_service.mark_batch_start_failure"):
            resp = client.post(f"{BASE}/b-001/run", json={})
        assert resp.status_code == 200

    def test_not_found_returns_404(self, client):
        with patch("app.services.batch_service.run_batch", side_effect=BatchNotFoundError()):
            resp = client.post(f"{BASE}/missing/run", json={})
        assert resp.status_code == 404

    def test_run_state_error_returns_400(self, client):
        with patch("app.services.batch_service.run_batch",
                   side_effect=BatchRunStateError("no valid contacts")):
            resp = client.post(f"{BASE}/b-001/run", json={})
        assert resp.status_code == 400
        assert "no valid contacts" in resp.json()["detail"]

    def test_voice_server_unreachable_returns_502(self, client):
        with patch("app.services.batch_service.run_batch", return_value=RUN_RESULT), \
             patch("app.routers.batches.requests.post",
                   side_effect=_requests.RequestException("timeout")), \
             patch("app.services.batch_service.mark_batch_start_failure"):
            resp = client.post(f"{BASE}/b-001/run", json={})
        assert resp.status_code == 502

    def test_concurrency_forwarded_to_service(self, client):
        with patch("app.services.batch_service.run_batch", return_value=RUN_RESULT) as mock_svc, \
             patch("app.routers.batches.requests.post", return_value=_voice_ok()), \
             patch("app.services.batch_service.mark_batch_start_failure"):
            client.post(f"{BASE}/b-001/run", json={"concurrency": 3})
        kwargs = mock_svc.call_args[1]
        assert kwargs.get("concurrency") == 3


# ── POST /batches/{batch_id}/schedule ─────────────────────────────────────

class TestScheduleBatch:
    def test_success_returns_200(self, client):
        with patch("app.services.batch_service.schedule_batch", return_value=SCHEDULE_RESULT):
            resp = client.post(f"{BASE}/b-001/schedule", json=SCHEDULE_BODY)
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_not_found_returns_404(self, client):
        with patch("app.services.batch_service.schedule_batch", side_effect=BatchNotFoundError()):
            resp = client.post(f"{BASE}/missing/schedule", json=SCHEDULE_BODY)
        assert resp.status_code == 404

    def test_state_error_returns_400(self, client):
        with patch("app.services.batch_service.schedule_batch",
                   side_effect=BatchRunStateError("already running")):
            resp = client.post(f"{BASE}/b-001/schedule", json=SCHEDULE_BODY)
        assert resp.status_code == 400

    def test_missing_scheduled_at_returns_422(self, client):
        resp = client.post(f"{BASE}/b-001/schedule", json={"timezone": "UTC"})
        assert resp.status_code == 422

    def test_missing_timezone_returns_422(self, client):
        resp = client.post(f"{BASE}/b-001/schedule",
                           json={"scheduled_at_local": "2099-01-01T10:00:00"})
        assert resp.status_code == 422


# ── POST /batches/{batch_id}/schedule/cancel ──────────────────────────────

class TestCancelBatchSchedule:
    def test_success_returns_200(self, client):
        ok = {"status": "success", "message": "Scheduled batch canceled"}
        with patch("app.services.batch_service.cancel_scheduled_batch", return_value=ok):
            resp = client.post(f"{BASE}/b-001/schedule/cancel")
        assert resp.status_code == 200

    def test_not_found_returns_404(self, client):
        with patch("app.services.batch_service.cancel_scheduled_batch",
                   side_effect=BatchNotFoundError()):
            resp = client.post(f"{BASE}/missing/schedule/cancel")
        assert resp.status_code == 404

    def test_state_error_returns_400(self, client):
        with patch("app.services.batch_service.cancel_scheduled_batch",
                   side_effect=BatchRunStateError("not schedulable")):
            resp = client.post(f"{BASE}/b-001/schedule/cancel")
        assert resp.status_code == 400


# ── POST /batches/{batch_id}/schedule/reschedule ──────────────────────────

class TestRescheduleBatch:
    def test_success_returns_200(self, client):
        with patch("app.services.batch_service.reschedule_batch", return_value=SCHEDULE_RESULT):
            resp = client.post(f"{BASE}/b-001/schedule/reschedule", json=SCHEDULE_BODY)
        assert resp.status_code == 200

    def test_not_found_returns_404(self, client):
        with patch("app.services.batch_service.reschedule_batch", side_effect=BatchNotFoundError()):
            resp = client.post(f"{BASE}/missing/schedule/reschedule", json=SCHEDULE_BODY)
        assert resp.status_code == 404

    def test_not_editable_returns_400(self, client):
        with patch("app.services.batch_service.reschedule_batch",
                   side_effect=BatchRunStateError("not editable")):
            resp = client.post(f"{BASE}/b-001/schedule/reschedule", json=SCHEDULE_BODY)
        assert resp.status_code == 400


# ── POST /batches/{batch_id}/stop ─────────────────────────────────────────

class TestStopBatch:
    def test_success_returns_200(self, client):
        with patch("app.services.batch_service.stop_batch",
                   return_value={"status": "success", "message": "Batch stop requested"}), \
             patch("app.routers.batches.requests.post",
                   return_value=_voice_ok({"status": "stopped"})):
            resp = client.post(f"{BASE}/b-001/stop")
        assert resp.status_code == 200

    def test_not_running_returns_400(self, client):
        with patch("app.services.batch_service.stop_batch",
                   side_effect=BatchRunStateError("not running")):
            resp = client.post(f"{BASE}/b-001/stop")
        assert resp.status_code == 400

    def test_not_found_returns_404(self, client):
        with patch("app.services.batch_service.stop_batch", side_effect=BatchNotFoundError()):
            resp = client.post(f"{BASE}/missing/stop")
        assert resp.status_code == 404

    def test_voice_server_unreachable_returns_502(self, client):
        with patch("app.services.batch_service.stop_batch",
                   return_value={"status": "success", "message": "stop requested"}), \
             patch("app.routers.batches.requests.post",
                   side_effect=_requests.RequestException("unreachable")):
            resp = client.post(f"{BASE}/b-001/stop")
        assert resp.status_code == 502


# ── Worker: POST /batches/worker/claim-next ───────────────────────────────

class TestWorkerClaimNext:
    def test_success_returns_contact(self, client):
        with patch("app.services.batch_service.claim_next_contact_for_execution",
                   return_value=CONTACT_DOC):
            resp = client.post(
                f"{BASE}/worker/claim-next",
                json={"org_id": "testorg1", "batch_id": "b-001"},
            )
        assert resp.status_code == 200
        assert resp.json()["contact"]["contact_number"] == "+12345678901"

    def test_no_contact_returns_null(self, client):
        with patch("app.services.batch_service.claim_next_contact_for_execution",
                   return_value=None):
            resp = client.post(
                f"{BASE}/worker/claim-next",
                json={"org_id": "testorg1", "batch_id": "b-001"},
            )
        assert resp.status_code == 200
        assert resp.json()["contact"] is None

    def test_missing_batch_id_returns_400(self, client):
        resp = client.post(f"{BASE}/worker/claim-next", json={"org_id": "testorg1"})
        assert resp.status_code == 400

    def test_missing_org_id_returns_400(self, client):
        resp = client.post(f"{BASE}/worker/claim-next", json={"batch_id": "b-001"})
        assert resp.status_code == 400


# ── Worker: POST /batches/worker/agent-config ─────────────────────────────

class TestWorkerAgentConfig:
    def test_success_returns_agent_config(self, client):
        config = {"agent_id": "agent-001", "caller_id": "+15550001234"}
        with patch("app.services.batch_service.get_agent_call_config_for_batch",
                   return_value=config):
            resp = client.post(
                f"{BASE}/worker/agent-config",
                json={"org_id": "testorg1", "agent_type": "sales_bot"},
            )
        assert resp.status_code == 200
        assert resp.json()["agent_id"] == "agent-001"

    def test_missing_agent_type_returns_400(self, client):
        resp = client.post(f"{BASE}/worker/agent-config", json={"org_id": "testorg1"})
        assert resp.status_code == 400

    def test_agent_not_found_returns_400(self, client):
        with patch("app.services.batch_service.get_agent_call_config_for_batch",
                   side_effect=ValueError("Agent not found for organization")):
            resp = client.post(
                f"{BASE}/worker/agent-config",
                json={"org_id": "testorg1", "agent_type": "ghost_bot"},
            )
        assert resp.status_code == 400

    def test_missing_org_id_returns_400(self, client):
        resp = client.post(f"{BASE}/worker/agent-config", json={"agent_type": "sales_bot"})
        assert resp.status_code == 400


# ── Worker: POST /batches/worker/report ──────────────────────────────────

class TestWorkerReport:
    def test_success_returns_updated_true(self, client):
        with patch("app.services.batch_service.report_contact_execution_result"):
            resp = client.post(
                f"{BASE}/worker/report",
                json={"org_id": "testorg1", "batch_id": "b-001", "row_number": 2, "ok": True},
            )
        assert resp.status_code == 200
        assert resp.json()["updated"] is True

    def test_missing_row_number_returns_400(self, client):
        resp = client.post(
            f"{BASE}/worker/report",
            json={"org_id": "testorg1", "batch_id": "b-001", "ok": True},
        )
        assert resp.status_code == 400

    def test_missing_batch_id_returns_400(self, client):
        resp = client.post(
            f"{BASE}/worker/report",
            json={"org_id": "testorg1", "row_number": 2, "ok": True},
        )
        assert resp.status_code == 400


# ── Worker: POST /batches/worker/finalize ────────────────────────────────

class TestWorkerFinalize:
    def test_success_returns_status(self, client):
        with patch("app.services.batch_service.finalize_batch_execution",
                   return_value={"status": "completed"}):
            resp = client.post(
                f"{BASE}/worker/finalize",
                json={"org_id": "testorg1", "batch_id": "b-001"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_stopped_flag_forwarded(self, client):
        with patch("app.services.batch_service.finalize_batch_execution",
                   return_value={"status": "stopped"}) as mock_svc:
            client.post(
                f"{BASE}/worker/finalize",
                json={"org_id": "testorg1", "batch_id": "b-001", "stopped": True},
            )
        kwargs = mock_svc.call_args[1]
        assert kwargs.get("stopped") is True

    def test_missing_org_id_returns_400(self, client):
        resp = client.post(f"{BASE}/worker/finalize", json={"batch_id": "b-001"})
        assert resp.status_code == 400

    def test_missing_batch_id_returns_400(self, client):
        resp = client.post(f"{BASE}/worker/finalize", json={"org_id": "testorg1"})
        assert resp.status_code == 400
