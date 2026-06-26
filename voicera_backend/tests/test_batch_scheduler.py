"""Tests for app/services/batch_scheduler.py"""

import pytest
from unittest.mock import patch, MagicMock

import app.services.batch_scheduler as scheduler_mod
from app.services import batch_scheduler


@pytest.fixture(autouse=True)
def reset_scheduler():
    """Reset global scheduler state between tests."""
    original = scheduler_mod._scheduler
    scheduler_mod._scheduler = None
    yield
    scheduler_mod._scheduler = None


class TestVoiceServerHeaders:
    def test_includes_api_key_when_set(self):
        with patch.object(batch_scheduler.settings, "INTERNAL_API_KEY", "my-key"):
            headers = batch_scheduler._voice_server_headers()
        assert headers["X-API-Key"] == "my-key"
        assert headers["Content-Type"] == "application/json"

    def test_no_api_key_when_empty(self):
        with patch.object(batch_scheduler.settings, "INTERNAL_API_KEY", ""):
            headers = batch_scheduler._voice_server_headers()
        assert "X-API-Key" not in headers


class TestStartBatchOnVoiceServer:
    def test_success_returns_json(self):
        mock_resp = MagicMock()
        mock_resp.text = '{"started": true}'
        mock_resp.json.return_value = {"started": True}
        with patch("app.services.batch_scheduler.requests.post", return_value=mock_resp) as mock_post:
            result = batch_scheduler._start_batch_on_voice_server(
                org_id="org1", batch_id="b1", agent_type="sales", concurrency=5
            )
        assert result == {"started": True}
        mock_post.assert_called_once()

    def test_empty_text_returns_empty_dict(self):
        mock_resp = MagicMock()
        mock_resp.text = ""
        with patch("app.services.batch_scheduler.requests.post", return_value=mock_resp):
            result = batch_scheduler._start_batch_on_voice_server(
                org_id="org1", batch_id="b1", agent_type="sales", concurrency=5
            )
        assert result == {}

    def test_non_dict_json_returns_empty_dict(self):
        mock_resp = MagicMock()
        mock_resp.text = "[1, 2]"
        mock_resp.json.return_value = [1, 2]
        with patch("app.services.batch_scheduler.requests.post", return_value=mock_resp):
            result = batch_scheduler._start_batch_on_voice_server(
                org_id="org1", batch_id="b1", agent_type="sales", concurrency=5
            )
        assert result == {}


class TestPollDueBatches:
    def test_no_claimed_batch_returns_immediately(self):
        with patch("app.services.batch_scheduler.batch_service.claim_next_due_scheduled_batch", return_value=None):
            batch_scheduler._poll_due_batches()

    def test_malformed_batch_skipped(self):
        claimed = [{"batch_id": "", "org_id": ""}, None]
        call_count = 0

        def side_effect():
            nonlocal call_count
            if call_count < len(claimed):
                val = claimed[call_count]
                call_count += 1
                return val
            return None

        with patch("app.services.batch_scheduler.batch_service.claim_next_due_scheduled_batch", side_effect=side_effect):
            batch_scheduler._poll_due_batches()

    def test_successful_batch_triggers_voice_server(self):
        claimed = {"batch_id": "b1", "org_id": "org1", "agent_type": "sales", "concurrency": 3}
        run_result = {"agent_type": "sales", "concurrency": 3}
        call_count = 0

        def claim_side_effect():
            nonlocal call_count
            call_count += 1
            return claimed if call_count == 1 else None

        with patch("app.services.batch_scheduler.batch_service.claim_next_due_scheduled_batch", side_effect=claim_side_effect), \
             patch("app.services.batch_scheduler.batch_service.run_batch", return_value=run_result) as mock_run, \
             patch("app.services.batch_scheduler._start_batch_on_voice_server", return_value={}) as mock_start:
            batch_scheduler._poll_due_batches()

        mock_run.assert_called_once_with(
            org_id="org1", batch_id="b1", agent_type="sales", concurrency=3, preserve_schedule=True
        )
        mock_start.assert_called_once()

    def test_voice_server_failure_marks_batch_failed(self):
        claimed = {"batch_id": "b1", "org_id": "org1", "agent_type": "sales", "concurrency": 3}
        call_count = 0

        def claim_side_effect():
            nonlocal call_count
            call_count += 1
            return claimed if call_count == 1 else None

        with patch("app.services.batch_scheduler.batch_service.claim_next_due_scheduled_batch", side_effect=claim_side_effect), \
             patch("app.services.batch_scheduler.batch_service.run_batch", side_effect=RuntimeError("fail")), \
             patch("app.services.batch_scheduler.batch_service.mark_batch_start_failure") as mock_fail:
            batch_scheduler._poll_due_batches()

        kwargs = mock_fail.call_args.kwargs
        assert kwargs["org_id"] == "org1"
        assert kwargs["batch_id"] == "b1"
        assert "Failed to start scheduled batch" in kwargs["error_message"]


class TestStartBatchScheduler:
    def test_starts_scheduler(self):
        mock_scheduler = MagicMock()
        mock_scheduler.running = False
        with patch("app.services.batch_scheduler.BackgroundScheduler", return_value=mock_scheduler), \
             patch.object(batch_scheduler.settings, "BATCH_SCHEDULER_POLL_SECONDS", 10, create=True):
            batch_scheduler.start_batch_scheduler()
        mock_scheduler.start.assert_called_once()
        assert scheduler_mod._scheduler is mock_scheduler

    def test_does_not_restart_when_running(self):
        mock_scheduler = MagicMock()
        mock_scheduler.running = True
        scheduler_mod._scheduler = mock_scheduler
        with patch("app.services.batch_scheduler.BackgroundScheduler") as mock_cls:
            batch_scheduler.start_batch_scheduler()
        mock_cls.assert_not_called()


class TestStopBatchScheduler:
    def test_stop_shuts_down_scheduler(self):
        mock_scheduler = MagicMock()
        scheduler_mod._scheduler = mock_scheduler
        batch_scheduler.stop_batch_scheduler()
        mock_scheduler.shutdown.assert_called_once_with(wait=False)
        assert scheduler_mod._scheduler is None

    def test_stop_when_no_scheduler_is_noop(self):
        scheduler_mod._scheduler = None
        batch_scheduler.stop_batch_scheduler()

