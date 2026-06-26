"""Tests for utils/batching.py."""

import threading
import pytest
from unittest.mock import MagicMock, patch, call
from fastapi import HTTPException

from utils.batching import (
    BatchWorker,
    BatchRunRequest,
    BatchStopRequest,
    _get_voice_server_internal_url,
    create_batch_router,
)


# ---------------------------------------------------------------------------
# _get_voice_server_internal_url
# ---------------------------------------------------------------------------

def test_get_voice_server_internal_url_default(monkeypatch):
    monkeypatch.delenv("VOICE_SERVER_INTERNAL_URL", raising=False)
    assert _get_voice_server_internal_url() == "http://127.0.0.1:7860"


def test_get_voice_server_internal_url_env(monkeypatch):
    monkeypatch.setenv("VOICE_SERVER_INTERNAL_URL", "http://myserver:9000/")
    assert _get_voice_server_internal_url() == "http://myserver:9000"


# ---------------------------------------------------------------------------
# BatchRunRequest / BatchStopRequest validation
# ---------------------------------------------------------------------------

def test_batch_run_request_defaults():
    req = BatchRunRequest(org_id="org1", batch_id="b1", agent_type="inbound")
    assert req.concurrency == 5


def test_batch_run_request_concurrency_bounds():
    with pytest.raises(Exception):
        BatchRunRequest(org_id="org1", batch_id="b1", agent_type="inbound", concurrency=0)
    with pytest.raises(Exception):
        BatchRunRequest(org_id="org1", batch_id="b1", agent_type="inbound", concurrency=21)


# ---------------------------------------------------------------------------
# BatchWorker.run
# ---------------------------------------------------------------------------

def test_batch_worker_run_starts_thread():
    worker = BatchWorker()
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = False

    with patch("utils.batching.threading.Thread", return_value=mock_thread) as mock_cls:
        result = worker.run("org1", "batch1", "inbound", 3)

    mock_thread.start.assert_called_once()
    assert result["status"] == "success"
    assert "3" in result["message"]


def test_batch_worker_run_already_running_raises():
    worker = BatchWorker()
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    worker._runners["batch1"] = {"thread": mock_thread, "stop_event": threading.Event(), "concurrency": 3}

    with pytest.raises(HTTPException) as exc_info:
        worker.run("org1", "batch1", "inbound", 3)
    assert exc_info.value.status_code == 400


def test_batch_worker_run_restarts_dead_thread():
    worker = BatchWorker()
    dead_thread = MagicMock()
    dead_thread.is_alive.return_value = False
    worker._runners["batch1"] = {"thread": dead_thread, "stop_event": threading.Event(), "concurrency": 3}

    new_thread = MagicMock()
    with patch("utils.batching.threading.Thread", return_value=new_thread):
        result = worker.run("org1", "batch1", "inbound", 5)

    new_thread.start.assert_called_once()
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# BatchWorker.stop
# ---------------------------------------------------------------------------

def test_batch_worker_stop_success():
    worker = BatchWorker()
    stop_event = threading.Event()
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    worker._runners["batch1"] = {"thread": mock_thread, "stop_event": stop_event, "concurrency": 3}

    result = worker.stop("batch1")
    assert result["status"] == "success"
    assert stop_event.is_set()


def test_batch_worker_stop_not_running_raises():
    worker = BatchWorker()
    with pytest.raises(HTTPException) as exc_info:
        worker.stop("nonexistent_batch")
    assert exc_info.value.status_code == 400


def test_batch_worker_stop_dead_thread_raises():
    worker = BatchWorker()
    dead_thread = MagicMock()
    dead_thread.is_alive.return_value = False
    worker._runners["batch1"] = {"thread": dead_thread, "stop_event": threading.Event(), "concurrency": 3}

    with pytest.raises(HTTPException) as exc_info:
        worker.stop("batch1")
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# BatchWorker._cleanup_runner
# ---------------------------------------------------------------------------

def test_cleanup_runner_removes_entry():
    worker = BatchWorker()
    worker._runners["batch1"] = {"thread": MagicMock(), "stop_event": threading.Event()}
    worker._cleanup_runner("batch1")
    assert "batch1" not in worker._runners


def test_cleanup_runner_missing_key_safe():
    worker = BatchWorker()
    worker._cleanup_runner("nonexistent")  # Should not raise


# ---------------------------------------------------------------------------
# BatchWorker._run_worker — unit test with mocked dependencies
# ---------------------------------------------------------------------------

def test_run_worker_no_agent_config():
    worker = BatchWorker()
    with patch("utils.batching.fetch_batch_agent_call_config", return_value=None), \
         patch("utils.batching.finalize_batch_execution") as mock_finalize:
        stop_event = threading.Event()
        worker._run_worker("org1", "batch1", "inbound", 2, stop_event)
    mock_finalize.assert_called_once_with(org_id="org1", batch_id="batch1", stopped=False)


def test_run_worker_no_agent_id():
    worker = BatchWorker()
    config = {"agent_id": "", "caller_id": "+1234"}
    with patch("utils.batching.fetch_batch_agent_call_config", return_value=config), \
         patch("utils.batching.finalize_batch_execution") as mock_finalize:
        stop_event = threading.Event()
        worker._run_worker("org1", "batch1", "inbound", 2, stop_event)
    mock_finalize.assert_called_once_with(org_id="org1", batch_id="batch1", stopped=False)


def test_run_worker_no_contacts():
    worker = BatchWorker()
    config = {"agent_id": "agent_abc", "caller_id": "+1234"}
    with patch("utils.batching.fetch_batch_agent_call_config", return_value=config), \
         patch("utils.batching.claim_next_batch_contact", return_value=None), \
         patch("utils.batching.finalize_batch_execution") as mock_finalize:
        stop_event = threading.Event()
        worker._run_worker("org1", "batch1", "inbound", 2, stop_event)
    mock_finalize.assert_called_once_with(org_id="org1", batch_id="batch1", stopped=False)


def test_run_worker_stop_event_sets_stopped():
    worker = BatchWorker()
    config = {"agent_id": "agent_abc", "caller_id": "+1234"}
    stop_event = threading.Event()
    stop_event.set()  # Pre-set stop

    with patch("utils.batching.fetch_batch_agent_call_config", return_value=config), \
         patch("utils.batching.claim_next_batch_contact", return_value=None), \
         patch("utils.batching.finalize_batch_execution") as mock_finalize:
        worker._run_worker("org1", "batch1", "inbound", 2, stop_event)

    mock_finalize.assert_called_once_with(org_id="org1", batch_id="batch1", stopped=True)


def test_run_worker_missing_contact_number():
    """Contact with empty number → report error, continue to next contact."""
    worker = BatchWorker()
    config = {"agent_id": "agent_abc", "caller_id": "+1234"}

    contacts_iter = iter([
        {"row_number": 1, "contact_number": ""},  # bad
        None,  # end
    ])

    with patch("utils.batching.fetch_batch_agent_call_config", return_value=config), \
         patch("utils.batching.claim_next_batch_contact", side_effect=contacts_iter), \
         patch("utils.batching.report_batch_contact_result") as mock_report, \
         patch("utils.batching.finalize_batch_execution"):
        stop_event = threading.Event()
        worker._run_worker("org1", "batch1", "inbound", 2, stop_event)

    mock_report.assert_called_once_with(
        org_id="org1",
        batch_id="batch1",
        row_number=1,
        ok=False,
        error="Missing contact_number",
    )


# ---------------------------------------------------------------------------
# create_batch_router
# ---------------------------------------------------------------------------

def test_create_batch_router_returns_router():
    router = create_batch_router()
    assert router is not None
    routes = [r.path for r in router.routes]
    assert "/outbound/batch/run/" in routes
    assert "/outbound/batch/stop/" in routes
