"""Contract tests for server.py: error containment, load-shedding, readiness.

Run: python3 -m pytest ai4bharat_stt_server/test_server_contracts.py

NeMo and the .nemo checkpoints are not needed -- the NeMo import surface is
stubbed and a fake model is injected, so these run on CPU anywhere. They cover
the failure modes from SCALABILITY_OPS_ISSUES.md that are otherwise only
reachable under production load:

  O1  worker survives an inference exception (pre-fix: thread died, request hung)
  S8  requests past their deadline are dropped un-inferred
  S7  the response path holds no threadpool thread per in-flight request
  O5  /health is 503 until loaded, 503 again when the worker heartbeat is stale
  O3  /metrics exposes counters in Prometheus text format
  O7  shutdown drains the queue instead of letting callers time out
      oversize -> 413, malformed -> 400 (pre-fix: 500 + traceback)
"""

import base64
import sys
import threading
import time
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

# --- stub the NeMo import surface before importing server ---------------------

_nemo = types.ModuleType("nemo")
_collections = types.ModuleType("nemo.collections")
_asr = types.ModuleType("nemo.collections.asr")
_models = types.ModuleType("nemo.collections.asr.models")


class _FakeASRModel:
    @staticmethod
    def restore_from(*_args, **_kwargs):
        raise AssertionError("tests inject models directly; restore_from must not run")


class _EncDecHybridRNNTCTCBPEModel(_FakeASRModel):
    pass


_models.EncDecHybridRNNTCTCBPEModel = _EncDecHybridRNNTCTCBPEModel
_asr.models = types.SimpleNamespace(ASRModel=_FakeASRModel)
_collections.asr = _asr
_nemo.collections = _collections
sys.modules.setdefault("nemo", _nemo)
sys.modules.setdefault("nemo.collections", _collections)
sys.modules.setdefault("nemo.collections.asr", _asr)
sys.modules.setdefault("nemo.collections.asr.models", _models)

import server  # noqa: E402


SAMPLE_RATE = server.TARGET_SAMPLE_RATE


def _pcm_b64(seconds: float = 1.0) -> str:
    samples = np.zeros(int(SAMPLE_RATE * seconds), dtype=np.int16)
    return base64.b64encode(samples.tobytes()).decode()


def _audio(seconds: float = 1.0) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_state():
    """Fresh queues, stats and shutdown flag around every test."""
    server._shutdown.clear()
    server.main_request_queue = server.queue.Queue(maxsize=server.QUEUE_MAXSIZE)
    server.bhili_request_queue = server.queue.Queue(maxsize=server.QUEUE_MAXSIZE)
    server.main_stats = server.WorkerStats("main")
    server.bhili_stats = server.WorkerStats("bhili")
    server.main_model = object()
    server.bhili_model = None
    for key in server.rejected_counts:
        server.rejected_counts[key] = 0
    yield
    server._shutdown.set()
    time.sleep(0.05)


def _run_worker(infer_fn):
    """Start a batch worker against the current main queue; returns a stopper."""
    thread = threading.Thread(
        target=server.batch_worker,
        args=(server.main_request_queue, infer_fn, server.main_stats),
        daemon=True,
    )
    thread.start()
    deadline = time.time() + 2
    while not server.main_stats.started and time.time() < deadline:
        time.sleep(0.01)
    return thread


def _client() -> TestClient:
    """TestClient WITHOUT entering the context manager: that would run the
    lifespan, which tries to restore a real .nemo checkpoint. Requests still
    work -- TestClient spins up a portal per call."""
    return TestClient(server.app)


# --- O1: the worker must survive inference failures --------------------------


def test_worker_survives_inference_exception():
    """Pre-fix this hung forever: the exception killed the thread and the
    untimed queue.get never returned."""
    calls = {"n": 0}

    def flaky(audio_arrays, _language_ids):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA error: out of memory")
        return ["ok"] * len(audio_arrays)

    _run_worker(flaky)
    client = _client()
    first = client.post("/transcribe", json={"audio_b64": _pcm_b64(), "language_id": "hi"})
    second = client.post("/transcribe", json={"audio_b64": _pcm_b64(), "language_id": "hi"})

    assert first.status_code == 200
    assert first.json() == {"text": ""}          # failed batch answered, not hung
    assert second.status_code == 200
    assert second.json() == {"text": "ok"}       # worker still alive afterwards
    assert server.main_stats.failures == 1
    assert server.main_stats.alive()


def test_health_degraded_and_503_when_worker_heartbeat_is_stale():
    server.main_stats.started = True
    server.main_stats.last_beat = time.monotonic() - (server.WORKER_STALL_SECONDS + 5)
    client = _client()
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["main_worker_alive"] is False


def test_transcribe_refuses_when_worker_is_not_alive():
    client = _client()
    response = client.post("/transcribe", json={"audio_b64": _pcm_b64()})
    assert response.status_code == 503
    assert server.rejected_counts["unavailable"] == 1


# --- S8: stale requests are dropped before inference -------------------------


def test_stale_requests_dropped_before_inference():
    seen = {"batches": 0}

    def spy(audio_arrays, _language_ids):
        seen["batches"] += 1
        return ["should not happen"] * len(audio_arrays)

    expired = {
        "audio_np": _audio(),
        "language_id": "hi",
        "loop": None,
        "future": None,
        "replied": True,          # already answered; _reply must be a no-op
        "deadline": time.monotonic() - 1.0,
        "enqueued_at": time.monotonic() - 11.0,
    }
    server.main_request_queue.put_nowait(expired)
    _run_worker(spy)
    time.sleep(0.4)

    assert seen["batches"] == 0, "expired request was sent to the model"
    assert server.main_stats.stale_dropped == 1
    assert server.main_stats.alive()


def test_queue_full_sheds_with_503():
    server.main_request_queue = server.queue.Queue(maxsize=1)
    server.main_stats.started = True
    server.main_stats.beat()
    server.main_request_queue.put_nowait({"filler": True})

    client = _client()
    response = client.post("/transcribe", json={"audio_b64": _pcm_b64()})
    assert response.status_code == 503
    assert response.json()["detail"] == "STT queue is full"
    assert server.rejected_counts["queue_full"] == 1


# --- input guards ------------------------------------------------------------


def test_oversized_payload_rejected_with_413():
    server.main_stats.started = True
    server.main_stats.beat()
    oversized = "A" * (server.MAX_AUDIO_B64_CHARS + 4)
    client = _client()
    response = client.post("/transcribe", json={"audio_b64": oversized})
    assert response.status_code == 413
    assert server.rejected_counts["too_large"] == 1


def test_malformed_payload_rejected_with_400():
    """Odd byte count used to raise ValueError from np.frombuffer -> 500."""
    server.main_stats.started = True
    server.main_stats.beat()
    odd = base64.b64encode(b"\x01\x02\x03").decode()
    client = _client()
    response = client.post("/transcribe", json={"audio_b64": odd})
    assert response.status_code == 400
    assert "16-bit PCM" in response.json()["detail"]
    assert server.rejected_counts["bad_request"] == 1


def test_decode_rejects_bad_base64():
    with pytest.raises(ValueError, match="not valid base64"):
        server._decode_audio_b64("!!!! not base64 !!!!")


# --- O5 / O3: readiness + metrics -------------------------------------------


def test_health_is_503_while_model_is_loading():
    server.main_model = None
    client = _client()
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "loading"
    assert body["ready"] is False
    assert body["main_loaded"] is False       # key kept for setup.sh


def test_metrics_exposes_counters_in_prometheus_format():
    server.main_stats.started = True
    server.main_stats.beat()
    server.main_stats.requests = 7
    server.main_stats.audio_seconds = 12.0
    server.main_stats.infer_seconds = 3.0
    client = _client()
    body = client.get("/metrics").text

    assert 'stt_requests_total{worker="main"} 7' in body
    assert 'stt_audio_seconds_total{worker="main"} 12.000' in body
    assert 'stt_infer_seconds_total{worker="main"} 3.000' in body   # RTFx = 4.0x
    assert 'stt_worker_up{worker="main"} 1' in body
    assert 'stt_rejected_total{reason="queue_full"} 0' in body
    assert "stt_ready 1" in body


# --- O7: shutdown drains instead of stranding callers ------------------------


def test_shutdown_drains_pending_requests():
    replied = []

    class _FakeLoop:
        def call_soon_threadsafe(self, fn, *args):
            replied.append(args[-1])

    item = {
        "audio_np": _audio(),
        "language_id": "hi",
        "loop": _FakeLoop(),
        "future": object(),
        "replied": False,
        "deadline": time.monotonic() + 10,
        "enqueued_at": time.monotonic(),
    }
    server.main_request_queue.put_nowait(item)
    server._drain_queue(server.main_request_queue)

    assert replied == [""]
    assert item["replied"] is True
    assert server.main_request_queue.qsize() == 0
