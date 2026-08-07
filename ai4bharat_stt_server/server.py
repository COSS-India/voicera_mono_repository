import asyncio
import base64
import binascii
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from pydantic import BaseModel
import uvicorn

import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from dotenv import load_dotenv

load_dotenv()

# =========================
# Request/Response Models
# =========================

class TranscribeRequest(BaseModel):
    audio_b64: str
    language_id: str = "hi"


class TranscribeResponse(BaseModel):
    text: str

# =========================
# Config
# =========================


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        logger.warning("Invalid {}, using default {}", name, default)
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        logger.warning("Invalid {}, using default {}", name, default)
        return default


TARGET_SAMPLE_RATE = 16000
MIN_SAMPLES = 1600

QUEUE_MAXSIZE = _env_int("STT_QUEUE_MAXSIZE", 256)
MAX_BATCH_SIZE = _env_int("STT_MAX_BATCH_SIZE", 16)
BATCH_TIMEOUT = _env_int("STT_BATCH_TIMEOUT_MS", 100) / 1000.0

# Longest single request accepted. 60 s of 16 kHz mono int16 = 1.92 MB raw,
# ~2.56 MB of base64. Anything larger is rejected with 413 before it can pin the
# GPU for minutes or blow up VRAM.
MAX_AUDIO_SECONDS = _env_float("STT_MAX_AUDIO_SECONDS", 60.0)
MAX_AUDIO_BYTES = int(MAX_AUDIO_SECONDS * TARGET_SAMPLE_RATE * 2)
MAX_AUDIO_B64_CHARS = ((MAX_AUDIO_BYTES + 2) // 3) * 4 + 16

# Matched to the voice server's aiohttp ClientTimeout(total=10). Queued items
# older than this are dropped un-inferred: under overload the GPU must not burn
# cycles on work nobody is waiting for.
REQUEST_DEADLINE_S = _env_float("STT_REQUEST_DEADLINE_S", 10.0)
# Handler-side ceiling, so a wedged CUDA call returns 504 instead of hanging.
REQUEST_TIMEOUT_S = REQUEST_DEADLINE_S + _env_float("STT_RESPONSE_GRACE_S", 5.0)

# A worker that has not touched its heartbeat in this long is reported unhealthy.
WORKER_STALL_SECONDS = _env_float("STT_WORKER_STALL_SECONDS", 60.0)
# Idle seconds before releasing cached CUDA memory (see _maybe_release_idle_memory).
IDLE_EMPTY_CACHE_SECONDS = _env_float("STT_IDLE_EMPTY_CACHE_SECONDS", 60.0)
LOG_SUMMARY_EVERY = _env_int("STT_LOG_SUMMARY_EVERY", 100)
MATMUL_PRECISION = (os.environ.get("STT_MATMUL_PRECISION") or "high").strip()

# Set to "yes" or "no" in .env
BHILI_ENABLE = os.environ.get("BHILI_ENABLE", "no").strip().lower()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
main_model = None
bhili_model = None

_shutdown = threading.Event()

# App-level rejection counters (queue-full, oversize, malformed, timeout).
rejected_counts = {
    "queue_full": 0,
    "too_large": 0,
    "bad_request": 0,
    "timeout": 0,
    "unavailable": 0,
}

# =========================
# Model loading
# =========================


def _required_model_path(env_var_name: str) -> Path:
    env_value = (os.environ.get(env_var_name) or "").strip()
    if not env_value:
        raise RuntimeError(
            f"Missing required environment variable: {env_var_name}. "
            f"Please set it in ai4bharat_stt_server/.env"
        )

    path = Path(env_value).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    else:
        path = path.resolve()

    if not path.is_file():
        raise RuntimeError(
            f"Invalid {env_var_name}: file not found at {path}. "
            "Please update ai4bharat_stt_server/.env"
        )

    return path


def load_main_model():
    model_path = _required_model_path("INDIC_NEMO_PATH")
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(model_path),
        map_location=torch.device(device),   # <-- add this
    )
    model = model.to(device)
    model.freeze()
    model.cur_decoder = "rnnt"
    return model


def load_bhili_model():
    model_path = _required_model_path("BHILI_NEMO_PATH")
    model = EncDecHybridRNNTCTCBPEModel.restore_from(
        str(model_path),
        map_location=torch.device(device),
    )
    model = model.to(device)
    model.freeze()
    model.cur_decoder = "rnnt"
    return model


def _is_bhili_language(language_id: str) -> bool:
    return (language_id or "").strip().lower() in {"bhb", "bhili"}


def _nemo_language_id(language_id: str) -> str:
    if _is_bhili_language(language_id):
        return "mr"
    return language_id or "hi"


def _decode_audio_b64(audio_b64: str) -> np.ndarray:
    """Decode base64 16 kHz mono int16 PCM. Raises ValueError on bad input.

    Runs off the event loop (see the request handlers): base64 decoding a
    multi-megabyte payload is CPU-bound and used to stall every other request
    sharing the worker process.
    """
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"audio_b64 is not valid base64: {exc}") from exc

    if len(audio_bytes) % 2 != 0:
        raise ValueError(
            "audio_b64 must decode to 16-bit PCM: got an odd number of bytes "
            f"({len(audio_bytes)})"
        )
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise ValueError(
            f"audio exceeds STT_MAX_AUDIO_SECONDS={MAX_AUDIO_SECONDS}: "
            f"{len(audio_bytes)} > {MAX_AUDIO_BYTES} bytes"
        )

    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

# =========================
# Queues, worker state
# =========================


class WorkerStats:
    """Liveness heartbeat + counters for one batch worker."""

    def __init__(self, name: str):
        self.name = name
        self.started = False
        self.last_beat = 0.0
        self.batches = 0
        self.requests = 0
        self.failures = 0
        self.stale_dropped = 0
        self.audio_seconds = 0.0
        self.infer_seconds = 0.0

    def beat(self) -> None:
        self.last_beat = time.monotonic()

    def alive(self) -> bool:
        if not self.started:
            return False
        return (time.monotonic() - self.last_beat) <= WORKER_STALL_SECONDS

    def rtfx(self) -> float:
        return self.audio_seconds / self.infer_seconds if self.infer_seconds > 0 else 0.0


main_request_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
bhili_request_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
main_stats = WorkerStats("main")
bhili_stats = WorkerStats("bhili")

# =========================
# Request plumbing
# =========================


def _settle(future: asyncio.Future, text: str) -> None:
    if not future.done():
        future.set_result(text)


def _reply(item: dict, text: str) -> None:
    """Complete a request's future from the worker thread. Idempotent."""
    if item["replied"]:
        return
    item["replied"] = True
    try:
        item["loop"].call_soon_threadsafe(_settle, item["future"], text)
    except RuntimeError:
        # Event loop already closed (shutdown race); the caller is gone anyway.
        pass


def _enqueue_request(
    request_queue: "queue.Queue", audio_np: np.ndarray, language_id: str
) -> asyncio.Future:
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    item = {
        "audio_np": audio_np,
        "language_id": language_id,
        "loop": loop,
        "future": future,
        "replied": False,
        "deadline": time.monotonic() + REQUEST_DEADLINE_S,
        "enqueued_at": time.monotonic(),
    }

    try:
        # put_nowait, not put(timeout=1.0): blocking here would stall the event
        # loop for a second per request once the queue is full. Shed instead.
        request_queue.put_nowait(item)
    except queue.Full:
        rejected_counts["queue_full"] += 1
        logger.warning("STT queue full (maxsize={}), shedding request", QUEUE_MAXSIZE)
        raise HTTPException(status_code=503, detail="STT queue is full")

    return future


async def _await_result(future: asyncio.Future) -> str:
    try:
        return await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        rejected_counts["timeout"] += 1
        logger.error(
            "STT request timed out after {}s waiting for a worker result", REQUEST_TIMEOUT_S
        )
        raise HTTPException(status_code=504, detail="STT inference timed out")

# =========================
# Inference
# =========================


def _transcribe_batch(model, audio_arrays, language_id: str):
    valid_indices = [i for i, arr in enumerate(audio_arrays) if len(arr) >= MIN_SAMPLES]
    if not valid_indices:
        return [""] * len(audio_arrays)

    valid_audio = [audio_arrays[i] for i in valid_indices]
    with torch.no_grad():
        transcriptions = model.transcribe(
            audio=valid_audio,
            batch_size=len(valid_audio),
            language_id=language_id,
        )[0]

    results = [""] * len(audio_arrays)
    for idx, text in zip(valid_indices, transcriptions):
        results[idx] = str(text).strip() if text is not None else ""
    return results


def main_infer(audio_arrays, language_ids):
    return _transcribe_batch(main_model, audio_arrays, language_ids[0])


def bhili_infer(audio_arrays, language_ids):
    return _transcribe_batch(
        bhili_model,
        audio_arrays,
        _nemo_language_id(language_ids[0] if language_ids else "bhb"),
    )

# =========================
# Batcher + worker thread
# =========================


def _collect_batch(request_queue: "queue.Queue", stats: WorkerStats) -> list:
    batch = []
    start = time.time()

    while len(batch) < MAX_BATCH_SIZE:
        remaining = BATCH_TIMEOUT - (time.time() - start)
        if remaining <= 0:
            break
        try:
            batch.append(request_queue.get(timeout=remaining))
        except queue.Empty:
            break

    stats.beat()
    return batch


def _drop_stale(batch: list, stats: WorkerStats) -> list:
    """Answer expired items immediately; return only the ones still awaited."""
    now = time.monotonic()
    fresh = [item for item in batch if item["deadline"] > now]
    stale = len(batch) - len(fresh)
    if stale:
        stats.stale_dropped += stale
        logger.warning(
            "{}: dropped {}/{} stale request(s) past their {}s deadline",
            stats.name,
            stale,
            len(batch),
            REQUEST_DEADLINE_S,
        )
        for item in batch:
            if item["deadline"] <= now:
                _reply(item, "")
    return fresh


def _release_cuda_cache() -> None:
    """Release cached-but-unused CUDA memory after a sustained idle period.

    PyTorch's caching allocator keeps the peak working set reserved forever,
    which is why nvidia-smi can show far more "used" memory per worker than a
    fresh process needs long after a burst has passed. This used to run once a
    second, which handed segments back to the driver during every short gap
    between utterances and made the next request pay cudaMalloc + fragmentation.
    Now it only fires after STT_IDLE_EMPTY_CACHE_SECONDS of continuous idle.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _log_summary(stats: WorkerStats) -> None:
    logger.info(
        "{}: {} batches, {} requests, {} failures, {} stale-dropped | "
        "audio={:.1f}s infer={:.1f}s RTFx={:.1f}x queue={}",
        stats.name,
        stats.batches,
        stats.requests,
        stats.failures,
        stats.stale_dropped,
        stats.audio_seconds,
        stats.infer_seconds,
        stats.rtfx(),
        _queue_for(stats).qsize(),
    )


def _queue_for(stats: WorkerStats) -> "queue.Queue":
    return bhili_request_queue if stats is bhili_stats else main_request_queue


def batch_worker(request_queue, infer_fn, stats: WorkerStats):
    """Collect requests, batch them, run the model, return results to callers.

    Never exits on an inference error. A single unhandled exception here used to
    kill the thread, after which every request blocked forever on an untimed
    queue.get and the service 503'd permanently while still reporting healthy.
    """
    stats.started = True
    stats.beat()
    idle_since = time.monotonic()
    last_release = time.monotonic()

    while not _shutdown.is_set():
        batch = []
        try:
            batch = _collect_batch(request_queue, stats)

            if not batch:
                now = time.monotonic()
                if (
                    now - idle_since >= IDLE_EMPTY_CACHE_SECONDS
                    and now - last_release >= IDLE_EMPTY_CACHE_SECONDS
                ):
                    _release_cuda_cache()
                    last_release = now
                continue

            idle_since = time.monotonic()

            pending = _drop_stale(batch, stats)
            if not pending:
                continue

            audio_arrays = [item["audio_np"] for item in pending]
            language_ids = [item["language_id"] for item in pending]
            audio_seconds = sum(len(a) for a in audio_arrays) / TARGET_SAMPLE_RATE

            t0 = time.monotonic()
            transcriptions = infer_fn(audio_arrays, language_ids)
            infer_s = time.monotonic() - t0

            stats.batches += 1
            stats.requests += len(pending)
            stats.audio_seconds += audio_seconds
            stats.infer_seconds += infer_s
            stats.beat()

            for item, text in zip(pending, transcriptions):
                _reply(item, text)

            logger.debug(
                "{}: batch size={} audio={:.2f}s infer={:.0f}ms RTFx={:.1f}x "
                "queue_wait={:.0f}ms queue_depth={}",
                stats.name,
                len(pending),
                audio_seconds,
                infer_s * 1000,
                audio_seconds / infer_s if infer_s > 0 else 0.0,
                (t0 - min(i["enqueued_at"] for i in pending)) * 1000,
                request_queue.qsize(),
            )
            if LOG_SUMMARY_EVERY > 0 and stats.batches % LOG_SUMMARY_EVERY == 0:
                _log_summary(stats)

        except Exception as exc:  # noqa: BLE001 - worker must never die
            stats.failures += 1
            logger.opt(exception=True).error(
                "{}: batch of {} failed ({}); answering empty and continuing",
                stats.name,
                len(batch),
                exc,
            )
            for item in batch:
                _reply(item, "")
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            stats.beat()

    _drain_queue(request_queue)
    logger.info("{}: worker stopped", stats.name)


def _drain_queue(request_queue: "queue.Queue") -> None:
    """Answer everything still queued so callers do not wait out their timeout."""
    drained = 0
    while True:
        try:
            item = request_queue.get_nowait()
        except queue.Empty:
            break
        _reply(item, "")
        drained += 1
    if drained:
        logger.info("Drained {} queued request(s) on shutdown", drained)


def _start_workers():
    threading.Thread(
        target=batch_worker,
        args=(main_request_queue, main_infer, main_stats),
        name="stt-worker-main",
        daemon=True,
    ).start()
    if BHILI_ENABLE == "yes":
        threading.Thread(
            target=batch_worker,
            args=(bhili_request_queue, bhili_infer, bhili_stats),
            name="stt-worker-bhili",
            daemon=True,
        ).start()

# =========================
# Lifespan
# =========================


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global main_model, bhili_model

    logger.info(
        "STT starting | device={} bhili={} batch_size={} batch_timeout={}ms "
        "queue={} max_audio={}s deadline={}s",
        device,
        BHILI_ENABLE,
        MAX_BATCH_SIZE,
        int(BATCH_TIMEOUT * 1000),
        QUEUE_MAXSIZE,
        MAX_AUDIO_SECONDS,
        REQUEST_DEADLINE_S,
    )

    if torch.cuda.is_available():
        # TF32 matmuls on Ampere+; set STT_MATMUL_PRECISION=highest to revert.
        torch.set_float32_matmul_precision(MATMUL_PRECISION)
        logger.info("matmul precision set to {}", MATMUL_PRECISION)

    t0 = time.monotonic()
    main_model = load_main_model()
    logger.info("main model loaded in {:.1f}s", time.monotonic() - t0)

    if BHILI_ENABLE == "yes":
        t0 = time.monotonic()
        bhili_model = load_bhili_model()
        logger.info("bhili model loaded in {:.1f}s", time.monotonic() - t0)
    else:
        bhili_model = None

    _start_workers()
    logger.info("STT ready")

    try:
        yield
    finally:
        logger.info("STT shutting down; draining queues")
        _shutdown.set()
        _drain_queue(main_request_queue)
        _drain_queue(bhili_request_queue)
        _log_summary(main_stats)
        if BHILI_ENABLE == "yes":
            _log_summary(bhili_stats)


app = FastAPI(lifespan=lifespan)

# =========================
# Routes
# =========================


@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}


async def _handle_transcribe(request: TranscribeRequest, request_queue, stats) -> str:
    if len(request.audio_b64) > MAX_AUDIO_B64_CHARS:
        rejected_counts["too_large"] += 1
        logger.warning(
            "Rejected oversized request: {} b64 chars > {} (STT_MAX_AUDIO_SECONDS={})",
            len(request.audio_b64),
            MAX_AUDIO_B64_CHARS,
            MAX_AUDIO_SECONDS,
        )
        raise HTTPException(
            status_code=413,
            detail=f"audio_b64 exceeds STT_MAX_AUDIO_SECONDS={MAX_AUDIO_SECONDS}",
        )

    try:
        audio_np = await asyncio.to_thread(_decode_audio_b64, request.audio_b64)
    except ValueError as exc:
        rejected_counts["bad_request"] += 1
        logger.warning("Rejected malformed request: {}", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    if not stats.alive():
        rejected_counts["unavailable"] += 1
        logger.error("{}: worker not alive, refusing request", stats.name)
        raise HTTPException(status_code=503, detail="STT worker unavailable")

    future = _enqueue_request(request_queue, audio_np, request.language_id)
    return await _await_result(future)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    text = await _handle_transcribe(request, main_request_queue, main_stats)
    return TranscribeResponse(text=text)


@app.post("/transcribe/bhili", response_model=TranscribeResponse)
async def transcribe_bhili(request: TranscribeRequest):
    if BHILI_ENABLE != "yes":
        raise HTTPException(status_code=503, detail="Bhili model is disabled")
    if bhili_model is None:
        raise HTTPException(status_code=503, detail="Bhili model not loaded")

    text = await _handle_transcribe(request, bhili_request_queue, bhili_stats)
    return TranscribeResponse(text=text)


def _health_payload() -> dict:
    main_alive = main_stats.alive()
    bhili_expected = BHILI_ENABLE == "yes"
    bhili_alive = bhili_stats.alive() if bhili_expected else True

    loaded = main_model is not None and (not bhili_expected or bhili_model is not None)
    if not loaded:
        status = "loading"
    elif not (main_alive and bhili_alive):
        status = "degraded"
    else:
        status = "healthy"

    return {
        "status": status,
        "ready": status == "healthy",
        "device": device,
        "bhili_enabled": BHILI_ENABLE,
        "main_loaded": main_model is not None,
        "bhili_loaded": bhili_model is not None,
        "main_worker_alive": main_alive,
        "bhili_worker_alive": bhili_stats.alive(),
        "main_queue_size": main_request_queue.qsize(),
        "bhili_queue_size": bhili_request_queue.qsize(),
        "max_batch_size": MAX_BATCH_SIZE,
        "batch_timeout_ms": int(BATCH_TIMEOUT * 1000),
        "main_batch_failures": main_stats.failures,
        "bhili_batch_failures": bhili_stats.failures,
        "main_stale_dropped": main_stats.stale_dropped,
        "main_rtfx": round(main_stats.rtfx(), 2),
        "rejected": dict(rejected_counts),
    }


@app.get("/health")
def health():
    """200 only when loaded and both expected workers are beating.

    Returning 200 while the model was still loading (2-3 min) made this useless
    as a readiness probe, and returning 200 after the worker thread had died
    meant a wedged container was never recycled.
    """
    payload = _health_payload()
    code = 200 if payload["status"] == "healthy" else 503
    return JSONResponse(status_code=code, content=payload)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus text format. Deliberately dependency-free.

    RTFx (audio-seconds per wall-second) is the number that says whether the GPU
    is saturated: stt_audio_seconds_total / stt_infer_seconds_total.
    """
    lines = [
        "# HELP stt_requests_total Requests completed by the batch worker.",
        "# TYPE stt_requests_total counter",
        "# HELP stt_batches_total Inference batches executed.",
        "# TYPE stt_batches_total counter",
        "# HELP stt_batch_failures_total Batches that raised and were answered empty.",
        "# TYPE stt_batch_failures_total counter",
        "# HELP stt_stale_dropped_total Requests dropped un-inferred past their deadline.",
        "# TYPE stt_stale_dropped_total counter",
        "# HELP stt_audio_seconds_total Audio seconds submitted to the model.",
        "# TYPE stt_audio_seconds_total counter",
        "# HELP stt_infer_seconds_total Wall seconds spent inside the model.",
        "# TYPE stt_infer_seconds_total counter",
        "# HELP stt_queue_depth Current queue depth.",
        "# TYPE stt_queue_depth gauge",
        "# HELP stt_worker_up Worker heartbeat is fresh.",
        "# TYPE stt_worker_up gauge",
        "# HELP stt_rejected_total Requests rejected before inference.",
        "# TYPE stt_rejected_total counter",
        "# HELP stt_model_loaded Model checkpoint is loaded.",
        "# TYPE stt_model_loaded gauge",
        "# HELP stt_ready Service is loaded and workers are alive.",
        "# TYPE stt_ready gauge",
    ]

    for stats, request_queue in ((main_stats, main_request_queue), (bhili_stats, bhili_request_queue)):
        label = f'{{worker="{stats.name}"}}'
        lines += [
            f"stt_requests_total{label} {stats.requests}",
            f"stt_batches_total{label} {stats.batches}",
            f"stt_batch_failures_total{label} {stats.failures}",
            f"stt_stale_dropped_total{label} {stats.stale_dropped}",
            f"stt_audio_seconds_total{label} {stats.audio_seconds:.3f}",
            f"stt_infer_seconds_total{label} {stats.infer_seconds:.3f}",
            f"stt_queue_depth{label} {request_queue.qsize()}",
            f"stt_worker_up{label} {int(stats.alive())}",
        ]

    for reason, count in rejected_counts.items():
        lines.append(f'stt_rejected_total{{reason="{reason}"}} {count}')

    lines.append(f'stt_model_loaded{{model="main"}} {int(main_model is not None)}')
    lines.append(f'stt_model_loaded{{model="bhili"}} {int(bhili_model is not None)}')
    lines.append(f"stt_ready {int(_health_payload()['ready'])}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    num_workers = int(os.environ.get("STT_NUM_WORKERS", "4"))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        workers=num_workers,
    )
