import asyncio
import base64
import gc
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# The normal venv intentionally keeps CPU ONNX Runtime. LIVE auto-language
# opts into the already-vetted, same-version GPU package before NeMo or the HF
# remote model can import onnxruntime.
if (
    os.environ.get("ENABLE_AUTO_LANGUAGE", "false").strip().lower()
    in {"1", "true", "yes", "on"}
):
    bundled_ort_gpu = Path(__file__).resolve().parent / ".onnxruntime-gpu"
    if bundled_ort_gpu.is_dir():
        sys.path.insert(0, str(bundled_ort_gpu))

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

logger = logging.getLogger(__name__)

# =========================
# FastAPI setup
# =========================

app = FastAPI()

# =========================
# Request/Response Models
# =========================

class TranscribeRequest(BaseModel):
    audio_b64: str
    language_id: str = "hi"


class TranscribeResponse(BaseModel):
    text: str


class ShadowLanguageProbeRequest(BaseModel):
    audio_b64: str
    session_id: str = "unknown"
    current_language: str
    candidate_languages: list[str] = Field(
        default_factory=lambda: ["hi", "kn", "mr", "ta", "te"]
    )
    min_duration_ms: int | None = None
    require_cuda: bool = False


class ShadowLanguageProbeResponse(BaseModel):
    predicted_language: str | None
    top_score: float | None
    second_score: float | None
    margin: float | None
    confidence: float | None
    preprocessing_ms: float
    inference_ms: float
    encoder_ms: float | None
    ctc_ms: float | None
    probe_ms: float | None
    reason: str | None
    top_candidates: list[dict]
    device: str
    gpu_device: str | None
    providers: list[str]

# =========================
# Model loading
# =========================

TARGET_SAMPLE_RATE = 16000
MIN_SAMPLES = 1600
QUEUE_MAXSIZE = 256
MAX_BATCH_SIZE = 16
BATCH_TIMEOUT = 0.100  # 100 ms

# Set to "yes" or "no" in .env
BHILI_ENABLE = os.environ.get("BHILI_ENABLE", "no").strip().lower()
ENABLE_AUTO_LANGUAGE_SHADOW = (
    os.environ.get("ENABLE_AUTO_LANGUAGE_SHADOW", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)
ENABLE_AUTO_LANGUAGE = (
    os.environ.get("ENABLE_AUTO_LANGUAGE", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)
AUTO_LANGUAGE_DEVICE = os.environ.get("AUTO_LANGUAGE_DEVICE", "cuda").strip().lower()
AUTO_LANGUAGE_MIN_DURATION_MS = int(
    os.environ.get("AUTO_LANGUAGE_MIN_DURATION_MS", "2000")
)
AUTO_LANGUAGE_SHADOW_MODEL_ID = os.environ.get(
    "AUTO_LANGUAGE_SHADOW_MODEL_ID",
    "ai4bharat/indic-conformer-600m-multilingual",
)
AUTO_LANGUAGE_CANDIDATE_LANGUAGES = tuple(
    language.strip()
    for language in os.environ.get(
        "AUTO_LANGUAGE_CANDIDATE_LANGUAGES",
        "hi,kn,mr,ta,te",
    ).split(",")
    if language.strip()
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
main_model = None
bhili_model = None
shadow_model = None
shadow_probes = {}
shadow_probe_lock = threading.Lock()
auto_language_runtime_enabled = False
auto_language_runtime_error = None
auto_language_providers = []
auto_language_gpu_device = None
auto_language_gpu_memory_mb = None
shadow_model_device = "cpu"


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


def load_shadow_model(target_device: str):
    """Load the benchmarked HF model once on the validated target device."""

    from transformers import AutoModel

    return (
        AutoModel.from_pretrained(
            AUTO_LANGUAGE_SHADOW_MODEL_ID,
            trust_remote_code=True,
        )
        .to(torch.device(target_device))
        .eval()
    )


def _available_ort_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


def _model_ort_providers(model) -> list[str]:
    providers: set[str] = set()
    sessions = getattr(model, "models", {})
    if isinstance(sessions, dict):
        for session in sessions.values():
            getter = getattr(session, "get_providers", None)
            if callable(getter):
                providers.update(getter())
    return sorted(providers)


def _gpu_free_memory_mb() -> float:
    free_bytes, _ = torch.cuda.mem_get_info()
    return free_bytes / (1024 * 1024)


def _validate_live_cuda_environment() -> None:
    if AUTO_LANGUAGE_DEVICE != "cuda":
        raise RuntimeError("AUTO_LANGUAGE_DEVICE must be cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    providers = _available_ort_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. Restart with "
            "PYTHONPATH=$PWD/.onnxruntime-gpu:$PYTHONPATH"
        )


def _warm_up_live_probe() -> None:
    warmup_audio = np.random.default_rng(42).normal(0.0, 0.01, TARGET_SAMPLE_RATE * 2)
    _run_shadow_probe(
        warmup_audio.astype(np.float32),
        list(AUTO_LANGUAGE_CANDIDATE_LANGUAGES),
        0,
    )
    torch.cuda.synchronize()


def _get_shadow_probe(candidate_languages: list[str], min_duration_ms: int):
    from indicconformer_language_probe import (
        IndicConformerLanguageProbe,
        LanguageProbeConfig,
    )

    languages = tuple(candidate_languages)
    cache_key = (languages, min_duration_ms)
    probe = shadow_probes.get(cache_key)
    if probe is None:
        probe = IndicConformerLanguageProbe(
            shadow_model,
            LanguageProbeConfig(
                min_probe_duration_ms=min_duration_ms,
                margin_threshold=None,
                scoring_method="normalized_ctc_score",
            ),
            languages=languages,
        )
        shadow_probes[cache_key] = probe
    return probe


def _run_shadow_probe(
    audio_np: np.ndarray,
    candidate_languages: list[str],
    min_duration_ms: int,
) -> dict:
    if shadow_model is None:
        raise RuntimeError("Automatic-language shadow model is unavailable")

    preprocessing_started = time.perf_counter()
    wav = torch.from_numpy(audio_np).unsqueeze(0).to(torch.device(shadow_model_device))
    preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000
    started = time.perf_counter()
    # The HF probe itself performs one shared encoder and one shared CTC pass.
    # This cannot reuse the production NeMo model.transcribe() encoder tensor:
    # the two installed model APIs/checkpoint formats expose incompatible contracts.
    with shadow_probe_lock, torch.no_grad():
        probe = _get_shadow_probe(candidate_languages, min_duration_ms)
        result = probe.detect_language(
            wav,
            scoring_method="normalized_ctc_score",
        )
    inference_ms = (time.perf_counter() - started) * 1000
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError("Shadow endpoint accepts one audio item per request")
        result = result[0]
    return {
        "predicted_language": result.get("top_language"),
        "top_score": result.get("top_score"),
        "second_score": result.get("second_score"),
        "margin": result.get("margin"),
        "confidence": result.get("confidence"),
        "preprocessing_ms": preprocessing_ms,
        "inference_ms": inference_ms,
        "encoder_ms": result.get("encoder_ms"),
        "ctc_ms": result.get("ctc_ms"),
        "probe_ms": result.get("probe_ms"),
        "reason": result.get("reason"),
        "top_candidates": list(result.get("candidates") or [])[:3],
        "device": "cuda" if shadow_model_device.startswith("cuda") else "cpu",
        "gpu_device": auto_language_gpu_device,
        "providers": list(auto_language_providers or _model_ort_providers(shadow_model)),
    }


def _is_bhili_language(language_id: str) -> bool:
    return (language_id or "").strip().lower() in {"bhb", "bhili"}


def _nemo_language_id(language_id: str) -> str:
    if _is_bhili_language(language_id):
        return "mr"
    return language_id or "hi"


def _decode_audio_b64(audio_b64: str) -> np.ndarray:
    audio_bytes = base64.b64decode(audio_b64)
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _enqueue_request(request_queue: queue.Queue, audio_np: np.ndarray, language_id: str) -> queue.Queue:
    response_queue = queue.Queue(maxsize=1)
    request_item = {
        "audio_np": audio_np,
        "language_id": language_id,
        "response_queue": response_queue,
    }

    try:
        request_queue.put(request_item, timeout=1.0)
    except queue.Full:
        raise HTTPException(status_code=503, detail="STT queue is full")

    return response_queue

# =========================
# Queues and batching config
# =========================

main_request_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
bhili_request_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

# =========================
# Batcher + worker thread
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
    # Concurrent sessions may now have different active languages. Never
    # decode an entire mixed-language batch using language_ids[0].
    results = [""] * len(audio_arrays)
    grouped_indices: dict[str, list[int]] = {}
    for index, language_id in enumerate(language_ids):
        grouped_indices.setdefault(language_id or "hi", []).append(index)
    for language_id, indices in grouped_indices.items():
        group_results = _transcribe_batch(
            main_model,
            [audio_arrays[index] for index in indices],
            language_id,
        )
        for index, text in zip(indices, group_results):
            results[index] = text
    return results


def bhili_infer(audio_arrays, language_ids):
    return _transcribe_batch(
        bhili_model,
        audio_arrays,
        _nemo_language_id(language_ids[0] if language_ids else "bhb"),
    )


def batch_worker(request_queue, infer_fn):
    """
    Collects requests, batches them, runs the model,
    and returns results to waiting callers.
    """
    while True:
        batch = []
        start = time.time()

        # Collect batch
        while len(batch) < MAX_BATCH_SIZE:
            remaining = BATCH_TIMEOUT - (time.time() - start)
            if remaining <= 0:
                break

            try:
                item = request_queue.get(timeout=remaining)
                batch.append(item)
            except queue.Empty:
                break

        if not batch:
            continue

        # Unpack batch
        audio_arrays = [item["audio_np"] for item in batch]
        language_ids = [item["language_id"] for item in batch]

        transcriptions = infer_fn(audio_arrays, language_ids)

        # Return results
        for item, text in zip(batch, transcriptions):
            item["response_queue"].put(text)


def _start_workers():
    threading.Thread(
        target=batch_worker,
        args=(main_request_queue, main_infer),
        daemon=True,
    ).start()
    if BHILI_ENABLE == "yes":
        threading.Thread(
            target=batch_worker,
            args=(bhili_request_queue, bhili_infer),
            daemon=True,
        ).start()


@app.on_event("startup")
async def startup_event():
    global main_model, bhili_model, shadow_model, shadow_model_device
    global auto_language_runtime_enabled, auto_language_runtime_error
    global auto_language_providers, auto_language_gpu_device
    global auto_language_gpu_memory_mb

    main_model = load_main_model()
    if BHILI_ENABLE == "yes":
        bhili_model = load_bhili_model()
    else:
        bhili_model = None
    shadow_model = None
    auto_language_runtime_enabled = False
    auto_language_runtime_error = None
    auto_language_providers = []
    auto_language_gpu_device = None
    auto_language_gpu_memory_mb = None
    shadow_probes.clear()

    if ENABLE_AUTO_LANGUAGE:
        try:
            _validate_live_cuda_environment()
            memory_before_mb = _gpu_free_memory_mb()
            shadow_model_device = "cuda:0"
            shadow_model = load_shadow_model(shadow_model_device)
            auto_language_providers = _model_ort_providers(shadow_model)
            if "CUDAExecutionProvider" not in auto_language_providers:
                raise RuntimeError(
                    "HF model ONNX sessions did not activate CUDAExecutionProvider: "
                    f"{auto_language_providers}"
                )
            auto_language_gpu_device = torch.cuda.get_device_name(0)
            _warm_up_live_probe()
            memory_after_mb = _gpu_free_memory_mb()
            auto_language_gpu_memory_mb = max(0.0, memory_before_mb - memory_after_mb)
            auto_language_runtime_enabled = True
            logger.info(
                "Auto-language probe device: CUDA | cuda_device=%s providers=%s "
                "model=%s loaded_successfully=true gpu_memory_mb=%.1f",
                auto_language_gpu_device,
                auto_language_providers,
                AUTO_LANGUAGE_SHADOW_MODEL_ID,
                auto_language_gpu_memory_mb,
            )
        except Exception as exc:
            auto_language_runtime_error = str(exc)
            shadow_model = None
            logger.exception(
                "LIVE auto-language initialization failed safely; feature disabled "
                "and explicit-language ASR remains available: %s",
                exc,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if (
        shadow_model is None
        and ENABLE_AUTO_LANGUAGE_SHADOW
    ):
        try:
            providers = _available_ort_providers()
            shadow_model_device = (
                "cuda:0"
                if torch.cuda.is_available() and "CUDAExecutionProvider" in providers
                else "cpu"
            )
            shadow_model = load_shadow_model(shadow_model_device)
            auto_language_providers = _model_ort_providers(shadow_model)
            auto_language_gpu_device = (
                torch.cuda.get_device_name(0)
                if shadow_model_device.startswith("cuda")
                else None
            )
            logger.info(
                "Automatic-language shadow model loaded: %s device=%s providers=%s",
                AUTO_LANGUAGE_SHADOW_MODEL_ID,
                shadow_model_device,
                auto_language_providers,
            )
        except Exception:
            shadow_model = None
            logger.exception(
                "Automatic-language shadow model failed to load; "
                "explicit-language ASR remains available"
            )
    _start_workers()

# =========================
# Routes
# =========================

@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    audio_np = _decode_audio_b64(request.audio_b64)
    response_queue = _enqueue_request(main_request_queue, audio_np, request.language_id)
    result = await asyncio.to_thread(response_queue.get)
    return TranscribeResponse(text=result)


@app.post("/transcribe/bhili", response_model=TranscribeResponse)
async def transcribe_bhili(request: TranscribeRequest):
    if BHILI_ENABLE != "yes":
        raise HTTPException(status_code=503, detail="Bhili model is disabled")
    if bhili_model is None:
        raise HTTPException(status_code=503, detail="Bhili model not loaded")

    audio_np = _decode_audio_b64(request.audio_b64)
    response_queue = _enqueue_request(bhili_request_queue, audio_np, request.language_id)
    result = await asyncio.to_thread(response_queue.get)
    return TranscribeResponse(text=result)


@app.post(
    "/shadow/language-probe",
    response_model=ShadowLanguageProbeResponse,
)
async def shadow_language_probe(request: ShadowLanguageProbeRequest):
    return await _handle_language_probe(request, live_request=False)


@app.post(
    "/language-probe",
    response_model=ShadowLanguageProbeResponse,
)
async def live_language_probe(request: ShadowLanguageProbeRequest):
    return await _handle_language_probe(request, live_request=True)


async def _handle_language_probe(
    request: ShadowLanguageProbeRequest,
    *,
    live_request: bool,
):
    if live_request and not ENABLE_AUTO_LANGUAGE:
        raise HTTPException(
            status_code=404,
            detail="LIVE automatic-language mode is disabled",
        )
    if not live_request and not (
        ENABLE_AUTO_LANGUAGE_SHADOW or ENABLE_AUTO_LANGUAGE
    ):
        raise HTTPException(
            status_code=404,
            detail="Automatic-language probe is disabled",
        )
    if (live_request or request.require_cuda) and not auto_language_runtime_enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "LIVE automatic-language CUDA runtime is unavailable: "
                f"{auto_language_runtime_error or 'not initialized'}"
            ),
        )
    if shadow_model is None:
        raise HTTPException(
            status_code=503,
            detail="Automatic-language shadow model is unavailable",
        )
    audio_np = _decode_audio_b64(request.audio_b64)
    duration_ms = len(audio_np) * 1000.0 / TARGET_SAMPLE_RATE
    min_duration_ms = (
        request.min_duration_ms
        if request.min_duration_ms is not None
        else AUTO_LANGUAGE_MIN_DURATION_MS
    )
    if min_duration_ms < 0:
        raise HTTPException(
            status_code=422,
            detail="min_duration_ms must be non-negative",
        )
    if duration_ms < min_duration_ms:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Shadow probe audio is {duration_ms:.0f} ms; "
                f"minimum is {min_duration_ms} ms"
            ),
        )
    if not request.candidate_languages:
        raise HTTPException(
            status_code=422,
            detail="candidate_languages must not be empty",
        )
    try:
        result = await asyncio.to_thread(
            _run_shadow_probe,
            audio_np,
            request.candidate_languages,
            min_duration_ms,
        )
    except Exception as exc:
        logger.exception(
            "Automatic-language probe failed for session=%s current_lang=%s",
            request.session_id,
            request.current_language,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ShadowLanguageProbeResponse(**result)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": device,
        "bhili_enabled": BHILI_ENABLE,
        "main_loaded": main_model is not None,
        "bhili_loaded": bhili_model is not None,
        "main_queue_size": main_request_queue.qsize(),
        "bhili_queue_size": bhili_request_queue.qsize(),
        "max_batch_size": MAX_BATCH_SIZE,
        "batch_timeout_ms": int(BATCH_TIMEOUT * 1000),
        "auto_language_requested": ENABLE_AUTO_LANGUAGE,
        "auto_language_runtime_enabled": auto_language_runtime_enabled,
        "auto_language_runtime_error": auto_language_runtime_error,
        "auto_language_device": (
            "cuda" if shadow_model_device.startswith("cuda") else "cpu"
        ),
        "auto_language_gpu_device": auto_language_gpu_device,
        "auto_language_providers": auto_language_providers,
        "auto_language_gpu_memory_mb": auto_language_gpu_memory_mb,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)