import asyncio
import base64
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from dotenv import load_dotenv

load_dotenv()

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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
main_model = None
bhili_model = None


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
    return _transcribe_batch(main_model, audio_arrays, language_ids[0])


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
    global main_model, bhili_model

    main_model = load_main_model()
    if BHILI_ENABLE == "yes":
        bhili_model = load_bhili_model()
    else:
        bhili_model = None
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
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)