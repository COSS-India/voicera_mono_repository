import os
import random
import threading
import queue
import time
import base64

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import torch
import nemo.collections.asr as nemo_asr

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

def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    nemo_path = "models/IndicConformer.nemo"

    model = nemo_asr.models.ASRModel.restore_from(restore_path=nemo_path)
    model = model.to(device)
    model.freeze()
    model.cur_decoder = "rnnt"
    return model

model = load_model()

# =========================
# Queues and batching config
# =========================

# Incoming transcription requests
request_queue = queue.Queue(maxsize=256)

# Dynamic batching params
MAX_BATCH_SIZE = 16
BATCH_TIMEOUT = 0.100  # 100 ms

# =========================
# Batcher + worker thread
# =========================

def batch_worker():
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
            if batch:
                print(len(batch), remaining)
            if remaining <= 0:
                print("out of time")
                break

            try:
                item = request_queue.get(timeout=remaining)
                batch.append(item)
                print("added to batch")
            except queue.Empty:
                break

        if not batch:
            continue

        # Unpack batch
        audio_arrays = [item["audio_np"] for item in batch]
        language_ids = [item["language_id"] for item in batch]

        # Run model (using first language_id for the batch - assumes homogeneous batch)
        with torch.no_grad():
            print(f"running {len(audio_arrays)}")
            transcriptions = model.transcribe(
                audio=audio_arrays,
                batch_size=len(audio_arrays),
                language_id=language_ids[0]
            )[0]

        # Return results
        for item, text in zip(batch, transcriptions):
            item["response_queue"].put(text)

# Start batch worker
threading.Thread(target=batch_worker, daemon=True).start()

# =========================
# Routes
# =========================

@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(request: TranscribeRequest):
    # Decode base64 to raw PCM bytes and convert to float32 numpy array
    audio_bytes = base64.b64decode(request.audio_b64)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Per-request response queue
    response_queue = queue.Queue(maxsize=1)

    # Enqueue work
    request_queue.put({
        "audio_np": audio_np,
        "language_id": request.language_id,
        "response_queue": response_queue
    })

    # Block until result is ready
    result = response_queue.get()
    return TranscribeResponse(text=result)


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)