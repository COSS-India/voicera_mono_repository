"""Minimal REST-based Indic Conformer STT Server"""

import asyncio
import base64
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from pydantic import BaseModel
from transformers import AutoModel

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8001)
args = parser.parse_args()

TARGET_SAMPLE_RATE = 16000
MIN_SAMPLES = 1600

app = FastAPI()
model = None
bhili_model = None
device = None


class TranscribeRequest(BaseModel):
    audio_b64: str
    language_id: str = "hi"


class TranscribeResponse(BaseModel):
    text: str


def transcribe_sync(audio_np: np.ndarray, language_id: str) -> str:
    try:
        if len(audio_np) < MIN_SAMPLES:
            return ""
        
        wav = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            result = model(wav, language_id, "rnnt")
        
        if isinstance(result, str):
            return result.strip()
        elif isinstance(result, (list, tuple)) and result:
            return str(result[0]).strip()
        return str(result).strip()
        
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""


def _bhili_neemo_language_id(requested: str) -> str:
    """NeMo expects e.g. mr; clients may send bhb as the canonical Bhili code."""
    lid = (requested or "").strip()
    if lid.lower() == "bhb":
        return "mr"
    return lid if lid else "mr"


def _bhili_nemo_path() -> Path:
    env = os.environ.get("BHILI_NEMO_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent / "bhili-asr" / "bhili_asr_finetune_v1_updated.nemo"


def bhili_transcribe_sync(audio_np: np.ndarray, language_id: str) -> str:
    try:
        if len(audio_np) < MIN_SAMPLES:
            return ""

        wav = torch.from_numpy(audio_np).float().unsqueeze(0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            torchaudio.save(tmp_path, wav, TARGET_SAMPLE_RATE)
            neemo_lid = _bhili_neemo_language_id(language_id)
            with torch.no_grad():
                result = bhili_model.transcribe([tmp_path], language_id=neemo_lid)
            if result and result[0]:
                out = result[0][0]
            else:
                out = result
            return str(out).strip() if out is not None else ""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        print(f"Bhili transcription error: {e}")
        return ""


@app.on_event("startup")
async def load_model():
    global model, bhili_model, device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True
    ).to(device).eval()
    
    dummy = torch.zeros(1, 16000).to(device)
    with torch.no_grad():
        model(dummy, "hi", "rnnt")
    print("Model ready")

    nemo_path = _bhili_nemo_path()
    if not nemo_path.is_file():
        raise RuntimeError(
            f"Bhili model not found at {nemo_path}. "
            "Set BHILI_NEMO_PATH or place the .nemo file under ai4bharat_stt_server/bhili-asr/."
        )
    print(f"Loading Bhili ASR from {nemo_path}...")
    bhili_model = EncDecHybridRNNTCTCBPEModel.restore_from(str(nemo_path)).to(device).eval()
    print("Bhili ASR ready")


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    audio_bytes = base64.b64decode(request.audio_b64)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    text = await asyncio.get_event_loop().run_in_executor(
        None, transcribe_sync, audio_np, request.language_id
    )
    
    return TranscribeResponse(text=text)


@app.post("/transcribe/bhili", response_model=TranscribeResponse)
async def transcribe_bhili(request: TranscribeRequest):
    if bhili_model is None:
        raise HTTPException(status_code=503, detail="Bhili model not loaded")

    audio_bytes = base64.b64decode(request.audio_b64)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    text = await asyncio.get_event_loop().run_in_executor(
        None, bhili_transcribe_sync, audio_np, request.language_id
    )

    return TranscribeResponse(text=text)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": str(device),
        "bhili_loaded": bhili_model is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)