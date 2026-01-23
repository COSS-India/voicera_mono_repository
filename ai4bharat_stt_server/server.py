"""Minimal REST-based Indic Conformer STT Server"""

import asyncio
import base64
import argparse
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoModel

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8001)
parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
args = parser.parse_args()

TARGET_SAMPLE_RATE = 16000
MIN_SAMPLES = 1600

app = FastAPI()
model = None
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


@app.on_event("startup")
async def load_model():
    global model, device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True
    ).to(device).eval()
    
    # Print model memory information
    if torch.cuda.is_available():
        # Check ONNX runtime GPU support
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            print(f"ONNX Runtime providers: {providers}")
            if 'CUDAExecutionProvider' in providers:
                print("✓ CUDAExecutionProvider is available - model will use GPU")
            else:
                print("⚠ WARNING: CUDAExecutionProvider not available - model will use CPU")
        except ImportError:
            print("Note: onnxruntime not imported")
        
        # Get GPU memory usage
        torch.cuda.reset_peak_memory_stats(device)
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
        
        print(f"GPU memory allocated: {memory_allocated:.2f} MB")
        print(f"GPU memory reserved: {memory_reserved:.2f} MB")
        
        # Try to get model parameters info (may not be available for all model types)
        try:
            params = list(model.parameters())
            if params:
                total_params = sum(p.numel() for p in params)
                trainable_params = sum(p.numel() for p in params if p.requires_grad)
                param_dtype = params[0].dtype
                
                print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
                print(f"Model dtype: {param_dtype}")
                
                # Calculate expected memory (rough estimate)
                if param_dtype == torch.float32:
                    expected_memory = total_params * 4 / 1024**2  # 4 bytes per float32
                elif param_dtype == torch.float16 or param_dtype == torch.bfloat16:
                    expected_memory = total_params * 2 / 1024**2  # 2 bytes per float16/bfloat16
                else:
                    expected_memory = 0
                
                if expected_memory > 0:
                    print(f"Expected memory (parameters only): {expected_memory:.2f} MB")
            else:
                print("Note: Model parameters not accessible (may use ONNX or custom architecture)")
        except (StopIteration, AttributeError, TypeError) as e:
            print(f"Note: Could not access model parameters: {type(e).__name__}")
            print("Model may use ONNX runtime or custom architecture")
    
    dummy = torch.zeros(1, 16000).to(device)
    with torch.no_grad():
        model(dummy, "hi", "rnnt")
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
        print(f"Peak GPU memory after warmup: {peak_memory:.2f} MB")
    
    print("Model ready")


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    audio_bytes = base64.b64decode(request.audio_b64)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    text = await asyncio.get_event_loop().run_in_executor(
        None, transcribe_sync, audio_np, request.language_id
    )
    
    return TranscribeResponse(text=text)


@app.get("/health")
async def health():
    return {"status": "healthy", "device": str(device)}


if __name__ == "__main__":
    if args.workers > 1:
        # When using workers, uvicorn requires app as import string
        uvicorn.run("server:app", host="0.0.0.0", port=args.port, workers=args.workers)
    else:
        # Single worker can use app object directly
        uvicorn.run(app, host="0.0.0.0", port=args.port)