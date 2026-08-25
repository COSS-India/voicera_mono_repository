"""
Simple Indic Conformer Multilingual STT Server
Load once, transcribe any of 22 Indian languages
"""

import torch
import torchaudio
from transformers import AutoModel
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import io

app = FastAPI()

# Global model
model = None
device = None


@app.on_event("startup")
def load_model():
    global model, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    
    print("Model loaded!")


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(default="hi"),
    decoder: str = Form(default="ctc")
):
    """
    Transcribe audio file
    
    - audio: Audio file (wav, flac, mp3, etc.)
    - language: Language code (hi, ta, bn, te, mr, etc.)
    - decoder: 'ctc' or 'rnnt'
    """
    # Read audio file
    audio_bytes = await audio.read()
    
    # Load audio with torchaudio
    wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
    
    # Convert to mono
    wav = torch.mean(wav, dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    
    # Move to device
    wav = wav.to(device)
    
    # Transcribe
    with torch.no_grad():
        text = model(wav, language, decoder)
    
    return {"text": text, "language": language}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)