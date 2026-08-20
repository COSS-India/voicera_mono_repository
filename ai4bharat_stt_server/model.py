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

from indicconformer_language_probe import (
    IndicConformerLanguageProbe,
    LanguageProbeConfig,
)

app = FastAPI()

# Global model
model = None
device = None
language_probe = None


@app.on_event("startup")
def load_model():
    global model, device, language_probe
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    # Constructing the probe does not execute the encoder or CTC head.  The
    # default endpoint path remains model(wav, language, decoder) unchanged.
    language_probe = IndicConformerLanguageProbe(model, LanguageProbeConfig())
    
    print("Model loaded!")


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(default="hi"),
    decoder: str = Form(default="ctc"),
    enable_auto_language: bool = Form(default=False),
    scoring_method: str = Form(default="non_blank_mass"),
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
    
    # Production compatibility: unless this explicit experimental flag is
    # enabled, preserve the original public model(wav, lang, strategy) path.
    with torch.no_grad():
        if not enable_auto_language:
            text = model(wav, language, decoder)
            return {"text": text, "language": language}

        result = language_probe.transcribe_auto_language(
            wav,
            strategy=decoder,
            scoring_method=scoring_method,
            return_diagnostics=True,
        )

    return {
        "text": result["transcript"],
        "language": result["language"],
        "reason": result.get("reason"),
        "probe": result.get("probe"),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
