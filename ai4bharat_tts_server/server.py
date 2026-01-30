import asyncio
import base64
import json
import os
import socket
from contextlib import asynccontextmanager
from threading import Thread, Event
from typing import AsyncGenerator

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from loguru import logger


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    description: str = Field(default="A clear, natural voice with good audio quality.")
    speaker: str = Field(default="Divya")
    play_steps_in_s: float = Field(default=0.3, gt=0, le=2.0)


class ModelState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None
        self.device = None
        self.torch_dtype = None
        self.frame_rate = None
        self.sample_rate = None
        self.is_loaded = False


state = ModelState()


async def load_model():
    state.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    hf_token = os.getenv("HF_TOKEN")
    token_kwargs = {"token": hf_token} if hf_token else {}

    state.model = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts",
        torch_dtype=state.torch_dtype,
        attn_implementation={"decoder": "sdpa", "text_encoder": "eager"},
        **token_kwargs,
    ).to(state.device)

    state.tokenizer = AutoTokenizer.from_pretrained(
        "ai4bharat/indic-parler-tts",
        **token_kwargs,
    )
    state.description_tokenizer = AutoTokenizer.from_pretrained(
        state.model.config.text_encoder._name_or_path,
        **token_kwargs,
    )

    state.frame_rate = state.model.audio_encoder.config.frame_rate
    state.sample_rate = state.model.config.sampling_rate
    state.is_loaded = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_model()
    yield
    if state.model is not None:
        del state.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(title="Indic Parler TTS API", version="2.0.0", lifespan=lifespan)


async def generate_audio_chunks(
    request: Request,
    text: str,
    description: str,
    speaker: str,
    play_steps_in_s: float,
) -> AsyncGenerator[bytes, None]:
    full_description = f"{speaker}'s voice. {description}"
    play_steps = int(state.frame_rate * play_steps_in_s)

    streamer = ParlerTTSStreamer(state.model, device=state.device, play_steps=play_steps)

    description_inputs = state.description_tokenizer(
        full_description, return_tensors="pt"
    ).to(state.device)

    prompt_inputs = state.tokenizer(text, return_tensors="pt").to(state.device)

    generation_kwargs = {
        "input_ids": description_inputs.input_ids,
        "attention_mask": description_inputs.attention_mask,
        "prompt_input_ids": prompt_inputs.input_ids,
        "prompt_attention_mask": prompt_inputs.attention_mask,
        "streamer": streamer,
        "do_sample": True,
        "temperature": 0.7,
    }

    generation_complete = Event()

    def run_generation():
        try:
            state.model.generate(**generation_kwargs)
        finally:
            generation_complete.set()

    thread = Thread(target=run_generation)
    thread.start()

    client_disconnected = False

    try:
        for new_audio in streamer:
            if new_audio.shape[0] == 0:
                break

            if await request.is_disconnected():
                client_disconnected = True
                break

            audio_int16 = (np.clip(new_audio, -1.0, 1.0) * 32767).astype(np.int16)
            chunk_data = {
                "audio": base64.b64encode(audio_int16.tobytes()).decode("utf-8"),
                "sample_rate": state.sample_rate,
                "samples": new_audio.shape[0],
            }
            logger.info(f"Audio chunk going out: {len(audio_int16.tobytes())} bytes")
            yield json.dumps(chunk_data) + "\n"

        if not client_disconnected:
            yield json.dumps({"done": True}) + "\n"

    finally:
        generation_complete.wait()
        thread.join()

        del description_inputs, prompt_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.post("/tts/stream")
async def stream_tts(request: Request, tts_request: TTSRequest):
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not tts_request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    return StreamingResponse(
        generate_audio_chunks(
            request=request,
            text=tts_request.text,
            description=tts_request.description,
            speaker=tts_request.speaker,
            play_steps_in_s=tts_request.play_steps_in_s,
        ),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy" if state.is_loaded else "loading",
        "device": str(state.device) if state.device else None,
        "sample_rate": state.sample_rate,
        "model_loaded": state.is_loaded,
    }


if __name__ == "__main__":
    config = uvicorn.Config(app, host="0.0.0.0", port=8002)
    server = uvicorn.Server(config)
    
    sock = config.bind_socket()
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    logger.info("TCP_NODELAY enabled - Nagle's algorithm disabled")
    
    asyncio.run(server.serve(sockets=[sock]))