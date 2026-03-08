"""
FastAPI WebSocket server for Parler TTS.
- Accepts text (+ optional description) over WebSocket.
- Streams audio chunks; sends is_final=True when streamer.end() has been reached for that request.
- If a new text request arrives mid-stream, prefills, merges with current generation, and streams both.
"""

import asyncio
import base64
import json
import concurrent.futures
import contextlib
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer

from ragged_parler_utils import ParlerTTSForConditionalGeneration
from ragged_parler_tts import ParlerTTSModelRunner

# --- Config ---
DEFAULT_DESCRIPTION = "A neutral voice with clear delivery."
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_ID = "ai4bharat/indic-parler-tts"

_runner = None
_request_id = 0
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _load_model():
    global _runner
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    _runner = ParlerTTSModelRunner(model, tokenizer, description_tokenizer)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    # shutdown: executor could be closed here if desired


app = FastAPI(title="Parler TTS WebSocket", lifespan=lifespan)


@app.get("/")
async def root():
    return {"service": "Parler TTS WebSocket", "ws": "/ws"}


def get_runner():
    assert _runner is not None, "Model not loaded"
    return _runner


def _prefill(runner, prompts, descriptions):
    return runner.model_prefill(prompts, descriptions)


def _step(runner, model_state):
    return runner.model_step(model_state)


def _merge(runner, model_state, model_state_enter):
    return runner.merge_model_states(model_state, model_state_enter)


def _drain_streamers(model_state, request_ids, sampling_rate=None):
    """Yield (request_id, chunk_b64, is_final, sampling_rate) for each chunk from each streamer."""
    num_seq = model_state["num_seq"]
    for seq_id in range(num_seq):
        streamer = model_state["streamers"][seq_id]
        rid = request_ids[seq_id]
        rate = sampling_rate or getattr(streamer.audio_encoder.config, "sampling_rate", 24000)
        while streamer.chunk_list_size() > 0:
            chunk = streamer.get_chunk()
            if chunk is None:
                break
            if chunk.dtype == np.float32 or chunk.dtype == np.float64:
                audio_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
            else:
                audio_int16 = chunk.astype(np.int16)
            chunk_b64 = base64.b64encode(audio_int16.tobytes()).decode("ascii")
            is_final = streamer.stramer_eos_flag and (streamer.chunk_list_size() == 0)
            yield (rid, chunk_b64, is_final, rate)


async def run_tts_loop(ws: WebSocket, request_queue: asyncio.Queue, loop, executor):
    global _request_id
    runner = get_runner()
    model_state = None
    request_ids = []

    while True:
        try:
            if model_state is None:
                # Wait for first request
                try:
                    msg = await asyncio.wait_for(request_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                text = msg.get("text", "").strip()
                if not text:
                    continue
                desc = msg.get("description", DEFAULT_DESCRIPTION).strip() or DEFAULT_DESCRIPTION
                rid = _request_id
                _request_id += 1
                model_state = await loop.run_in_executor(
                    executor,
                    lambda t=text, d=desc: _prefill(runner, [t], [d]),
                )
                request_ids = [rid]

                sampling_rate = runner.model.audio_encoder.config.sampling_rate
                for rid_out, chunk_b64, is_final, rate in _drain_streamers(model_state, request_ids, sampling_rate):
                    await ws.send_json({"request_id": rid_out, "chunk": chunk_b64, "is_final": is_final, "sampling_rate": rate})

            else:
                # Merge any new requests from queue
                while not request_queue.empty():
                    try:
                        msg = request_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    text = msg.get("text", "").strip()
                    if not text:
                        continue
                    desc = msg.get("description", DEFAULT_DESCRIPTION).strip() or DEFAULT_DESCRIPTION
                    rid = _request_id
                    _request_id += 1
                    state_enter = await loop.run_in_executor(
                        executor,
                        lambda t=text, d=desc: _prefill(runner, [t], [d]),
                    )
                    model_state = await loop.run_in_executor(
                        executor,
                        lambda: _merge(runner, model_state, state_enter),
                    )
                    request_ids.append(rid)

                with torch.no_grad():
                    model_state = await loop.run_in_executor(
                        executor,
                        lambda: _step(runner, model_state),
                    )
                    # Flush final chunks into streamers before we drain (eviction would remove them)
                    model_state = await loop.run_in_executor(
                        executor,
                        lambda: runner.indicate_stream_ended(model_state),
                    )
                sampling_rate = runner.model.audio_encoder.config.sampling_rate
                for rid_out, chunk_b64, is_final, rate in _drain_streamers(model_state, request_ids, sampling_rate):
                    await ws.send_json({"request_id": rid_out, "chunk": chunk_b64, "is_final": is_final, "sampling_rate": rate})

                is_all_ended = await loop.run_in_executor(
                    executor,
                    lambda: runner.is_all_sequences_ended(model_state),
                )
                if is_all_ended:
                    model_state = await loop.run_in_executor(
                        executor,
                        lambda: runner.clear_all_model_states_at_once(model_state),
                    )
                    model_state = None
                    request_ids = []

        except WebSocketDisconnect:
            break
        except Exception as e:
            await ws.send_json({"error": str(e)})
            break


@app.websocket("/ws")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    request_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    executor = _executor

    async def receive_loop():
        try:
            while True:
                data = await ws.receive_text()
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    msg = {"text": data}
                await request_queue.put(msg)
        except WebSocketDisconnect:
            pass

    recv_task = asyncio.create_task(receive_loop())
    try:
        await run_tts_loop(ws, request_queue, loop, executor)
    finally:
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
