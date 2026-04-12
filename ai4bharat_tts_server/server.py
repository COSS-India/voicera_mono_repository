"""
WebSocket TTS server: continuous batching with the same loop as test_parler_tts.py
(prefill when a request arrives, step all running requests together, stream PCM chunks).

Client sends one JSON object per utterance:
  {"prompt": "...", "description": "..."}

Server first sends a small JSON metadata frame, then binary frames (float32 mono PCM),
then a final JSON {"type": "done"}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import threading
import uuid

import numpy as np
import torch
import websockets
from dotenv import load_dotenv

from inference.runner import ParlerTTSModelRunner, TTSRequest

# Hugging Face DAC for Parler-style models is typically 24 kHz mono.
AUDIO_SAMPLE_RATE = 44100

here = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def inference_worker(
    runner: ParlerTTSModelRunner,
    prefill_q: queue.Queue,
    stop_evt: threading.Event,
    decode_every: int,
) -> None:
    """
    Runs forever: drain new requests (prefill), then one decode step for the whole batch.
    New jobs can arrive anytime; each iteration prefills everything pending before stepping.

    audio_decode() runs every ``decode_every`` steps (same idea as test_parler_tts.py using 60),
    and always on a step that evicts a request so DAC still sees final _pending_audio_decode.
    """
    pending_out: dict[str, queue.Queue] = {}
    step_count = 0

    while not stop_evt.is_set():
        # 1) Prefill every new request waiting in the queue (continuous batching intake).
        while True:
            try:
                job = prefill_q.get_nowait()
            except queue.Empty:
                break
            if job is None:
                return
            req: TTSRequest
            out_q: queue.Queue
            req, out_q = job
            pending_out[req.pid] = out_q
            try:
                runner.prefill(req)
            except Exception as e:
                out_q.put(("error", str(e)))
                pending_out.pop(req.pid, None)

        # 2) One global step (batched over all running sequences), same as the test file.
        if runner.running_requests:
            pids_before = set(runner.running_requests.keys())
            runner.step()
            runner.check_stopping_criteria()
            pids_after = set(runner.running_requests.keys())
            evicted = pids_before - pids_after
            step_count += 1

            should_audio_decode = bool(evicted) or (step_count % decode_every == 0)
            audio_dict = runner.audio_decode() if should_audio_decode else {}

            for pid, arr in audio_dict.items():
                q_out = pending_out.get(pid)
                if q_out is not None:
                    q_out.put(("audio", arr))

            for pid in evicted:
                q_out = pending_out.pop(pid, None)
                if q_out is not None:
                    q_out.put(("done", None))
        else:
            stop_evt.wait(0.005)


async def handle_client(
    websocket: websockets.ServerProtocol,
    runner: ParlerTTSModelRunner,
    prefill_q: queue.Queue,
) -> None:
    try:
        raw = await websocket.recv()
    except websockets.ConnectionClosed:
        return

    try:
        msg = json.loads(raw)
        prompt = msg["prompt"]
        description = msg["description"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        await websocket.send(json.dumps({"type": "error", "message": f"bad request: {e}"}))
        return

    out_q: queue.Queue = queue.Queue()
    pid = uuid.uuid4().hex[:8]
    req = TTSRequest(prompt=prompt, description=description, pid=pid)
    prefill_q.put((req, out_q))

    await websocket.send(
        json.dumps(
            {
                "type": "meta",
                "pid": pid,
                "sample_rate": AUDIO_SAMPLE_RATE,
                "dtype": "float32",
                "channels": 1,
            }
        )
    )

    while True:
        kind, payload = await asyncio.to_thread(out_q.get)
        if kind == "error":
            await websocket.send(json.dumps({"type": "error", "message": payload}))
            return
        if kind == "audio":
            await websocket.send(payload.astype(np.float32).tobytes())
        elif kind == "done":
            await websocket.send(json.dumps({"type": "done", "pid": pid}))
            return


async def main_async(
    host: str,
    port: int,
    checkpoint_path: str,
    decode_every: int,
) -> None:
    runner = ParlerTTSModelRunner(checkpoint_path, play_steps=decode_every)
    prefill_q: queue.Queue = queue.Queue()
    stop_evt = threading.Event()

    thread = threading.Thread(
        target=inference_worker,
        args=(runner, prefill_q, stop_evt, decode_every),
        daemon=True,
    )
    thread.start()

    async with websockets.serve(
        lambda ws: handle_client(ws, runner, prefill_q),
        host,
        port,
        max_size=None,
    ):
        print(
            f"TTS WebSocket server ws://{host}:{port} "
            f"(checkpoints={checkpoint_path}, decode_every={decode_every})"
        )
        await asyncio.Future()


def main() -> None:
    load_dotenv(os.path.join(here, ".env"))
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "").strip()
    if not checkpoint_path:
        raise SystemExit(
            "CHECKPOINT_PATH must be set in .env (same directory as server.py). "
            "No default or CLI override is used."
        )

    parser = argparse.ArgumentParser(description="Parler TTS WebSocket server (continuous batching)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument(
        "--decode-every",
        type=int,
        default=60,
        metavar="N",
        help=(
            "Call audio_decode every N global steps (test_parler_tts.py uses 60). "
            "Always decodes on steps that finish a request. Default 1 = decode every step."
        ),
    )
    args = parser.parse_args()
    if args.decode_every < 1:
        parser.error("--decode-every must be >= 1")
    asyncio.run(
        main_async(args.host, args.port, checkpoint_path, args.decode_every),
    )


if __name__ == "__main__":
    main()
