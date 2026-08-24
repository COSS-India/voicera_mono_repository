"""
WebSocket TTS server: continuous batching with the same loop as test_parler_tts.py
(prefill when a request arrives, step all running requests together, stream PCM chunks).

Runs N independent worker PROCESSES (not threads), each with its own CUDA context
and its own copy of the model, all sharing this one GPU. A single incoming job queue
and a single results queue connect them to the one asyncio websocket-accepting process.

Why: the previous design ran one Python thread doing the continuous-batching loop.
Under concurrent load that thread pinned ~1 CPU core (Python's GIL) no matter how
many requests arrived, while `docker stats` showed 13+ of 16 vCPUs sitting idle and
`nvidia-smi` showed ~75GB of GPU memory free. Multiple OS processes each get their
own GIL, so concurrent load can actually spread across the idle cores; the GPU has
enough spare memory/compute to host several model replicas at once.

Client sends one JSON object per utterance:
  {"prompt": "...", "description": "..."}

Server first sends a small JSON metadata frame, then binary frames (float32 mono PCM),
then a final JSON {"type": "done"}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
import uuid

import numpy as np
import websockets

AUDIO_SAMPLE_RATE = 44100
here = os.path.dirname(os.path.abspath(__file__))


def worker_main(
    worker_id: int,
    checkpoint_path: str,
    decode_every: int,
    job_q,
    results_q,
    ready_q,
) -> None:
    """Entry point for one worker process.

    Loads its own model onto the GPU and runs the same continuous-batching loop
    the old single-thread server used. The only structural difference: requests
    arrive as plain (pid, prompt, description) tuples off a shared multiprocessing
    queue (fed by every websocket connection, from whichever worker happens to be
    free -- this is what gives load balancing across workers for free), and every
    result (audio chunk / done / error) is sent back tagged with its pid over a
    second shared queue instead of being written straight into a local dict.
    """
    import torch  # noqa: PLC0415 -- imported inside the spawned child process only

    from inference.runner import ParlerTTSModelRunner, TTSRequest  # noqa: PLC0415

    use_cuda_graph = os.environ.get("TTS_USE_CUDA_GRAPH", "1").strip().lower() not in ("0", "false", "no")
    runner = ParlerTTSModelRunner(checkpoint_path, play_steps=decode_every, use_cuda_graph=use_cuda_graph)
    print(f"[worker startup] use_cuda_graph={use_cuda_graph}", flush=True)
    pending_pids: set[str] = set()
    # Evicted (EOS) requests whose "done" is held back until their final tail has
    # actually been decoded and dispatched -- see runner.pending_final_pids().
    done_pending: set[str] = set()
    step_count = 0
    last_empty_cache = 0.0
    ready_q.put(worker_id)
    print(f"[worker {worker_id}] ready (pid={os.getpid()})", flush=True)

    with torch.no_grad():
        while True:
            # 1) Drain every new job waiting for THIS worker (continuous batching intake).
            batch_jobs = []
            while True:
                try:
                    job = job_q.get_nowait()
                except queue.Empty:
                    break
                if job is None:
                    return
                batch_jobs.append(job)

            if batch_jobs:
                reqs = []
                for pid, prompt, description in batch_jobs:
                    req = TTSRequest(prompt=prompt, description=description, pid=pid)
                    pending_pids.add(pid)
                    reqs.append(req)
                try:
                    runner.prefill_batch(reqs)
                except Exception as e:
                    traceback.print_exc()
                    for req in reqs:
                        try:
                            runner.free(req)
                        except Exception:
                            pass
                        runner.running_requests.pop(req.pid, None)
                        pending_pids.discard(req.pid)
                        results_q.put((req.pid, "error", str(e)))

            # 2) One global step, batched over whatever this worker is running.
            audio_dict = {}
            evicted: set[str] = set()
            if runner.running_requests:
                pids_before = set(runner.running_requests.keys())
                try:
                    runner.step()
                except Exception as e:
                    traceback.print_exc()
                    for p in pids_before:
                        req_p = runner.running_requests.get(p)
                        if req_p is not None:
                            try:
                                runner.free(req_p)
                            except Exception:
                                pass
                        runner.running_requests.pop(p, None)
                        pending_pids.discard(p)
                        results_q.put((p, "error", f"inference step failed: {e}"))
                    continue

                runner.check_stopping_criteria()
                pids_after = set(runner.running_requests.keys())
                evicted = pids_before - pids_after
                step_count += 1

                # Keep decoding while any EOS tail is still queued, not just on the
                # periodic tick: a held-back "done" waits on that decode.
                should_audio_decode = (
                    bool(evicted)
                    or bool(runner.pending_final_pids())
                    or (step_count % decode_every == 0)
                )
                try:
                    audio_dict = runner.audio_decode() if should_audio_decode else {}
                except Exception as e:
                    traceback.print_exc()
                    for p in list(runner.running_requests.keys()):
                        req_p = runner.running_requests.get(p)
                        if req_p is not None:
                            try:
                                runner.free(req_p)
                            except Exception:
                                pass
                        runner.running_requests.pop(p, None)
                        pending_pids.discard(p)
                        results_q.put((p, "error", f"audio decode failed: {e}"))
                    # Their tails are unrecoverable, but these clients have already
                    # had most of their audio: close them out rather than leaving
                    # them waiting on a "done" that can no longer come.
                    for p in list(done_pending):
                        done_pending.discard(p)
                        pending_pids.discard(p)
                        results_q.put((p, "done", None))
                    continue
            elif done_pending or runner.pending_final_pids():
                # The batch has drained but EOS tails are still queued. Without
                # this the last chunk of the final utterances -- and their "done"
                # -- would never be sent, because the decode only ran inside the
                # branch above.
                try:
                    audio_dict = runner.audio_decode()
                except Exception as e:
                    traceback.print_exc()
                    for p in list(done_pending):
                        done_pending.discard(p)
                        pending_pids.discard(p)
                        results_q.put((p, "error", f"audio decode failed: {e}"))
                    continue
            else:
                now = time.monotonic()
                if now - last_empty_cache > 1.0:
                    torch.cuda.empty_cache()
                    last_empty_cache = now
                time.sleep(0.005)
                continue

            for p, arr in audio_dict.items():
                if p in pending_pids:
                    results_q.put((p, "audio", arr))

            # "done" only once this pid has no tail left to decode, so a client
            # never stops reading with its last ~100 ms still on the GPU.
            done_pending |= evicted
            if done_pending:
                for p in sorted(done_pending - runner.pending_final_pids()):
                    done_pending.discard(p)
                    pending_pids.discard(p)
                    results_q.put((p, "done", None))


def results_dispatcher(
    results_q,
    pending_out: dict,
    pending_out_lock: threading.Lock,
    stop_evt: threading.Event,
) -> None:
    """Background thread in the parent process: pulls (pid, kind, payload) tuples
    off the shared results queue -- fed by every worker process -- and routes each
    to that request's own thread-safe queue.Queue, which handle_client already
    reads from via asyncio.to_thread(out_q.get), unchanged from the single-thread
    version.
    """
    while not stop_evt.is_set():
        try:
            item = results_q.get(timeout=0.5)
        except queue.Empty:
            continue
        pid, kind, payload = item
        with pending_out_lock:
            out_q = pending_out.get(pid)
        if out_q is not None:
            out_q.put((kind, payload))


async def handle_client(
    websocket: websockets.ServerProtocol,
    job_q,
    pending_out: dict,
    pending_out_lock: threading.Lock,
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
    with pending_out_lock:
        pending_out[pid] = out_q
    job_q.put((pid, prompt, description))

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

    try:
        while True:
            kind, payload = await asyncio.to_thread(out_q.get)
            if kind == "error":
                try:
                    await websocket.send(json.dumps({"type": "error", "message": payload}))
                except websockets.ConnectionClosed:
                    pass
                return
            if kind == "audio":
                try:
                    await websocket.send(payload.astype(np.float32).tobytes())
                except websockets.ConnectionClosed:
                    # Client hung up mid-stream; stop pushing audio for it.
                    return
            elif kind == "done":
                try:
                    await websocket.send(json.dumps({"type": "done", "pid": pid}))
                except websockets.ConnectionClosed:
                    pass
                return
    finally:
        with pending_out_lock:
            pending_out.pop(pid, None)


async def main_async(
    host: str,
    port: int,
    checkpoint_path: str,
    decode_every: int,
    num_workers: int,
) -> None:
    ctx = mp.get_context("spawn")  # required: CUDA contexts are not fork-safe
    job_q = ctx.Queue()
    results_q = ctx.Queue()
    ready_q = ctx.Queue()

    workers = []
    for i in range(num_workers):
        p = ctx.Process(
            target=worker_main,
            args=(i, checkpoint_path, decode_every, job_q, results_q, ready_q),
            daemon=True,
        )
        p.start()
        workers.append(p)

    for _ in range(num_workers):
        ready_q.get()
    print(f"All {num_workers} TTS worker processes ready.", flush=True)

    pending_out: dict[str, queue.Queue] = {}
    pending_out_lock = threading.Lock()
    stop_evt = threading.Event()
    dispatcher_thread = threading.Thread(
        target=results_dispatcher,
        args=(results_q, pending_out, pending_out_lock, stop_evt),
        daemon=True,
    )
    dispatcher_thread.start()

    async with websockets.serve(
        lambda ws: handle_client(ws, job_q, pending_out, pending_out_lock),
        host,
        port,
        max_size=None,
    ):
        print(
            f"TTS WebSocket server ws://{host}:{port} "
            f"(checkpoints={checkpoint_path}, decode_every={decode_every}, "
            f"num_workers={num_workers})",
            flush=True,
        )
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parler TTS WebSocket server (continuous batching, multi-process)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(here, "checkpoints"),
        help="Model checkpoint directory",
    )
    parser.add_argument(
        "--decode-every",
        type=int,
        default=60,
        metavar="N",
        help=(
            "Call audio_decode every N global steps (test_parler_tts.py uses 60). "
            "Always decodes on steps that finish a request."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("TTS_NUM_WORKERS", "3")),
        help=(
            "Number of independent worker processes, each holding its own model "
            "copy and CUDA context, all sharing this GPU. Also settable via the "
            "TTS_NUM_WORKERS env var. Each replica currently costs ~18-20GB GPU "
            "memory at idle (check `nvidia-smi` headroom before raising this; "
            "start conservative and increase once memory usage under real load "
            "has been observed)."
        ),
    )
    args = parser.parse_args()
    if args.decode_every < 1:
        parser.error("--decode-every must be >= 1")
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")

    asyncio.run(
        main_async(args.host, args.port, args.checkpoint, args.decode_every, args.num_workers),
    )


if __name__ == "__main__":
    main()
