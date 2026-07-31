"""
server.py  —  OmniVoice FastAPI + WebSocket inference server

Architecture (mirrors infer_batch.py exactly):
  • spawn  8 worker processes across 4 GPUs (2 per GPU) at startup
  • each worker owns its own OmniVoice model instance (process_init)
  • incoming requests are collected into a pending queue
  • a batcher thread groups them using the same cluster_samples_by_duration /
    cluster_samples_by_batch_size logic from infer_batch.py
  • batches are dispatched to the ProcessPoolExecutor just like infer_batch.py
  • results are returned to each waiting WebSocket client

REST endpoint  POST /tts
  Body (multipart/form-data):
    text         str   required
    ref_audio    file  optional  (for voice cloning)
    ref_text     str   optional
    instruct     str   optional  (for voice design, e.g. "male, young adult")
    language_id  str   optional  default "en"
    duration     float optional
    speed        float optional

WebSocket endpoint  WS /ws/tts
  Client sends JSON:
    { "text": "...", "ref_text": "...", "language_id": "en",
      "instruct": null, "duration": null, "speed": null }
  NOTE: ref_audio must be uploaded via REST for WS requests; pass ref_audio_path
  for a file already on disk.
  Server streams back:
    { "status": "queued",     "request_id": "..." }
    { "status": "processing", "request_id": "..." }
    { "status": "done",       "request_id": "...",
      "audio_b64": "<base64 wav>", "audio_duration": 4.2,
      "synth_time": 0.8, "rtf": 0.19 }
    { "status": "error",      "request_id": "...", "detail": "..." }

Usage:
    python server.py [--model k2-fsa/OmniVoice] [--num_gpus 4] \
                     [--nj_per_gpu 2] [--num_step 16] [--max_batch_per_worker 4] \
                     [--batch_duration 60] [--batch_size 0] [--warmup 1] [--port 8005]

Default layout: 4 GPUs × 2 models = 8 workers. Default --max_batch_per_worker 2
keeps per-request latency low for longer utterances. --num_step 16 +
--guidance_scale 0 skips CFG's unconditional forward → ~2× vs default CFG=2.
"""

import argparse
import asyncio
import base64
import functools
import io
import logging
import multiprocessing as mp
import os
import signal
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.audio import load_audio
from omnivoice.utils.common import get_best_device_with_count
from omnivoice.utils.duration import RuleDurationEstimator

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
SAMPLING_RATE = 24_000
worker_model: Optional[OmniVoice] = None   # process-local, set by process_init
worker_num_step: int = 16                  # process-local diffusion steps
worker_guidance_scale: float = 0.0         # 0 → skip CFG uncond forward (~2×)

def _noop():
    """Picklable no-op used to trigger worker pool initialisation."""
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    await shutdown()


app = FastAPI(title="OmniVoice Server", lifespan=lifespan)
logger = logging.getLogger("omnivoice_server")

# Set by startup, used by handlers
_executor: Optional[ProcessPoolExecutor] = None
_pending_queue: asyncio.Queue = None          # (request_id, sample_tuple, future)
_result_map: Dict[str, asyncio.Future] = {}   # request_id → asyncio.Future
_loop: asyncio.AbstractEventLoop = None
_duration_estimator: Optional[RuleDurationEstimator] = None
_args: Optional[argparse.Namespace] = None
_batcher_task: Optional[asyncio.Task] = None
_num_workers: int = 1  # ProcessPoolExecutor worker count (GPUs × nj_per_gpu)

# Voice-clone prompt cache: maps prompt_id → path to saved .pt file
PROMPT_CACHE_DIR: str = "/tmp/omnivoice_prompts"
_prompt_cache: Dict[str, str] = {}  # prompt_id → absolute .pt path

# ---------------------------------------------------------------------------
# Worker process helpers  (identical to infer_batch.py)
# ---------------------------------------------------------------------------

def process_init(
    rank_queue,
    model_checkpoint: str,
    warmup: int = 0,
    num_step: int = 16,
    guidance_scale: float = 0.0,
):
    global worker_model, worker_num_step, worker_guidance_scale

    worker_num_step = num_step
    worker_guidance_scale = guidance_scale
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    fmt = ("%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] "
           "[Worker %(process)d] %(message)s")
    logging.basicConfig(format=fmt, level=logging.INFO, force=True)

    rank = rank_queue.get()
    device_type, device_id = rank
    if device_type == "cpu":
        worker_device = "cpu"
    elif device_type == "mps":
        worker_device = "mps"
    else:
        worker_device = f"{device_type}:{device_id}"

    logging.info(
        f"Initializing worker on {worker_device} "
        f"(num_step={num_step}, guidance_scale={guidance_scale})"
    )
    # Serialize from_pretrained per GPU: transformers' caching_allocator_warmup
    # can briefly reserve ~10GB+, and 2 workers on a 16GB card OOM if they overlap.
    import fcntl
    lock_path = f"/tmp/omnivoice_init_gpu_{device_id}.lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            worker_model = OmniVoice.from_pretrained(
                model_checkpoint, device_map=worker_device, dtype=torch.float16
            )
            if warmup > 0:
                logging.info(f"Running {warmup} warmup pass(es) on {worker_device}")
                dummy = (torch.randn(1, SAMPLING_RATE), SAMPLING_RATE)
                for _ in range(warmup):
                    with torch.inference_mode():
                        worker_model.generate(
                            text=["hello"], language=["en"],
                            ref_audio=[dummy], ref_text=["hello"],
                            num_step=num_step,
                            guidance_scale=guidance_scale,
                        )
                logging.info(f"Warmup done on {worker_device}")
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)

    logging.info(f"Worker on {worker_device} ready.")


def worker_create_voice_clone_prompt(
    ref_audio_path: str,
    ref_text: Optional[str],
    prompt_path: str,
) -> None:
    """
    Runs inside a worker process.
    Creates a VoiceClonePrompt from ref_audio_path, saves it to prompt_path.

    Note: OmniVoice stores the original (often quiet) mic RMS and later
    attenuates ALL generated speech to match it. Quiet refs (rms < 0.1) cause
    short utterances to be wiped by silence-removal → audio_dur=0 (missing
    greetings). Clamp saved rms to >= 0.1; tokens are already loudness-normalised
    during create_voice_clone_prompt when rms was low.
    """
    global worker_model
    from omnivoice.models.omnivoice import VoiceClonePrompt  # noqa: F401
    prompt = worker_model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text if ref_text else None,
    )
    if prompt.ref_rms is not None and 0 < prompt.ref_rms < 0.1:
        prompt = VoiceClonePrompt(
            ref_audio_tokens=prompt.ref_audio_tokens,
            ref_text=prompt.ref_text,
            ref_rms=0.1,
        )
    prompt.save(prompt_path)


def run_inference_batch(batch_samples: List[Tuple]) -> List[Tuple]:
    """
    Runs inside the worker process.
    Sample tuple is 9-tuple:
      (req_id, ref_text, ref_audio_path, text, lang_id, dur, spd, instruct, prompt_path)
    prompt_path takes priority over ref_audio_path for voice cloning.
    Returns list of (request_id, audio_bytes, audio_dur, synth_time).
    """
    global worker_model, worker_num_step, worker_guidance_scale
    from omnivoice.models.omnivoice import VoiceClonePrompt  # noqa: F401

    request_ids, ref_texts, ref_audio_paths = [], [], []
    texts, langs, durations, speeds, instructs, prompt_paths = [], [], [], [], [], []

    for sample in batch_samples:
        # Support both old 8-tuple and new 9-tuple
        if len(sample) == 9:
            req_id, ref_text, ref_audio_path, text, lang_id, dur, spd, instruct, prompt_path = sample
        else:
            req_id, ref_text, ref_audio_path, text, lang_id, dur, spd, instruct = sample
            prompt_path = None
        request_ids.append(req_id)
        ref_texts.append(ref_text)
        ref_audio_paths.append(ref_audio_path)
        texts.append(text)
        langs.append(lang_id)
        durations.append(dur)
        speeds.append(spd)
        instructs.append(instruct)
        prompt_paths.append(prompt_path)

    # Load VoiceClonePrompts where prompt_path is provided
    voice_clone_prompts = []
    has_prompts = any(p is not None for p in prompt_paths)
    if has_prompts:
        for p in prompt_paths:
            if p is not None:
                voice_clone_prompts.append(VoiceClonePrompt.load(p))
            else:
                voice_clone_prompts.append(None)

    gen_kwargs = dict(
        num_step=worker_num_step,
        guidance_scale=worker_guidance_scale,
    )

    t0 = time.time()
    with torch.inference_mode():
        if has_prompts:
            audios = worker_model.generate(
                text=texts,
                language=langs,
                voice_clone_prompt=voice_clone_prompts,
                duration=durations if any(d is not None for d in durations) else None,
                speed=speeds       if any(s is not None for s in speeds)    else None,
                instruct=instructs if any(i is not None for i in instructs) else None,
                **gen_kwargs,
            )
        else:
            # Voice-design / auto: must pass num_step — default is 32 and ~2x slower.
            audios = worker_model.generate(
                text=texts,
                language=langs,
                ref_audio=ref_audio_paths if any(p is not None for p in ref_audio_paths) else None,
                ref_text=ref_texts        if any(t is not None for t in ref_texts)        else None,
                duration=durations        if any(d is not None for d in durations)        else None,
                speed=speeds              if any(s is not None for s in speeds)            else None,
                instruct=instructs        if any(i is not None for i in instructs)        else None,
                **gen_kwargs,
            )

        # OmniVoice sometimes returns near-empty audio after silence removal
        # (especially short greetings with quiet clone prompts). Retry those once
        # with post-processing disabled so the greeting is still audible.
        empty_idx = [
            i for i, a in enumerate(audios)
            if a is None or getattr(a, "size", 0) == 0 or a.shape[-1] < int(0.05 * worker_model.sampling_rate)
        ]
        if empty_idx:
            logging.warning(
                "Empty/near-empty audio for %d sample(s); retrying without postprocess: %s",
                len(empty_idx),
                [texts[i][:40] for i in empty_idx],
            )
            retry_texts = [texts[i] for i in empty_idx]
            retry_langs = [langs[i] for i in empty_idx]
            if has_prompts:
                retry_prompts = [voice_clone_prompts[i] for i in empty_idx]
                retry_audios = worker_model.generate(
                    text=retry_texts,
                    language=retry_langs,
                    voice_clone_prompt=retry_prompts,
                    postprocess_output=False,
                    **gen_kwargs,
                )
            else:
                retry_ref_audio = [ref_audio_paths[i] for i in empty_idx]
                retry_ref_text = [ref_texts[i] for i in empty_idx]
                retry_audios = worker_model.generate(
                    text=retry_texts,
                    language=retry_langs,
                    ref_audio=retry_ref_audio if any(p is not None for p in retry_ref_audio) else None,
                    ref_text=retry_ref_text if any(t is not None for t in retry_ref_text) else None,
                    postprocess_output=False,
                    **gen_kwargs,
                )
            for j, i in enumerate(empty_idx):
                audios[i] = retry_audios[j]

    batch_synth_time = time.time() - t0
    synth_per_sample = batch_synth_time / len(batch_samples)

    results = []
    for req_id, audio in zip(request_ids, audios):
        buf = io.BytesIO()
        sf.write(buf, audio, worker_model.sampling_rate, format="WAV")
        audio_bytes = buf.getvalue()
        audio_dur = float(audio.shape[-1]) / worker_model.sampling_rate
        results.append((req_id, audio_bytes, audio_dur, synth_per_sample))

    return results


# ---------------------------------------------------------------------------
# Duration estimation helpers  (same as infer_batch.py)
# ---------------------------------------------------------------------------

def _get_audio_duration(audio_path: str) -> float:
    try:
        info = sf.info(audio_path)
        return info.frames / info.samplerate
    except Exception:
        wav = load_audio(audio_path, SAMPLING_RATE)
        return wav.shape[-1] / SAMPLING_RATE


def _estimate_total_duration(
    estimator: RuleDurationEstimator,
    text: str,
    ref_text: Optional[str],
    ref_audio_path: Optional[str],
    gen_duration: Optional[float] = None,
) -> float:
    ref_dur = _get_audio_duration(ref_audio_path) if ref_audio_path else 0.0
    if gen_duration is None:
        if ref_audio_path:
            gen_duration = estimator.estimate_duration(
                text, ref_text or "", ref_dur, low_threshold=2.0
            )
        else:
            gen_duration = estimator.estimate_duration(
                text, "Nice to meet you.", 0.5, low_threshold=2.0
            )
    return ref_dur + gen_duration


def cluster_by_duration(
    samples: List[Tuple],
    estimator: RuleDurationEstimator,
    batch_duration: float,
) -> List[List[Tuple]]:
    """Identical logic to infer_batch.cluster_samples_by_duration."""
    sample_with_dur = []
    for s in samples:
        ref_text, ref_audio_path, text, dur = s[1], s[2], s[3], s[5]
        total = _estimate_total_duration(estimator, text, ref_text, ref_audio_path, dur)
        sample_with_dur.append((s, total))
    sample_with_dur.sort(key=lambda x: x[1], reverse=True)

    batches, current, current_total = [], [], 0.0
    for sample, duration in sample_with_dur:
        if duration > batch_duration:
            batches.append([sample])
            continue
        if current_total + duration <= batch_duration:
            current.append(sample)
            current_total += duration
        else:
            batches.append(current)
            current, current_total = [sample], duration
    if current:
        batches.append(current)
    return batches


def cluster_by_batch_size(
    samples: List[Tuple],
    estimator: RuleDurationEstimator,
    batch_size: int,
) -> List[List[Tuple]]:
    """Identical logic to infer_batch.cluster_samples_by_batch_size."""
    sample_with_dur = []
    for s in samples:
        ref_text, ref_audio_path, text, dur = s[1], s[2], s[3], s[5]
        total = _estimate_total_duration(estimator, text, ref_text, ref_audio_path, dur)
        sample_with_dur.append((s, total))
    sample_with_dur.sort(key=lambda x: x[1], reverse=True)
    sorted_samples = [s for s, _ in sample_with_dur]
    return [sorted_samples[i: i + batch_size] for i in range(0, len(sorted_samples), batch_size)]


def ensure_worker_parallelism(
    batches: List[List[Tuple]],
    num_workers: int,
) -> List[List[Tuple]]:
    """
    After infer_batch-style clustering, re-split if needed so concurrent load
    actually fans out across ProcessPoolExecutor workers (one batch → one GPU
    worker, same as infer_batch submitting many futures).

    Why: with batch_duration=60 and 30 short utterances (~2s each), duration
    clustering packs everything into ~1 batch → only 1 GPU is used. Offline
    infer_batch avoids this because it has thousands of samples → many batches.
    A live server burst needs an extra split so idle workers get work.
    """
    if num_workers <= 1 or not batches:
        return batches

    flat = [s for b in batches for s in b]
    n = len(flat)
    target = min(num_workers, n)
    if len(batches) >= target:
        return batches

    # Evenly slice the duration-sorted flat list into `target` batches
    out: List[List[Tuple]] = []
    for i in range(target):
        start = (i * n) // target
        end = ((i + 1) * n) // target
        if start < end:
            out.append(flat[start:end])
    return out


# ---------------------------------------------------------------------------
# Batcher  —  collects pending requests and dispatches batches to the pool
# ---------------------------------------------------------------------------

async def batcher_loop():
    """
    Runs as an asyncio task.
    Waits for the first queued request, then coalesces arrivals for
    BATCH_WINDOW_MS before clustering + dispatch (same as infer_batch.py).

    Dispatch model (same as infer_batch.py):
      one clustered batch  →  one ProcessPoolExecutor.submit / run_in_executor
      →  one worker process on one GPU. Multiple batches → multiple GPUs.
    """
    global _pending_queue, _executor, _result_map, _duration_estimator, _args, _loop, _num_workers

    # Coalesce window after the first arrival. Barrier-fire clients all land
    # within a few ms; a short trailing drain catches stragglers.
    BATCH_WINDOW_MS = 25
    TRAIL_MS = 5

    logger.info("Batcher loop started.")

    while True:
        # Block until at least one request is queued
        first = await _pending_queue.get()
        pending: List[Tuple] = [first]

        deadline = _loop.time() + (BATCH_WINDOW_MS / 1000.0)
        while True:
            timeout = deadline - _loop.time()
            if timeout <= 0:
                break
            try:
                item = await asyncio.wait_for(_pending_queue.get(), timeout=timeout)
                pending.append(item)
            except asyncio.TimeoutError:
                break

        # Trailing drain for stragglers still being accepted on the event loop
        await asyncio.sleep(TRAIL_MS / 1000.0)
        while True:
            try:
                pending.append(_pending_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        # Separate into sample tuples, keeping request_id as the first field
        samples = [item[1] for item in pending]   # each is a 9-tuple (req_id, ...)

        # Same clone/design split as infer_batch.py
        # Clone = has ref_audio_path (index 2) OR prompt_path (index 8)
        clone_samples = [s for s in samples if s[2] is not None or (len(s) > 8 and s[8] is not None)]
        other_samples = [s for s in samples if s[2] is     None and (len(s) <= 8 or s[8] is None)]

        batches: List[List[Tuple]] = []
        for subset in (clone_samples, other_samples):
            if not subset:
                continue
            if _args.batch_size > 0:
                clustered = cluster_by_batch_size(subset, _duration_estimator, _args.batch_size)
            else:
                clustered = cluster_by_duration(subset, _duration_estimator, _args.batch_duration)
            # Fan out across idle GPU workers, then cap batch size.
            # Packing 4 long utterances (~5s) into one worker is much slower
            # than 2 waves of size 2 (attention cost grows super-linearly).
            max_per_worker = max(1, getattr(_args, "max_batch_per_worker", 2))
            clustered = ensure_worker_parallelism(clustered, _num_workers)
            refined: List[List[Tuple]] = []
            for b in clustered:
                if len(b) <= max_per_worker:
                    refined.append(b)
                else:
                    for i in range(0, len(b), max_per_worker):
                        refined.append(b[i : i + max_per_worker])
            batches.extend(refined)

        logger.info(
            f"Batcher: {len(samples)} request(s) → {len(batches)} batch(es) "
            f"(workers={_num_workers}, sizes={[len(b) for b in batches]})"
        )

        for batch in batches:
            req_ids_in_batch = [s[0] for s in batch]

            # Notify waiting clients that their request is now processing
            for req_id in req_ids_in_batch:
                fut = _result_map.get(req_id)
                if fut and not fut.done():
                    _status_map[req_id] = "processing"

            # Submit to the process pool — do NOT await here so multiple
            # batches run in parallel across GPUs (same as infer_batch futures).
            loop_fut = _loop.run_in_executor(
                _executor,
                run_inference_batch,
                batch,
            )

            # When the executor future completes, resolve each request's future
            def make_callback(rids):
                def on_done(f):
                    try:
                        results = f.result()
                        for req_id, audio_bytes, audio_dur, synth_time in results:
                            client_fut = _result_map.get(req_id)
                            if client_fut and not client_fut.done():
                                _loop.call_soon_threadsafe(
                                    client_fut.set_result,
                                    {
                                        "audio_bytes": audio_bytes,
                                        "audio_dur": audio_dur,
                                        "synth_time": synth_time,
                                    }
                                )
                    except Exception as e:
                        err = traceback.format_exc()
                        logger.error(f"Batch failed: {e}\n{err}")
                        for req_id in rids:
                            client_fut = _result_map.get(req_id)
                            if client_fut and not client_fut.done():
                                _loop.call_soon_threadsafe(
                                    client_fut.set_exception,
                                    RuntimeError(str(e))
                                )
                return on_done

            loop_fut.add_done_callback(make_callback(req_ids_in_batch))


# ---------------------------------------------------------------------------
# Request submission helper
# ---------------------------------------------------------------------------

_status_map: Dict[str, str] = {}   # request_id → "queued" | "processing"


async def submit_request(
    text: str,
    language_id: str = "en",
    ref_audio_path: Optional[str] = None,
    ref_text: Optional[str] = None,
    instruct: Optional[str] = None,
    duration: Optional[float] = None,
    speed: Optional[float] = None,
    prompt_path: Optional[str] = None,
) -> Tuple[str, asyncio.Future]:
    """Enqueue a request and return (request_id, future)."""
    req_id = str(uuid.uuid4())[:8]
    # 9-tuple: (id, ref_text, ref_audio_path, text, language_id, duration, speed, instruct, prompt_path)
    sample = (req_id, ref_text, ref_audio_path, text, language_id, duration, speed, instruct, prompt_path)

    fut = _loop.create_future()
    _result_map[req_id] = fut
    _status_map[req_id] = "queued"
    await _pending_queue.put((req_id, sample))
    return req_id, fut


# ---------------------------------------------------------------------------
# FastAPI  —  Voice-clone prompt cache endpoints
# ---------------------------------------------------------------------------

@app.post("/voice-prompt")
async def create_voice_prompt(
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
):
    """
    Encode a reference audio clip into a reusable VoiceClonePrompt (.pt).

    Returns { "prompt_id": "<id>" } that can be passed as prompt_id in /ws/tts
    requests. The prompt is stored server-side; call DELETE /voice-prompt/{id}
    to release it when done (or it persists until the server restarts).
    """
    audio_bytes = await ref_audio.read()
    if not audio_bytes:
        from fastapi import HTTPException as _HTTPException
        raise _HTTPException(status_code=400, detail="Empty audio file")

    # Keep the real extension — soundfile fails on webm labelled as .wav
    # (librosa/audioread can recover, but quality/RMS suffer).
    orig_name = ref_audio.filename or "ref.wav"
    ext = os.path.splitext(orig_name)[-1].lower() or ".wav"
    if ext not in (".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac"):
        ext = ".wav"

    tmp_audio = f"/tmp/ref_enc_{uuid.uuid4().hex}{ext}"
    with open(tmp_audio, "wb") as fh:
        fh.write(audio_bytes)

    os.makedirs(PROMPT_CACHE_DIR, exist_ok=True)
    prompt_id = uuid.uuid4().hex
    prompt_path = os.path.join(PROMPT_CACHE_DIR, f"{prompt_id}.pt")

    try:
        # Run in the executor so we use a live model instance (process-level model)
        await _loop.run_in_executor(
            _executor,
            functools.partial(
                worker_create_voice_clone_prompt,
                tmp_audio,
                ref_text,
                prompt_path,
            ),
        )
    finally:
        if os.path.exists(tmp_audio):
            os.unlink(tmp_audio)

    _prompt_cache[prompt_id] = prompt_path
    logger.info(f"Voice prompt created: {prompt_id} → {prompt_path}")
    return {"prompt_id": prompt_id}


@app.delete("/voice-prompt/{prompt_id}")
async def delete_voice_prompt(prompt_id: str):
    """Delete a cached voice-clone prompt to free disk space."""
    path = _prompt_cache.pop(prompt_id, None)
    if path and os.path.exists(path):
        os.unlink(path)
        logger.info(f"Voice prompt deleted: {prompt_id}")
        return {"deleted": True}
    return {"deleted": False}


# ---------------------------------------------------------------------------
# FastAPI  —  REST endpoint
# ---------------------------------------------------------------------------

@app.post("/tts")
async def tts_rest(
    text:        str            = Form(...),
    language_id: str            = Form("en"),
    ref_text:    Optional[str]  = Form(None),
    instruct:    Optional[str]  = Form(None),
    duration:    Optional[float]= Form(None),
    speed:       Optional[float]= Form(None),
    ref_audio:   Optional[UploadFile] = File(None),
):
    """Submit a TTS request and wait for the audio (WAV bytes returned directly)."""
    ref_audio_path = None
    if ref_audio is not None:
        orig_name = ref_audio.filename or "ref.wav"
        ext = os.path.splitext(orig_name)[-1].lower() or ".wav"
        if ext not in (".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac"):
            ext = ".wav"
        tmp_path = f"/tmp/ref_{uuid.uuid4().hex}{ext}"
        with open(tmp_path, "wb") as f:
            f.write(await ref_audio.read())
        ref_audio_path = tmp_path

    req_id, fut = await submit_request(
        text=text,
        language_id=language_id,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        instruct=instruct,
        duration=duration,
        speed=speed,
    )

    logger.info(f"[{req_id}] REST request queued: '{text[:60]}'")
    result = await fut
    logger.info(f"[{req_id}] Done — audio_dur={result['audio_dur']:.2f}s  "
                f"synth_time={result['synth_time']:.2f}s  "
                f"RTF={result['synth_time']/result['audio_dur']:.4f}")

    if ref_audio_path:
        os.unlink(ref_audio_path)

    _result_map.pop(req_id, None)
    _status_map.pop(req_id, None)

    return Response(content=result["audio_bytes"], media_type="audio/wav",
                    headers={"X-Request-Id": req_id,
                             "X-Audio-Duration": str(result["audio_dur"]),
                             "X-Synth-Time": str(result["synth_time"])})


# ---------------------------------------------------------------------------
# FastAPI  —  WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/tts")
async def tts_ws(websocket: WebSocket):
    """
    WebSocket TTS endpoint.

    Client sends JSON:
      { "text": "...", "language_id": "en",
        "ref_text": null, "ref_audio_path": null,
        "prompt_id": null,
        "instruct": null, "duration": null, "speed": null }

    prompt_id: a voice-clone prompt ID returned by POST /voice-prompt.
    When provided, voice cloning uses the cached prompt (zero audio transfer).

    Server replies with status messages then final audio as base64.
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        text           = data.get("text", "")
        language_id    = data.get("language_id", "en")
        ref_text       = data.get("ref_text")
        ref_audio_path = data.get("ref_audio_path")   # path on server disk
        prompt_id      = data.get("prompt_id")        # cached VoiceClonePrompt id
        instruct       = data.get("instruct")
        duration       = data.get("duration")
        speed          = data.get("speed")

        if not text:
            await websocket.send_json({"status": "error", "detail": "'text' is required"})
            return

        # Resolve prompt_id → disk path
        prompt_path = None
        if prompt_id:
            prompt_path = _prompt_cache.get(prompt_id)
            if not prompt_path:
                await websocket.send_json({
                    "status": "error",
                    "detail": f"Unknown prompt_id '{prompt_id}'. "
                              "Create one via POST /voice-prompt first.",
                })
                return

        req_id, fut = await submit_request(
            text=text,
            language_id=language_id,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            instruct=instruct,
            duration=duration,
            speed=speed,
            prompt_path=prompt_path,
        )

        logger.info(f"[{req_id}] WS request queued: '{text[:60]}'")
        await websocket.send_json({"status": "queued", "request_id": req_id})

        # Poll until processing starts (batcher picks it up)
        while _status_map.get(req_id) == "queued":
            await asyncio.sleep(0.02)
        await websocket.send_json({"status": "processing", "request_id": req_id})

        # Wait for result
        result = await fut
        audio_b64 = base64.b64encode(result["audio_bytes"]).decode()
        rtf = result["synth_time"] / result["audio_dur"] if result["audio_dur"] > 0 else 0

        logger.info(f"[{req_id}] WS done — audio_dur={result['audio_dur']:.2f}s  "
                    f"synth_time={result['synth_time']:.2f}s  RTF={rtf:.4f}")

        await websocket.send_json({
            "status": "done",
            "request_id": req_id,
            "audio_b64": audio_b64,
            "audio_duration": result["audio_dur"],
            "synth_time": result["synth_time"],
            "rtf": rtf,
        })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}\n{traceback.format_exc()}")
        try:
            await websocket.send_json({"status": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        _result_map.pop(req_id, None)
        _status_map.pop(req_id, None)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

async def startup():
    global _executor, _pending_queue, _loop, _duration_estimator, _batcher_task, _num_workers

    _loop = asyncio.get_event_loop()
    _pending_queue = asyncio.Queue()
    _duration_estimator = RuleDurationEstimator()

    device_type, num_devices = get_best_device_with_count()
    if _args.num_gpus is not None:
        num_devices = min(_args.num_gpus, num_devices)

    num_processes = num_devices * _args.nj_per_gpu
    _num_workers = num_processes
    logger.info(
        f"Starting {num_processes} worker process(es) "
        f"({_args.nj_per_gpu}/GPU across {num_devices} GPU(s))."
    )

    manager = mp.Manager()
    rank_queue = manager.Queue()
    for rank in list(range(num_devices)) * _args.nj_per_gpu:
        rank_queue.put((device_type, rank))

    _executor = ProcessPoolExecutor(
        max_workers=num_processes,
        initializer=process_init,
        initargs=(rank_queue, _args.model, _args.warmup, _args.num_step, _args.guidance_scale),
    )

    # Trigger worker initialisation immediately by submitting dummy futures
    dummy_futs = [_loop.run_in_executor(_executor, _noop) for _ in range(num_processes)]
    await asyncio.gather(*dummy_futs)
    logger.info("All workers initialised.")

    _batcher_task = asyncio.create_task(batcher_loop())
    logger.info("Server ready.")


async def shutdown():
    global _executor, _batcher_task
    if _batcher_task:
        _batcher_task.cancel()
    if _executor:
        _executor.shutdown(wait=False)
    logger.info("Server shut down.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser():
    p = argparse.ArgumentParser(description="OmniVoice FastAPI + WebSocket server")
    p.add_argument("--model",          type=str,   default="k2-fsa/OmniVoice")
    p.add_argument("--num_gpus",       type=int,   default=None,
                   help="Number of GPUs to use (default: all).")
    p.add_argument("--nj_per_gpu",     type=int,   default=2,
                   help="Model instances per GPU (default: 2 → 8 models on 4 GPUs).")
    p.add_argument("--batch_duration", type=float, default=60.0,
                   help="Max total duration per batch in seconds (duration-based batching).")
    p.add_argument("--batch_size",     type=int,   default=0,
                   help="Fixed batch size. 0 = use duration-based batching.")
    p.add_argument("--max_batch_per_worker", type=int, default=2,
                   help="Max samples packed per GPU worker (default 2: lower latency for long audio; "
                        "use 4 only for short utterances).")
    p.add_argument("--num_step",       type=int,   default=16,
                   help="Diffusion decoding steps (default 16 ≈ 2x vs model default 32).")
    p.add_argument("--guidance_scale", type=float, default=0.0,
                   help="CFG scale. 0 skips the unconditional forward (~2x faster). "
                        "Model default is 2.0 (higher quality, slower).")
    p.add_argument("--warmup",         type=int,   default=1,
                   help="Warmup iterations per worker.")
    p.add_argument("--host",           type=str,   default="0.0.0.0")
    p.add_argument("--port",           type=int,   default=8005)
    return p


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    fmt = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    _args = get_parser().parse_args()

    logger.info(
        f"Config: model={_args.model}  num_gpus={_args.num_gpus}  "
        f"nj_per_gpu={_args.nj_per_gpu}  batch_size={_args.batch_size}  "
        f"max_batch_per_worker={_args.max_batch_per_worker}  "
        f"num_step={_args.num_step}  guidance_scale={_args.guidance_scale}  "
        f"batch_duration={_args.batch_duration}  warmup={_args.warmup}"
    )

    uvicorn.run(
        app,
        host=_args.host,
        port=_args.port,
        log_level="info",
    )
