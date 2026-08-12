"""SNAC audio codec: generated token ids -> 24 kHz PCM.

Orpheus does not emit audio samples. It emits *SNAC codes* encoded as ordinary
LLM token ids, seven codes per 85.3 ms frame. This module owns that arithmetic
and the streaming decode loop.

Verified model facts these rest on:

  * 7 SNAC codes = 1 frame = 2048 samples = 85.33 ms at 24 kHz
  * ``code = token_id - AUDIO_BASE - (index % 7) * 4096``
  * valid codes are 0..4095. Code 0 IS valid; 4096 triggers a CUDA assert
    inside SNAC, so out-of-range windows are dropped rather than decoded.
"""
from __future__ import annotations

import queue as _queue
import threading
from typing import Optional

import numpy as np

# --- Audio-code arithmetic --------------------------------------------------
AUDIO_BASE = 128266          # token id of SNAC code 0 at frame-phase 0 (<|snac_0|>)
CODES_PER_FRAME = 7
CODEBOOK_SIZE = 4096
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 2048
FRAME_MS = SAMPLES_PER_FRAME / SAMPLE_RATE * 1000.0   # 85.333 ms

# Streaming window: each time a frame completes, decode the last 4 frames but
# emit only the middle frame's samples. The extra context on both sides is what
# removes the boundary click you get from decoding frames in isolation.
DECODE_WINDOW_FRAMES = 4
DECODE_WINDOW_CODES = DECODE_WINDOW_FRAMES * CODES_PER_FRAME          # 28
EMIT_SLICE = slice(SAMPLES_PER_FRAME, 2 * SAMPLES_PER_FRAME)          # [2048:4096]


def token_id_to_code(token_id: int, index: int) -> Optional[int]:
    """Map a generated token id to a SNAC code given its frame-phase ``index``.

    Returns None for non-audio tokens (text / control / wrong region), which
    land as negative codes. Code 0 is a valid audio code - do not drop it.
    """
    code = token_id - AUDIO_BASE - (index % CODES_PER_FRAME) * CODEBOOK_SIZE
    return None if code < 0 else code


class StreamingAudioBuffer:
    """Accumulates token ids and hands back the window to decode next.

    Decoding deliberately does not happen here: the caller dispatches windows to
    the batched decoder so the event loop is never blocked, which keeps this on
    the hot path cheap and pure.
    """

    def __init__(self) -> None:
        self.codes: list[int] = []
        self.count = 0          # accepted audio codes so far; drives the frame phase

    def push_token(self, token_id: int) -> Optional[list[int]]:
        code = token_id_to_code(token_id, self.count)
        if code is None:
            return None
        self.codes.append(code)
        self.count += 1
        if self.count % CODES_PER_FRAME == 0 and self.count >= DECODE_WINDOW_CODES:
            return self.codes[-DECODE_WINDOW_CODES:]
        return None


class SnacDecoder:
    """Loads the SNAC 24 kHz codec onto a torch device."""

    def __init__(self, device: str = "cuda", model_id: str = "hubertsiuzdak/snac_24khz") -> None:
        from snac import SNAC          # imported lazily: keeps this module importable without torch
        import torch

        self.torch = torch
        self.device = device
        self.model = SNAC.from_pretrained(model_id).eval().to(device)
        # Decode is called from a worker thread; serialise access to the module.
        self.lock = threading.Lock()

    def decode_windows(self, windows: list[list[int]]) -> list[bytes]:
        """Decode a batch of equal-length code windows to int16 PCM (middle frame each).

        Rows containing an out-of-range code are returned as b"" instead of being
        decoded, so one bad row cannot take down the whole batch via the SNAC
        CUDA assert.
        """
        torch = self.torch
        arr = np.asarray(windows, dtype=np.int64)                    # [B, 28]
        batch = arr.shape[0]
        results: list[bytes] = [b""] * batch
        valid = [i for i in range(batch) if arr[i].min() >= 0 and arr[i].max() < CODEBOOK_SIZE]
        if not valid:
            return results

        sub = arr[valid].reshape(len(valid), DECODE_WINDOW_FRAMES, CODES_PER_FRAME)
        # SNAC's hierarchy: level 0 carries 1 code per frame, level 1 carries 2,
        # level 2 carries 4. Vectorised gather beats the reference implementation's
        # per-element torch.cat loop, which costs an alloc and a sync per code.
        level0 = sub[:, :, 0]
        level1 = sub[:, :, [1, 4]].reshape(len(valid), -1)
        level2 = sub[:, :, [2, 3, 5, 6]].reshape(len(valid), -1)

        def to_gpu(a: np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(a).astype(np.int32)).to(self.device)

        layers = [to_gpu(level0), to_gpu(level1), to_gpu(level2)]
        with self.lock, torch.inference_mode():
            audio = self.model.decode(layers)                        # [V, 1, frames*2048]

        middle = audio[:, :, EMIT_SLICE]                             # -> [V, 1, 2048]
        pcm = middle.squeeze(1).float().cpu().numpy() * 32767.0
        pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
        for row, index in enumerate(valid):
            results[index] = pcm[row].tobytes()
        return results


class BatchedSnacDecoder:
    """Coalesces decode requests from every concurrent stream into one GPU call.

    Every window is exactly ``DECODE_WINDOW_CODES`` long, so windows stack
    cleanly on a batch dimension: N streams cost one decode instead of N. Without
    this, SNAC decode contends with generation and streams stop being real-time
    well before the engine's admission limit is reached.

    There is no timer. The worker takes one request, drains whatever else has
    queued up behind it, and decodes that. Under load the GPU is busy long enough
    for the next batch to fill naturally; when idle, a lone request goes straight
    through.
    """

    def __init__(self, decoder: SnacDecoder, max_batch: int = 96) -> None:
        self.decoder = decoder
        self.max_batch = max_batch
        self._queue: "_queue.Queue[tuple[list[int], object]]" = _queue.Queue()
        self._loop = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        import asyncio

        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="snac-decode")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    async def decode(self, window: list[int]) -> bytes:
        future = self._loop.create_future()
        self._queue.put((window, future))
        return await future

    def _worker(self) -> None:
        def settle(future, result=None, error=None):
            if future.done():
                return
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(result)

        while not self._stop.is_set():
            try:
                window, future = self._queue.get(timeout=0.5)
            except _queue.Empty:
                continue
            batch = [(window, future)]
            while len(batch) < self.max_batch:
                try:
                    batch.append(self._queue.get_nowait())
                except _queue.Empty:
                    break
            try:
                results = self.decoder.decode_windows([w for w, _ in batch])
            except Exception as exc:  # noqa: BLE001 - propagate to every waiter in the batch
                for _, fut in batch:
                    self._loop.call_soon_threadsafe(settle, fut, None, exc)
                continue
            for (_, fut), pcm in zip(batch, results):
                self._loop.call_soon_threadsafe(settle, fut, pcm, None)
