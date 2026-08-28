#!/usr/bin/env python3
"""A single streaming WebSocket client, paced at real time.

This is the primitive Run B (concurrency) is built from, so its measurement discipline matters
more than its convenience:

* **Real-time pacing is the default.** Feeding faster than 1x measures batch throughput and
  calls it streaming.
* **Open-loop timing.** Every latency is measured against `t_sched`, the arrival time fixed
  *in advance*, not against when this client actually managed to send. Closing that loop is
  how coordinated omission hides a stall: a server that freezes stops receiving sends during
  exactly the slow window, so the samples that would have been slow are never taken.
* **Client-drift guard.** 8 vCPUs run both the harness and the gateway here, so a client that
  falls behind its own schedule is recorded as `client_bound` and must not be reported as a
  server limit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import websockets

SAMPLE_RATE = 16000


def read_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} is {sr} Hz; expected {SAMPLE_RATE}")
    return wav


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 2)


async def stream_one(url: str, wav: np.ndarray, lang: str, *, rate: float = 1.0,
                     block_ms: int = 100, start_at: float | None = None,
                     label: str = "c0", detail: bool = False) -> dict:
    """Stream one clip and summarise it.

    `detail=True` additionally returns the raw per-partial list, including each partial's
    `turn` index. Runs F and G need that: a turn increment IS a decoder-state rotation, so it
    is the only marker that separates the pause a rotation costs from an ordinary gap. It is
    off by default because Run B opens up to 32 of these and the raw lists would dominate the
    output JSON.
    """
    block = max(1, int(SAMPLE_RATE * block_ms / 1000))
    pcm16 = (np.clip(wav, -1, 1) * 32767).astype(np.int16)

    if start_at is not None:                      # staggered / Poisson arrival
        delay = start_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    t_sched0 = time.monotonic()                   # the OPEN-LOOP origin
    partials, drift = [], []
    ttfp_ms = None
    final_text = ""
    # Set by the sender the instant the last audio block goes out, so the tail below is
    # "how long after the audio ended did the transcript settle" and not "how long was the clip".
    t_audio_end: list[float] = []

    async with websockets.connect(f"{url}?language={lang}&endpoint=0", max_size=None,
                                  open_timeout=30, close_timeout=10) as ws:
        hello = json.loads(await ws.recv())
        if hello.get("type") != "ready":
            raise RuntimeError(f"unexpected greeting: {hello}")
        t_audio0 = time.monotonic()

        async def sender():
            for i in range(0, len(pcm16), block):
                # Arrival time fixed in advance -- never derived from when the previous send
                # completed, which is what makes this open-loop.
                target = t_audio0 + (i / SAMPLE_RATE) / rate
                slack = target - time.monotonic()
                if slack > 0:
                    await asyncio.sleep(slack)
                else:
                    drift.append(-slack * 1000)   # we are LATE: client-side, not server-side
                await ws.send(pcm16[i:i + block].tobytes())
            t_audio_end.append(time.monotonic())
            # "stop", not "finalize": since turn rollover landed, `finalize` commits the
            # current turn and KEEPS the stream open, which is right for a live client and
            # wrong for a benchmark that wants the clip to end.
            await ws.send(json.dumps({"type": "stop"}))

        send_task = asyncio.create_task(sender())
        t_closed = None
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                now = time.monotonic()
                kind = msg.get("type")
                if kind == "error":
                    raise RuntimeError(msg.get("error"))
                if kind == "closed":
                    final_text = msg.get("transcript", final_text)
                    t_closed = now
                    break
                if kind not in ("partial", "turn_final"):
                    continue
                if ttfp_ms is None and (msg.get("text") or "").strip():
                    ttfp_ms = (now - t_audio0) * 1000
                partials.append({
                    "t_ms": round((now - t_sched0) * 1000, 1),
                    "latency_ms": msg.get("latency_ms"),
                    "n_words": len((msg.get("text") or "").split()),
                    "final": kind == "turn_final",
                    # A turn increment is a decoder-state rotation. Run G partitions gaps on it.
                    "turn": msg.get("turn"),
                })
                # `transcript` spans every turn, so a clip that endpointed mid-way still
                # reports its whole text rather than only the last turn.
                final_text = msg.get("transcript") or msg.get("full_text", "")
        finally:
            send_task.cancel()

    body = [p for p in partials if not p["final"]]
    lags = [p["latency_ms"] for p in body if p["latency_ms"] is not None]
    gaps = [body[i]["t_ms"] - body[i - 1]["t_ms"] for i in range(1, len(body))]
    words = [p["n_words"] for p in body if p["n_words"]]
    audio_s = len(wav) / SAMPLE_RATE
    e2e_s = time.monotonic() - t_sched0
    max_drift = max(drift) if drift else 0.0
    # Tail latency: last audio sample sent -> stream closed. None if the clip never closed
    # cleanly, which must not silently read as zero.
    tail_ms = (round((t_closed - t_audio_end[0]) * 1000, 1)
               if t_closed is not None and t_audio_end else None)

    out = {
        "label": label,
        "text": final_text,
        "audio_s": round(audio_s, 3),
        "e2e_s": round(e2e_s, 3),
        # Normalized latency: if SHORT clips score worse than long ones, they are stuck behind
        # long ones -- head-of-line blocking.
        "normalized_latency": round(e2e_s / audio_s, 3),
        "ttfp_ms": round(ttfp_ms, 1) if ttfp_ms else None,
        "n_partials": len(body),
        "words_per_partial": round(sum(words) / len(words), 2) if words else 0.0,
        "max_inter_partial_gap_ms": round(max(gaps), 1) if gaps else None,
        "delta_lag_p50_ms": pct(lags, 0.50),
        "delta_lag_p95_ms": pct(lags, 0.95),
        "delta_lag_p99_ms": pct(lags, 0.99),
        "max_send_drift_ms": round(max_drift, 1),
        "tail_ms": tail_ms,
        # >100 ms behind its own schedule means this row measures the client, not the server.
        "client_bound": bool(max_drift > 100),
    }
    if detail:
        out["partials"] = partials
    return out


async def main_async(args) -> int:
    wav = read_audio(args.audio)
    res = await stream_one(args.url, wav, args.lang, rate=args.rate,
                           block_ms=args.block_ms, label="c0")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--url", default="ws://localhost:9002/v1/asr/ws")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--block-ms", type=int, default=100)
    ap.add_argument("--out", type=Path)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
