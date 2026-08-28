#!/usr/bin/env python3
"""Run D — soak. Does anything drift, leak, or grow without bound?

Steady load for a fixed duration, sampling /metrics throughout. Four things are being watched,
each of which is invisible in a short run:

* **VRAM drift** -- a slow allocator leak looks fine for 60 s and kills the service overnight.
* **Latency drift** -- percentiles that creep upward mean work is accumulating somewhere.
* **Session leaks** -- `sessions_active` must return to zero between waves.
* **`decoder_mems_list` growth** -- the decoder KV cache grows with decode steps, against a
  `max_sequence_length: 1024`. Long or many-turn sessions are the case that would hit it.

Reported as first-half vs second-half, plus the slope, because "did it drift" is a question
about the shape of the run and a single average answers it wrongly.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, "/app/bench")

from ws_client import read_audio, stream_one  # noqa: E402


async def sample_loop(base: str, stop: asyncio.Event, out: list, period: float = 2.0):
    async with httpx.AsyncClient(timeout=10) as c:
        while not stop.is_set():
            try:
                m = (await c.get(f"{base}/metrics")).json()
                m["_t"] = time.monotonic()
                out.append(m)
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=period)
            except asyncio.TimeoutError:
                pass


async def main_async(args) -> int:
    man = json.loads(args.corpus.read_text())
    clips = [i for i in man["items"] if i["lang"] == args.lang
             and i["bucket"] in {"short", "medium"}]
    wavs = [(c, read_audio(Path(c["path"]))) for c in clips]
    if not wavs:
        raise SystemExit("no clips")

    stop = asyncio.Event()
    samples: list = []
    sampler = asyncio.create_task(sample_loop(args.base, stop, samples))

    t_end = time.monotonic() + args.seconds
    waves, errors, completed = 0, 0, 0
    print(f"[runD] soaking {args.seconds:.0f}s at N={args.streams}", file=sys.stderr)
    while time.monotonic() < t_end:
        tasks = []
        for i in range(args.streams):
            c, wav = wavs[i % len(wavs)]
            tasks.append(stream_one(args.url, wav, c["lang"], rate=1.0, block_ms=100,
                                    label=f"w{waves}c{i}"))
        rows = await asyncio.gather(*tasks, return_exceptions=True)
        completed += sum(1 for r in rows if isinstance(r, dict))
        errors += sum(1 for r in rows if not isinstance(r, dict))
        waves += 1
        print(f"[runD] wave {waves}: ok={completed} err={errors} "
              f"{time.monotonic() - (t_end - args.seconds):.0f}s elapsed",
              file=sys.stderr, flush=True)

    stop.set()
    await sampler

    def half_stats(key):
        xs = [(s["_t"], s.get(key)) for s in samples if s.get(key) is not None]
        if len(xs) < 4:
            return None
        mid = len(xs) // 2
        a = [v for _, v in xs[:mid]]
        b = [v for _, v in xs[mid:]]
        return {
            "first_half_mean": round(sum(a) / len(a), 3),
            "second_half_mean": round(sum(b) / len(b), 3),
            "delta": round(sum(b) / len(b) - sum(a) / len(a), 3),
            "min": round(min(v for _, v in xs), 3),
            "max": round(max(v for _, v in xs), 3),
        }

    result = {
        "seconds": args.seconds,
        "streams": args.streams,
        "waves": waves,
        "completed": completed,
        "errors": errors,
        "n_metric_samples": len(samples),
        "vram_allocated_gb": half_stats("vram_allocated_gb"),
        "vram_reserved_gb": half_stats("vram_reserved_gb"),
        "tick_ms_p95": half_stats("tick_ms_p95"),
        "avg_decode_ms": half_stats("avg_decode_ms"),
        "avg_encode_ms": half_stats("avg_encode_ms"),
        "sessions_active_max": max((s.get("sessions_active", 0) for s in samples), default=None),
        "sessions_active_final": samples[-1].get("sessions_active") if samples else None,
        "session_leak": bool(samples and samples[-1].get("sessions_active", 0) > 0),
        "decoder_mems_note": (
            "decoder_mems_list length is not exposed by /metrics; VRAM reserved is the proxy "
            "watched here. Direct measurement is NOT MEASURED."),
    }
    print(json.dumps(result, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": result, "samples": samples}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("/corpus/manifest.json"))
    ap.add_argument("--url", default="ws://core-asr:9002/v1/asr/ws")
    ap.add_argument("--base", default="http://core-asr:9002")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=Path("/results/runD_soak.json"))
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
