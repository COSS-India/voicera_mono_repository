#!/usr/bin/env python3
"""Stream one clip through the engine in-process and report the streaming metrics.

This is both the Stage 3 smoke test and the primitive Run A (the geometry sweep) is built from.
It drives `StreamingEngine` directly rather than over a socket so the numbers isolate the
model+policy from the gateway.

Pacing: audio is fed in `--feed-ms` blocks at `--rate`x real time. Feeding faster than 1x
measures batch throughput and calls it streaming, so the default is 1x.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")

from core_engine import SAMPLE_RATE, EngineConfig, StreamingEngine  # noqa: E402
from vad import VadConfig  # noqa: E402


def read_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} is {sr} Hz; this harness expects {SAMPLE_RATE} Hz "
                         "(the corpus builder resamples once, offline)")
    return wav


def percentile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 2)


def run(engine: StreamingEngine, wav: np.ndarray, lang: str, *,
        rate: float, feed_ms: int) -> dict:
    sess = engine.create_session(lang=lang)
    block = max(1, int(SAMPLE_RATE * feed_ms / 1000))

    partials: list[dict] = []
    deltas_ms: list[float] = []
    ttfp_ms = None
    t_start = time.monotonic()
    fed = 0

    def drain():
        nonlocal ttfp_ms
        for sid, d in engine.tick().items():
            if sid != sess.sid:
                continue
            now = time.monotonic()
            if ttfp_ms is None and d.text.strip():
                ttfp_ms = (now - sess.t_audio0) * 1000
            if not d.is_final:
                deltas_ms.append(d.latency_ms)
            partials.append({
                "t_ms": round((now - t_start) * 1000, 1),
                "text": d.text,
                "full": d.full_text,
                "latency_ms": round(d.latency_ms, 1),
                "n_words": len(d.text.split()),
                "final": d.is_final,
            })

    while fed < len(wav):
        chunk = wav[fed:fed + block]
        fed += len(chunk)
        sess.feed(chunk)
        drain()
        if rate > 0:
            # sleep so that wall-clock tracks audio time at `rate`x
            target = t_start + (fed / SAMPLE_RATE) / rate
            slack = target - time.monotonic()
            if slack > 0:
                time.sleep(slack)

    sess.request_finalize()
    deadline = time.monotonic() + 30
    while not sess._finalized and time.monotonic() < deadline:
        drain()
        time.sleep(0.005)
    drain()

    final = next((p for p in reversed(partials) if p["final"]), None)
    text = final["full"] if final else sess._full_text
    body = [p for p in partials if not p["final"]]
    gaps = [body[i]["t_ms"] - body[i - 1]["t_ms"] for i in range(1, len(body))]
    words = [p["n_words"] for p in body if p["n_words"]]

    engine.close_session(sess.sid)
    return {
        "text": text,
        "audio_s": round(len(wav) / SAMPLE_RATE, 3),
        "ttfp_ms": round(ttfp_ms, 1) if ttfp_ms else None,
        "n_partials": len(body),
        "words_per_partial": round(sum(words) / len(words), 2) if words else 0.0,
        "max_inter_partial_gap_ms": round(max(gaps), 1) if gaps else None,
        "delta_lag_p50_ms": percentile(deltas_ms, 0.50),
        "delta_lag_p95_ms": percentile(deltas_ms, 0.95),
        "delta_lag_p99_ms": percentile(deltas_ms, 0.99),
        "partials": partials,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--ckpt", default="/artifacts/indic_transcribe_core.nemo")
    ap.add_argument("--hf-dir", default="/models/core")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--chunk", type=float, default=1.0)
    ap.add_argument("--right", type=float, default=0.5)
    ap.add_argument("--left", type=float, default=10.0)
    ap.add_argument("--alignatt-thr", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=None,
                    help="pin the AlignAtt budget; default is max(4, round(10*(chunk+right)))")
    ap.add_argument("--rate", type=float, default=1.0, help="x real time; 1.0 = live")
    ap.add_argument("--feed-ms", type=int, default=20)
    ap.add_argument("--vad", type=int, default=0,
                    help="off by default here so the geometry sweep measures the policy, "
                         "not the silence gate")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    cfg = EngineConfig(
        ckpt_path=args.ckpt, hf_dir=args.hf_dir, language=args.lang,
        left_context_secs=args.left, chunk_secs=args.chunk, right_context_secs=args.right,
        alignatt_thr=args.alignatt_thr, token_budget=args.token_budget,
        vad=VadConfig(enabled=bool(args.vad)),
    )
    engine = StreamingEngine(cfg)
    engine.load()

    res = run(engine, read_audio(args.audio), args.lang, rate=args.rate, feed_ms=args.feed_ms)
    res["config"] = {
        "chunk_requested": args.chunk, "right_requested": args.right,
        "chunk_effective": engine.chunk_eff, "right_effective": engine.right_eff,
        "theoretical_latency_s": round(engine.theoretical_latency, 3),
        "alignatt_thr": args.alignatt_thr,
        "token_budget": engine.token_budget,
        "token_budget_nemo_default": 10 * int(engine.chunk_eff + engine.right_eff),
        "rate": args.rate, "vad": bool(args.vad), "lang": args.lang,
    }
    res["engine_metrics"] = engine.metrics()

    summary = {k: v for k, v in res.items() if k != "partials"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
