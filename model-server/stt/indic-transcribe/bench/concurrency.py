#!/usr/bin/env python3
"""Run B — concurrency, measured in a way that does not flatter the server.

The inherited harness started every client simultaneously on one looped clip. Real traffic does
neither, and both shortcuts bias the result in the server's favour. This one differs on five
points, each of which changes the number:

1. **Real-time pacing, mandatory.** Feeding faster than 1x measures batch throughput and calls
   it streaming.

2. **Staggered starts** jittered across one chunk period, plus a Poisson arrival option.
   Synchronised starts make every stream hit its chunk boundary together -- an artificial herd
   that hands the batcher a full batch it would never see in production. Kept as a separate
   `--arrival sync` STRESS row rather than deleted, because it is a real worst case.

3. **Mixed clip lengths**, with the distribution published alongside the result.

4. **Open loop.** Arrival times are fixed before the run and latency is measured from
   `t_sched`. Otherwise a stalled server stops receiving sends during exactly the slow window,
   so the slow samples are never taken and p99 looks fine -- coordinated omission.

5. **Head-of-line detector.** Normalised latency (`e2e / audio_duration`) bucketed by clip
   duration. If SHORT clips score worse than long ones they are stuck behind long ones.

Plus the client-drift guard: 8 vCPUs run both this harness and the gateway, so any row whose
client fell >100 ms behind its own schedule is marked `client_bound` and excluded from the
server-side aggregates.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, "/app/bench")

import metrics_lib as M  # noqa: E402
from gpu_sampler import GpuSampler  # noqa: E402
from ws_client import read_audio, stream_one  # noqa: E402


async def reset_stats(base: str) -> None:
    """Drop the server's accumulated percentile samples before a cell runs.

    Without this every cell inherits the previous cell's samples, and a sweep of five levels
    reports five increasingly-smeared versions of the first one. It matters most at N=1, whose
    tick p95 would otherwise carry the tail of the 24-stream cell before it.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(f"{base}/admin/reset_stats")
    except Exception as e:
        print(f"[runB] WARNING: could not reset stats: {e!r}", file=sys.stderr, flush=True)


async def fetch_metrics(base: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{base}/metrics")
            return r.json()
    except Exception as e:
        return {"error": repr(e)}


async def level(url: str, base: str, clips: list, n: int, *, arrival: str, chunk_s: float,
                lam: float, rate: float, seed: int, gpu: bool = False) -> dict:
    rng = random.Random(seed)
    wavs = [(c, read_audio(Path(c["path"]))) for c in clips]

    await reset_stats(base)

    t0 = time.monotonic() + 1.0        # a beat to let every task be scheduled first
    starts = []
    for i in range(n):
        if arrival == "sync":
            starts.append(t0)                                   # the artificial herd
        elif arrival == "poisson":
            starts.append(t0 + rng.expovariate(lam) if lam > 0 else t0)
        else:                                                    # staggered (default)
            starts.append(t0 + rng.uniform(0, chunk_s))

    tasks = []
    meta = {}
    for i in range(n):
        c, wav = wavs[i % len(wavs)]
        meta[f"c{i}"] = {"bucket": c["bucket"], "lang": c["lang"],
                         "lead_ms": M.leading_silence_ms(wav)}
        # detail=True so the rotation pause -- the thing users actually notice -- can be
        # measured UNDER LOAD rather than only at N=1.
        tasks.append(stream_one(url, wav, c["lang"], rate=rate, block_ms=100,
                                start_at=starts[i], label=f"c{i}", detail=True))

    # Sample /metrics mid-flight, when the server is actually loaded rather than draining.
    mid = asyncio.ensure_future(_sample_mid(base, delay=6.0))
    # GPU counters are sampled around the SAME occupancy that produces the latency numbers.
    # Sampling them in a separate pass would describe a differently-loaded machine.
    sampler = GpuSampler(10.0) if gpu else None
    if sampler is not None:
        sampler.__enter__()
    try:
        rows = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if sampler is not None:
            sampler.__exit__(None, None, None)
    mid_metrics = await mid

    ok = [r for r in rows if isinstance(r, dict)]
    errs = [repr(r) for r in rows if not isinstance(r, dict)]
    clean = [r for r in ok if not r["client_bound"]]

    def agg(key, src=None):
        xs = [r[key] for r in (src or clean) if r.get(key) is not None]
        return round(sum(xs) / len(xs), 2) if xs else None

    ttfps = [r["ttfp_ms"] for r in clean if r.get("ttfp_ms")]
    e2es = [r["e2e_s"] for r in clean]

    # ---- smoothness and the rotation pause, pooled across every stream at this level ----
    steady_all, boundary_all, rotations = [], [], []
    for r in clean:
        st, bd = M.split_gaps(r.get("partials", []))
        steady_all += st
        boundary_all += bd
        turns = {p.get("turn") for p in r.get("partials", []) if p.get("turn") is not None}
        rotations.append(max(0, len(turns) - 1))

    # ---- per audio length, because short and long streams do not degrade alike ----
    by_bucket: dict = {}
    for r in clean:
        mt = meta.get(r["label"], {})
        b = mt.get("bucket", "?")
        d = by_bucket.setdefault(b, {"ttfp": [], "ttfp_speech": [], "nl": [], "gap": []})
        if r.get("ttfp_ms"):
            d["ttfp"].append(r["ttfp_ms"])
            d["ttfp_speech"].append(r["ttfp_ms"] - mt.get("lead_ms", 0.0))
        d["nl"].append(r["normalized_latency"])
        st, _ = M.split_gaps(r.get("partials", []))
        d["gap"] += st
    bucket_summary = {
        b: {
            "n": len(v["nl"]),
            "ttfp_ms_p50": M.percentile(v["ttfp"], 0.50),
            "ttfp_ms_p95": M.percentile(v["ttfp"], 0.95),
            "ttfp_from_speech_ms_p50": M.percentile(v["ttfp_speech"], 0.50),
            "normalized_latency_p50": M.percentile(v["nl"], 0.50),
            "gap_steady_ms_p50": M.percentile(v["gap"], 0.50),
            "gap_steady_ms_p95": M.percentile(v["gap"], 0.95),
        }
        for b, v in sorted(by_bucket.items())
    }

    # Head-of-line: normalised latency by duration bucket.
    hol: dict[str, list] = {}
    for r in clean:
        b = "short" if r["audio_s"] < 8 else ("medium" if r["audio_s"] < 25 else "long")
        hol.setdefault(b, []).append(r["normalized_latency"])
    hol_summary = {k: round(sum(v) / len(v), 3) for k, v in sorted(hol.items())}

    return {
        "n_streams": n,
        "arrival": arrival,
        "rate": rate,
        "n_ok": len(ok),
        "n_client_bound": len(ok) - len(clean),
        "n_errors": len(errs),
        "errors": errs[:3],
        "ttfp_ms_p50": M.percentile(ttfps, 0.50),
        "ttfp_ms_p95": M.percentile(ttfps, 0.95),
        "ttfp_ms_p99": M.percentile(ttfps, 0.99),
        "ttfp_ms_p50_ci": M.bootstrap_ci(ttfps, 0.50),
        "delta_lag_p50_ms": agg("delta_lag_p50_ms"),
        "delta_lag_p95_ms": agg("delta_lag_p95_ms"),
        "e2e_s_p50": M.percentile(e2es, 0.50),
        "e2e_s_p95": M.percentile(e2es, 0.95),
        "normalized_latency_by_bucket": hol_summary,
        # If short clips are WORSE than long ones, they are trapped behind long ones.
        "hol_suspected": bool(
            len(hol_summary) > 1 and "short" in hol_summary
            and hol_summary["short"] > max(v for k, v in hol_summary.items() if k != "short")),
        # --- smoothness ---
        "gap_steady_ms_p50": M.percentile(steady_all, 0.50),
        "gap_steady_ms_p90": M.percentile(steady_all, 0.90),
        "gap_steady_ms_p99": M.percentile(steady_all, 0.99),
        "n_steady_gaps": len(steady_all),
        # --- the rotation pause, under load ---
        "gap_boundary_ms_p50": M.percentile(boundary_all, 0.50),
        "gap_boundary_ms_p95": M.percentile(boundary_all, 0.95),
        "gap_boundary_ms_max": round(max(boundary_all), 1) if boundary_all else None,
        "n_boundary_gaps": len(boundary_all),
        "rotations_per_stream_mean": (round(sum(rotations) / len(rotations), 2)
                                      if rotations else None),
        "by_bucket": bucket_summary,
        "audio_s_distribution": sorted(round(r["audio_s"], 1) for r in clean),
        # The raw per-partial lists are deliberately NOT stored -- everything derived from
        # them is above, and 60 streams x several hundred partials would dominate the file.
        "server_metrics_midflight": mid_metrics,
        "server_metrics_end": await fetch_metrics(base),
        "gpu": sampler.summary() if sampler is not None else {
            "note": "NOT MEASURED (run without --gpu-sample)"},
    }


async def _sample_mid(base: str, delay: float) -> dict:
    await asyncio.sleep(delay)
    return await fetch_metrics(base)


async def main_async(args) -> int:
    man = json.loads(args.corpus.read_text())
    buckets = {b.strip() for b in args.buckets.split(",") if b.strip()}
    clips = [i for i in man["items"] if i["bucket"] in buckets
             and (not args.lang or i["lang"] == args.lang)]
    if not clips:
        raise SystemExit(f"no clips for buckets={sorted(buckets)} lang={args.lang}")

    # Interleave the buckets round-robin. Streams are assigned `clips[i % len(clips)]`, and the
    # manifest is grouped by bucket, so in manifest order a level of 8 draws six medium clips
    # and two long ones and never sees a short one at all -- which is exactly the bucket that
    # degrades first under load. Deterministic, so a re-run measures the same mix.
    by_bucket: dict = {}
    for c in sorted(clips, key=lambda c: c["path"]):
        by_bucket.setdefault(c["bucket"], []).append(c)
    order, rings = [], [by_bucket[b] for b in sorted(by_bucket)]
    for k in range(max(len(r) for r in rings)):
        for r in rings:
            if k < len(r):
                order.append(r[k])
    clips = order

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    out = {"levels": [], "corpus": {"n_clips": len(clips), "buckets": sorted(buckets)}}
    for n in levels:
        for k in range(args.repeat):
            print(f"[runB] N={n} arrival={args.arrival} repeat={k} ...",
                  file=sys.stderr, flush=True)
            # Seed varies per repeat so the arrival jitter differs; a repeat that re-used the
            # same stagger would re-measure one arrival pattern rather than sample the spread.
            res = await level(args.url, args.base, clips, n, arrival=args.arrival,
                              chunk_s=args.chunk_s, lam=args.lam, rate=args.rate,
                              seed=args.seed + n + 1000 * k, gpu=args.gpu_sample)
            res["repeat"] = k
            sm = res["server_metrics_midflight"]
            print(f"[runB]   ttfp_p50={res['ttfp_ms_p50']} ttfp_p95={res['ttfp_ms_p95']} "
                  f"tick_p95={sm.get('tick_ms_p95')} "
                  f"budget_used={sm.get('tick_budget_used_p95')}% "
                  f"sess/tick={sm.get('avg_sessions_per_tick')} "
                  f"client_bound={res['n_client_bound']} err={res['n_errors']}",
                  file=sys.stderr, flush=True)
            out["levels"].append(res)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
            await asyncio.sleep(args.cooldown)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("/corpus/manifest.json"))
    ap.add_argument("--url", default="ws://core-asr:9002/v1/asr/ws")
    ap.add_argument("--base", default="http://core-asr:9002")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--buckets", default="short,medium")
    ap.add_argument("--levels", default="1,8,16,24,32")
    ap.add_argument("--arrival", default="stagger", choices=["stagger", "sync", "poisson"])
    ap.add_argument("--chunk-s", type=float, default=0.96,
                    help="one chunk period; the stagger window")
    ap.add_argument("--lam", type=float, default=8.0, help="Poisson arrival rate (per second)")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpu-sample", action="store_true",
                    help="sample nvidia-smi at 10 Hz around each level (Run I)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat each level this many times; every repeat is kept as its own row")
    ap.add_argument("--out", type=Path, default=Path("/results/runB_concurrency.json"))
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
