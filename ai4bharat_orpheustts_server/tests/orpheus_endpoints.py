#!/usr/bin/env python3
"""Every endpoint the OpenAPI (Swagger) schema declares, timed idle and under load.

Two things are measured that the streaming benchmarks do not cover:

  * the data plane in all its documented shapes - /v1/tts, /v1/tts/stream and
    /v1/audio/speech across wav / mp3 / flac / opus / pcm, buffered and chunked;
  * the control plane (/health, /metrics, /v1/voices, ...) while the GPU is
    saturated, which is what a load balancer and a metrics scraper actually see
    during a traffic spike.

Usage:
    python3 tests/orpheus_endpoints.py --out endpoints.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

TEXT = ("नमस्ते, आज मौसम बहुत अच्छा है और मैं बिल्कुल ठीक हूँ। "
        "आज हम आपको एक नई तकनीक के बारे में बताने जा रहे हैं जो भारत की "
        "बाईस भाषाओं में काम करती है।")
TEXT_LONG = TEXT * 3
VOICE = "Amit"

CONTROL = [("GET", "/health", None), ("GET", "/metrics", None),
           ("GET", "/v1/models", None), ("GET", "/v1/languages", None),
           ("GET", "/v1/voices", None), ("GET", "/v1/voices?language=ta", None),
           ("GET", "/v1/styles", None)]


def med(vals, nd=1):
    """Median over the non-None values, or None if there are none."""
    v = [x for x in vals if x is not None]
    return round(statistics.median(v), nd) if v else None


def pct(v, p):
    s = sorted(v)
    if not s:
        return None
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def timed_get(base, path, timeout=30):
    t0 = time.perf_counter()
    r = requests.get(f"{base}{path}", timeout=timeout)
    ms = (time.perf_counter() - t0) * 1000.0
    return {"ms": ms, "status": r.status_code, "bytes": len(r.content)}


def timed_speech(base, fmt, stream, text=TEXT, timeout=600):
    """Returns wall time, time-to-first-byte and the server's own headers."""
    t0 = time.perf_counter()
    body = {"model": "orpheus", "voice": VOICE, "input": text,
            "response_format": fmt}
    if stream:
        first = None
        n = 0
        with requests.post(f"{base}/v1/audio/speech", json=body, stream=True,
                           timeout=timeout) as r:
            st = r.status_code
            h = dict(r.headers)
            for c in r.iter_content(chunk_size=None):
                if not c:
                    continue
                if first is None:
                    first = (time.perf_counter() - t0) * 1000.0
                n += len(c)
        wall = (time.perf_counter() - t0) * 1000.0
        return {"wall_ms": wall, "ttfb_ms": first, "bytes": n, "status": st,
                "hdr": h}
    r = requests.post(f"{base}/v1/audio/speech", json=body, timeout=timeout)
    wall = (time.perf_counter() - t0) * 1000.0
    return {"wall_ms": wall, "ttfb_ms": None, "bytes": len(r.content),
            "status": r.status_code, "hdr": dict(r.headers)}


def timed_native(base, path, method, timeout=600):
    t0 = time.perf_counter()
    if method == "POST":
        r = requests.post(f"{base}{path}", json={"text": TEXT, "voice": VOICE},
                          timeout=timeout)
    else:
        r = requests.get(f"{base}{path}", params={"text": TEXT, "voice": VOICE},
                         timeout=timeout)
    wall = (time.perf_counter() - t0) * 1000.0
    return {"wall_ms": wall, "bytes": len(r.content), "status": r.status_code,
            "hdr": dict(r.headers)}


def hdr_nums(h):
    # The server sends these lowercased, and dict(requests.headers) drops the
    # CaseInsensitiveDict wrapper - so match on lowercase keys explicitly.
    low = {str(k).lower(): v for k, v in h.items()}

    def g(k):
        try:
            return float(low[k])
        except Exception:
            return None
    return {"server_ttfa_ms": g("x-ttfa-ms"), "server_rtf": g("x-rtf"),
            "gen_ms": g("x-generation-ms"), "audio_s": g("x-audio-duration-sec")}


def control_sweep(base, label, reps=5):
    rows = []
    for _, path, _ in CONTROL:
        samples = []
        for _ in range(reps):
            try:
                samples.append(timed_get(base, path)["ms"])
            except Exception as exc:
                samples.append(None)
        ok = [s for s in samples if s is not None]
        rows.append({"endpoint": f"GET {path}", "phase": label,
                     "n": len(ok),
                     "p50_ms": round(statistics.median(ok), 2) if ok else None,
                     "max_ms": round(max(ok), 2) if ok else None,
                     "errors": len(samples) - len(ok)})
    return rows


def data_sweep(base, label, reps=3):
    rows = []
    matrix = [("POST /v1/audio/speech", "wav", False),
              ("POST /v1/audio/speech", "mp3", False),
              ("POST /v1/audio/speech", "flac", False),
              ("POST /v1/audio/speech", "opus", False),
              ("POST /v1/audio/speech", "pcm", False),
              ("POST /v1/audio/speech (chunked)", "pcm", True),
              ("POST /v1/audio/speech (chunked)", "mp3", True)]
    for name, fmt, stream in matrix:
        runs = []
        for _ in range(reps):
            try:
                runs.append(timed_speech(base, fmt, stream))
            except Exception:
                pass
        if not runs:
            rows.append({"endpoint": name, "format": fmt, "phase": label,
                         "errors": reps})
            continue
        hn = [hdr_nums(r["hdr"]) for r in runs]
        audio_s = hn[0]["audio_s"]
        rows.append({
            "endpoint": name, "format": fmt, "phase": label, "n": len(runs),
            "wall_p50_ms": med([r["wall_ms"] for r in runs]),
            "ttfb_p50_ms": med([r["ttfb_ms"] for r in runs]),
            "server_ttfa_p50_ms": med([x["server_ttfa_ms"] for x in hn]),
            "server_rtf_p50": med([x["server_rtf"] for x in hn], 3),
            "server_gen_p50_ms": med([x["gen_ms"] for x in hn]),
            "audio_s": audio_s,
            "bytes": runs[0]["bytes"],
            "kb_per_audio_s": (round(runs[0]["bytes"] / 1024 / audio_s, 1)
                               if audio_s else None),
            "status": runs[0]["status"]})
    for name, path, method in (("POST /v1/tts", "/v1/tts", "POST"),
                               ("GET /v1/tts/stream", "/v1/tts/stream", "GET")):
        runs = []
        for _ in range(reps):
            try:
                runs.append(timed_native(base, path, method))
            except Exception:
                pass
        if not runs:
            continue
        hn = [hdr_nums(r["hdr"]) for r in runs]
        audio_s = hn[0]["audio_s"]
        rows.append({
            "endpoint": name, "format": "wav", "phase": label, "n": len(runs),
            "wall_p50_ms": med([r["wall_ms"] for r in runs]),
            "server_ttfa_p50_ms": med([x["server_ttfa_ms"] for x in hn]),
            "server_rtf_p50": med([x["server_rtf"] for x in hn], 3),
            "server_gen_p50_ms": med([x["gen_ms"] for x in hn]),
            "audio_s": audio_s, "bytes": runs[0]["bytes"],
            "kb_per_audio_s": (round(runs[0]["bytes"] / 1024 / audio_s, 1)
                               if audio_s else None),
            "status": runs[0]["status"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:9000")
    ap.add_argument("--load", type=int, default=64,
                    help="background streams during the loaded phase")
    ap.add_argument("--out", default="endpoints.json")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    health = requests.get(f"{base}/health", timeout=10).json()
    if not health.get("ready"):
        sys.exit("server not ready")

    schema = requests.get(f"{base}/openapi.json", timeout=10).json()
    declared = sorted(f"{m.upper()} {p}" for p, v in schema["paths"].items()
                      for m in v)
    print(f"OpenAPI declares {len(declared)} operations: {declared}")

    out = {"declared": declared, "health": health, "load_streams": args.load}

    print("\n=== IDLE")
    out["control_idle"] = control_sweep(base, "idle")
    for r in out["control_idle"]:
        print(f"  {r['endpoint']:28s} p50 {r['p50_ms']}ms  max {r['max_ms']}ms")
    out["data_idle"] = data_sweep(base, "idle")
    for r in out["data_idle"]:
        print(f"  {r['endpoint']:34s} {r['format']:5s} wall {r.get('wall_p50_ms')}ms "
              f"ttfb {r.get('ttfb_p50_ms')} rtf {r.get('server_rtf_p50')} "
              f"{r.get('kb_per_audio_s')} KB/audio-s")

    # ---- loaded phase ---------------------------------------------------- #
    print(f"\n=== UNDER LOAD ({args.load} concurrent streams)")
    stop = threading.Event()

    def bg():
        while not stop.is_set():
            try:
                with requests.post(f"{base}/v1/audio/speech", stream=True,
                                   timeout=600,
                                   json={"model": "orpheus", "voice": VOICE,
                                         "input": TEXT_LONG,
                                         "response_format": "pcm"}) as r:
                    for _ in r.iter_content(chunk_size=65536):
                        if stop.is_set():
                            break
            except Exception:
                pass

    pool = ThreadPoolExecutor(max_workers=args.load + 4)
    for _ in range(args.load):
        pool.submit(bg)
    time.sleep(6)                     # let the batch fill up
    m = requests.get(f"{base}/metrics", timeout=10).json()
    print(f"  streams_active at probe time: {m.get('streams_active')}")
    out["streams_active_during_load"] = m.get("streams_active")

    out["control_loaded"] = control_sweep(base, f"loaded_{args.load}")
    for r in out["control_loaded"]:
        print(f"  {r['endpoint']:28s} p50 {r['p50_ms']}ms  max {r['max_ms']}ms")
    out["data_loaded"] = data_sweep(base, f"loaded_{args.load}", reps=2)
    for r in out["data_loaded"]:
        print(f"  {r['endpoint']:34s} {r['format']:5s} wall {r.get('wall_p50_ms')}ms "
              f"ttfb {r.get('ttfb_p50_ms')} rtf {r.get('server_rtf_p50')}")

    stop.set()
    pool.shutdown(wait=False)
    time.sleep(2)
    out["metrics_final"] = requests.get(f"{base}/metrics", timeout=10).json()

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
