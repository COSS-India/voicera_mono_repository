#!/usr/bin/env python3
"""Functional and performance test suite for the Orpheus Indic TTS server.

Usage:
    python3 tests/orpheus_test.py --suite all
    python3 tests/orpheus_test.py --suite api
    python3 tests/orpheus_test.py --suite concurrency --concurrency 1,8,32,64
    python3 tests/orpheus_test.py --suite all --json results.json

Requires: requests (pip install requests). No other dependencies.

Measurement basis
-----------------
Performance suites stream response_format=pcm, where every byte maps to a known
frame: 4096 bytes = one 85.333 ms frame. The first and last chunks carry two
frames each, because the SNAC decoder emits a widened window at both ends of a
stream. Frame arrival timestamps drive the latency, jitter and late-frame
statistics.

A frame is counted LATE if it arrives after the moment a real-time player would
need it: first_audio + PREBUFFER_MS + frame_index * 85.333 ms. A stream is
"clean" only if no frame in it was late.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import statistics
import struct
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

try:
    import requests
except ImportError:
    sys.exit("This suite requires 'requests'.  pip install requests")

FRAME_BYTES = 4096
FRAME_MS = 2048 / 24000 * 1000          # 85.3333 ms
SAMPLE_RATE = 24000
PREBUFFER_MS = 100                      # playback lead-in before frame 0 is due

TEXT_SHORT = "नमस्ते, आज मौसम बहुत अच्छा है।"
TEXT_MEDIUM = ("नमस्ते, आज मौसम बहुत अच्छा है और मैं बिल्कुल ठीक हूँ। "
               "आज हम आपको एक नई तकनीक के बारे में बताने जा रहे हैं जो भारत की "
               "बाईस भाषाओं में काम करती है।")
TEXT_LONG = TEXT_MEDIUM * 3

RESULTS: dict = {}
PASSED = FAILED = 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
    else:
        FAILED += 1
        print(f"  FAIL  {name}   {detail}")
    return condition


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def rnd(obj, n=2):
    if isinstance(obj, float):
        return round(obj, n)
    if isinstance(obj, dict):
        return {k: rnd(v, n) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rnd(v, n) for v in obj]
    return obj


def emit(row):
    print("  " + json.dumps(rnd(row), ensure_ascii=False))


# --------------------------------------------------------------------------- #
# single request primitives
# --------------------------------------------------------------------------- #
def stream_pcm(base, text, timeout=600, voice="Amit"):
    """Stream one pcm request. Returns per-frame arrival times (ms from send)."""
    t0 = time.perf_counter()
    arrivals, nbytes, chunk_sizes = [], 0, []
    with requests.post(f"{base}/v1/audio/speech", stream=True, timeout=timeout,
                       json={"model": "orpheus", "voice": voice, "input": text,
                             "response_format": "pcm"}) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=None):
            if not chunk:
                continue
            now = (time.perf_counter() - t0) * 1000.0
            nbytes += len(chunk)
            chunk_sizes.append(len(chunk))
            for _ in range(max(1, len(chunk) // FRAME_BYTES)):
                arrivals.append(now)
    if not arrivals:
        return None

    frames = len(arrivals)
    audio_ms = frames * FRAME_MS
    wall_ms = arrivals[-1]
    deadline0 = arrivals[0] + PREBUFFER_MS
    late = [i for i, t in enumerate(arrivals) if t > deadline0 + i * FRAME_MS]
    gaps = [arrivals[i] - arrivals[i - 1] for i in range(1, frames)]
    return {
        "ttfa_ms": arrivals[0],
        "wall_ms": wall_ms,
        "audio_ms": audio_ms,
        "rtf": wall_ms / audio_ms,
        "frames": frames,
        "late_frames": len(late),
        "clean": len(late) == 0,
        "jitter_p99_ms": pct(gaps, 99) if gaps else 0.0,
        "gap_max_ms": max(gaps) if gaps else 0.0,
        "bytes": nbytes,
        "chunk_sizes": chunk_sizes,
    }


def buffered(base, text, fmt="wav", timeout=600, voice="Amit"):
    """One non-streaming request. Returns wall time and server-reported headers."""
    t0 = time.perf_counter()
    r = requests.post(f"{base}/v1/audio/speech", timeout=timeout,
                      json={"model": "orpheus", "voice": voice, "input": text,
                            "response_format": fmt})
    wall = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    h = r.headers
    return {
        "wall_ms": wall,
        "bytes": len(r.content),
        "server_ttfa_ms": float(h["X-TTFA-Ms"]) if "X-TTFA-Ms" in h else None,
        "server_rtf": float(h["X-RTF"]) if "X-RTF" in h else None,
        "audio_s": float(h["X-Audio-Duration-Sec"]) if "X-Audio-Duration-Sec" in h else None,
        "gen_ms": float(h["X-Generation-Ms"]) if "X-Generation-Ms" in h else None,
        "content": r.content,
    }


def ws_request(host, port, payload, timeout=600):
    """Minimal RFC6455 client. Returns (events, binary_frame_sizes, ttfa_ms)."""
    s = socket.create_connection((host, port), timeout=timeout)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET /v1/tts/ws HTTP/1.1\r\nHost: {host}:{port}\r\n"
               f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += s.recv(4096)
    head, buf = buf.split(b"\r\n\r\n", 1)
    if b"101" not in head.split(b"\r\n")[0]:
        s.close()
        raise RuntimeError(f"websocket upgrade failed: {head[:120]!r}")

    body = json.dumps(payload).encode()
    mask = os.urandom(4)
    n = len(body)
    hdr = bytes([0x81]) + (bytes([0x80 | n]) if n < 126
                           else bytes([0x80 | 126]) + struct.pack(">H", n))
    s.sendall(hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(body)))

    t0 = time.perf_counter()
    events, sizes, ttfa = [], [], None

    def need(k):
        nonlocal buf
        while len(buf) < k:
            d = s.recv(65536)
            if not d:
                raise EOFError
            buf += d

    try:
        while True:
            need(2)
            opcode = buf[0] & 0x0F
            ln = buf[1] & 0x7F
            off = 2
            if ln == 126:
                need(4); ln = struct.unpack(">H", buf[2:4])[0]; off = 4
            elif ln == 127:
                need(10); ln = struct.unpack(">Q", buf[2:10])[0]; off = 10
            need(off + ln)
            data = buf[off:off + ln]
            buf = buf[off + ln:]
            if opcode == 8:
                break
            if opcode == 2:
                if ttfa is None:
                    ttfa = (time.perf_counter() - t0) * 1000.0
                sizes.append(len(data))
            elif opcode == 1:
                ev = json.loads(data)
                events.append(ev)
                if ev.get("type") in ("done", "error"):
                    break
    except EOFError:
        pass
    finally:
        s.close()
    return events, sizes, ttfa


# --------------------------------------------------------------------------- #
# suite: api
# --------------------------------------------------------------------------- #
def suite_api(base, host, port):
    section("API — endpoints, schemas and error handling")
    schemas = {}

    h = requests.get(f"{base}/health", timeout=30).json()
    check("GET /health", h.get("ready") is True,
          f"model={h.get('model')} quant={h.get('quantization')} max_num_seqs={h.get('max_num_seqs')}")
    schemas["GET /health"] = h

    m = requests.get(f"{base}/metrics", timeout=30).json()
    check("GET /metrics", "requests_total" in m, f"uptime={m.get('uptime_seconds')}s")
    schemas["GET /metrics"] = m

    mo = requests.get(f"{base}/v1/models", timeout=30).json()
    check("GET /v1/models", mo.get("object") == "list" and len(mo["data"]) == 1,
          mo["data"][0]["id"])
    schemas["GET /v1/models"] = mo

    langs = requests.get(f"{base}/v1/languages", timeout=30).json()
    check("GET /v1/languages", len(langs) == 22, f"{len(langs)} languages")
    schemas["GET /v1/languages"] = langs[:1] + ["...", f"{len(langs)} total"]

    voices = requests.get(f"{base}/v1/voices", timeout=30).json()
    total = sum(len(v["voices"]) for v in voices.values())
    check("GET /v1/voices", total == 40, f"{total} speakers / {len(voices)} languages")
    schemas["GET /v1/voices"] = {k: voices[k] for k in list(voices)[:2]}

    ta = requests.get(f"{base}/v1/voices?language=ta", timeout=30).json()
    check("GET /v1/voices?language=ta", list(ta) == ["ta"], str(ta["ta"]["voices"]))
    check("GET /v1/voices?language=zz -> 404",
          requests.get(f"{base}/v1/voices?language=zz", timeout=30).status_code == 404)

    st = requests.get(f"{base}/v1/styles", timeout=30).json()
    check("GET /v1/styles", len(st["styles"]) == 12, f"default={st['default']}")
    schemas["GET /v1/styles"] = st

    # native synthesis
    r = requests.post(f"{base}/v1/tts", json={"text": TEXT_SHORT, "voice": "Amit"}, timeout=300)
    check("POST /v1/tts", r.status_code == 200 and r.content[:4] == b"RIFF",
          f"{len(r.content)}B rtf={r.headers.get('X-RTF')}")
    r = requests.get(f"{base}/v1/tts/stream",
                     params={"text": TEXT_SHORT, "voice": "Amit"}, timeout=300)
    check("GET /v1/tts/stream", r.status_code == 200 and r.content[:4] == b"RIFF",
          f"{len(r.content)}B")

    # websocket
    ev, sizes, ttfa = ws_request(host, port, {"text": TEXT_SHORT, "voice": "Amit"})
    done = next((e for e in ev if e["type"] == "done"), None)
    check("WS /v1/tts/ws", done is not None and len(sizes) > 3,
          f"{len(sizes)} frames ttfa={ttfa:.0f}ms events={[e['type'] for e in ev]}")
    if done:
        schemas["WS start frame"] = ev[0]
        schemas["WS done frame"] = done
        check("  WS head/tail frames are double-width",
              sizes[0] == 2 * FRAME_BYTES and sizes[-1] == 2 * FRAME_BYTES,
              f"first={sizes[0]}B mid={sizes[len(sizes)//2]}B last={sizes[-1]}B")
    ev_bad, _, _ = ws_request(host, port, {"text": TEXT_SHORT, "voice": "Nope"})
    check("WS bad voice -> error frame",
          any(e["type"] == "error" for e in ev_bad), str(ev_bad)[:70])

    section("API — response_format x stream_format matrix")
    matrix = {}
    for fmt in ("pcm", "mp3", "wav", "flac", "opus"):
        # buffered
        b = buffered(base, TEXT_SHORT, fmt)
        sig = {"wav": b"RIFF", "flac": b"fLaC", "opus": b"OggS"}.get(fmt)
        okb = len(b["content"]) > 1000 and (sig is None or b["content"][:4] == sig)
        # sse
        r = requests.post(f"{base}/v1/audio/speech", stream=True, timeout=300,
                          json={"model": "orpheus", "voice": "Amit", "input": TEXT_SHORT,
                                "response_format": fmt, "stream_format": "sse"})
        if r.status_code == 200:
            deltas = 0
            done_ev = None
            for line in r.iter_lines():
                if line and line.startswith(b"data: "):
                    e = json.loads(line[6:])
                    if e["type"] == "speech.audio.delta":
                        deltas += 1
                    elif e["type"] == "speech.audio.done":
                        done_ev = e
            sse = f"{deltas} deltas"
            if fmt == "pcm" and done_ev:
                schemas["SSE speech.audio.done"] = done_ev
        else:
            sse = f"HTTP {r.status_code}"
            r.close()
        matrix[fmt] = {"buffered": "ok" if okb else "FAIL", "buffered_bytes": b["bytes"],
                       "sse": sse}
        check(f"format {fmt}", okb, f"buffered={b['bytes']}B  sse={sse}")
    RESULTS["format_matrix"] = matrix

    # chunked streaming behaviour
    for fmt, should_stream in (("pcm", True), ("mp3", True), ("wav", False),
                               ("flac", False), ("opus", False)):
        t0 = time.perf_counter()
        first = None
        n = 0
        with requests.post(f"{base}/v1/audio/speech", stream=True, timeout=300,
                           json={"model": "orpheus", "voice": "Amit", "input": TEXT_SHORT,
                                 "response_format": fmt}) as r:
            for c in r.iter_content(chunk_size=None):
                if not c:
                    continue
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                n += 1
        total = (time.perf_counter() - t0) * 1000
        early = first < total * 0.5
        check(f"chunked {fmt}: {'streams' if should_stream else 'buffered'}",
              early == should_stream,
              f"{n} chunks, first byte at {first:.0f}ms of {total:.0f}ms")

    section("API — request validation and error envelope")
    cases = [
        ("empty input", {"voice": "Amit", "input": ""}, 400),
        ("whitespace input", {"voice": "Amit", "input": "   "}, 400),
        ("unknown voice", {"voice": "Nope", "input": TEXT_SHORT}, 400),
        ("unknown style", {"voice": "Amit", "input": TEXT_SHORT, "style": "XX"}, 400),
        ("unknown language", {"voice": "Amit", "input": TEXT_SHORT, "language": "zz"}, 400),
        ("prompt over max_model_len", {"voice": "Amit", "input": "नमस्ते " * 9000}, 400),
        ("sse + flac", {"voice": "Amit", "input": TEXT_SHORT,
                        "response_format": "flac", "stream_format": "sse"}, 400),
        ("sse + opus", {"voice": "Amit", "input": TEXT_SHORT,
                        "response_format": "opus", "stream_format": "sse"}, 400),
        ("missing input field", {"voice": "Amit"}, 400),
        ("speed out of range", {"voice": "Amit", "input": TEXT_SHORT, "speed": 9.0}, 400),
        ("valid request", {"voice": "Amit", "input": TEXT_SHORT}, 200),
    ]
    for name, body, want in cases:
        body = {"model": "orpheus", "response_format": body.pop("response_format", "pcm"), **body}
        r = requests.post(f"{base}/v1/audio/speech", json=body, timeout=300)
        detail = f"HTTP {r.status_code}"
        if r.status_code >= 400:
            try:
                e = r.json()["error"]
                detail = f'{r.status_code} {e["type"]}: {e["message"][:52]}'
                if name == "unknown voice":
                    schemas["error response"] = r.json()
            except Exception:
                detail = f"HTTP {r.status_code} (not an OpenAI envelope)"
        check(name, r.status_code == want, detail)

    e = requests.post(f"{base}/v1/audio/speech",
                      json={"model": "orpheus", "voice": "Nope", "input": TEXT_SHORT},
                      timeout=60).json()
    check("error envelope shape",
          "error" in e and {"message", "type", "param", "code"} <= set(e["error"]))

    RESULTS["schemas"] = {k: v for k, v in schemas.items()}
    return schemas


# --------------------------------------------------------------------------- #
# suite: non-live (buffered / batch)
# --------------------------------------------------------------------------- #
def suite_batch(base, repeats=3):
    section("NON-LIVE (buffered) — one complete file per request")
    rows = []
    for label, text in (("short", TEXT_SHORT), ("medium", TEXT_MEDIUM), ("long", TEXT_LONG)):
        for fmt in ("wav", "mp3", "flac", "opus", "pcm"):
            runs = [buffered(base, text, fmt) for _ in range(repeats)]
            row = {"case": label, "format": fmt,
                   "audio_s": runs[0]["audio_s"],
                   "wall_ms_med": statistics.median(r["wall_ms"] for r in runs),
                   "server_rtf": runs[0]["server_rtf"],
                   "bytes": runs[0]["bytes"]}
            rows.append(row)
            emit(row)
    # correctness of the wav container
    b = buffered(base, TEXT_MEDIUM, "wav")
    w = wave.open(BytesIO(b["content"]))
    dur = w.getnframes() / w.getframerate()
    check("wav container is 24 kHz mono s16 with truthful length",
          w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
          and w.getsampwidth() == 2 and abs(dur - (b["audio_s"] or dur)) < 0.2,
          f"{dur:.2f}s @ {w.getframerate()}Hz vs header {b['audio_s']}s")
    RESULTS["batch"] = rows
    return rows


# --------------------------------------------------------------------------- #
# suite: live (streaming latency)
# --------------------------------------------------------------------------- #
def suite_live(base, repeats=5):
    section("LIVE (streaming) — single-stream latency and playback smoothness")
    rows = []
    for label, text in (("short", TEXT_SHORT), ("medium", TEXT_MEDIUM), ("long", TEXT_LONG)):
        runs = [r for r in (stream_pcm(base, text) for _ in range(repeats)) if r]
        row = {"case": label,
               "audio_s": runs[0]["audio_ms"] / 1000,
               "frames": runs[0]["frames"],
               "ttfa_ms_p50": statistics.median(r["ttfa_ms"] for r in runs),
               "ttfa_ms_min": min(r["ttfa_ms"] for r in runs),
               "ttfa_ms_max": max(r["ttfa_ms"] for r in runs),
               "rtf_p50": statistics.median(r["rtf"] for r in runs),
               "jitter_p99_ms": statistics.median(r["jitter_p99_ms"] for r in runs),
               "gap_max_ms": max(r["gap_max_ms"] for r in runs),
               "clean_streams": f"{sum(r['clean'] for r in runs)}/{len(runs)}"}
        rows.append(row)
        emit(row)
    sizes = runs[-1]["chunk_sizes"]
    check("head and tail chunks carry two frames",
          sizes[0] == 2 * FRAME_BYTES and sizes[-1] == 2 * FRAME_BYTES,
          f"first={sizes[0]}B mid={sizes[len(sizes)//2]}B last={sizes[-1]}B")
    RESULTS["live"] = rows
    return rows


# --------------------------------------------------------------------------- #
# suite: concurrency (continuous batching)
# --------------------------------------------------------------------------- #
def _burst(base, n, text):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = [f.result() for f in [ex.submit(stream_pcm, base, text) for _ in range(n)]]
    wall = time.perf_counter() - t0
    res = [r for r in res if r]
    if not res:
        return None
    audio_s = sum(r["audio_ms"] for r in res) / 1000
    return {"concurrency": n, "wall_s": wall, "completed": len(res),
            "agg_audio_s_per_s": audio_s / wall,
            "ttfa_p50_ms": pct([r["ttfa_ms"] for r in res], 50),
            "ttfa_p95_ms": pct([r["ttfa_ms"] for r in res], 95),
            "ttfa_p99_ms": pct([r["ttfa_ms"] for r in res], 99),
            "rtf_p50": pct([r["rtf"] for r in res], 50),
            "rtf_worst": max(r["rtf"] for r in res),
            "jitter_p99_ms": pct([r["jitter_p99_ms"] for r in res], 99),
            "clean_pct": 100.0 * sum(r["clean"] for r in res) / len(res)}


def suite_concurrency(base, levels, text=TEXT_MEDIUM):
    section("CONTINUOUS BATCHING — simultaneous burst at each concurrency level")
    print("  All N streams start at the same instant. agg_audio_s_per_s is total audio")
    print("  produced divided by wall clock; clean_pct is streams with zero late frames.\n")
    rows = []
    for n in levels:
        r = _burst(base, n, text)
        if r:
            rows.append(r)
            emit(r)
        time.sleep(2)
    if len(rows) >= 2:
        base_row = rows[0]
        best = max(rows, key=lambda r: r["agg_audio_s_per_s"])
        print(f"\n  peak throughput {best['agg_audio_s_per_s']:.1f} audio-s/s at "
              f"concurrency {best['concurrency']} "
              f"({best['agg_audio_s_per_s'] / base_row['agg_audio_s_per_s']:.1f}x "
              f"the single-stream rate)")
    RESULTS["concurrency"] = rows

    section("CONTINUOUS BATCHING — sequential vs concurrent, same 8 requests")
    seq = [stream_pcm(base, text) for _ in range(8)]
    seq_s = sum(r["wall_ms"] for r in seq) / 1000
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        con = [f.result() for f in [ex.submit(stream_pcm, base, text) for _ in range(8)]]
    con_s = time.perf_counter() - t0
    row = {"requests": 8, "sequential_total_s": seq_s, "concurrent_wall_s": con_s,
           "speedup_x": seq_s / con_s,
           "per_request_latency_increase_pct":
               100 * (statistics.mean(r["wall_ms"] for r in con)
                      / statistics.mean(r["wall_ms"] for r in seq) - 1)}
    emit(row)
    check("continuous batching gives a real speedup", row["speedup_x"] > 2.0,
          f"{row['speedup_x']:.2f}x")
    RESULTS["batching_proof"] = row
    return rows


# --------------------------------------------------------------------------- #
# suite: delayed latency (arrival-rate and sustained overload)
# --------------------------------------------------------------------------- #
def suite_latency(base, levels, duration=25, text=TEXT_MEDIUM):
    section("DELAYED LATENCY — sustained load, N requests kept in flight")
    print("  Each level runs for the full duration with N workers looping. This is the")
    print("  harsh case: demand never stops, so admission queueing shows up in ttfa_p95.\n")
    rows = []
    for n in levels:
        stop = time.perf_counter() + duration
        out = []

        def worker():
            while time.perf_counter() < stop:
                try:
                    r = stream_pcm(base, text)
                    if r:
                        out.append(r)
                except Exception:
                    pass

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            [f.result() for f in [ex.submit(worker) for _ in range(n)]]
        wall = time.perf_counter() - t0
        if not out:
            continue
        row = {"concurrency": n, "duration_s": wall, "completed": len(out),
               "agg_audio_s_per_s": sum(r["audio_ms"] for r in out) / 1000 / wall,
               "ttfa_p50_ms": pct([r["ttfa_ms"] for r in out], 50),
               "ttfa_p95_ms": pct([r["ttfa_ms"] for r in out], 95),
               "ttfa_p99_ms": pct([r["ttfa_ms"] for r in out], 99),
               "started_within_500ms_pct":
                   100.0 * sum(1 for r in out if r["ttfa_ms"] <= 500) / len(out),
               "clean_pct": 100.0 * sum(r["clean"] for r in out) / len(out),
               "rtf_worst": max(r["rtf"] for r in out)}
        rows.append(row)
        emit(row)
        time.sleep(3)
    RESULTS["sustained"] = rows

    section("DELAYED LATENCY — fixed arrival rate (open-loop)")
    print("  Requests are issued on a clock regardless of whether earlier ones finished.")
    print("  Queue growth appears as ttfa_p95 climbing well above ttfa_p50.\n")
    rate_rows = []
    for rate in (2, 5, 10, 20):
        n_req = max(8, int(rate * 8))
        results = []
        pool = ThreadPoolExecutor(max_workers=n_req)
        futures = []
        t0 = time.perf_counter()
        for i in range(n_req):
            target = t0 + i / rate
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futures.append(pool.submit(stream_pcm, base, text))
        for f in futures:
            try:
                r = f.result()
                if r:
                    results.append(r)
            except Exception:
                pass
        wall = time.perf_counter() - t0
        pool.shutdown()
        if not results:
            continue
        row = {"arrival_rate_rps": rate, "requests": n_req, "completed": len(results),
               "wall_s": wall,
               "agg_audio_s_per_s": sum(r["audio_ms"] for r in results) / 1000 / wall,
               "ttfa_p50_ms": pct([r["ttfa_ms"] for r in results], 50),
               "ttfa_p95_ms": pct([r["ttfa_ms"] for r in results], 95),
               "clean_pct": 100.0 * sum(r["clean"] for r in results) / len(results)}
        rate_rows.append(row)
        emit(row)
        time.sleep(3)
    RESULTS["arrival_rate"] = rate_rows
    return rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:9000")
    ap.add_argument("--suite", default="all",
                    choices=["all", "api", "batch", "live", "concurrency", "latency"])
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64",
                    help="comma-separated levels for the concurrency and latency suites")
    ap.add_argument("--duration", type=int, default=25,
                    help="seconds per sustained-load level")
    ap.add_argument("--json", help="write all results to this file")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    hostport = base.split("://", 1)[1]
    host = hostport.split(":")[0]
    port = int(hostport.split(":")[1]) if ":" in hostport else 80
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]

    try:
        health = requests.get(f"{base}/health", timeout=10).json()
    except Exception as exc:
        sys.exit(f"cannot reach {base}/health: {exc}")
    if not health.get("ready"):
        sys.exit(f"server is not ready: {health}")

    print(f"target      {base}")
    print(f"model       {health['model']}  ({health['model_path']})")
    print(f"quantization {health['quantization']}   max_num_seqs {health['max_num_seqs']}")
    RESULTS["server"] = health

    t0 = time.perf_counter()
    if args.suite in ("all", "api"):
        suite_api(base, host, port)
    if args.suite in ("all", "batch"):
        suite_batch(base)
    if args.suite in ("all", "live"):
        suite_live(base)
    if args.suite in ("all", "concurrency"):
        suite_concurrency(base, levels)
    if args.suite in ("all", "latency"):
        suite_latency(base, levels, args.duration)

    RESULTS["final_metrics"] = requests.get(f"{base}/metrics", timeout=10).json()
    section("SUMMARY")
    print(f"  checks       {PASSED} passed, {FAILED} failed")
    print(f"  elapsed      {time.perf_counter() - t0:.1f}s")
    print(f"  server total {RESULTS['final_metrics']['requests_total']} requests, "
          f"{RESULTS['final_metrics']['errors_total']} errors, "
          f"{RESULTS['final_metrics']['audio_seconds_total']:.0f} audio-seconds")

    if args.json:
        RESULTS["passed"] = PASSED
        RESULTS["failed"] = FAILED
        with open(args.json, "w") as fh:
            json.dump(RESULTS, fh, indent=2, ensure_ascii=False, default=str)
        print(f"  results      {args.json}")

    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
