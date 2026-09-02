#!/usr/bin/env python3
"""Concurrency, batching and arrival-pattern load generator for the Orpheus TTS server.

Nothing here touches the server: every number is either measured at the socket or
read back from the WebSocket ``done`` frame the server already sends.

Why WebSocket and not /v1/audio/speech
--------------------------------------
Both transports run the same engine.stream_pcm(). Only the WS path returns a
per-stream ``done`` frame carrying the server's own view: tokens, gen_ms, ttfa_ms,
rtf, tokens_per_s. Those token counts are what make TTFT recoverable (see below).
A transport-parity mode re-runs a few levels over plain HTTP chunked PCM to show
the two paths measure the same.

Measurement basis
-----------------
PCM, 24 kHz mono s16le. 4096 bytes = one 85.333 ms frame. The SNAC decoder emits
a widened window at both ends, so the first and last binary messages carry two
frames each (codec.py EMIT_HEAD / EMIT_TAIL).

TTFA  time to first audio, measured at the socket: send -> first PCM byte.
ITL   inter-token latency, from the server's own counters:
          ITL = (gen_ms - ttfa_ms) / (tokens - 28)
      i.e. steady-state only, excluding the 28-token first window.
TTFT  time to first token, derived:
          TTFT = TTFA - 27 * ITL
      The first audio chunk cannot exist until 28 audio tokens are generated
      (DECODE_WINDOW_CODES = 4 frames * 7 codes). Token 1 lands at TTFT; the
      remaining 27 land one ITL apart. Verified self-consistent by the frame gap:
      mid-stream gap == 7 * ITL within jitter, which also shows SNAC decode is
      overlapped with generation rather than added to it. TTFT therefore carries
      queue wait + prefill + first sample step, and is a slight UNDER-estimate by
      the cost of the first window's decode (~1-3 ms, not overlapped).
RTF   real-time factor. Two of them, both reported:
          rtf_gen  = gen_ms / audio_ms   (server: generation vs audio produced)
          rtf_wall = wall_ms / audio_ms  (client: whole stream vs audio produced)
      Below 1.0 means faster than real time.

Connections are opened and the WS handshake completed BEFORE the firing deadline,
so a burst measures engine admission and not TCP/HTTP setup (~0.3 ms locally).

Clock: time.monotonic() is CLOCK_MONOTONIC on Linux, i.e. comparable across
processes, which is what lets N worker processes fire one synchronised burst.

Usage
-----
    python3 tests/orpheus_load.py --mode all --out results.json
    python3 tests/orpheus_load.py --mode burst --levels 1,64,256
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import multiprocessing as mp
import os
import random
import struct
import sys
import time

FRAME_BYTES = 4096
FRAME_MS = 2048 / 24000 * 1000.0        # 85.3333
PREBUFFER_MS = 100.0                    # player lead-in before frame 0 is due
FIRST_WINDOW_TOKENS = 28                # codec.DECODE_WINDOW_CODES
CODES_PER_FRAME = 7

TEXT_SHORT = "नमस्ते, आज मौसम बहुत अच्छा है।"
TEXT_MEDIUM = ("नमस्ते, आज मौसम बहुत अच्छा है और मैं बिल्कुल ठीक हूँ। "
               "आज हम आपको एक नई तकनीक के बारे में बताने जा रहे हैं जो भारत की "
               "बाईस भाषाओं में काम करती है।")
TEXT_LONG = TEXT_MEDIUM * 3
TEXTS = {"short": TEXT_SHORT, "medium": TEXT_MEDIUM, "long": TEXT_LONG}
VOICE = "Amit"

# Unloaded inter-token latency, calibrated at startup by a single stream. Used as
# the floor of the TTFT bracket: no token can arrive faster than the idle rate.
ITL_UNLOADED_MS = 4.2


# --------------------------------------------------------------------------- #
# one stream, asyncio
# --------------------------------------------------------------------------- #
async def ws_connect(host, port):
    """Open a socket and finish the WS upgrade. Returns (reader, writer)."""
    reader, writer = await asyncio.open_connection(host, port)
    sock = writer.get_extra_info("socket")
    if sock is not None:
        import socket as _s
        sock.setsockopt(_s.IPPROTO_TCP, _s.TCP_NODELAY, 1)
    key = base64.b64encode(os.urandom(16)).decode()
    writer.write((f"GET /v1/tts/ws HTTP/1.1\r\nHost: {host}:{port}\r\n"
                  f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    await writer.drain()
    head = await reader.readuntil(b"\r\n\r\n")
    if b"101" not in head.split(b"\r\n")[0]:
        raise RuntimeError(f"ws upgrade failed: {head[:100]!r}")
    return reader, writer


def ws_frame(opcode: int, payload: bytes = b"") -> bytes:
    # RFC6455 requires the mask BIT to be set on client frames but places no
    # constraint on the key. A zero key makes the XOR a no-op, which keeps a
    # Python-level per-byte loop (milliseconds on a 12 KB prompt) out of the
    # measurement path entirely.
    mask = b"\x00\x00\x00\x00"
    n = len(payload)
    b0 = 0x80 | opcode
    if n < 126:
        hdr = bytes([b0, 0x80 | n])
    elif n < 65536:
        hdr = bytes([b0, 0x80 | 126]) + struct.pack(">H", n)
    else:
        hdr = bytes([b0, 0x80 | 127]) + struct.pack(">Q", n)
    return hdr + mask + payload


def ws_text_frame(payload: bytes) -> bytes:
    return ws_frame(0x1, payload)


async def ws_read_frame(reader):
    """Returns (opcode, payload). Assumes unfragmented server frames."""
    h = await reader.readexactly(2)
    opcode = h[0] & 0x0F
    ln = h[1] & 0x7F
    if ln == 126:
        ln = struct.unpack(">H", await reader.readexactly(2))[0]
    elif ln == 127:
        ln = struct.unpack(">Q", await reader.readexactly(8))[0]
    data = await reader.readexactly(ln) if ln else b""
    return opcode, data


async def run_stream(host, port, text, fire_at, job):
    """Pre-connected stream: waits for the deadline, sends, times every frame."""
    rec = {"job": job, "ok": False}
    reader = writer = None
    try:
        # Connect shortly before firing rather than at phase start. uvicorn pings
        # idle websockets (ws_ping_interval ~20 s) and a raw client that never
        # pongs gets closed, so a connection held open for a spike scheduled 30 s
        # out would be dead on arrival. A short lead keeps setup out of the timed
        # window without leaving the socket idle long enough to be pinged.
        lead = job.get("connect_lead", 1.0)
        d = (fire_at - lead) - time.monotonic()
        if d > 0:
            await asyncio.sleep(d)
        t_conn0 = time.monotonic()
        reader, writer = await ws_connect(host, port)
        rec["connect_ms"] = (time.monotonic() - t_conn0) * 1000.0

        # Serialise and frame BEFORE the deadline so no client CPU lands inside
        # the timed window.
        payload = {"text": text, "voice": VOICE}
        if job.get("max_tokens"):
            payload["max_tokens"] = job["max_tokens"]
        frame = ws_text_frame(json.dumps(payload).encode())

        delay = fire_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

        t0 = time.monotonic()
        rec["t_send"] = t0
        rec["fire_skew_ms"] = (t0 - fire_at) * 1000.0
        writer.write(frame)
        await writer.drain()

        arrivals, frame_bytes = [], []
        done = None
        while True:
            opcode, data = await ws_read_frame(reader)
            now = (time.monotonic() - t0) * 1000.0
            if opcode == 8:
                break
            if opcode == 9:                       # ping -> pong, stay alive
                writer.write(ws_frame(0xA, data))
                await writer.drain()
                continue
            if opcode == 2:
                arrivals.append(round(now, 3))
                frame_bytes.append(len(data))
            elif opcode == 1:
                ev = json.loads(data)
                if ev.get("type") == "start":
                    # Sent after preflight/tokenisation and before engine.generate()
                    # is entered, so this is a MEASURED lower bound on TTFT and an
                    # exact read on app-layer (non-GPU) overhead.
                    rec["t_start_frame_ms"] = now
                elif ev.get("type") == "done":
                    done = ev["metrics"]
                    rec["t_done_rel_ms"] = now
                    break
                if ev.get("type") == "error":
                    rec["error"] = ev.get("message", "")[:200]
                    break
        rec["arrivals_ms"] = arrivals
        rec["msg_bytes"] = frame_bytes
        rec["done"] = done
        rec["ok"] = bool(arrivals and done)
    except Exception as exc:                                  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
    return rec


# --------------------------------------------------------------------------- #
# HTTP chunked PCM stream, for transport parity
# --------------------------------------------------------------------------- #
async def run_stream_http(host, port, text, fire_at, job):
    rec = {"job": job, "ok": False, "transport": "http"}
    writer = None
    try:
        reader, writer = await asyncio.open_connection(host, port)
        import socket as _s
        s = writer.get_extra_info("socket")
        if s is not None:
            s.setsockopt(_s.IPPROTO_TCP, _s.TCP_NODELAY, 1)
        body = json.dumps({"model": "orpheus", "voice": VOICE, "input": text,
                           "response_format": "pcm"}).encode()
        req = (b"POST /v1/audio/speech HTTP/1.1\r\nHost: " + host.encode() +
               b"\r\nContent-Type: application/json\r\nConnection: close\r\n"
               b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)

        delay = fire_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        t0 = time.monotonic()
        rec["t_send"] = t0
        rec["fire_skew_ms"] = (t0 - fire_at) * 1000.0
        writer.write(req)
        await writer.drain()

        head = await reader.readuntil(b"\r\n\r\n")
        status = int(head.split(b" ")[1])
        rec["status"] = status
        chunked = b"chunked" in head.lower()
        arrivals, nbytes, sizes = [], 0, []
        if status == 200 and chunked:
            while True:
                line = (await reader.readuntil(b"\r\n")).strip()
                size = int(line, 16)
                if size == 0:
                    await reader.readexactly(2)
                    break
                data = await reader.readexactly(size)
                await reader.readexactly(2)
                now = (time.monotonic() - t0) * 1000.0
                arrivals.append(round(now, 3))
                sizes.append(size)
                nbytes += size
        rec["arrivals_ms"] = arrivals
        rec["msg_bytes"] = sizes
        rec["bytes"] = nbytes
        rec["done"] = None
        rec["ok"] = bool(arrivals)
    except Exception as exc:                                  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
    return rec


# --------------------------------------------------------------------------- #
# worker process: runs a slice of the schedule on one event loop
# --------------------------------------------------------------------------- #
async def run_slot(host, port, text, fire_at, job):
    """One closed-loop client: request, wait for it to finish, request again,
    until the slot deadline. Keeps exactly one stream in flight per slot, so
    offered concurrency equals N instead of growing without bound."""
    out = []
    until = job["until"]
    k = 0
    while time.monotonic() < until:
        sub = dict(job, i=job["i"] * 10000 + k)
        r = await run_stream(host, port, text, max(fire_at, time.monotonic()), sub)
        out.append(r)
        k += 1
        if not r.get("ok"):
            break
    return out


def worker(host, port, jobs, out_q):
    async def main():
        tasks = []
        for j in jobs:
            if j.get("until"):
                fn = run_slot
            elif j.get("transport") == "http":
                fn = run_stream_http
            else:
                fn = run_stream
            tasks.append(asyncio.create_task(
                fn(host, port, TEXTS[j["text"]], j["fire_at"], j)))
        res = await asyncio.gather(*tasks, return_exceptions=True)
        flat = []
        for r in res:
            if isinstance(r, list):
                flat.extend(r)
            elif isinstance(r, dict):
                flat.append(r)
            else:
                flat.append({"ok": False, "error": repr(r)})
        return flat

    try:
        out_q.put(asyncio.run(main()))
    except Exception as exc:                                  # noqa: BLE001
        out_q.put([{"ok": False, "error": f"worker died: {exc}"}])


def fan_out(host, port, jobs, n_procs):
    """Run a schedule across n_procs processes. Returns flat list of records."""
    if not jobs:
        return []
    n_procs = max(1, min(n_procs, len(jobs)))
    shards = [[] for _ in range(n_procs)]
    for i, j in enumerate(jobs):
        shards[i % n_procs].append(j)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(host, port, s, q), daemon=True)
             for s in shards if s]
    for p in procs:
        p.start()
    out = []
    for _ in procs:
        out.extend(q.get())
    for p in procs:
        p.join(timeout=30)
    return out


# --------------------------------------------------------------------------- #
# per-stream derived metrics
# --------------------------------------------------------------------------- #
def enrich(rec):
    """Fill TTFA/TTFT/ITL/RTF/jitter/late-frame fields on one record."""
    if not rec.get("ok"):
        return rec
    arr = rec["arrivals_ms"]
    sizes = rec.get("msg_bytes") or []
    frames = sum(max(1, s // FRAME_BYTES) for s in sizes) if sizes else len(arr)

    # one timestamp per frame, so late-frame accounting is per frame not per message
    per_frame = []
    for t, s in zip(arr, sizes or [FRAME_BYTES] * len(arr)):
        for _ in range(max(1, s // FRAME_BYTES)):
            per_frame.append(t)

    rec["ttfa_ms"] = arr[0]
    rec["wall_ms"] = arr[-1]
    rec["frames"] = frames
    rec["audio_ms"] = frames * FRAME_MS
    rec["rtf_wall"] = rec["wall_ms"] / rec["audio_ms"] if frames else None

    gaps = [arr[i] - arr[i - 1] for i in range(1, len(arr))]
    rec["gap_p50_ms"] = pct(gaps, 50)
    rec["gap_p99_ms"] = pct(gaps, 99)
    rec["gap_max_ms"] = max(gaps) if gaps else 0.0

    deadline0 = per_frame[0] + PREBUFFER_MS
    late = [i for i, t in enumerate(per_frame) if t > deadline0 + i * FRAME_MS]
    rec["late_frames"] = len(late)
    rec["late_frac"] = len(late) / len(per_frame) if per_frame else 0.0
    rec["clean"] = not late
    # worst underrun: how far behind the real-time schedule the stream ever fell
    rec["max_underrun_ms"] = max(
        (t - (deadline0 + i * FRAME_MS) for i, t in enumerate(per_frame)), default=0.0)

    d = rec.get("done") or {}
    tok, gen, sttfa = d.get("tokens"), d.get("gen_ms"), d.get("ttfa_ms")
    rec["server_ttfa_ms"] = sttfa
    rec["server_rtf_gen"] = d.get("rtf")
    rec["tokens"] = tok
    rec["tokens_per_s"] = d.get("tokens_per_s")
    rec["gen_ms"] = gen
    if tok and gen and sttfa is not None and tok > FIRST_WINDOW_TOKENS:
        itl = (gen - sttfa) / (tok - FIRST_WINDOW_TOKENS)
        rec["itl_ms"] = itl
        rec["itl_mean_ms"] = gen / tok
        rec["transport_ms"] = rec["ttfa_ms"] - sttfa
        # self-consistency check on the model TTFA = TTFT + 27*ITL + decode:
        # a mid-stream gap should equal 7 ITL if SNAC decode is overlapped.
        rec["gap_over_7itl"] = rec["gap_p50_ms"] / (CODES_PER_FRAME * itl)

        # Token-accumulation term: the 27 token intervals the decoder must wait
        # through before frame 0 can exist at all. Measured, not assumed.
        rec["accum_ms"] = (FIRST_WINDOW_TOKENS - 1) * itl

        # TTFT bracket. ITL is not constant across a stream: the first window is
        # generated before any SNAC decode contention exists, so ITL_first lies
        # between the unloaded rate and the steady-state rate. That puts TTFT in
        #     [ TTFA - 27*ITL_steady ,  TTFA - 27*ITL_unloaded ]
        # and the WS start frame gives a hard MEASURED floor, since generate() is
        # only entered after it is sent.
        floor = rec.get("t_start_frame_ms", 0.0)
        lo = rec["ttfa_ms"] - (FIRST_WINDOW_TOKENS - 1) * itl
        hi = rec["ttfa_ms"] - (FIRST_WINDOW_TOKENS - 1) * ITL_UNLOADED_MS
        rec["ttft_lo_ms"] = max(floor, lo)
        rec["ttft_hi_ms"] = max(rec["ttft_lo_ms"], hi)
        rec["ttft_ms"] = rec["ttft_lo_ms"]          # conservative point estimate
        rec["ttft_mid_ms"] = (rec["ttft_lo_ms"] + rec["ttft_hi_ms"]) / 2.0
        rec["ttft_floor_measured_ms"] = floor
        rec["ttft_server_ms"] = max(0.0, sttfa - (FIRST_WINDOW_TOKENS - 1) * itl)
    return rec


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def pct(vals, p):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def mean(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def col(recs, key):
    return [r.get(key) for r in recs if r.get("ok") and r.get(key) is not None]


def summarise(recs, wall_s, offered, label):
    ok = [r for r in recs if r.get("ok")]
    errs = [r.get("error") for r in recs if not r.get("ok")]
    if not ok:
        # Keep the full key set so a wholly-failed group still tabulates as
        # blanks instead of raising a KeyError in a caller.
        blank = {k: None for k in (
            "wall_s", "arrival_span_s", "fire_skew_p99_ms", "audio_s_produced",
            "audio_s_per_s", "tokens_per_s_agg", "rps_completed", "ttfa_p50_ms",
            "ttfa_p90_ms", "ttfa_p95_ms", "ttfa_p99_ms", "ttfa_max_ms",
            "ttfa_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
            "ttft_max_ms", "ttft_hi_p50_ms", "ttft_hi_p95_ms", "ttft_mid_p50_ms",
            "start_frame_p50_ms", "start_frame_p99_ms", "accum_p50_ms",
            "itl_p50_ms", "itl_p95_ms", "tokens_per_s_stream_p50", "rtf_gen_p50",
            "rtf_gen_p95", "rtf_wall_p50", "rtf_wall_p95", "rtf_wall_max",
            "gap_p50_ms", "jitter_p99_ms", "gap_max_ms", "late_frame_pct",
            "clean_stream_pct", "max_underrun_ms", "e2e_p50_s", "e2e_p95_s",
            "gap_over_7itl_p50", "transport_ms_p50", "connect_ms_p99")}
        return {"label": label, "offered": offered, "completed": 0,
                "errors": len(errs), "error_sample": errs[:3], **blank}
    audio_s = sum(r["audio_ms"] for r in ok) / 1000.0
    tokens = sum(r["tokens"] or 0 for r in ok)
    starts = [r["t_send"] for r in ok]
    span = max(starts) - min(starts)
    s = {
        "label": label,
        "offered": offered,
        "completed": len(ok),
        "errors": len(errs),
        "wall_s": round(wall_s, 3),
        "arrival_span_s": round(span, 3),
        "fire_skew_p99_ms": rnd(pct(col(recs, "fire_skew_ms"), 99)),
        # throughput
        "audio_s_produced": round(audio_s, 2),
        "audio_s_per_s": rnd(audio_s / wall_s),
        "tokens_per_s_agg": rnd(tokens / wall_s, 1),
        "rps_completed": rnd(len(ok) / wall_s, 3),
        # TTFA (client socket)
        "ttfa_p50_ms": rnd(pct(col(ok, "ttfa_ms"), 50), 1),
        "ttfa_p90_ms": rnd(pct(col(ok, "ttfa_ms"), 90), 1),
        "ttfa_p95_ms": rnd(pct(col(ok, "ttfa_ms"), 95), 1),
        "ttfa_p99_ms": rnd(pct(col(ok, "ttfa_ms"), 99), 1),
        "ttfa_max_ms": rnd(max(col(ok, "ttfa_ms")), 1),
        "ttfa_mean_ms": rnd(mean(col(ok, "ttfa_ms")), 1),
        # TTFT (derived, bracketed) + the measured floor from the WS start frame
        "ttft_p50_ms": rnd(pct(col(ok, "ttft_lo_ms"), 50), 1),
        "ttft_p95_ms": rnd(pct(col(ok, "ttft_lo_ms"), 95), 1),
        "ttft_p99_ms": rnd(pct(col(ok, "ttft_lo_ms"), 99), 1),
        "ttft_max_ms": rnd(max(col(ok, "ttft_lo_ms")) if col(ok, "ttft_lo_ms") else None, 1),
        "ttft_hi_p50_ms": rnd(pct(col(ok, "ttft_hi_ms"), 50), 1),
        "ttft_hi_p95_ms": rnd(pct(col(ok, "ttft_hi_ms"), 95), 1),
        "ttft_mid_p50_ms": rnd(pct(col(ok, "ttft_mid_ms"), 50), 1),
        "start_frame_p50_ms": rnd(pct(col(ok, "t_start_frame_ms"), 50), 2),
        "start_frame_p99_ms": rnd(pct(col(ok, "t_start_frame_ms"), 99), 2),
        "accum_p50_ms": rnd(pct(col(ok, "accum_ms"), 50), 1),
        # token pacing
        "itl_p50_ms": rnd(pct(col(ok, "itl_ms"), 50), 3),
        "itl_p95_ms": rnd(pct(col(ok, "itl_ms"), 95), 3),
        "tokens_per_s_stream_p50": rnd(pct(col(ok, "tokens_per_s"), 50), 1),
        # RTF
        "rtf_gen_p50": rnd(pct(col(ok, "server_rtf_gen"), 50), 3),
        "rtf_gen_p95": rnd(pct(col(ok, "server_rtf_gen"), 95), 3),
        "rtf_wall_p50": rnd(pct(col(ok, "rtf_wall"), 50), 3),
        "rtf_wall_p95": rnd(pct(col(ok, "rtf_wall"), 95), 3),
        "rtf_wall_max": rnd(max(col(ok, "rtf_wall")), 3),
        # smoothness
        "gap_p50_ms": rnd(pct(col(ok, "gap_p50_ms"), 50), 1),
        "jitter_p99_ms": rnd(pct(col(ok, "gap_p99_ms"), 99), 1),
        "gap_max_ms": rnd(max(col(ok, "gap_max_ms")), 1),
        "late_frame_pct": rnd(100.0 * sum(r["late_frames"] for r in ok)
                              / max(1, sum(r["frames"] for r in ok)), 2),
        "clean_stream_pct": rnd(100.0 * sum(r["clean"] for r in ok) / len(ok), 1),
        "max_underrun_ms": rnd(max(col(ok, "max_underrun_ms")), 1),
        # e2e
        "e2e_p50_s": rnd(pct(col(ok, "wall_ms"), 50) / 1000.0, 3),
        "e2e_p95_s": rnd(pct(col(ok, "wall_ms"), 95) / 1000.0, 3),
        # bookkeeping
        "gap_over_7itl_p50": rnd(pct(col(ok, "gap_over_7itl"), 50), 3),
        "transport_ms_p50": rnd(pct(col(ok, "transport_ms"), 50), 2),
        "connect_ms_p99": rnd(pct(col(recs, "connect_ms"), 99), 2),
    }
    if errs:
        s["error_sample"] = errs[:3]
    return s


def rnd(x, n=2):
    return round(x, n) if isinstance(x, (int, float)) else x


# --------------------------------------------------------------------------- #
# background sampler: /metrics + nvidia-smi
# --------------------------------------------------------------------------- #
def sampler(base, out_path, stop_evt, interval=0.25):
    import subprocess
    import urllib.request
    rows = []
    while not stop_evt.is_set():
        t = time.monotonic()
        row = {"t": round(t, 3)}
        try:
            with urllib.request.urlopen(f"{base}/metrics", timeout=2) as r:
                m = json.load(r)
            row.update(req_total=m.get("requests_total"),
                       errors=m.get("errors_total"),
                       streams_active=m.get("streams_active"),
                       audio_s_total=m.get("audio_seconds_total"))
        except Exception:
            pass
        try:
            q = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,utilization.memory,memory.used,"
                 "power.draw,clocks.sm,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3)
            f = [x.strip() for x in q.stdout.strip().split(",")]
            if len(f) == 6:
                row.update(gpu_util=float(f[0]), mem_util=float(f[1]),
                           mem_used_mb=float(f[2]), power_w=float(f[3]),
                           sm_mhz=float(f[4]), temp_c=float(f[5]))
        except Exception:
            pass
        rows.append(row)
        time.sleep(interval)
    with open(out_path, "w") as fh:
        json.dump(rows, fh)


# --------------------------------------------------------------------------- #
# schedule builders
# --------------------------------------------------------------------------- #
def sched_burst(n, text, fire_at, transport="ws"):
    """All n requests fire at the same instant."""
    return [{"i": i, "text": text, "fire_at": fire_at, "transport": transport}
            for i in range(n)]


def sched_offsets(offsets, text, fire_at, transport="ws"):
    return [{"i": i, "text": text, "fire_at": fire_at + off, "transport": transport}
            for i, off in enumerate(offsets)]


def exp_offsets(rate, n, rng):
    """Poisson process: exponential inter-arrivals, mean 1/rate."""
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(rate)
    return out


def lognorm_offsets(rate, n, rng, sigma=1.0):
    """Same mean inter-arrival as `rate`, lognormal spread (burstier than Poisson)."""
    mu = math.log(1.0 / rate) - sigma * sigma / 2.0     # E[X] = 1/rate
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        t += math.exp(rng.gauss(mu, sigma))
    return out


def uniform_offsets(rate, n, rng):
    """Same mean, bounded spread: U(0, 2/rate)."""
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.uniform(0.0, 2.0 / rate)
    return out


def pareto_offsets(rate, n, rng, alpha=1.5):
    """Heavy-tailed inter-arrivals, same mean: long lulls then tight clumps."""
    scale = (alpha - 1.0) / alpha / rate                # E[X] = 1/rate
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        t += scale * rng.paretovariate(alpha)
    return out


def cv(offsets):
    """Coefficient of variation of inter-arrival times: 0=clockwork, 1=Poisson."""
    d = [offsets[i] - offsets[i - 1] for i in range(1, len(offsets))]
    if len(d) < 2:
        return None
    m = sum(d) / len(d)
    var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
    return math.sqrt(var) / m if m else None


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def run_phase(host, port, jobs, n_procs, prep_s=1.0):
    """Give workers prep_s to connect, then fire. Returns (records, wall_s)."""
    fire_at = time.monotonic() + prep_s
    for j in jobs:
        j["fire_at"] = fire_at + j.get("offset", 0.0)
    t0 = time.monotonic()
    recs = [enrich(r) for r in fan_out(host, port, jobs, n_procs)]
    wall = time.monotonic() - max(t0, fire_at)
    return recs, wall


def mode_burst(host, port, levels, procs, text, out, settle):
    print(f"\n=== BURST (simultaneous, closed-loop)  text={text}", flush=True)
    rows = []
    for n in levels:
        jobs = [{"i": i, "text": text, "offset": 0.0} for i in range(n)]
        recs, wall = run_phase(host, port, jobs, min(procs, n))
        s = summarise(recs, wall, n, f"burst_{n}")
        s["concurrency"] = n
        rows.append(s)
        print(f"  n={n:4d} ttfa p50/p95 {s['ttfa_p50_ms']}/{s['ttfa_p95_ms']}ms  "
              f"ttft p50 {s['ttft_p50_ms']}ms  itl {s['itl_p50_ms']}ms  "
              f"rtf_gen {s['rtf_gen_p50']}  audio-s/s {s['audio_s_per_s']}  "
              f"tok/s {s['tokens_per_s_agg']}  clean {s['clean_stream_pct']}%", flush=True)
        out["raw"].setdefault("burst", {})[str(n)] = recs
        time.sleep(settle)
    out["burst"] = rows


def mode_parity(host, port, levels, procs, text, out, settle):
    print(f"\n=== TRANSPORT PARITY (HTTP chunked pcm vs WS)", flush=True)
    rows = []
    for n in levels:
        jobs = [{"i": i, "text": text, "offset": 0.0, "transport": "http"}
                for i in range(n)]
        recs, wall = run_phase(host, port, jobs, min(procs, n))
        s = summarise(recs, wall, n, f"http_{n}")
        s["concurrency"] = n
        rows.append(s)
        print(f"  n={n:4d} HTTP ttfa p50/p95 {s['ttfa_p50_ms']}/{s['ttfa_p95_ms']}ms  "
              f"rtf_wall p50 {s['rtf_wall_p50']}  audio-s/s {s['audio_s_per_s']}  "
              f"clean {s['clean_stream_pct']}%", flush=True)
        time.sleep(settle)
    out["parity_http"] = rows


def mode_batching(host, port, procs, text, out, settle):
    """Continuous batching: sequential-vs-concurrent, and mid-flight joiners."""
    print(f"\n=== CONTINUOUS BATCHING", flush=True)
    n = 8
    seq = []
    t0 = time.monotonic()
    for i in range(n):
        r, _ = run_phase(host, port, [{"i": i, "text": text, "offset": 0.0}], 1, prep_s=0.3)
        seq.extend(r)
    seq_wall = time.monotonic() - t0
    con, con_wall = run_phase(host, port,
                              [{"i": i, "text": text, "offset": 0.0} for i in range(n)],
                              min(procs, n))
    s_seq = summarise(seq, seq_wall, n, "sequential_8")
    s_con = summarise(con, con_wall, n, "concurrent_8")
    out["batching_speedup"] = {
        "requests": n,
        "sequential": s_seq, "concurrent": s_con,
        "speedup_x": rnd(s_seq["wall_s"] / s_con["wall_s"], 2),
        "throughput_gain_x": rnd(s_con["audio_s_per_s"] / s_seq["audio_s_per_s"], 2),
        "per_stream_itl_penalty_pct": rnd(
            100 * (s_con["itl_p50_ms"] / s_seq["itl_p50_ms"] - 1), 1),
        "ttfa_penalty_pct": rnd(100 * (s_con["ttfa_p50_ms"] / s_seq["ttfa_p50_ms"] - 1), 1),
    }
    print(f"  sequential 8: {s_seq['wall_s']}s  {s_seq['audio_s_per_s']} audio-s/s  "
          f"itl {s_seq['itl_p50_ms']}ms")
    print(f"  concurrent 8: {s_con['wall_s']}s  {s_con['audio_s_per_s']} audio-s/s  "
          f"itl {s_con['itl_p50_ms']}ms  -> {out['batching_speedup']['speedup_x']}x")
    time.sleep(settle)

    # mid-flight joiners: does a new stream get admitted without waiting for the
    # in-flight batch to drain? Uses TEXT_LONG for incumbents so they are still
    # generating when the joiners arrive.
    print("  mid-flight joiners (incumbents=long, joiners at +2s/+4s/+6s)", flush=True)
    joiner_rows = []
    for base_n, join_n in ((32, 8), (64, 8), (128, 8)):
        jobs = [{"i": i, "text": "long", "offset": 0.0, "role": "incumbent"}
                for i in range(base_n)]
        for k, off in enumerate((2.0, 4.0, 6.0)):
            jobs += [{"i": 1000 + k * 100 + j, "text": "medium", "offset": off,
                      "role": f"joiner_t{off:g}"} for j in range(join_n)]
        recs, wall = run_phase(host, port, jobs, procs)
        groups = {}
        for r in recs:
            groups.setdefault(r["job"].get("role", "?"), []).append(r)
        row = {"incumbents": base_n, "joiners_per_wave": join_n, "wall_s": rnd(wall, 2)}
        for role, rs in sorted(groups.items()):
            g = summarise(rs, wall, len(rs), role)
            row[role] = {"ttfa_p50_ms": g["ttfa_p50_ms"], "ttfa_p95_ms": g["ttfa_p95_ms"],
                         "ttft_p50_ms": g["ttft_p50_ms"], "itl_p50_ms": g["itl_p50_ms"],
                         "rtf_gen_p50": g["rtf_gen_p50"], "clean_pct": g["clean_stream_pct"],
                         "completed": g["completed"]}
        joiner_rows.append(row)
        print(f"    incumbents={base_n}: " + "  ".join(
            f"{k.replace('joiner_t','+')}{'' if k=='incumbent' else 's'}"
            f"={v['ttfa_p50_ms']}ms" for k, v in row.items()
            if isinstance(v, dict)), flush=True)
        time.sleep(settle)
    out["batching_joiners"] = joiner_rows


def mode_arrivals(host, port, procs, text, out, settle, seed):
    """Open-loop arrivals: fixed-rate, Poisson, lognormal, uniform, Pareto."""
    print(f"\n=== ARRIVAL PATTERNS (open-loop, requests issued on a clock)", flush=True)
    rows = []
    specs = []
    for rate in (2, 4, 8, 12, 16):
        specs.append(("poisson", rate, 1.0))
    for dist in ("fixed", "uniform", "lognormal", "pareto"):
        specs.append((dist, 8, 1.0))
    for i, (dist, rate, _) in enumerate(specs):
        rng = random.Random(seed + i)
        n = min(96, max(12, int(rate * 12)))
        if dist == "poisson":
            offs = exp_offsets(rate, n, rng)
        elif dist == "lognormal":
            offs = lognorm_offsets(rate, n, rng)
        elif dist == "uniform":
            offs = uniform_offsets(rate, n, rng)
        elif dist == "pareto":
            offs = pareto_offsets(rate, n, rng)
        else:
            offs = [i / rate for i in range(n)]
        jobs = [{"i": k, "text": text, "offset": o} for k, o in enumerate(offs)]
        recs, wall = run_phase(host, port, jobs, procs)
        s = summarise(recs, wall, n, f"{dist}_{rate}rps")
        s.update(distribution=dist, offered_rps=rate, requests=n,
                 inter_arrival_cv=rnd(cv(offs), 3),
                 max_inflight_est=rnd(max_inflight(recs), 1))
        rows.append(s)
        print(f"  {dist:9s} {rate:2d} rps (cv={s['inter_arrival_cv']}) "
              f"ttfa p50/p95/p99 {s['ttfa_p50_ms']}/{s['ttfa_p95_ms']}/{s['ttfa_p99_ms']}ms  "
              f"rtf_gen {s['rtf_gen_p50']}  achieved {s['rps_completed']} rps  "
              f"peak-inflight {s['max_inflight_est']}  clean {s['clean_stream_pct']}%",
              flush=True)
        out["raw"].setdefault("arrivals", {})[s["label"]] = recs
        time.sleep(settle)
    out["arrivals"] = rows


def max_inflight(recs):
    """Peak overlapping streams, from send/finish timestamps."""
    ev = []
    for r in recs:
        if r.get("ok"):
            ev.append((r["t_send"], 1))
            ev.append((r["t_send"] + r["wall_ms"] / 1000.0, -1))
    ev.sort()
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


def mode_stagger(host, port, procs, text, out, settle):
    """Staggered ramp: 256 requests spaced by a fixed interval."""
    print(f"\n=== STAGGERED RAMP (256 streams, fixed spacing)", flush=True)
    rows = []
    for step_ms in (5, 10, 25, 50, 100):
        n = 256
        offs = [i * step_ms / 1000.0 for i in range(n)]
        jobs = [{"i": k, "text": text, "offset": o} for k, o in enumerate(offs)]
        recs, wall = run_phase(host, port, jobs, procs, prep_s=2.0)
        s = summarise(recs, wall, n, f"stagger_{step_ms}ms")
        s.update(step_ms=step_ms, implied_rps=rnd(1000.0 / step_ms, 1),
                 max_inflight_est=rnd(max_inflight(recs), 1))
        rows.append(s)
        print(f"  step={step_ms:3d}ms ({s['implied_rps']} rps) ttfa p50/p95 "
              f"{s['ttfa_p50_ms']}/{s['ttfa_p95_ms']}ms  ttft p50 {s['ttft_p50_ms']}ms  "
              f"peak-inflight {s['max_inflight_est']}  audio-s/s {s['audio_s_per_s']}  "
              f"clean {s['clean_stream_pct']}%", flush=True)
        time.sleep(settle)
    out["stagger"] = rows


def mode_spike(host, port, procs, text, out, settle):
    """Burst train: idle gap, then N at once, repeated. Shows spike-to-spike drift."""
    print(f"\n=== BURST TRAIN (spike / idle duty cycle)", flush=True)
    rows = []
    # Each spike is its own phase: fire n streams, let them finish, idle, repeat.
    # Running them as one pre-scheduled batch would hold sockets open across the
    # websocket ping interval, so every spike gets fresh connections instead.
    for n, idle_s, spikes in ((64, 4.0, 4), (128, 4.0, 4), (256, 5.0, 3)):
        row = {"spike_size": n, "idle_between_s": idle_s, "spikes": spikes,
               "per_spike": []}
        t_train = time.monotonic()
        for s_i in range(spikes):
            jobs = [{"i": s_i * 1000 + j, "text": text, "offset": 0.0,
                     "spike": s_i} for j in range(n)]
            recs, wall = run_phase(host, port, jobs, procs, prep_s=2.0)
            g = summarise(recs, wall, n, f"spike{s_i}")
            row["per_spike"].append({
                "spike": s_i, "ttfa_p50_ms": g["ttfa_p50_ms"],
                "ttfa_p95_ms": g["ttfa_p95_ms"], "ttfa_p99_ms": g["ttfa_p99_ms"],
                "ttft_p50_ms": g["ttft_p50_ms"], "itl_p50_ms": g["itl_p50_ms"],
                "rtf_gen_p50": g["rtf_gen_p50"], "clean_pct": g["clean_stream_pct"],
                "audio_s_per_s": g["audio_s_per_s"], "wall_s": g["wall_s"],
                "completed": g["completed"], "errors": g["errors"]})
            time.sleep(idle_s)
        row["wall_s"] = rnd(time.monotonic() - t_train, 2)
        rows.append(row)
        print(f"  spike={n}, {idle_s}s idle between: " + " | ".join(
            f"#{p['spike']+1} ttfa {p['ttfa_p50_ms']}ms rtf {p['rtf_gen_p50']}"
            for p in row["per_spike"]), flush=True)
        time.sleep(settle)
    out["spike_train"] = rows


def mode_sustained(host, port, levels, procs, text, out, duration, settle):
    """Closed-loop sustained load: N workers loop for `duration` seconds."""
    print(f"\n=== SUSTAINED (N kept in flight for {duration}s)", flush=True)
    rows = []
    for n in levels:
        until = time.monotonic() + 2.0 + duration
        jobs = [{"i": s, "text": text, "offset": 0.0, "slot": s, "until": until}
                for s in range(n)]
        recs, wall = run_phase(host, port, jobs, procs, prep_s=2.0)
        s = summarise(recs, wall, len(jobs), f"sustained_{n}")
        s["concurrency"] = n
        s["max_inflight_est"] = rnd(max_inflight(recs), 1)
        rows.append(s)
        print(f"  n={n:4d} {s['completed']} done  ttfa p50/p95/p99 {s['ttfa_p50_ms']}/"
              f"{s['ttfa_p95_ms']}/{s['ttfa_p99_ms']}ms  rtf_gen {s['rtf_gen_p50']}  "
              f"audio-s/s {s['audio_s_per_s']}  peak-inflight {s['max_inflight_est']}  "
              f"clean {s['clean_stream_pct']}%", flush=True)
        time.sleep(settle)
    out["sustained"] = rows


def mode_mixed(host, port, procs, out, settle):
    """Heterogeneous text lengths in one batch: head-of-line behaviour."""
    print(f"\n=== MIXED LENGTHS (short/medium/long together)", flush=True)
    rows = []
    for n in (32, 64, 128):
        kinds = ["short", "medium", "long"]
        jobs = [{"i": i, "text": kinds[i % 3], "offset": 0.0} for i in range(n)]
        recs, wall = run_phase(host, port, jobs, procs, prep_s=1.5)
        by = {}
        for r in recs:
            by.setdefault(r["job"]["text"], []).append(r)
        row = {"concurrency": n, "wall_s": rnd(wall, 2)}
        for k, rs in by.items():
            g = summarise(rs, wall, len(rs), k)
            row[k] = {"ttfa_p50_ms": g["ttfa_p50_ms"], "ttfa_p95_ms": g["ttfa_p95_ms"],
                      "ttft_p50_ms": g["ttft_p50_ms"], "itl_p50_ms": g["itl_p50_ms"],
                      "rtf_gen_p50": g["rtf_gen_p50"], "audio_s": g["audio_s_produced"],
                      "e2e_p50_s": g["e2e_p50_s"], "clean_pct": g["clean_stream_pct"],
                      "completed": g["completed"]}
        rows.append(row)
        print(f"  n={n}: " + "  ".join(
            f"{k}: ttfa {v['ttfa_p50_ms']}ms itl {v['itl_p50_ms']}ms rtf {v['rtf_gen_p50']}"
            for k, v in row.items() if isinstance(v, dict)), flush=True)
        time.sleep(settle)
    out["mixed_lengths"] = rows


def mode_prefill(host, port, procs, out, settle):
    """Prompt-length scaling. The DIFFERENCE in TTFA between two prompt lengths at
    the same batch width is pure prefill cost: every term after prefill (the
    27-token wait, the SNAC decode) is identical, so it cancels. This is the one
    assumption-free read on the prefill component of TTFT.

    Generation is capped at 64 tokens so each probe costs ~0.3 s instead of
    synthesising 100 s of audio.
    """
    print(f"\n=== PREFILL PROBE (TTFA vs prompt length, generation capped)", flush=True)
    global TEXTS
    rows = []
    variants = {}
    for mult, name in ((1, "p1"), (4, "p4"), (12, "p12"), (30, "p30")):
        variants[name] = TEXT_MEDIUM * mult
    TEXTS.update(variants)
    for name, txt in variants.items():
        for n in (1, 32, 128):
            jobs = [{"i": i, "text": name, "offset": 0.0, "max_tokens": 64}
                    for i in range(n)]
            recs, wall = run_phase(host, port, jobs, min(procs, n), prep_s=1.0)
            s = summarise(recs, wall, n, f"prefill_{name}_n{n}")
            s.update(prompt_chars=len(txt), concurrency=n, max_tokens=64)
            rows.append(s)
            print(f"  chars={len(txt):5d} n={n:3d}  ttfa p50 {s['ttfa_p50_ms']}ms  "
                  f"start_frame {s['start_frame_p50_ms']}ms  itl {s['itl_p50_ms']}ms  "
                  f"tokens/s {s['tokens_per_s_stream_p50']}", flush=True)
            time.sleep(max(1.0, settle / 3))
    out["prefill"] = rows


def calibrate(host, port):
    """Measure the unloaded ITL, which anchors the TTFT bracket."""
    global ITL_UNLOADED_MS
    recs = []
    for _ in range(3):
        r, _ = run_phase(host, port, [{"i": 0, "text": "medium", "offset": 0.0}],
                         1, prep_s=0.2)
        recs.extend(r)
    itls = col(recs, "itl_ms")
    if itls:
        ITL_UNLOADED_MS = min(itls)
    ttfa = col(recs, "ttfa_ms")
    cal = {"itl_unloaded_ms": rnd(ITL_UNLOADED_MS, 4),
           "itl_samples_ms": [rnd(x, 4) for x in itls],
           "ttfa_ms": [rnd(x, 2) for x in ttfa],
           "start_frame_ms": [rnd(x, 3) for x in col(recs, "t_start_frame_ms")],
           "gap_over_7itl": [rnd(x, 3) for x in col(recs, "gap_over_7itl")],
           "tokens_per_s": col(recs, "tokens_per_s"),
           "transport_ms": [rnd(x, 3) for x in col(recs, "transport_ms")]}
    print(f"  calibration: unloaded ITL {cal['itl_unloaded_ms']} ms/token, "
          f"gap/(7*ITL) {cal['gap_over_7itl']}, "
          f"start-frame {cal['start_frame_ms']} ms, "
          f"ws transport {cal['transport_ms']} ms", flush=True)
    return cal


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:9000")
    ap.add_argument("--mode", default="all",
                    help="all | burst,batching,arrivals,stagger,spike,sustained,"
                         "mixed,prefill,parity (comma-separated)")
    ap.add_argument("--levels", default="1,2,4,8,16,24,32,48,64,96,128,160,192,224,256")
    ap.add_argument("--sustained-levels", default="16,32,64,128,256")
    ap.add_argument("--parity-levels", default="1,32,128,256")
    ap.add_argument("--text", default="medium", choices=["short", "medium", "long"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="orpheus_load_results.json")
    ap.add_argument("--raw-out", default=None, help="write per-request records here")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    hostport = base.split("://", 1)[1]
    host = hostport.split(":")[0]
    port = int(hostport.split(":")[1]) if ":" in hostport else 80
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    slevels = [int(x) for x in args.sustained_levels.split(",") if x.strip()]
    plevels = [int(x) for x in args.parity_levels.split(",") if x.strip()]
    modes = ([m.strip() for m in args.mode.split(",")] if args.mode != "all" else
             ["burst", "batching", "arrivals", "stagger", "spike", "sustained",
              "mixed", "prefill", "parity"])

    import urllib.request
    with urllib.request.urlopen(f"{base}/health", timeout=10) as r:
        health = json.load(r)
    if not health.get("ready"):
        sys.exit(f"server not ready: {health}")
    with urllib.request.urlopen(f"{base}/metrics", timeout=10) as r:
        m0 = json.load(r)

    out = {"meta": {"target": base, "health": health, "metrics_before": m0,
                    "text_case": args.text, "procs": args.procs,
                    "client_cpus": os.cpu_count(), "seed": args.seed,
                    "started_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "frame_ms": FRAME_MS, "prebuffer_ms": PREBUFFER_MS,
                    "first_window_tokens": FIRST_WINDOW_TOKENS,
                    "modes": modes, "levels": levels},
           "raw": {}}

    stop = mp.Event()
    samp_path = args.out + ".samples.json"
    sp = mp.Process(target=sampler, args=(base, samp_path, stop), daemon=True)
    sp.start()

    t_start = time.monotonic()
    try:
        print("=== CALIBRATION (single stream, idle server)", flush=True)
        out["calibration"] = calibrate(host, port)
        out["meta"]["itl_unloaded_ms"] = ITL_UNLOADED_MS
        if "burst" in modes:
            mode_burst(host, port, levels, args.procs, args.text, out, args.settle)
        if "batching" in modes:
            mode_batching(host, port, args.procs, args.text, out, args.settle)
        if "arrivals" in modes:
            mode_arrivals(host, port, args.procs, args.text, out, args.settle, args.seed)
        if "stagger" in modes:
            mode_stagger(host, port, args.procs, args.text, out, args.settle)
        if "spike" in modes:
            mode_spike(host, port, args.procs, args.text, out, args.settle)
        if "sustained" in modes:
            mode_sustained(host, port, slevels, args.procs, args.text, out,
                           args.duration, args.settle)
        if "mixed" in modes:
            mode_mixed(host, port, args.procs, out, args.settle)
        if "prefill" in modes:
            mode_prefill(host, port, args.procs, out, args.settle)
        if "parity" in modes:
            mode_parity(host, port, plevels, args.procs, args.text, out, args.settle)
    finally:
        stop.set()
        sp.join(timeout=10)
        with urllib.request.urlopen(f"{base}/metrics", timeout=10) as r:
            out["meta"]["metrics_after"] = json.load(r)
        out["meta"]["elapsed_s"] = round(time.monotonic() - t_start, 1)
        out["meta"]["samples_file"] = samp_path
        raw = out.pop("raw")
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
        if args.raw_out:
            with open(args.raw_out, "w") as fh:
                json.dump(raw, fh, default=str)
        print(f"\nwrote {args.out}  ({out['meta']['elapsed_s']}s)")


if __name__ == "__main__":
    main()
