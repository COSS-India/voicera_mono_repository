"""
test_server_concurrent.py

Accurate concurrent OmniVoice server benchmark (voice-design mode).

Timing protocol (barrier / one-shot fire):
  1. Open ALL WebSocket connections first  (connect time is NOT in latency)
  2. Wait until every socket is ready      (asyncio.Barrier)
  3. Record t_fire, then send ALL payloads in the same instant
  4. Latency per request = time from t_fire → that request's "done" message

This removes handshake skew so numbers reflect true concurrent TTS latency.

Default text:  हॅलो, तुम्ही अजून कॉलवर आहात का
Default design: female, young adult, high pitch

Usage (from OmniVoice/ with the omnivoice venv):
    python test_server_concurrent.py
    python test_server_concurrent.py --num_requests 30 --url http://localhost:8005
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import statistics
import time
import wave
from typing import Any, Optional

import websockets

DEFAULT_TEXT = "हॅलो, तुम्ही अजून कॉलवर आहात का हॅलो, तुम्ही अजून कॉलवर आहात का हॅलो"
DEFAULT_INSTRUCT = "female, young adult, high pitch"
DEFAULT_LANGUAGE = "mr"  # Marathi


def _derive_ws_url(raw: str) -> str:
    base = raw.strip().rstrip("/")
    low = base.lower()
    if low.startswith("https://"):
        return "wss://" + base[8:] + "/ws/tts"
    if low.startswith("http://"):
        return "ws://" + base[7:] + "/ws/tts"
    if not base.endswith("/ws/tts"):
        return base + "/ws/tts"
    return base


def _wav_duration_seconds(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


async def _recv_done(ws) -> dict[str, Any]:
    """Read WS messages until status=done or error."""
    while True:
        raw = await ws.recv()
        data = json.loads(raw)
        status = data.get("status")
        if status in ("queued", "processing"):
            continue
        return data


async def run(args: argparse.Namespace) -> None:
    os.makedirs(args.res_dir, exist_ok=True)
    ws_url = _derive_ws_url(args.url)
    n = args.num_requests

    payload: dict[str, Any] = {
        "text": args.text,
        "language_id": args.language,
        "instruct": args.instruct,
    }
    if args.speed is not None and args.speed != 1.0:
        payload["speed"] = args.speed
    payload_str = json.dumps(payload)

    print("=" * 68)
    print("OmniVoice concurrent test  (barrier fire · accurate timing)")
    print("=" * 68)
    print(f"  URL:          {args.url}")
    print(f"  WS:           {ws_url}")
    print(f"  Requests:     {n}")
    print(f"  Language:     {args.language}")
    print(f"  Instruct:     {args.instruct}")
    print(f"  Text:         {args.text}")
    print(f"  Output dir:   {os.path.abspath(args.res_dir)}")
    print("=" * 68)
    print()
    print("Phase 1: opening all WebSocket connections ...")

    # ---- Phase 1: connect all sockets (not counted in latency) ----
    t_connect0 = time.perf_counter()
    sockets = []
    connect_errors: list[tuple[int, str]] = []
    for i in range(n):
        try:
            ws = await websockets.connect(
                ws_url, open_timeout=15, max_size=32 * 1024 * 1024
            )
            sockets.append((i, ws))
        except Exception as exc:
            connect_errors.append((i, str(exc)))
            sockets.append((i, None))
    connect_wall = time.perf_counter() - t_connect0
    ready = [(i, ws) for i, ws in sockets if ws is not None]
    print(f"  Connected {len(ready)}/{n} in {connect_wall:.3f}s")
    if connect_errors:
        for i, err in connect_errors:
            print(f"  connect FAIL req_{i:03d}: {err}")
    if not ready:
        print("No connections — aborting.")
        return

    # ---- Phase 2: barrier — all ready workers wait, then fire together ----
    barrier = asyncio.Barrier(len(ready) + 1)  # +1 for the coordinator
    fire_time: dict[str, float] = {}  # shared box: {"t": ...}

    async def worker(req_id: int, ws) -> dict[str, Any]:
        error: Optional[str] = None
        audio_bytes = b""
        server_audio_dur = 0.0
        server_synth_time = 0.0
        server_rtf = 0.0
        server_req_id = ""
        t_done = 0.0

        try:
            # Block until every connection is ready; coordinator releases us all at once
            await barrier.wait()
            await ws.send(payload_str)
            data = await _recv_done(ws)
            t_done = time.perf_counter()

            status = data.get("status")
            if status == "error":
                error = data.get("detail", "unknown error")
            elif status == "done":
                server_req_id = data.get("request_id", "")
                server_audio_dur = float(data.get("audio_duration") or 0)
                server_synth_time = float(data.get("synth_time") or 0)
                server_rtf = float(data.get("rtf") or 0)
                b64 = data.get("audio_b64") or ""
                if b64:
                    audio_bytes = base64.b64decode(b64)
            else:
                error = f"unexpected status: {status}"
        except Exception as exc:
            error = str(exc)
            t_done = time.perf_counter()
        finally:
            try:
                await ws.close()
            except Exception:
                pass

        t_fire = fire_time["t"]
        latency = t_done - t_fire if t_done else float("inf")

        actual_dur = 0.0
        if audio_bytes:
            try:
                actual_dur = _wav_duration_seconds(audio_bytes)
                path = os.path.join(args.res_dir, f"req_{req_id:03d}.wav")
                with open(path, "wb") as fh:
                    fh.write(audio_bytes)
            except Exception as exc:
                error = error or f"wav decode failed: {exc}"

        audio_dur = actual_dur or server_audio_dur
        return {
            "id": req_id,
            "server_req_id": server_req_id,
            "latency_s": latency,          # from shared t_fire
            "audio_dur_s": audio_dur,
            "server_synth_s": server_synth_time,
            "server_rtf": server_rtf,
            "client_rtf": latency / audio_dur if audio_dur > 0 else float("inf"),
            "bytes": len(audio_bytes),
            "error": error,
            "ok": error is None and audio_dur > 0,
            "t_done": t_done,
        }

    print(f"Phase 2: barrier armed — firing all {len(ready)} payloads at once ...\n")

    worker_tasks = [
        asyncio.create_task(worker(req_id, ws)) for req_id, ws in ready
    ]

    # Coordinator: set fire timestamp, then join the barrier (releases everyone)
    fire_time["t"] = time.perf_counter()
    await barrier.wait()

    results = await asyncio.gather(*worker_tasks)
    t_all_done = time.perf_counter()
    makespan = t_all_done - fire_time["t"]

    # Include connect failures as failed results
    for i, err in connect_errors:
        results.append({
            "id": i,
            "server_req_id": "",
            "latency_s": float("inf"),
            "audio_dur_s": 0.0,
            "server_synth_s": 0.0,
            "server_rtf": 0.0,
            "client_rtf": float("inf"),
            "bytes": 0,
            "error": f"connect: {err}",
            "ok": False,
            "t_done": 0.0,
        })

    results = sorted(results, key=lambda r: r["id"])
    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    col = "  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}"
    print(col.format("ID", "LATENCY", "AUDIO_DUR", "SRV_SYNTH", "CLIENT_RTF", "STATUS"))
    print("  " + "-" * 68)
    for r in results:
        status = "OK" if r["ok"] else f"ERR:{(r['error'] or '')[:24]}"
        lat = f"{r['latency_s']:.3f}s" if r["latency_s"] != float("inf") else "inf"
        rtf = f"{r['client_rtf']:.4f}" if r["audio_dur_s"] > 0 else "inf"
        print(
            col.format(
                f"req_{r['id']:03d}",
                lat,
                f"{r['audio_dur_s']:.3f}s",
                f"{r['server_synth_s']:.3f}s",
                rtf,
                status,
            )
        )

    print()
    print("=" * 68)
    print(f"SUMMARY  ({n} concurrent · barrier fire · connect excluded)")
    print("=" * 68)
    print(f"  Connected:                 {len(ready)}/{n}  ({connect_wall:.3f}s connect phase)")
    print(f"  Succeeded:                 {len(ok)}/{n}")
    print(f"  Failed:                    {len(failed)}")
    print(f"  Makespan (t_fire→all done):{makespan:.3f}s")

    if ok:
        lats = [r["latency_s"] for r in ok]
        audios = [r["audio_dur_s"] for r in ok]
        client_rtfs = [r["client_rtf"] for r in ok if r["audio_dur_s"] > 0]
        server_synths = [r["server_synth_s"] for r in ok if r["server_synth_s"] > 0]
        total_audio = sum(audios)
        first_done = min(lats)
        last_done = max(lats)

        print(f"  Total audio generated:     {total_audio:.3f}s")
        print()
        print("  Latency from shared t_fire → each response (accurate)")
        print(f"    Mean:                    {statistics.mean(lats):.3f}s")
        print(f"    Median:                  {statistics.median(lats):.3f}s")
        print(f"    First done (min):        {first_done:.3f}s")
        print(f"    Last done  (max):        {last_done:.3f}s")
        print(f"    Spread (max−min):        {last_done - first_done:.3f}s")
        if len(lats) >= 2:
            print(f"    Stdev:                   {statistics.stdev(lats):.3f}s")
        print()
        if server_synths:
            uniq = sorted(set(round(s, 4) for s in server_synths))
            print("  Server synth_time (from response; same value ⇒ same GPU batch)")
            print(f"    Mean:                    {statistics.mean(server_synths):.3f}s")
            print(f"    Unique values:           {uniq}")
            print()
        if client_rtfs:
            print("  Client RTF  (latency / audio_dur)")
            print(f"    Mean:                    {statistics.mean(client_rtfs):.4f}")
            print(f"    Best:                    {min(client_rtfs):.4f}")
            print(f"    Worst:                   {max(client_rtfs):.4f}")
        print()
        print(f"  Effective throughput:      {total_audio / makespan:.2f}x realtime")
        print(f"  (total_audio / makespan)")

    if failed:
        print()
        print("  Failures:")
        for r in failed:
            print(f"    req_{r['id']:03d}: {r['error']}")

    print(f"\n  WAVs saved to: {os.path.abspath(args.res_dir)}")
    print("=" * 68)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Barrier-fire concurrent OmniVoice TTS tester (accurate timing)"
    )
    p.add_argument("--url", type=str, default="http://localhost:8005")
    p.add_argument("--num_requests", type=int, default=30)
    p.add_argument("--text", type=str, default=DEFAULT_TEXT)
    p.add_argument("--instruct", type=str, default=DEFAULT_INSTRUCT)
    p.add_argument("--language", type=str, default=DEFAULT_LANGUAGE)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--res_dir", type=str, default="results_server_concurrent/")
    return p


def main() -> None:
    args = get_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
