#!/usr/bin/env python3
"""Test the TTS WebSocket server. Saves WAV to files/{text}.wav and prints TTFT + chunk intervals.
Usage: python test_ws_client.py [ws://localhost:8002/ws]"""

import base64
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    from scipy.io import wavfile
except ImportError:
    wavfile = None

try:
    from websocket import create_connection
except ImportError:
    print("Install: pip install websocket-client")
    sys.exit(1)

url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8002/ws"
text = "नमस्ते नमस्ते नमस्ते नमस्ते नमस्ते"
description = "Female voice"

# Sanitize text for filename: allow alphanumeric, spaces -> underscore, limit length
def safe_filename(s: str, max_len: int = 80) -> str:
    s = re.sub(r'[\s/\\:*?"<>|]+', "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_") or "audio"
    return s[:max_len] if len(s) > max_len else s


try:
    ws = create_connection(url)
except Exception as e:
    if "404" in str(e):
        print("404 Not Found. Start the TTS server with:")
        print("  uvicorn websocket_server:app --host 0.0.0.0 --port 8002")
        print("Then run this client again.")
    else:
        print("Connection failed:", e)
    sys.exit(1)

payload = {"text": text, "description": description}
t_send = time.perf_counter()
ws.send(json.dumps(payload))

chunks_int16 = []
sampling_rate = None
ttft_s = None
chunk_times = []

while True:
    msg = ws.recv()
    t_recv = time.perf_counter()
    d = json.loads(msg)
    # Print JSON as it comes (chunk abbreviated for readability)
    to_print = {k: f"<{len(v)} chars>" if k == "chunk" and isinstance(v, str) else v for k, v in d.items()}
    print(json.dumps(to_print, ensure_ascii=False))
    if "error" in d:
        print("Error:", d)
        break
    if ttft_s is None:
        ttft_s = t_recv - t_send
        print(f"Time to first token: {ttft_s*1000:.0f} ms")
    else:
        chunk_times.append(t_recv)
    chunk_b64 = d.get("chunk")
    if chunk_b64:
        raw = base64.b64decode(chunk_b64)
        arr = np.frombuffer(raw, dtype=np.int16)
        chunks_int16.append(arr)
    if d.get("sampling_rate") is not None:
        sampling_rate = d.get("sampling_rate")
    if d.get("is_final"):
        break

ws.close()

if chunk_times:
    intervals = np.diff([t_send] + chunk_times)
    print(f"Chunk intervals (ms): first={intervals[0]*1000:.0f} " + " ".join(f"{t*1000:.0f}" for t in intervals[1:]))
    if len(intervals) > 1 and chunks_int16:
        mean_interval_ms = np.mean(intervals[1:]) * 1000
        rate = sampling_rate or 24000
        total_samples = sum(len(c) for c in chunks_int16)
        audio_per_chunk_ms = (total_samples / rate * 1000) / len(chunks_int16)
        print(f"{mean_interval_ms:.0f} ms chunk interval to evict {audio_per_chunk_ms:.0f} ms of audio chunk")

if chunks_int16 and wavfile is not None:
    out_dir = Path("files")
    out_dir.mkdir(exist_ok=True)
    name = safe_filename(text)
    wav_path = out_dir / f"{name}.wav"
    rate = sampling_rate or 24000
    audio = np.concatenate(chunks_int16)
    wavfile.write(str(wav_path), rate, audio)
    print(f"Wrote {wav_path}")
elif chunks_int16 and wavfile is None:
    print("Install scipy to save WAV: pip install scipy")

print("Done.")
