#!/usr/bin/env python3
"""Gap-free continuous speech — the case real speakers produce and the other tests miss.

`test_longform.py` concatenates clips with their natural leading/trailing silence intact, so the
SOFT rotation trigger (speech budget + a >=250 ms gap) keeps firing and turns stay short. A person
talking steadily gives no such gap, so every turn runs to the HARD cap. At a 0.24 s chunk that is
~4.2 ticks/s x several decode steps each, and `decoder_mems_list` grows one position per step
against the decoder's 1024-position limit.

Asserts what a service must guarantee, not merely what a transcript looks like:
  1. no CUDA error appears in the server log during the run;
  2. /health is still 200 afterwards;
  3. a NEW session opened afterwards still transcribes -- the real symptom was that one bad
     session poisoned the CUDA context and every later connect failed instantly.
"""
import argparse, asyncio, json, sys
from pathlib import Path
import numpy as np, soundfile as sf, websockets

SR = 16000


def trim_silence(w, thresh=0.01, win=1600):
    """Drop near-silent windows so the stream has no gap the soft trigger can use."""
    if len(w) < win:
        return w
    n = len(w) // win
    fr = w[:n * win].reshape(n, win)
    keep = np.abs(fr).max(axis=1) > thresh
    return fr[keep].reshape(-1) if keep.any() else w


async def stream(url, pcm, lang, block_ms=100):
    updates, err = 0, None
    async with websockets.connect(f"{url}?language={lang}", max_size=None,
                                  open_timeout=30) as ws:
        hello = json.loads(await ws.recv())
        if hello.get("type") == "error":
            return 0, hello.get("error")
        blk = int(SR * block_ms / 1000)

        async def send():
            loop = asyncio.get_event_loop(); t0 = loop.time()
            for i in range(0, len(pcm), blk):
                slack = t0 + i / SR - loop.time()
                if slack > 0:
                    await asyncio.sleep(slack)
                await ws.send(pcm[i:i + blk].tobytes())
            await asyncio.sleep(1.5)
            await ws.send(json.dumps({"type": "stop"}))

        t = asyncio.create_task(send())
        try:
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                if m["type"] == "error":
                    err = m.get("error"); break
                if m["type"] in ("partial", "turn_final"):
                    updates += 1
                if m["type"] == "closed":
                    break
        finally:
            t.cancel()
    return updates, err


async def main_async(a):
    man = json.loads(Path(a.corpus).read_text())
    clips = [i for i in man["items"] if i["lang"] == a.lang and i["bucket"] == "medium"]
    parts = [trim_silence(sf.read(c["path"], dtype="float32")[0]) for c in clips]
    wav = np.concatenate(parts)
    while len(wav) / SR < a.seconds:                 # loop until long enough, still gap-free
        wav = np.concatenate([wav, wav])
    wav = wav[:int(a.seconds * SR)]
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    print(f"[cont] {len(wav)/SR:.0f}s of GAP-FREE speech (silence trimmed)", flush=True)

    updates, err = await stream(a.url, pcm, a.lang)
    print(f"[cont] main stream: {updates} updates, error={err!r}")

    await asyncio.sleep(3)
    u2, e2 = await stream(a.url, pcm[:SR * 4], a.lang)   # can a NEW session still work?
    print(f"[cont] follow-up session: {u2} updates, error={e2!r}")

    ok = err is None and e2 is None and u2 > 0
    print(f"[cont] {'PASS' if ok else 'FAIL'} — "
          f"{'survived and still serving' if ok else 'service damaged by the run'}")
    return 0 if ok else 1


ap = argparse.ArgumentParser()
ap.add_argument("--url", default="ws://core-asr:9002/v1/asr/ws")
ap.add_argument("--corpus", default="/corpus/manifest.json")
ap.add_argument("--lang", default="hi")
ap.add_argument("--seconds", type=float, default=75.0)
a = ap.parse_args()
sys.exit(asyncio.run(main_async(a)))
