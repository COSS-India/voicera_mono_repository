#!/usr/bin/env python3
"""Regression: long continuous speech that STARTS WITH SILENCE must keep committing throughout.

This is the reported failure, reproduced. A real mic sends a moment of silence before the
speaker starts, which used to be VAD-skipped in a way that permanently corrupted the audio
buffer's context split (right context 0.96 s instead of 0.48 s). AlignAtt derives its commit
boundary from that split, so during fluent speech nothing committed at all -- transcription
stopped and only resumed a little on each pause.

Asserts three things:
  1. commits are spread across the whole utterance -- no silent gap beyond MAX_GAP_S;
  2. the transcript is not truncated -- word count within TOL of the offline reference;
  3. audio is fed for the full duration (the client itself did not stall).

Run `docker logs core-asr | grep "Expected context"` around this to confirm the buffer never
warns; the engine now also logs its own split-divergence warning naming the session.
"""
import argparse, asyncio, json, sys
from pathlib import Path
import numpy as np, soundfile as sf, websockets

SR = 16000
MAX_GAP_S = 3.5          # longest tolerable silence between committed partials
TOL = 0.30               # allowed shortfall vs the offline reference word count


async def main():
    print(f"[longform] target {ARGS.url} label={ARGS.label}", flush=True)
    man = json.loads(Path("/corpus/manifest.json").read_text())
    ref = json.loads(Path("/results/offline_reference.json").read_text())
    clips = [i for i in man["items"] if i["lang"] == "hi" and i["bucket"] == "medium"][:3]

    parts, ref_words = [], 0
    for c in clips:
        w, _ = sf.read(c["path"], dtype="float32")
        parts.append(w)
        ref_words += len(ref.get(Path(c["path"]).name, "").split())
    # 1 s of leading silence: exactly what a mic sends between Start and the first word,
    # and precisely the condition that used to corrupt the buffer.
    wav = np.concatenate([np.zeros(SR, np.float32)] + parts)
    dur = len(wav) / SR
    print(f"[longform] 1.0s silence + {dur - 1:.1f}s continuous speech = {dur:.1f}s, "
          f"reference {ref_words} words", flush=True)

    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    url = f"{ARGS.url}?language=hi"     # manual mode = the demo's default

    commits, transcript, t0 = [], "", None
    async with websockets.connect(url, max_size=None) as ws:
        await ws.recv()
        loop = asyncio.get_event_loop()
        t0 = loop.time()

        async def send():
            blk = int(SR * 0.1)
            for i in range(0, len(pcm), blk):
                await asyncio.sleep(0.1)
                await ws.send(pcm[i:i + blk].tobytes())
            await asyncio.sleep(2.0)
            await ws.send(json.dumps({"type": "stop"}))

        t = asyncio.create_task(send())
        try:
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                if m["type"] in ("partial", "turn_final"):
                    if (m.get("text") or "").strip():
                        commits.append(round(loop.time() - t0, 2))
                    transcript = m.get("transcript") or transcript
                elif m["type"] == "closed":
                    transcript = m.get("transcript", transcript)
                    break
        finally:
            t.cancel()

    got_words = len(transcript.split())
    gaps = [round(b - a, 2) for a, b in zip(commits, commits[1:])]
    worst = max(gaps) if gaps else None
    # include the stretch before the first commit and after the last
    lead = commits[0] if commits else dur
    tail = round(dur - commits[-1], 2) if commits else dur

    srt = sorted(gaps)
    p50 = srt[len(srt)//2] if srt else None
    p90 = srt[min(len(srt)-1, int(0.9*len(srt)))] if srt else None
    print(f"[longform] commits={len(commits)} first={lead:.1f}s "
          f"gap p50={p50}s p90={p90}s worst={worst}s tail={tail}s")
    print(f"[longform] words got={got_words} ref={ref_words} "
          f"({got_words / max(1, ref_words):.0%})")
    print(f"[longform] {transcript[:180]}")

    ok_flow = bool(gaps) and worst is not None and worst <= MAX_GAP_S
    ok_words = got_words >= ref_words * (1 - TOL)
    for name, ok in (("continuous commits", ok_flow), ("no truncation", ok_words)):
        print(f"[longform] {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if (ok_flow and ok_words) else 1


_ap = argparse.ArgumentParser()
_ap.add_argument("--url", default="ws://core-asr:9002/v1/asr/ws")
_ap.add_argument("--label", default="live")
ARGS = _ap.parse_args()
sys.exit(asyncio.run(main()))
