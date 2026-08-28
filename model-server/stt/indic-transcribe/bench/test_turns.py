#!/usr/bin/env python3
"""Both stream-control modes, over the demo's own code path.

  manual (default, endpoint=0) -- a pause must NOT end anything. One continuous stream from
                                  connect to `stop`, and every word survives the gap.
  auto   (opt-in,  endpoint=1) -- a pause commits a turn, and the stream still continues.

The audio is speech, a 1.6 s gap, then more speech: the exact shape that used to close the
socket and silently drop the second half.
"""
import argparse, asyncio, json, sys
from pathlib import Path
import numpy as np, soundfile as sf, websockets

SR = 16000

_ap = argparse.ArgumentParser()
# Was hardcoded to ws://core-asr:9002, which silently ignored --url and made this the one gate
# in verify.sh that could not be pointed at a differently-reachable server.
_ap.add_argument("--url", default="ws://localhost:9002/v1/asr/ws")
_ap.add_argument("--corpus", default="/corpus/manifest.json")
ARGS = _ap.parse_args()


async def run(mode: str, wav, expect_turns) -> dict:
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    ep = "1" if mode == "auto" else "0"
    url = f"{ARGS.url}?language=hi&endpoint={ep}"
    turn_finals, partials, closed = [], 0, None

    async with websockets.connect(url, max_size=None) as ws:
        hello = json.loads(await ws.recv())

        async def send():
            blk = int(SR * 0.1)
            for i in range(0, len(pcm), blk):
                await asyncio.sleep(0.1)
                await ws.send(pcm[i:i + blk].tobytes())
            await asyncio.sleep(1.2)
            await ws.send(json.dumps({"type": "stop"}))

        t = asyncio.create_task(send())
        try:
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
                if m["type"] == "partial":
                    partials += 1
                elif m["type"] == "turn_final":
                    turn_finals.append(m["full_text"])
                elif m["type"] == "closed":
                    closed = m["transcript"]
                    break
        finally:
            t.cancel()

    words = len((closed or "").split())
    ok = len(turn_finals) == expect_turns and words > 6
    print(f"[{mode:6s}] endpointing={hello.get('endpointing')} turns={len(turn_finals)} "
          f"(expected {expect_turns}) partials={partials} words={words} "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"[{mode:6s}] {closed!r}")
    return {"ok": ok}


async def main():
    man = json.loads(Path(ARGS.corpus).read_text())
    clips = [i for i in man["items"] if i["lang"] == "hi" and i["bucket"] == "short"][:2]
    a, _ = sf.read(clips[0]["path"], dtype="float32")
    b, _ = sf.read(clips[1]["path"], dtype="float32")
    wav = np.concatenate([a, np.zeros(int(1.6 * SR), np.float32), b])
    print(f"[audio ] {len(a)/SR:.1f}s + 1.6s silence + {len(b)/SR:.1f}s = {len(wav)/SR:.1f}s\n")

    # manual: the gap must be ignored entirely -> exactly ONE turn, closed by `stop`
    r1 = await run("manual", wav, expect_turns=1)
    print()
    # auto: the gap commits turn 0, the stream continues into turn 1
    r2 = await run("auto", wav, expect_turns=2)
    return 0 if (r1["ok"] and r2["ok"]) else 1


sys.exit(asyncio.run(main()))
