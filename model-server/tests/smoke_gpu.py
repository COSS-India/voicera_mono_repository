"""On-GPU smoke test. Run this on the box, against real models.

Unlike everything else in tests/, this needs the containers actually running.
It round-trips audio: TTS speaks a sentence, STT transcribes it back, and the
two are compared. That covers both models plus the full gateway path without
needing a sample audio file.

Run from model-server/ with the stack up:

    docker compose -f compose.model-server.yml exec -T gateway python - < tests/smoke_gpu.py
    docker compose -f compose.model-server.yml cp gateway:/tmp/tts_out.wav .

Then listen to tts_out.wav and compare it against the same sentence from the
prod TTS. If they sound different, the revamp changed something.
"""

import array
import asyncio
import struct
import sys
import time
import wave

import httpx

GATEWAY = "http://localhost:8000"
SENTENCE = "नमस्ते, आप कैसे हैं? मैं ठीक हूँ, धन्यवाद।"
OUT_WAV = "/tmp/tts_out.wav"


def ok(label, passed, detail=""):
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    return passed


def resample_linear(samples, src_rate, dst_rate):
    """Pure-python linear resample. Crude, but this is a smoke test and the
    gateway image has no numpy."""
    if src_rate == dst_rate:
        return samples
    ratio = src_rate / dst_rate
    out = array.array("f")
    n = len(samples)
    i = 0.0
    while i < n - 1:
        lo = int(i)
        frac = i - lo
        out.append(samples[lo] * (1.0 - frac) + samples[lo + 1] * frac)
        i += ratio
    return out


def to_pcm16(samples):
    return b"".join(
        struct.pack("<h", max(-32768, min(32767, int(s * 32767)))) for s in samples
    )


async def main():
    results = []

    print("\n=== 1. gateway is up and knows its upstreams ===")
    async with httpx.AsyncClient(timeout=30) as c:
        h = (await c.get(f"{GATEWAY}/health")).json()
        print(f"  status: {h['status']}")
        for kind, v in h["upstreams"].items():
            print(f"    {kind}: deployed={v.get('deployed')} reachable={v.get('reachable')}")
        results.append(ok("both models reachable",
                          all(v.get("reachable") for v in h["upstreams"].values()
                              if v.get("deployed"))))

        m = (await c.get(f"{GATEWAY}/models")).json()
        live = [k for k, v in m["deployed"].items() if v]
        results.append(ok("catalogue reports what is deployed", bool(live), str(m["deployed"])))

    print("\n=== 2. TTS speaks the sentence ===")
    print(f"  asked for: {SENTENCE}")
    pcm = array.array("f")
    rate = None
    fmt = None
    t0 = time.perf_counter()
    ttfb = None
    async with httpx.AsyncClient(timeout=120) as c, c.stream(
        "POST",
        f"{GATEWAY}/v1/audio/speech",
        json={
            "input": SENTENCE,
            "voice": "Divya",
            "instructions": "A clear, natural voice with good audio quality.",
            "language": "hi",
            "response_format": "pcm_f32le",
        },
    ) as r:
        if r.status_code != 200:
            body = (await r.aread())[:200]
            print(f"  TTS error: HTTP {r.status_code} {body!r}")
        else:
            rate = int(r.headers.get("X-Sample-Rate", 0)) or None
            fmt = r.headers.get("X-Audio-Format")
            # Chunked HTTP can split a float across reads. Carry the tail
            # forward or every later sample is garbage.
            remainder = b""
            async for chunk in r.aiter_raw():
                if not chunk:
                    continue
                if ttfb is None:
                    ttfb = time.perf_counter() - t0
                buf = remainder + chunk
                usable = len(buf) - (len(buf) % 4)
                remainder = buf[usable:]
                if usable:
                    pcm.frombytes(buf[:usable])
    total = time.perf_counter() - t0
    dur = len(pcm) / rate if rate else 0
    print(f"  sample rate      : {rate} Hz  (from the response header, not assumed)")
    print(f"  audio format     : {fmt}")
    print(f"  time to first    : {ttfb * 1000:.0f} ms" if ttfb else "  no audio")
    print(f"  wall clock       : {total:.2f} s")
    print(f"  audio duration   : {dur:.2f} s")
    print(f"  realtime factor  : {total / dur:.2f}x" if dur else "")
    results.append(ok("TTS synthesised audio", dur > 0.5, f"{dur:.2f}s"))

    if dur > 0:
        with wave.open(OUT_WAV, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(to_pcm16(pcm))
        print(f"  written          : {OUT_WAV}  <- copy this out and listen to it")

    print("\n=== 3. STT transcribes it back ===")
    if dur > 0.5:
        raw = to_pcm16(resample_linear(pcm, rate, 16000))
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                f"{GATEWAY}/v1/audio/transcriptions",
                files={"file": ("audio.pcm", raw, "application/octet-stream")},
                data={"language": "hi"},
            )
            text = r.json().get("text", "") if r.status_code == 200 else f"HTTP {r.status_code}"
        print(f"  asked TTS to say : {SENTENCE}")
        print(f"  STT heard        : {text}")
        results.append(ok("STT returned a transcript", bool(text.strip()), f"{len(text)} chars"))
    else:
        results.append(ok("STT round-trip", False, "no audio to transcribe"))

    print("\n=== 4. hanging up mid-sentence frees the slot ===")
    # Barge-in. The server should evict the request rather than keep generating.
    async with httpx.AsyncClient(timeout=60) as c, c.stream(
        "POST",
        f"{GATEWAY}/v1/audio/speech",
        json={"input": SENTENCE * 4, "voice": "Divya", "language": "hi"},
    ) as r:
        n = 0
        async for chunk in r.aiter_raw():
            n += len(chunk)
            if n > 4096:
                break
    await asyncio.sleep(1.0)
    async with httpx.AsyncClient(timeout=30) as c:
        h = (await c.get(f"{GATEWAY}/health")).json()
    results.append(ok("server still healthy after an aborted stream",
                      h.get("status") == "healthy", h.get("status", "?")))

    print(f"\n=== {sum(results)}/{len(results)} passed ===")
    print("The transcript will not match word for word -- resampling here is crude "
          "and STT is imperfect. What matters is that it is recognisably the same "
          "sentence, and that tts_out.wav sounds like your prod TTS.\n")
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
