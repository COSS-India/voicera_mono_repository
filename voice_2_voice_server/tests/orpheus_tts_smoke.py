#!/usr/bin/env python3
"""Frame-level smoke test for OrpheusTTSService.

Drives the real Pipecat service class against a running Orpheus server and checks
the frame contract the pipeline depends on, without booting a pipeline. This is
the cheapest place to catch the mistake that matters most here: Orpheus emits
int16 PCM, so a stray float32 conversion produces white noise rather than an
error, and nothing downstream would complain.

Run from the voice_2_voice_server directory, with its venv active:

    export ORPHEUS_TTS_SERVER_URL=ws://localhost:8004
    python tests/orpheus_tts_smoke.py                       # Hindi, voice Amit
    python tests/orpheus_tts_smoke.py --voice Anitha --language ta --style NEWS
    python tests/orpheus_tts_smoke.py --fragment             # 3-word clause, the
                                                            # real pipeline cadence
    python tests/orpheus_tts_smoke.py --out /tmp/orpheus.pcm # then listen to it

Play the dump with either of:

    ffplay -f s16le -ar 24000 -ac 1 /tmp/orpheus.pcm
    aplay  -f S16_LE -r 24000 -c 1 /tmp/orpheus.pcm
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipecat.frames.frames import (  # noqa: E402
    ErrorFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)

from services.orpheus.tts import (  # noqa: E402
    ORPHEUS_SAMPLE_RATE,
    OrpheusTTSService,
)

# A full sentence and a bare clause. FastPunctuationAggregator splits on .!?,। and
# strips the punctuation, so in a live call run_tts sees fragments like the second
# one far more often than the first.
SENTENCE = "नमस्ते, आज मौसम बहुत अच्छा है।"
FRAGMENT = "नमस्ते जी"

SAMPLES = {
    "hi": "नमस्ते, आज मौसम बहुत अच्छा है।",
    "ta": "வணக்கம், இன்று வானிலை மிகவும் நன்றாக உள்ளது.",
    "bn": "নমস্কার, আজ আবহাওয়া খুব ভালো।",
    "mr": "नमस्कार, आज हवामान खूप छान आहे.",
}


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")
    globals()["FAILURES"] += 1


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


FAILURES = 0


async def run(args: argparse.Namespace) -> int:
    if not os.getenv("ORPHEUS_TTS_SERVER_URL"):
        print("ORPHEUS_TTS_SERVER_URL is not set. Try:")
        print("  export ORPHEUS_TTS_SERVER_URL=ws://localhost:8004")
        return 2

    text = args.text or (FRAGMENT if args.fragment else SAMPLES.get(args.language, SENTENCE))

    svc = OrpheusTTSService(
        speaker=args.voice,
        language_id=args.language,
        style=args.style,
    )
    print(f"endpoint : {svc._ws_url}")
    print(f"voice    : {args.voice}   language: {args.language}   style: {args.style}")
    print(f"text     : {text!r}\n")

    # StartFrame's required fields have moved between Pipecat versions; the service
    # only needs `start` to have opened its aiohttp session.
    try:
        await svc.start(StartFrame(audio_out_sample_rate=ORPHEUS_SAMPLE_RATE))
    except TypeError:
        await svc.start(StartFrame())

    started = stopped = 0
    audio_frames = 0
    total_bytes = 0
    rates: set[int] = set()
    channels: set[int] = set()
    errors: list[str] = []
    pcm = bytearray()

    t0 = time.perf_counter()
    ttfa_ms = None
    try:
        async for frame in svc.run_tts(text):
            if isinstance(frame, TTSStartedFrame):
                started += 1
            elif isinstance(frame, TTSAudioRawFrame):
                if ttfa_ms is None:
                    ttfa_ms = (time.perf_counter() - t0) * 1000.0
                audio_frames += 1
                total_bytes += len(frame.audio)
                rates.add(frame.sample_rate)
                channels.add(frame.num_channels)
                pcm += frame.audio
            elif isinstance(frame, TTSStoppedFrame):
                stopped += 1
            elif isinstance(frame, ErrorFrame):
                errors.append(str(getattr(frame, "error", frame)))
    finally:
        await svc.stop(None)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    audio_sec = total_bytes / 2 / ORPHEUS_SAMPLE_RATE
    print("--- frames ---")
    print(f"TTSStartedFrame  : {started}")
    print(f"TTSAudioRawFrame : {audio_frames}  ({total_bytes} bytes = {audio_sec:.2f}s audio)")
    print(f"TTSStoppedFrame  : {stopped}")
    print(f"ErrorFrame       : {len(errors)}{'  ' + '; '.join(errors) if errors else ''}")
    if ttfa_ms is not None:
        print(f"\nTTFA {ttfa_ms:.0f} ms   wall {wall_ms:.0f} ms   "
              f"RTF {(wall_ms / 1000.0) / audio_sec:.2f}" if audio_sec else "")

    print("\n--- assertions ---")
    if not errors:
        ok("no ErrorFrame")
    else:
        fail(f"{len(errors)} ErrorFrame(s): {errors}")
    if started == 1:
        ok("exactly one TTSStartedFrame")
    else:
        fail(f"expected 1 TTSStartedFrame, got {started}")
    if stopped == 1:
        ok("exactly one TTSStoppedFrame")
    else:
        fail(f"expected 1 TTSStoppedFrame, got {stopped}")
    if audio_frames >= 1:
        ok(f"{audio_frames} audio frames")
    else:
        fail("no audio frames")
    if rates == {ORPHEUS_SAMPLE_RATE}:
        ok(f"every frame declares {ORPHEUS_SAMPLE_RATE} Hz")
    else:
        fail(f"expected all frames at {ORPHEUS_SAMPLE_RATE} Hz, saw {sorted(rates)}")
    if channels == {1}:
        ok("every frame is mono")
    else:
        fail(f"expected mono, saw num_channels {sorted(channels)}")
    # 4096 bytes = 2048 samples = one 85.33 ms Orpheus frame. Head and tail chunks
    # carry two frames, so the total is a multiple of 4096 but frames may not be.
    if total_bytes and total_bytes % 4096 == 0:
        ok(f"total is a whole number of 85.33 ms frames ({total_bytes // 4096})")
    else:
        fail(f"{total_bytes} bytes is not a multiple of 4096 - dtype or framing is wrong")
    # int16 speech should not be dominated by extremes; float32 misread as int16 is.
    if pcm:
        import array
        s = array.array("h")
        s.frombytes(bytes(pcm))
        peak = max(max(s), -min(s))
        clipped = sum(1 for v in s if abs(v) > 32000)
        frac = clipped / len(s)
        print(f"       peak |amplitude| {peak}, {frac:.4%} of samples near full scale")
        if peak > 500 and frac < 0.02:
            ok("waveform looks like int16 speech (not a float32 misread)")
        else:
            fail("waveform looks wrong - near-silent, or float32 misread as int16")

    if args.out:
        with open(args.out, "wb") as fh:
            fh.write(bytes(pcm))
        print(f"\nwrote {len(pcm)} bytes to {args.out}")
        print(f"  ffplay -f s16le -ar {ORPHEUS_SAMPLE_RATE} -ac 1 {args.out}")

    print(f"\n{'ALL CHECKS PASSED' if FAILURES == 0 else f'{FAILURES} CHECK(S) FAILED'}")
    return 1 if FAILURES else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default="Amit", help="speaker name from GET /v1/voices")
    ap.add_argument("--language", default="hi", help="language code, e.g. hi / ta / bn")
    ap.add_argument("--style", default="CONV", help="style from GET /v1/styles")
    ap.add_argument("--text", default=None, help="override the sample text")
    ap.add_argument("--fragment", action="store_true",
                    help="use a 3-word clause, the cadence the live pipeline produces")
    ap.add_argument("--out", default=None, help="write raw s16le PCM here to listen to")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
