#!/usr/bin/env python3
"""Wire-contract check for the Orpheus TTS WebSocket, with no Pipecat needed.

Mirrors the aiohttp receive loop and byte handling in services/orpheus/tts.py, so
it verifies the contract that class depends on without importing it. Useful when
Pipecat is not installed, when the voice server venv is not set up, or from inside
the Orpheus container itself:

    ORPHEUS_WS=ws://localhost:8004/v1/tts/ws python tests/orpheus_protocol_check.py

    # or, with no host-side aiohttp:
    docker cp tests/orpheus_protocol_check.py voicera_orpheus_tts:/tmp/c.py
    docker exec voicera_orpheus_tts python /tmp/c.py

What it asserts:

  1. the start frame declares 24000 Hz / s16le / mono, and echoes voice+language
  2. every binary frame is a whole number of 4096-byte (85.33 ms) Orpheus frames
  3. the PCM really is int16 speech, not float32 misread as int16 - checked by
     peak, clipping fraction and zero-crossing rate, because a dtype mistake here
     produces plausible-looking white noise instead of an error
  4. `style` is honoured and measurably changes the audio
  5. a speaker name alone resolves the language, with no `language` field sent
  6. Kashmiri works (the language this integration newly exposed to the dashboard)
  7. an unknown voice yields an error frame and no audio, never silent success
  8. a 3-word clause synthesises cleanly - the cadence FastPunctuationAggregator
     actually produces in a live call, rather than whole sentences

Exit status is non-zero if any check fails.

See also tests/orpheus_tts_smoke.py, which drives the real OrpheusTTSService and
checks the Pipecat frame sequence.
"""
import asyncio, json, os, sys, time
import aiohttp, numpy as np

WS = os.environ.get("ORPHEUS_WS", "ws://localhost:9000/v1/tts/ws")
FAIL = 0
def ok(m): print(f"  ok    {m}")
def bad(m):
    global FAIL; FAIL += 1; print(f"  FAIL  {m}")

async def synth(session, payload):
    """The service class's loop, verbatim in structure."""
    start = None; frames = []; done = None; err = None
    t0 = time.perf_counter(); ttfa = None
    async with session.ws_connect(WS, autoping=True) as ws:
        await ws.send_str(json.dumps(payload))
        while True:
            msg = await ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                d = json.loads(msg.data)
                k = d.get("type")
                if k == "error": err = d.get("message"); break
                if k == "start": start = d
                elif k == "done": done = d; break
            elif msg.type == aiohttp.WSMsgType.BINARY:
                if not msg.data: continue
                if ttfa is None: ttfa = (time.perf_counter()-t0)*1000
                frames.append(msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                err = str(ws.exception()); break
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                break
    return start, frames, done, err, ttfa, (time.perf_counter()-t0)*1000

async def main():
    conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    to = aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)
    async with aiohttp.ClientSession(connector=conn, timeout=to) as s:

        print("=== 1. full sentence, Hindi / Amit / CONV ===")
        p = {"text":"नमस्ते, आज मौसम बहुत अच्छा है।","voice":"Amit","language":"hi","style":"CONV"}
        start, frames, done, err, ttfa, wall = await synth(s, p)
        if err: bad(f"unexpected error: {err}"); return
        ok(f"start frame: {start}")
        if start.get("sample_rate")==24000: ok("sample_rate 24000")
        else: bad(f"sample_rate {start.get('sample_rate')}")
        if start.get("format")=="s16le": ok("format s16le (int16 -> passthrough is correct)")
        else: bad(f"format {start.get('format')}")
        if start.get("channels")==1: ok("channels 1")
        else: bad(f"channels {start.get('channels')}")
        if start.get("language")=="hi" and start.get("voice")=="Amit": ok("language/voice echoed back")
        else: bad(f"echo mismatch {start}")

        total = sum(len(f) for f in frames)
        audio_s = total/2/24000
        print(f"       {len(frames)} frames, {total} bytes, {audio_s:.2f}s audio, "
              f"TTFA {ttfa:.0f}ms, wall {wall:.0f}ms, RTF {(wall/1000)/audio_s:.2f}")
        if total % 4096 == 0: ok(f"total is {total//4096} whole 85.33ms frames")
        else: bad(f"{total} not a multiple of 4096")
        if all(len(f) % 4096 == 0 for f in frames): ok("every frame is a whole number of 85.33ms frames")
        else: bad(f"ragged frame sizes: {sorted({len(f) for f in frames})}")
        if done and "metrics" in done: ok(f"done frame carries metrics ({sorted(done['metrics'])[:4]}...)")
        else: bad(f"done frame {done}")

        pcm = b"".join(frames)
        a = np.frombuffer(pcm, dtype=np.int16)
        peak = int(max(abs(int(a.max())), abs(int(a.min()))))
        clip = float((np.abs(a.astype(np.int32))>32000).mean())
        rms = float(np.sqrt((a.astype(np.float64)**2).mean()))
        print(f"       int16 peak {peak}, rms {rms:.0f}, {clip:.4%} near full scale")
        if peak>500 and clip<0.02: ok("waveform is plausible int16 speech")
        else: bad("waveform wrong (silent, or float32 misread as int16)")
        # A float32 buffer reinterpreted as int16 is near-white; speech is not.
        zc = float((np.diff(np.signbit(a)) != 0).mean())
        print(f"       zero-crossing rate {zc:.3f}")
        if zc < 0.35: ok(f"zero-crossing rate {zc:.3f} is speech-like, not noise")
        else: bad(f"zero-crossing rate {zc:.3f} looks like noise - dtype bug")

        print("\n=== 2. style is honoured and changes the audio ===")
        pn = dict(p); pn["style"]="NEWS"
        s2, f2, d2, e2, _, _ = await synth(s, pn)
        if e2: bad(f"NEWS failed: {e2}")
        else:
            if s2.get("style")=="NEWS": ok("start frame echoes style NEWS")
            else: bad(f"style echo {s2.get('style')}")
            b2=b"".join(f2)
            if b2 != pcm: ok(f"NEWS audio differs from CONV ({len(b2)} vs {len(pcm)} bytes)")
            else: bad("NEWS produced byte-identical audio - style ignored")

        print("\n=== 3. language inferred from speaker name (no language field) ===")
        s3,f3,d3,e3,_,_ = await synth(s, {"text":"வணக்கம், இன்று வானிலை நன்றாக உள்ளது.","voice":"Anitha"})
        if e3: bad(f"Anitha failed: {e3}")
        elif s3.get("language")=="ta": ok(f"voice 'Anitha' resolved to language 'ta' ({sum(len(x) for x in f3)} bytes)")
        else: bad(f"expected ta, got {s3.get('language')}")

        print("\n=== 4. Kashmiri, the language this integration newly exposed ===")
        s4,f4,d4,e4,_,_ = await synth(s, {"text":"آداب، از چھُ موسم ٲنٛدِ خٕش۔","voice":"Ishfaq","language":"ks"})
        if e4: bad(f"Kashmiri failed: {e4}")
        elif f4: ok(f"Ishfaq/ks produced {sum(len(x) for x in f4)} bytes")
        else: bad("Kashmiri produced no audio")

        print("\n=== 5. error path: unknown voice must yield an error frame, not silence ===")
        s5,f5,d5,e5,_,_ = await synth(s, {"text":"test","voice":"NotARealSpeaker"})
        if e5 and not f5: ok(f"error frame, no audio: {e5[:80]}")
        else: bad(f"expected error frame; got err={e5!r} frames={len(f5)}")

        print("\n=== 6. short clause, the cadence FastPunctuationAggregator produces ===")
        s6,f6,d6,e6,t6,w6 = await synth(s, {"text":"नमस्ते जी","voice":"Amit","language":"hi","style":"CONV"})
        if e6: bad(f"fragment failed: {e6}")
        else:
            tb=sum(len(x) for x in f6)
            print(f"       {tb} bytes = {tb/2/24000:.2f}s audio, TTFA {t6:.0f}ms, wall {w6:.0f}ms")
            if tb>0 and tb%4096==0: ok("fragment synthesised cleanly")
            else: bad(f"fragment produced {tb} bytes")

    print(f"\n{'ALL PROTOCOL CHECKS PASSED' if FAIL==0 else f'{FAIL} CHECK(S) FAILED'}")
    return 1 if FAIL else 0

sys.exit(asyncio.run(main()))
