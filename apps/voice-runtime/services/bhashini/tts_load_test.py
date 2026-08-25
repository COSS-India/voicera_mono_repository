"""
Bhashini TTS load tester (gRPC via BhashiniTTSService).

Edit the CONFIG block below, then run:
  cd voice_2_voice_server
  python services/bhashini/tts_load_test.py

CLI flags (--mode, -n, etc.) override CONFIG when passed explicitly.

Requires BHASHINI_TTS_FUNCTION_ID and BHASHINI_TTS_AUTH_TOKEN in voice_2_voice_server/.env
"""

from __future__ import annotations

# =============================================================================
# CONFIG — edit these variables for your test run
# =============================================================================

# Total number of TTS requests to send.
NUM_REQUESTS = 100

# Arrival pattern:
#   "burst"      — all requests start at once (max concurrency = NUM_REQUESTS)
#   "stagger"    — request i starts after i * INTERVAL_MS (real-world drip)
#   "sequential" — one request finishes before the next starts
MODE = "burst"

# Stagger only: milliseconds between starting each request (e.g. 100, 200, 300).
INTERVAL_MS = 200

# Sequential only: seconds to wait after each request completes before the next.
GAP_S = 0.0

# Voice settings (speaker name is prepended to description by BhashiniTTSService).
SPEAKER = "Radha"
DESCRIPTION = (
    "speaks with a slightly higher pitch in a close sounding environment. "
    "The voice is clear with subtle emotional depth and a normal pace "
    "all captured in high-quality recording."
)

# Input text. Leave empty to rotate through MR_SENTENCES below.
TEXT = ""

# Language code for Bhashini (Marathi = "mr").
LANGUAGE = "mr"
SAMPLE_RATE = 44100

# Set to a folder path to save WAV files, or "" to skip saving.
OUT_DIR = ""

# If True, stop on the first error or missing audio.
STRICT = False

# =============================================================================

import argparse
import asyncio
import re
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSStoppedFrame

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from services.bhashini.tts import BhashiniTTSService  # noqa: E402

MR_SENTENCES: list[str] = [
    "नमस्कार! तुम्ही माझा आवाज स्पष्टपणे ऐकू शकता का?",
    "आज हवामान खूप छान आहे आणि हलकी वारा वाहत आहे.",
    "कृपया हे वाक्य लक्षपूर्वक ऐका आणि मग सांगा कसे वाटले.",
    "हे एक चाचणी वाक्य आहे जेणेकरून आपण वेळ आणि गुणवत्ता दोन्ही तपासू शकू.",
    "धन्यवाद! तुमचा दिवस शुभ असो आणि तुम्ही निरोगी राहा.",
]


def _safe_filename(s: str, max_len: int = 120) -> str:
    s = s.strip()
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", s)
    s = re.sub(r"\s+", "_", s)
    return (s.strip("._") or "output")[:max_len]


def _fmt_seconds(x: float | None, *, ms: bool = False) -> str:
    if x is None:
        return "n/a"
    if ms:
        return f"{x * 1000.0:.2f}ms"
    return f"{x:.3f}s"


def _streaming_from_rtf(rtf: float | None) -> tuple[str, str]:
    if rtf is None:
        return "n/a", "RTF missing"
    if rtf < 1.0:
        return "yes", "RTF < 1 (faster than real-time playback length)"
    return "no", "RTF >= 1 (slower than real-time; needs buffering)"


def _write_wav(path: Path, sample_rate: int, pcm_int16: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())


async def run_one(
    *,
    index: int,
    start_delay_s: float,
    tts: BhashiniTTSService,
    text: str,
    out_dir: Path | None,
    strict: bool,
    active_counter: list[int],
    peak_counter: list[int],
) -> dict:
    await asyncio.sleep(start_delay_s)

    active_counter[0] += 1
    peak_counter[0] = max(peak_counter[0], active_counter[0])

    t_send = time.monotonic()
    t_first_audio: float | None = None
    t_last_audio: float | None = None
    t_done: float | None = None

    chunks: list[bytes] = []
    inter_chunk_gaps_s: list[float] = []
    last_chunk_mono: float | None = None
    sample_rate = tts.sample_rate
    error: str | None = None

    try:
        async for frame in tts.run_tts(text):
            now = time.monotonic()
            if isinstance(frame, TTSAudioRawFrame):
                if t_first_audio is None:
                    t_first_audio = now
                elif last_chunk_mono is not None:
                    inter_chunk_gaps_s.append(now - last_chunk_mono)
                last_chunk_mono = now
                t_last_audio = now
                sample_rate = frame.sample_rate
                chunks.append(frame.audio)
            elif isinstance(frame, ErrorFrame):
                error = frame.error
                break
            elif isinstance(frame, TTSStoppedFrame):
                t_done = now
    finally:
        active_counter[0] -= 1

    if error:
        if strict:
            raise RuntimeError(f"request {index}: {error}")
        return {
            "index": index,
            "text": text,
            "ok": False,
            "error": error,
            "start_delay_s": start_delay_s,
        }

    if not chunks:
        msg = f"request {index}: no audio received"
        if strict:
            raise RuntimeError(msg)
        return {
            "index": index,
            "text": text,
            "ok": False,
            "error": msg,
            "start_delay_s": start_delay_s,
        }

    pcm = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32767.0
    audio_s = float(pcm.size) / float(sample_rate)

    out_path: Path | None = None
    if out_dir is not None:
        out_path = out_dir / f"{_safe_filename(text)}_{index:03d}.wav"
        _write_wav(out_path, sample_rate, (pcm * 32767.0).astype(np.int16))

    ttft_s = (t_first_audio - t_send) if t_first_audio is not None else None
    total_s = (t_done - t_send) if t_done is not None else None
    recv_span_s = (
        (t_last_audio - t_first_audio)
        if (t_first_audio is not None and t_last_audio is not None)
        else 0.0
    )
    rtf = (total_s / audio_s) if (total_s is not None and audio_s > 0) else None
    stream_ok, stream_reason = _streaming_from_rtf(rtf)

    n_chunks = len(chunks)
    if inter_chunk_gaps_s:
        ic_ms = [g * 1000.0 for g in inter_chunk_gaps_s]
        ic_avg_ms = statistics.fmean(ic_ms)
        ic_min_ms = min(ic_ms)
        ic_max_ms = max(ic_ms)
    else:
        ic_avg_ms = ic_min_ms = ic_max_ms = None

    return {
        "index": index,
        "text": text,
        "ok": True,
        "start_delay_s": start_delay_s,
        "sample_rate": sample_rate,
        "samples": int(pcm.size),
        "audio_s": audio_s,
        "ttft_s": ttft_s,
        "total_s": total_s,
        "recv_span_s": recv_span_s,
        "rtf": rtf,
        "streamable": stream_ok,
        "streamable_reason": stream_reason,
        "n_chunks": n_chunks,
        "inter_chunk_ms_avg": ic_avg_ms,
        "inter_chunk_ms_min": ic_min_ms,
        "inter_chunk_ms_max": ic_max_ms,
        "wav_path": str(out_path) if out_path is not None else None,
    }


def _prompt_for_index(args: argparse.Namespace, index: int) -> str:
    if args.text:
        return args.text
    return MR_SENTENCES[index % len(MR_SENTENCES)]


async def run_burst_or_stagger(args: argparse.Namespace, tts: BhashiniTTSService) -> list[dict]:
    out_dir = Path(args.out_dir) if args.out_dir else None
    interval_s = args.interval_ms / 1000.0
    active_counter = [0]
    peak_counter = [0]

    if args.mode == "burst":
        print(f"Mode: burst — launching {args.requests} requests concurrently (no stagger).\n")
    else:
        print(
            f"Mode: stagger — {args.requests} requests, one every {args.interval_ms:.0f} ms "
            f"(request i starts after i × {args.interval_ms:.0f} ms).\n"
        )

    tasks = [
        asyncio.create_task(
            run_one(
                index=i,
                start_delay_s=i * interval_s,
                tts=tts,
                text=_prompt_for_index(args, i),
                out_dir=out_dir,
                strict=args.strict,
                active_counter=active_counter,
                peak_counter=peak_counter,
            )
        )
        for i in range(args.requests)
    ]

    wall_start = time.monotonic()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall_total_s = time.monotonic() - wall_start

    failures = [r for r in results if isinstance(r, BaseException)]
    if failures and args.strict:
        raise failures[0]

    oks = [r for r in results if isinstance(r, dict)]
    oks.sort(key=lambda r: r["index"])

    for r in oks:
        if not r.get("ok"):
            print(f"[{r['index']:03d}] ERROR {r.get('error')}")
            continue
        snippet = r["text"] if len(r["text"]) <= 52 else (r["text"][:49] + "...")
        rtf_disp = r["rtf"] if r["rtf"] is not None else float("nan")
        if r["inter_chunk_ms_avg"] is not None:
            ic_str = (
                f"chunks={r['n_chunks']} inter_chunk_ms "
                f"avg={r['inter_chunk_ms_avg']:.2f} min={r['inter_chunk_ms_min']:.2f} "
                f"max={r['inter_chunk_ms_max']:.2f}"
            )
        else:
            ic_str = f"chunks={r['n_chunks']} inter_chunk_ms=n/a (single chunk)"
        print(
            f"[{r['index']:03d}] start+{_fmt_seconds(r['start_delay_s'])}  "
            f"ttft={_fmt_seconds(r['ttft_s'], ms=True)}  total={_fmt_seconds(r['total_s'])}  "
            f"audio={r['audio_s']:.3f}s  rtf={rtf_disp:.3f}  "
            f"streamable={r['streamable']}  {ic_str}  | {snippet}"
        )

    _print_summary(
        args=args,
        results=oks,
        wall_total_s=wall_total_s,
        peak_concurrency=peak_counter[0],
    )
    return oks


async def run_sequential(args: argparse.Namespace, tts: BhashiniTTSService) -> list[dict]:
    out_dir = Path(args.out_dir) if args.out_dir else None
    active_counter = [0]
    peak_counter = [0]

    print(
        "Mode: sequential — each request runs to completion before the next starts. "
        f"Optional gap between requests: {args.gap_s:.3f}s.\n"
    )

    results: list[dict] = []
    wall_start = time.monotonic()

    for i in range(args.requests):
        if i > 0 and args.gap_s > 0:
            await asyncio.sleep(args.gap_s)

        r = await run_one(
            index=i,
            start_delay_s=0.0,
            tts=tts,
            text=_prompt_for_index(args, i),
            out_dir=out_dir,
            strict=args.strict,
            active_counter=active_counter,
            peak_counter=peak_counter,
        )
        results.append(r)

        if not r.get("ok"):
            print(f"[{i:03d}] ERROR {r.get('error')}")
            continue

        snippet = r["text"] if len(r["text"]) <= 52 else (r["text"][:49] + "...")
        rtf_disp = r["rtf"] if r["rtf"] is not None else float("nan")
        if r["inter_chunk_ms_avg"] is not None:
            ic_str = (
                f"chunks={r['n_chunks']} inter_chunk_ms "
                f"avg={r['inter_chunk_ms_avg']:.2f} min={r['inter_chunk_ms_min']:.2f} "
                f"max={r['inter_chunk_ms_max']:.2f}"
            )
        else:
            ic_str = f"chunks={r['n_chunks']} inter_chunk_ms=n/a (single chunk)"
        print(
            f"[{i:03d}] ttft={_fmt_seconds(r['ttft_s'], ms=True)}  total={_fmt_seconds(r['total_s'])}  "
            f"audio={r['audio_s']:.3f}s  rtf={rtf_disp:.3f}  "
            f"streamable={r['streamable']}  {ic_str}  | {snippet}"
        )

    wall_total_s = time.monotonic() - wall_start
    _print_summary(
        args=args,
        results=results,
        wall_total_s=wall_total_s,
        peak_concurrency=peak_counter[0],
    )
    return results


def _print_summary(
    *,
    args: argparse.Namespace,
    results: list[dict],
    wall_total_s: float,
    peak_concurrency: int,
) -> None:
    oks = [r for r in results if r.get("ok")]
    fails = [r for r in results if not r.get("ok")]

    print("\n--- summary ---")
    print(
        f"mode={args.mode} language={args.language} speaker={args.speaker!r} "
        f"ok={len(oks)}/{len(results)} failed={len(fails)} "
        f"wall_total={wall_total_s:.3f}s peak_concurrency={peak_concurrency}"
    )
    if args.mode == "stagger":
        print(f"interval_ms={args.interval_ms}")
    if args.mode == "sequential":
        print(f"gap_between_requests_s={args.gap_s}")

    if not oks:
        raise SystemExit("No successful requests.")

    def line(name: str, xs: list[float], unit: str) -> None:
        if not xs:
            print(f"{name}: n/a")
            return
        print(
            f"{name}: average={statistics.fmean(xs):.2f}{unit}  "
            f"min={min(xs):.2f}{unit}  max={max(xs):.2f}{unit}"
        )

    ttft_ms = [r["ttft_s"] * 1000.0 for r in oks if r.get("ttft_s") is not None]
    total_ms = [r["total_s"] * 1000.0 for r in oks if r.get("total_s") is not None]
    rtf_vals = [r["rtf"] for r in oks if r.get("rtf") is not None]
    gen_ms = [r["recv_span_s"] * 1000.0 for r in oks if r.get("recv_span_s") is not None]
    audio_s_vals = [r["audio_s"] for r in oks]

    line("ttft", ttft_ms, "ms")
    line("total", total_ms, "ms")
    line("gen_span (first->last audio)", gen_ms, "ms")
    line("audio_duration", audio_s_vals, "s")
    line("rtf (total/audio)", rtf_vals, "x")

    pooled_inter_chunk_ms: list[float] = []
    for r in oks:
        if r.get("inter_chunk_ms_avg") is not None and r.get("n_chunks", 0) > 1:
            for _ in range(r["n_chunks"] - 1):
                pooled_inter_chunk_ms.append(r["inter_chunk_ms_avg"])

    if pooled_inter_chunk_ms:
        line("approx inter-chunk gap (per-request avg pooled)", pooled_inter_chunk_ms, "ms")

    stream_yes = sum(1 for r in oks if r.get("rtf") is not None and r["rtf"] < 1.0)
    if rtf_vals:
        mean_rtf = statistics.fmean(rtf_vals)
        _, mean_reason = _streaming_from_rtf(mean_rtf)
        print(
            f"\nStreaming (RTF < 1): {stream_yes}/{len(oks)} requests streamable=yes; "
            f"{len(oks) - stream_yes} streamable=no."
        )
        print(f"Average RTF={mean_rtf:.3f} → {mean_reason}")


async def async_main(args: argparse.Namespace) -> None:
    print("--- active config ---")
    print(f"  mode={args.mode}  requests={args.requests}  language={args.language}")
    if args.mode == "stagger":
        print(f"  interval_ms={args.interval_ms}")
    if args.mode == "sequential":
        print(f"  gap_s={args.gap_s}")
    if args.mode == "burst":
        print(f"  concurrent_requests={args.requests}")
    print(f"  speaker={args.speaker!r}")
    print(f"  text={'(rotate sentences)' if not args.text else args.text[:60] + ('...' if len(args.text) > 60 else '')}")
    print(f"  out_dir={args.out_dir or '(none)'}  strict={args.strict}\n")

    tts = BhashiniTTSService(
        speaker=args.speaker,
        description=args.description,
        language=args.language,
        sample_rate=args.sample_rate,
    )

    if args.mode == "sequential":
        await run_sequential(args, tts)
    else:
        await run_burst_or_stagger(args, tts)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bhashini gRPC TTS load tester with burst, stagger, and sequential modes"
    )
    p.add_argument(
        "--mode",
        choices=("burst", "stagger", "sequential"),
        default=MODE,
        help=f"burst=all at once; stagger=interval between starts; sequential=one after another (CONFIG: {MODE})",
    )
    p.add_argument(
        "-n",
        "--requests",
        type=int,
        default=NUM_REQUESTS,
        help=f"Number of TTS requests to send (CONFIG: {NUM_REQUESTS})",
    )
    p.add_argument(
        "--interval-ms",
        type=float,
        default=INTERVAL_MS,
        help=f"Stagger mode only: ms between starting each request (CONFIG: {INTERVAL_MS})",
    )
    p.add_argument(
        "--gap-s",
        type=float,
        default=GAP_S,
        help=f"Sequential mode only: seconds to wait after each completed request (CONFIG: {GAP_S})",
    )
    p.add_argument(
        "--text",
        default=TEXT,
        help="Fixed Marathi input for every request (empty = rotate built-in sentences)",
    )
    p.add_argument(
        "--speaker",
        default=SPEAKER,
        help=f"Speaker name prepended to description (CONFIG: {SPEAKER})",
    )
    p.add_argument(
        "--description",
        default=DESCRIPTION,
        help="Voice/style description (speaker name is prepended automatically)",
    )
    p.add_argument(
        "--language",
        default=LANGUAGE,
        help=f"Bhashini language code (CONFIG: {LANGUAGE})",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help=f"Expected output sample rate (CONFIG: {SAMPLE_RATE})",
    )
    p.add_argument(
        "--out-dir",
        default=OUT_DIR,
        help="If set, write WAV files here",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        default=STRICT,
        help=f"Fail on first error or missing audio (CONFIG: {STRICT})",
    )
    args = p.parse_args()

    if args.requests < 1:
        p.error("--requests must be >= 1")
    if args.interval_ms < 0:
        p.error("--interval-ms must be >= 0")
    if args.gap_s < 0:
        p.error("--gap-s must be >= 0")
    if args.mode == "burst":
        args.interval_ms = 0.0

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
