"""
Sequential WS TTS load tester (no parallelism).

How requests run (strictly one after another):
  For each utterance: open WebSocket → send JSON → read meta → read all audio
  chunks until ``done`` → connection closes. Only then does the next utterance
  start (optional ``--gap-s`` waits *between* completed requests, not in parallel).

Protocol (see ai4bharat_tts_server/server.py):
1) client sends one JSON: {"prompt": "...", "description": "...", "language": "hi"}
2) server sends JSON {"type":"meta", ...}
3) server streams binary float32 PCM chunks (mono)
4) server sends JSON {"type":"done", ...} and closes

The script records wall-clock gaps between consecutive binary PCM messages
(inter-chunk receive times).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np
import websockets
from scipy.io import wavfile

DEFAULT_URI = "ws://127.0.0.1:8002"

# Rotate among these 5 Hindi sentences.
HI_SENTENCES: list[str] = [
    "नमस्ते! क्या आप मेरी आवाज़ साफ़ सुन पा रहे हैं?",
    "आज मौसम बहुत सुहावना है और हल्की हवा चल रही है।",
    "कृपया इस वाक्य को ध्यान से सुनें और फिर बताइए कि कैसा लगा।",
    "यह एक परीक्षण है ताकि हम समय और गुणवत्ता दोनों का आकलन कर सकें।",
    "धन्यवाद! आपका दिन शुभ हो और आप स्वस्थ रहें।",
]


def _safe_filename(s: str, max_len: int = 120) -> str:
    s = s.strip()
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip("._") or "output"
    return s[:max_len]


def streaming_from_rtf(rtf: float | None) -> tuple[str, str]:
    """
    RTF here = total wall time / audio duration (same as printed metrics).

    RTF < 1  → faster than playback length → can keep up for real-time streaming.
    RTF >= 1 → slower than playback → would fall behind without large buffering.
    """
    if rtf is None:
        return "n/a", "RTF missing"
    if rtf < 1.0:
        return "yes", "RTF < 1 (faster than real-time playback length)"
    return "no", "RTF >= 1 (slower than real-time; needs buffering or shorter audio)"


async def run_one(
    *,
    uri: str,
    prompt: str,
    description: str,
    language: str,
    out_dir: Path | None,
    index: int,
    strict: bool,
) -> dict:
    t_send = time.monotonic()
    t_meta: float | None = None
    t_first_audio: float | None = None
    t_last_audio: float | None = None
    t_done: float | None = None

    chunks: list[np.ndarray] = []
    inter_chunk_gaps_s: list[float] = []
    last_chunk_mono: float | None = None
    sample_rate: int | None = None
    pid: str | None = None

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "prompt": prompt,
                    "description": description,
                    "language": language,
                }
            )
        )

        meta_raw = await ws.recv()
        t_meta = time.monotonic()
        if not isinstance(meta_raw, str):
            raise RuntimeError(f"expected meta JSON (str), got {type(meta_raw)}")
        meta = json.loads(meta_raw)
        if meta.get("type") != "meta":
            raise RuntimeError(f"expected meta, got {meta}")
        sample_rate = int(meta.get("sample_rate", 24000))
        pid = str(meta.get("pid", f"req{index}"))

        while True:
            msg = await ws.recv()
            now = time.monotonic()
            if isinstance(msg, str):
                body = json.loads(msg)
                if body.get("type") == "error":
                    raise RuntimeError(f"server error: {body!r}")
                if body.get("type") != "done":
                    raise RuntimeError(f"expected done, got {body!r}")
                t_done = now
                break

            # Time between successive binary PCM frames (wall clock at client recv).
            if t_first_audio is None:
                t_first_audio = now
            elif last_chunk_mono is not None:
                inter_chunk_gaps_s.append(now - last_chunk_mono)
            last_chunk_mono = now
            t_last_audio = now
            chunks.append(np.frombuffer(msg, dtype=np.float32))

    pcm = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    if pcm.size == 0:
        msg = (
            "no PCM received (meta then done). "
            "This can happen with short prompts when server runs with a large --decode-every; "
            "try server `--decode-every 1` or use a longer prompt."
        )
        if strict:
            raise RuntimeError(msg)
        return {
            "index": index,
            "prompt": prompt,
            "pid": pid,
            "sample_rate": sample_rate,
            "ok": False,
            "error": msg,
        }

    assert sample_rate is not None
    audio_s = float(pcm.size) / float(sample_rate)

    out_path: Path | None = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{_safe_filename(prompt)}_{index:03d}_{pid}.wav"
        wavfile.write(out_path, sample_rate, pcm)

    # Metrics.
    ttft_s = (t_first_audio - t_send) if t_first_audio is not None else None
    total_s = (t_done - t_send) if t_done is not None else None
    recv_span_s = (t_last_audio - t_first_audio) if (t_first_audio is not None and t_last_audio is not None) else 0.0
    meta_latency_s = (t_meta - t_send) if t_meta is not None else None

    rtf = (total_s / audio_s) if (total_s is not None and audio_s > 0) else None
    stream_ok, stream_reason = streaming_from_rtf(rtf)

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
        "prompt": prompt,
        "pid": pid,
        "sample_rate": sample_rate,
        "samples": int(pcm.size),
        "audio_s": audio_s,
        "meta_latency_s": meta_latency_s,
        "ttft_s": ttft_s,
        "total_s": total_s,
        "recv_span_s": recv_span_s,
        "rtf": rtf,
        "streamable": stream_ok,
        "streamable_reason": stream_reason,
        "n_chunks": n_chunks,
        "inter_chunk_gaps_s": inter_chunk_gaps_s,
        "inter_chunk_ms_avg": ic_avg_ms,
        "inter_chunk_ms_min": ic_min_ms,
        "inter_chunk_ms_max": ic_max_ms,
        "wav_path": str(out_path) if out_path is not None else None,
        "ok": True,
    }


def _fmt(x: float | None, *, ms: bool = False) -> str:
    if x is None:
        return "n/a"
    if ms:
        return f"{x * 1000.0:.2f}ms"
    return f"{x:.3f}s"


async def async_main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else None

    print(
        "Mode: strictly sequential — each request runs to completion (through `done`), "
        "then the next starts. Optional --gap-s sleeps only between requests.\n"
    )

    results: list[dict] = []
    for i in range(args.requests):
        if i > 0 and args.gap_s > 0:
            await asyncio.sleep(args.gap_s)

        prompt = HI_SENTENCES[i % len(HI_SENTENCES)]
        r = await run_one(
            uri=args.uri,
            prompt=prompt,
            description=args.description,
            language=args.language,
            out_dir=out_dir,
            index=i,
            strict=args.strict,
        )
        results.append(r)

        if not r.get("ok"):
            print(f"[{i:03d}] ERROR {r.get('error')}")
            continue

        rtf_disp = r["rtf"] if r["rtf"] is not None else float("nan")
        snippet = prompt if len(prompt) <= 52 else (prompt[:49] + "...")
        nc = r["n_chunks"]
        if r["inter_chunk_ms_avg"] is not None:
            ic_str = (
                f"chunks={nc} inter_chunk_ms "
                f"avg={r['inter_chunk_ms_avg']:.2f} min={r['inter_chunk_ms_min']:.2f} max={r['inter_chunk_ms_max']:.2f}"
            )
        else:
            ic_str = f"chunks={nc} inter_chunk_ms=n/a (only one audio chunk)"
        print(
            f"[{i:03d}] ttft={_fmt(r['ttft_s'], ms=True)}  total={_fmt(r['total_s'])}  "
            f"audio={r['audio_s']:.3f}s  rtf={rtf_disp:.3f}  "
            f"streamable={r['streamable']} ({r['streamable_reason']})  "
            f"gen_span={_fmt(r['recv_span_s'])}  {ic_str}  | {snippet}"
        )

    oks = [r for r in results if r.get("ok")]
    if not oks:
        raise SystemExit("No successful requests.")

    ttft_ms = [r["ttft_s"] * 1000.0 for r in oks if r.get("ttft_s") is not None]
    total_ms = [r["total_s"] * 1000.0 for r in oks if r.get("total_s") is not None]
    rtf_vals = [r["rtf"] for r in oks if r.get("rtf") is not None]
    gen_ms = [r["recv_span_s"] * 1000.0 for r in oks if r.get("recv_span_s") is not None]
    audio_s_vals = [r["audio_s"] for r in oks]
    n_chunks_ints = [r["n_chunks"] for r in oks]

    pooled_inter_chunk_ms: list[float] = []
    for r in oks:
        for g in r.get("inter_chunk_gaps_s", []):
            pooled_inter_chunk_ms.append(g * 1000.0)

    def line(name: str, xs: list[float], unit: str) -> None:
        if not xs:
            print(f"{name}: n/a")
            return
        print(
            f"{name}: average={statistics.fmean(xs):.2f}{unit}  "
            f"min={min(xs):.2f}{unit}  max={max(xs):.2f}{unit}"
        )

    stream_yes = sum(1 for r in oks if r.get("rtf") is not None and r["rtf"] < 1.0)
    stream_no = len(oks) - stream_yes

    print("\n--- summary ---")
    print(f"uri={args.uri} ok={len(oks)}/{len(results)} rotate={len(HI_SENTENCES)} gap_between_requests_s={args.gap_s}")
    line("ttft", ttft_ms, "ms")
    line("total", total_ms, "ms")
    line("gen_span (first->last audio recv)", gen_ms, "ms")
    line("audio_duration", audio_s_vals, "s")
    line("rtf (total/audio)", rtf_vals, "x")
    print(
        f"audio chunks per request: average={statistics.fmean(n_chunks_ints):.2f}  "
        f"min={min(n_chunks_ints)}  max={max(n_chunks_ints)}"
    )
    if pooled_inter_chunk_ms:
        line("time between audio chunks (all gaps, pooled)", pooled_inter_chunk_ms, "ms")
    else:
        print("time between audio chunks: n/a (every request had a single chunk)")

    if rtf_vals:
        mean_rtf = statistics.fmean(rtf_vals)
        _, mean_reason = streaming_from_rtf(mean_rtf)
        print(
            f"\nStreaming (from RTF): {stream_yes}/{len(oks)} requests have RTF < 1 → streamable=yes; "
            f"{stream_no} have streamable=no."
        )
        print(f"Average RTF={mean_rtf:.3f} → {mean_reason}")
        if stream_yes == len(oks):
            print("Overall: all completed requests look OK for real-time streaming by this RTF rule.")
        elif stream_no == len(oks):
            print("Overall: none of the requests meet RTF < 1 for this rule (expect buffering or lag).")
        else:
            print("Overall: mixed — some utterances meet the rule, some do not.")


def main() -> None:
    p = argparse.ArgumentParser(description="Sequential Hindi WS TTS tester with timing metrics")
    p.add_argument("--uri", default=DEFAULT_URI, help="WebSocket URI (default ws://127.0.0.1:8002)")
    p.add_argument(
        "-n",
        "--requests",
        type=int,
        default=50,
        help="Number of sequential requests (rotates among 5 sentences)",
    )
    p.add_argument(
        "--gap-s",
        type=float,
        default=0.0,
        help="Seconds to sleep after each fully completed request before starting the next (default 0).",
    )
    p.add_argument(
        "--description",
        default="A calm, clear voice speaking at a normal pace.",
        help="Speaker/style description passed to the model",
    )
    p.add_argument("--language", default="hi", help="Language field sent to server (default hi)")
    p.add_argument(
        "--out-dir",
        default="",
        help="If set, write wavs to this directory (e.g. tests/files/hi_metrics)",
    )
    p.add_argument("--strict", action="store_true", help="Fail on server error or missing PCM")
    args = p.parse_args()

    if args.requests < 1:
        p.error("--requests must be >= 1")
    if args.gap_s < 0:
        p.error("--gap-s must be >= 0")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

