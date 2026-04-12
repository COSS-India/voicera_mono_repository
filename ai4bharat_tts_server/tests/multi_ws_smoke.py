"""
Parallel WebSocket TTS clients — fires 16 requests at a time, 4 waves = 64 total.
Each wave waits for all 16 to complete before the next wave starts.

Run server first, e.g. `python server.py`, then:
  python tests/multi_ws_smoke.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import numpy as np
import websockets


WAVE_SIZE   = 10
N_WAVES     = 1
DESCRIPTION = "Vidya's voice is monotone."

HINDI_PROMPTS: list[str] = [
    "नमस्ते, आप कैसे हैं? आज दिन कैसा रहा?",
    "आज मौसम बहुत सुहावना है, बाहर घूमने का मन कर रहा है।",
    "कृपया धीरे और साफ़ बोलें, मैं सुन रहा हूँ।",
    "यह एक छोटा परीक्षण वाक्य है, सब ठीक से सुनाई दे रहा है क्या?",
    "मेरा नाम विद्या है और मैं दिल्ली से हूँ।",
    "क्या आप मुझे रास्ता बता सकते हैं? मुझे स्टेशन जाना है।",
    "आपकी आवाज़ बहुत मधुर है, सुनकर अच्छा लगा।",
    "मैं कल आपसे मिलने आऊँगा, समय निकालिएगा।",
    "यह काम जल्दी करना ज़रूरी है, देर नहीं होनी चाहिए।",
    "बाज़ार से थोड़ा दूध और सब्ज़ी लेकर आना।",
    "परीक्षा की तैयारी अच्छे से करो, मेहनत रंग लाएगी।",
    "आज खाने में क्या बनाएंगे? मुझे दाल-चावल पसंद है।",
    "फ़ोन पर बात करना हो तो शाम को कॉल करना।",
    "बच्चे स्कूल से वापस आ गए हैं, उन्हें खाना दे दो।",
    "डॉक्टर ने कहा है कि आराम करना ज़रूरी है।",
    "इस महीने का बिजली का बिल बहुत ज़्यादा आया है।",
]


async def run_one_request(
    index: int,
    uri: str,
    prompt: str,
) -> tuple[int, float | None, float | None, int]:
    """
    Fire one request immediately (no gap — all 16 in a wave start together).
    Returns (index, ttft_ms, mean_inter_chunk_ms, total_samples).
    """
    chunks: list[np.ndarray] = []
    ttft_ms: float | None = None
    inter_chunk_ms: list[float] = []
    last_recv_mono: float | None = None

    async with websockets.connect(uri) as ws:
        t_before_send = time.monotonic()
        await ws.send(json.dumps({"prompt": prompt, "description": DESCRIPTION}))

        meta = json.loads(await ws.recv())
        if meta["type"] != "meta":
            raise RuntimeError(f"[{index}] expected meta, got {meta}")

        while True:
            msg = await ws.recv()
            now = time.monotonic()

            if isinstance(msg, str):
                body = json.loads(msg)
                if body["type"] == "error":
                    raise RuntimeError(f"[{index}] server error: {body!r}")
                if body["type"] == "done":
                    break
                raise RuntimeError(f"[{index}] unexpected message: {body}")

            if ttft_ms is None:
                ttft_ms = (now - t_before_send) * 1000.0
            else:
                if last_recv_mono is not None:
                    inter_chunk_ms.append((now - last_recv_mono) * 1000.0)
            last_recv_mono = now
            chunks.append(np.frombuffer(msg, dtype=np.float32))

    total_samples = sum(c.size for c in chunks)
    mean_chunk_ms = float(np.mean(inter_chunk_ms)) if inter_chunk_ms else None
    return index, ttft_ms, mean_chunk_ms, total_samples


def print_wave_results(
    wave: int,
    results: list,
    wave_wall_ms: float,
) -> list[float]:
    """Print per-request table for one wave. Returns list of ttft values."""
    oks = sorted(
        [r for r in results if not isinstance(r, BaseException)],
        key=lambda r: r[0],
    )
    failures = [r for r in results if isinstance(r, BaseException)]

    print(f"\n── Wave {wave} ──────────────────────────────────────────────────────")
    print(f"  {'idx':>3}  {'ttft_ms':>9}  {'inter_chunk_ms':>16}  {'samples':>8}")
    print("  " + "-" * 50)

    ttft_values = []
    for idx, ttft_ms, mean_chunk_ms, total_samples in oks:
        ttft_str  = f"{ttft_ms:.1f}"  if ttft_ms       is not None else "n/a"
        chunk_str = f"{mean_chunk_ms:.1f}" if mean_chunk_ms is not None else "n/a (1 chunk)"
        print(f"  [{idx:02d}]  {ttft_str:>9}  {chunk_str:>16}  {total_samples:>8}")
        if ttft_ms is not None:
            ttft_values.append(ttft_ms)

    if ttft_values:
        print(
            f"  TTFB  avg={np.mean(ttft_values):.1f}ms  "
            f"min={np.min(ttft_values):.1f}ms  "
            f"max={np.max(ttft_values):.1f}ms  "
            f"p50={np.percentile(ttft_values, 50):.1f}ms  "
            f"p95={np.percentile(ttft_values, 95):.1f}ms"
        )
    print(f"  Wave wall time : {wave_wall_ms:.1f} ms  |  OK: {len(oks)}/{WAVE_SIZE}", end="")
    if failures:
        print(f"  |  ERRORS: {len(failures)}")
        for f in failures:
            print(f"    ERROR: {f}")
    else:
        print()

    return ttft_values


async def async_main(uri: str, strict: bool, rng: random.Random) -> None:
    total_requests = WAVE_SIZE * N_WAVES
    print(
        f"Firing {total_requests} requests in {N_WAVES} waves of {WAVE_SIZE} "
        f"→ {uri}\n"
        f"Each wave starts only after the previous wave fully completes.\n"
    )

    all_ttft: list[float] = []
    total_wall_start = time.monotonic()

    for wave in range(1, N_WAVES + 1):
        prompts = [rng.choice(HINDI_PROMPTS) for _ in range(WAVE_SIZE)]

        wave_start = time.monotonic()
        tasks = [
            asyncio.create_task(run_one_request(i, uri, prompts[i]))
            for i in range(WAVE_SIZE)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wave_wall_ms = (time.monotonic() - wave_start) * 1000.0

        wave_ttft = print_wave_results(wave, results, wave_wall_ms)
        all_ttft.extend(wave_ttft)

        failures = [r for r in results if isinstance(r, BaseException)]
        if strict and failures:
            raise failures[0]

    total_wall_ms = (time.monotonic() - total_wall_start) * 1000.0

    # ── Overall summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'OVERALL SUMMARY  (64 requests, 4 waves)':^60}")
    print("=" * 60)
    if all_ttft:
        print(f"  TTFB avg   : {np.mean(all_ttft):.1f} ms")
        print(f"  TTFB min   : {np.min(all_ttft):.1f} ms")
        print(f"  TTFB max   : {np.max(all_ttft):.1f} ms")
        print(f"  TTFB p50   : {np.percentile(all_ttft, 50):.1f} ms")
        print(f"  TTFB p95   : {np.percentile(all_ttft, 95):.1f} ms")
    print(f"  Total wall : {total_wall_ms:.1f} ms")
    print(f"  Requests   : {len(all_ttft)} / {total_requests} succeeded")
    print("=" * 60)


def main() -> None:
    p = argparse.ArgumentParser(
        description=f"Fire {N_WAVES} waves of {WAVE_SIZE} WS TTS requests (total {WAVE_SIZE * N_WAVES})"
    )
    p.add_argument("--uri",    default="ws://127.0.0.1:8002")
    p.add_argument("--seed",   type=int, default=None,
                   help="RNG seed for reproducible prompt selection")
    p.add_argument("--strict", action="store_true",
                   help="Stop immediately on first error")
    args = p.parse_args()

    asyncio.run(async_main(
        uri=args.uri,
        strict=args.strict,
        rng=random.Random(args.seed),
    ))


if __name__ == "__main__":
    main()