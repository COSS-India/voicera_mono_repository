"""
Batch WebSocket TTS smoke test: fire ``batch_size`` requests at once, report TTFT + RTF.

TTFT = ms from send until first PCM chunk.
RTF  = wall_time_to_done / audio_duration  (< 1 means faster than real-time).

Run server first, e.g. ``python server.py``, then:
  python tests/multi_ws_smoke.py
  python tests/multi_ws_smoke.py --batch-size 16
  python tests/multi_ws_smoke.py -n 8 --seed 0 --strict
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import websockets
from scipy.io import wavfile

OUT_DIR = Path(__file__).resolve().parent / "files"

EXAMPLE_UTTERANCES: list[tuple[str, str]] = [
    (
        "नमस्ते, आप कैसे हैं? आज दिन कैसा रहा?",
        "A calm, clear female voice speaking at a normal pace.",
    ),
    (
        "आज मौसम बहुत सुहावना है, बाहर घूमने का मन कर रहा है।",
        "A warm, friendly male voice.",
    ),
    (
        "कृपया धीरे और साफ़ बोलें, मैं सुन रहा हूँ।",
        "A projected, articulate voice with crisp pronunciation.",
    ),
    (
        "यह एक छोटा परीक्षण वाक्य है, सब ठीक से सुनाई दे रहा है क्या?",
        "A soft, gentle female voice speaking slowly.",
    ),
    (
        "నమస్కారం, మీరు ఎలా ఉన్నారు? ఈ రోజు ఎలా గడిచింది?",
        "A peaceful female voice with clear articulation.",
    ),
    (
        "ఈ రోజు వాతావరణం చాలా బాగుంది, బయటకు వెళ్ళడానికి మంచి రోజు.",
        "A strong, friendly male voice.",
    ),
    (
        "దయచేసి నెమ్మదిగా మాట్లాడండి, నేను వింటున్నాను.",
        "A delicate, relaxed speaking tone.",
    ),
    (
        "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ? ಇಂದು ದಿನ ಹೇಗೆ ಕಳೆಯಿತು?",
        "A steady female voice with precise, clear speech.",
    ),
    (
        "ಇಂದು ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ, ಹೊರಗೆ ಹೋಗುವುದಕ್ಕೆ ಒಳ್ಳೆಯ ದಿನ.",
        "A warm male voice with an upbeat, positive tone.",
    ),
    (
        "ದಯವಿಟ್ಟು ನಿಧಾನವಾಗಿ ಹೇಳಿ, ನಾನು ಕೇಳುತ್ತಿದ್ದೇನೆ.",
        "A soft, clear voice at a slow, easy pace.",
    ),
]


def safe_filename_from_prompt(prompt: str, max_len: int = 120) -> str:
    s = prompt.strip()
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip("._") or "output"
    return s[:max_len]


async def run_one_request(
    index: int,
    uri: str,
    prompt: str,
    description: str,
    out_dir: Path,
    strict: bool,
    save_wav: bool,
) -> tuple[int, float | None, float | None, float | None, Path | None, str]:
    """One utterance starting immediately.

    Returns (index, ttft_ms, rtf, audio_s, wav_path, prompt).
    """
    chunks: list[np.ndarray] = []
    ttft_ms: float | None = None
    sample_rate = 44100
    pid = f"req{index}"

    async with websockets.connect(uri) as ws:
        t0 = time.monotonic()
        await ws.send(json.dumps({"prompt": prompt, "description": description}))

        meta = json.loads(await ws.recv())
        if meta["type"] != "meta":
            raise RuntimeError(f"expected meta, got {meta}")
        sample_rate = int(meta.get("sample_rate", sample_rate))
        pid = str(meta.get("pid", pid))

        while True:
            msg = await ws.recv()
            now = time.monotonic()
            if isinstance(msg, str):
                body = json.loads(msg)
                if body["type"] == "error":
                    raise RuntimeError(f"request {index}: server error {body!r}")
                if body["type"] != "done":
                    raise RuntimeError(f"expected done, got {body}")
                break
            if ttft_ms is None:
                ttft_ms = (now - t0) * 1000.0
            chunks.append(np.frombuffer(msg, dtype=np.float32))

        elapsed_s = time.monotonic() - t0

    pcm = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    if pcm.size == 0:
        msg = (
            f"request {index}: no PCM (use server --decode-every 1 or a longer prompt)"
        )
        if strict:
            raise RuntimeError(msg)
        print(f"WARN {msg}")
        return index, None, None, None, None, prompt

    audio_s = float(pcm.size) / float(sample_rate)
    rtf = elapsed_s / audio_s if audio_s > 0 else None

    out_path: Path | None = None
    if save_wav:
        base = safe_filename_from_prompt(prompt)
        out_path = out_dir / f"{base}_{index:02d}_{pid}.wav"
        wavfile.write(out_path, sample_rate, pcm)

    return index, ttft_ms, rtf, audio_s, out_path, prompt


async def async_main(
    batch_size: int,
    uri: str,
    strict: bool,
    save_wav: bool,
    rng: random.Random,
) -> None:
    out_dir = OUT_DIR
    if save_wav:
        out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [rng.choice(EXAMPLE_UTTERANCES) for _ in range(batch_size)]

    # All clients start together (true batch).
    tasks = [
        asyncio.create_task(
            run_one_request(
                i, uri, pairs[i][0], pairs[i][1], out_dir, strict, save_wav
            )
        )
        for i in range(batch_size)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        for r in failures:
            print(f"ERROR {r}")
        if strict:
            raise failures[0]

    oks = [r for r in results if not isinstance(r, BaseException)]
    oks.sort(key=lambda r: r[0])

    print(f"uri={uri} batch_size={batch_size}\n")
    ttfts: list[float] = []
    rtfs: list[float] = []
    for idx, ttft_ms, rtf, audio_s, path, prompt in oks:
        if ttft_ms is None or rtf is None:
            print(f"[{idx:02d}] ttft_ms=n/a  rtf=n/a")
            continue
        ttfts.append(ttft_ms)
        rtfs.append(rtf)
        print(f"[{idx:02d}] ttft_ms={ttft_ms:.2f}  rtf={rtf:.3f}")

    if ttfts:
        print(
            f"\nsummary  ttft_ms: mean={float(np.mean(ttfts)):.2f}  "
            f"median={float(np.median(ttfts)):.2f}  "
            f"rtf: mean={float(np.mean(rtfs)):.3f}  "
            f"median={float(np.median(rtfs)):.3f}"
        )

    if failures and not strict:
        print(f"finished with {len(failures)} error(s); use --strict to fail fast")
    elif not failures:
        print("ok")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch WS TTS clients at once; report TTFT + RTF per request"
    )
    p.add_argument(
        "-n",
        "--batch-size",
        type=int,
        default=16,
        help="Number of requests to start at once (default 16)",
    )
    p.add_argument("--uri", default="ws://127.0.0.1:8002")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for prompt/description picks",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Raise on server errors or empty PCM",
    )
    p.add_argument(
        "--save-wav",
        action="store_true",
        help="Write WAVs under tests/files/",
    )
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be >= 1")

    asyncio.run(
        async_main(
            batch_size=args.batch_size,
            uri=args.uri,
            strict=args.strict,
            save_wav=args.save_wav,
            rng=random.Random(args.seed),
        )
    )


if __name__ == "__main__":
    main()
