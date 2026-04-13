#!/usr/bin/env python3
"""Call POST /transcribe/bhili with the same int16 PCM contract as the server."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torchaudio

TARGET_SR = 16000
HERE = Path(__file__).resolve().parent
DEFAULT_WAV = HERE / "bhili-asr" / (
    "भारत_नी_राजधानी_दिल्ली_छे_अने_भारत_ना_शेजार_ना_देश_पाकिस्तान_अने_नेपाळ_छे.wav"
)


def wav_to_int16_pcm_bytes(path: Path) -> bytes:
    wav, sr = torchaudio.load(str(path))
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
    x = wav.squeeze(0).clamp(-1.0, 1.0).numpy().astype(np.float32)
    pcm = (x * 32767.0).round().astype(np.int16)
    return pcm.tobytes()


def post_transcribe_bhili(
    base_url: str,
    audio_b64: str,
    language_id: str,
    timeout_s: float = 120.0,
) -> dict:
    url = base_url.rstrip("/") + "/transcribe/bhili"
    body = json.dumps({"audio_b64": audio_b64, "language_id": language_id}).encode(
        "utf-8"
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="Test Bhili STT HTTP endpoint")
    p.add_argument(
        "--url",
        default="http://127.0.0.1:8001",
        help="Server base URL (no trailing path)",
    )
    p.add_argument(
        "--wav",
        type=Path,
        default=DEFAULT_WAV,
        help="Path to WAV (mono/stereo; resampled to 16 kHz)",
    )
    p.add_argument(
        "--language-id",
        default="bhb",
        help="Request language_id (bhb maps to mr for NeMo on the server)",
    )
    p.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
    args = p.parse_args()

    if not args.wav.is_file():
        print(f"WAV not found: {args.wav}", file=sys.stderr)
        return 1

    pcm_bytes = wav_to_int16_pcm_bytes(args.wav)
    audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
    print(f"POST {args.url.rstrip('/')}/transcribe/bhili  ({len(pcm_bytes)} bytes PCM)")

    try:
        out = post_transcribe_bhili(
            args.url, audio_b64, args.language_id, timeout_s=args.timeout
        )
    except urllib.error.HTTPError as e:
        print(e.code, e.reason, file=sys.stderr)
        print(e.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(e, file=sys.stderr)
        return 1

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
