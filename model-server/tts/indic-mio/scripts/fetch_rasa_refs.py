"""Fetch candidate reference clips for the preset voices from ai4bharat/Rasa.

Rasa (https://huggingface.co/datasets/ai4bharat/Rasa, CC-BY-4.0) is a GATED
dataset: accept the terms on the dataset page and set HF_TOKEN before running.

For each voice in voices/manifest.json this downloads ONE parquet shard of the
voice's source language (with a real progress bar), reads it locally, picks the
first clean clip matching the requested gender and target duration, trims it, and
writes voices/refs/<ref>.wav.

Direct shard download is used on purpose: HF's Xet backend makes parquet
*streaming* (`load_dataset(..., streaming=True)`) hang on range reads, so we grab
a whole shard once and read it with pyarrow instead.

It is a CONVENIENCE, not gospel: audition the output and replace anything off.
On first use it prints the parquet column names so you can adjust GENDER_KEYS /
AUDIO_KEYS if a lookup misses.

Usage (from indic_mio_tts_server dir):
    pip install huggingface_hub pyarrow soundfile numpy
    export HF_TOKEN=hf_...
    python scripts/fetch_rasa_refs.py
    python scripts/fetch_rasa_refs.py --voice Aditi --seconds 10
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

RASA_REPO = "ai4bharat/Rasa"
# Candidate parquet column names Rasa may use.
GENDER_KEYS = ("gender", "speaker_gender", "sex")
AUDIO_KEYS = ("audio", "wav", "speech")
# Prefer a calm reference clip; emotional styles make a worse general voice.
NEUTRAL_STYLES = ("neutral", "conversational", "normal", "default", "narration", "news")

_SOURCE_RE = re.compile(r"Rasa:\s*([A-Za-z]+)\s*\((male|female)\)", re.IGNORECASE)


def _parse_source(source: str):
    """'ai4bharat/Rasa: Hindi (female)' -> ('Hindi', 'female'). Config names are
    title-case; keep them, lower-case only the gender."""
    m = _SOURCE_RE.search(source or "")
    return (m.group(1).capitalize(), m.group(2).lower()) if m else None


def _load_manifest() -> dict:
    with open(os.path.join(_ROOT, "voices", "manifest.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _first_key(row: dict, keys) -> str | None:
    for k in keys:
        if k in row:
            return k
    return None


def _language_shard(api, hf_hub_download, language: str, token: str) -> str | None:
    """Return a local path to the first parquet shard for `language`."""
    files = api.list_repo_files(RASA_REPO, repo_type="dataset", token=token)
    hits = sorted(
        f for f in files
        if f.endswith(".parquet") and language in f.split("/")
    )
    if not hits:
        print(f"    no parquet shard found for config {language!r}")
        return None
    print(f"    downloading shard: {hits[0]}")
    return hf_hub_download(RASA_REPO, hits[0], repo_type="dataset", token=token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Rasa reference clips for preset voices")
    parser.add_argument("--voice", default=None, help="Only this voice id (default: all)")
    parser.add_argument("--seconds", type=float, default=10.0, help="Target clip length")
    parser.add_argument("--min-seconds", type=float, default=6.0, help="Skip clips shorter than this")
    parser.add_argument("--max-scan", type=int, default=2000, help="Max rows to scan per voice")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN (and accept the Rasa terms on its HF page) first.")

    try:
        import numpy as np
        import pyarrow.parquet as pq
        import soundfile as sf
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. Run: pip install huggingface_hub pyarrow soundfile numpy")

    api = HfApi()
    refs_dir = os.path.join(_ROOT, "voices", "refs")
    os.makedirs(refs_dir, exist_ok=True)

    voices = _load_manifest().get("voices", [])
    if args.voice:
        voices = [v for v in voices if v.get("name") == args.voice]
        if not voices:
            sys.exit(f"Voice {args.voice!r} not in manifest.")

    printed_schema = False
    for v in voices:
        name, ref = v.get("name"), v.get("ref")
        parsed = _parse_source(v.get("source", ""))
        if not parsed:
            print(f"[{name}] cannot parse language/gender from source={v.get('source')!r}; skipping")
            continue
        language, want_gender = parsed
        print(f"[{name}] Rasa '{language}', want {want_gender} ...")

        try:
            shard = _language_shard(api, hf_hub_download, language, token)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] shard lookup/download failed: {e}")
            continue
        if not shard:
            continue

        pf = pq.ParquetFile(shard)
        cols = list(pf.schema_arrow.names)
        if not printed_schema:
            print("  parquet columns:", cols)
            printed_schema = True
        audio_key = _first_key({c: 1 for c in cols}, AUDIO_KEYS)
        gender_key = _first_key({c: 1 for c in cols}, GENDER_KEYS)
        if not audio_key:
            print(f"[{name}] no audio column among {AUDIO_KEYS} in {cols}; skipping")
            continue

        read_cols = [audio_key] + ([gender_key] if gender_key else [])
        saved, scanned = False, 0
        for batch in pf.iter_batches(batch_size=64, columns=read_cols):
            for row in batch.to_pylist():
                scanned += 1
                if scanned > args.max_scan:
                    break
                if gender_key:
                    g = str(row.get(gender_key, "")).strip().lower()
                    if g and want_gender not in g:
                        continue
                audio = row.get(audio_key)
                data = audio.get("bytes") if isinstance(audio, dict) else None
                if not data:
                    continue
                arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
                if getattr(arr, "ndim", 1) > 1:
                    arr = arr.mean(axis=1)
                if arr.size / sr < args.min_seconds:
                    continue
                clip = arr[: int(args.seconds * sr)]
                out = os.path.join(refs_dir, ref)
                sf.write(out, clip, sr)
                print(f"[{name}] wrote {out} ({clip.size / sr:.1f}s @ {sr}Hz)"
                      + (f" gender={g}" if gender_key else " (no gender column)"))
                saved = True
                break
            if saved or scanned > args.max_scan:
                break

        if not saved:
            print(f"[{name}] no matching clip in {scanned} rows; adjust GENDER_KEYS or --max-scan")

    print("\nDone. Audition voices/refs/*.wav, then: python scripts/build_voices.py")


if __name__ == "__main__":
    main()
