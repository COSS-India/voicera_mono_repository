#!/usr/bin/env python3
"""Build the benchmark corpus from FLEURS test audio.

Why FLEURS: ungated, already 16 kHz, and it covers this checkpoint's languages. There is no
ffmpeg or sox on this box (8 vCPUs; we never transcode per request), so every audio operation
here is pure Python -- soundfile to read and write, numpy to slice and concatenate.

Three duration buckets, because latency behaves differently in each:

  short   3-5 s    the TTFP-dominated case
  medium  10-20 s  steady-state streaming
  long    30-45 s  built by CONCATENATING utterances

Long clips are deliberately **streaming-only**. This checkpoint trains at `max_duration: 30` and
whole-file decoding degrades badly past about a minute, but streaming never shows the encoder
more than `left + chunk + right`, so long-form is a fair streaming test and an unfair offline
one. Their reference is the concatenation of the per-utterance references, which is exactly what
the audio is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

#: FLEURS config name per checkpoint language code, for the languages both sides have.
#: FLEURS has no bhb/bho/brx/doi/kok/mai/mni/sa/sat/sd, so those cannot be covered here and are
#: reported as such rather than silently omitted.
FLEURS_CONFIG = {
    "as": "as_in", "bn": "bn_in", "en": "en_us", "gu": "gu_in", "hi": "hi_in",
    "kn": "kn_in", "ml": "ml_in", "mr": "mr_in", "ne": "ne_np", "or": "or_in",
    "pa": "pa_in", "ta": "ta_in", "te": "te_in", "ur": "ur_pk",
}

BUCKETS = {"short": (3.0, 5.0), "medium": (10.0, 20.0), "long": (30.0, 45.0)}


def write_wav(path: Path, wav: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")


def build_language(lang: str, cfg_name: str, out_dir: Path, *,
                   per_bucket: int, max_scan: int) -> list[dict]:
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset("google/fleurs", cfg_name, split="test", streaming=True)
    # Take the audio UNDECODED and decode it ourselves.
    #
    # datasets 5.x routes its Audio feature through torchcodec, which needs the FFmpeg shared
    # libraries. This image deliberately has no ffmpeg (8 vCPUs; we never transcode per
    # request), so the decode raises "To support decoding audio data, please install
    # 'torchcodec'". decode=False hands back the raw file bytes instead, and FLEURS ships 16 kHz
    # WAV, which soundfile reads directly -- no new dependency, and one less decoder between
    # the corpus and the model's front end.
    ds = ds.cast_column("audio", Audio(decode=False))

    pool: list[tuple[np.ndarray, str]] = []
    items: list[dict] = []
    made = {b: 0 for b in BUCKETS}

    for i, row in enumerate(ds):
        if i >= max_scan or all(made[b] >= per_bucket for b in BUCKETS):
            break
        a = row["audio"]
        raw = a.get("bytes")
        if raw is None:
            path = a.get("path")
            if not path:
                continue
            raw = Path(path).read_bytes()
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            # FLEURS is 16 kHz; anything else would need a resampler we deliberately do not
            # have here, so skip rather than silently degrade the front end's parity.
            continue
        ref = (row.get("transcription") or "").strip()
        dur = len(wav) / SAMPLE_RATE

        placed = False
        for name, (lo, hi) in BUCKETS.items():
            if name == "long":
                continue
            if lo <= dur <= hi and made[name] < per_bucket:
                p = out_dir / lang / name / f"{lang}_{name}_{made[name]:03d}.wav"
                write_wav(p, wav)
                items.append({"path": str(p), "lang": lang, "bucket": name,
                              "duration_s": round(dur, 3), "reference": ref,
                              "n_utterances": 1})
                made[name] += 1
                placed = True
                break
        if not placed and made["long"] < per_bucket:
            pool.append((wav, ref))
            total = sum(len(w) for w, _ in pool) / SAMPLE_RATE
            if total >= BUCKETS["long"][0]:
                cat = np.concatenate([w for w, _ in pool])
                cat = cat[:int(BUCKETS["long"][1] * SAMPLE_RATE)]
                p = out_dir / lang / "long" / f"{lang}_long_{made['long']:03d}.wav"
                write_wav(p, cat)
                items.append({
                    "path": str(p), "lang": lang, "bucket": "long",
                    "duration_s": round(len(cat) / SAMPLE_RATE, 3),
                    # The reference for a concatenation IS the concatenation of the references.
                    "reference": " ".join(r for _, r in pool).strip(),
                    "n_utterances": len(pool),
                    "streaming_only": True,
                })
                made["long"] += 1
                pool = []
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/corpus"))
    ap.add_argument("--langs", default="hi",
                    help="comma-separated checkpoint language codes, or 'all-fleurs'")
    ap.add_argument("--per-bucket", type=int, default=4)
    ap.add_argument("--max-scan", type=int, default=400)
    args = ap.parse_args()

    if args.langs == "all-fleurs":
        langs = list(FLEURS_CONFIG)
    else:
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    manifest: list[dict] = []
    unavailable = []
    for lang in langs:
        cfg = FLEURS_CONFIG.get(lang)
        if cfg is None:
            unavailable.append(lang)
            print(f"[corpus] {lang}: NOT IN FLEURS — skipped", flush=True)
            continue
        print(f"[corpus] {lang} ({cfg}) ...", flush=True)
        try:
            items = build_language(lang, cfg, args.out, per_bucket=args.per_bucket,
                                   max_scan=args.max_scan)
        except Exception as e:
            print(f"[corpus] {lang}: FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
            unavailable.append(lang)
            continue
        manifest.extend(items)
        by = {}
        for it in items:
            by[it["bucket"]] = by.get(it["bucket"], 0) + 1
        print(f"[corpus] {lang}: {len(items)} clips {by}", flush=True)

    out = args.out / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "sample_rate": SAMPLE_RATE,
        "buckets": {k: list(v) for k, v in BUCKETS.items()},
        "languages_requested": langs,
        "languages_unavailable_in_fleurs": unavailable,
        "note": ("long clips are concatenations and are STREAMING-ONLY: the checkpoint trains "
                 "at max_duration 30, so whole-file decoding of them is not a fair reference."),
        "items": manifest,
    }, ensure_ascii=False, indent=2))
    total_s = sum(i["duration_s"] for i in manifest)
    print(f"[corpus] {len(manifest)} clips, {total_s / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
