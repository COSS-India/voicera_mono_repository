"""Download specific Rasa clips (by their `filename`) as the preset voice refs.

Use this when you have hand-picked exact Rasa utterances. For each voice it scans
the parquet shards of the clip's language (test shards first, they're smaller),
reads only the `filename` column to find which shard holds the pick, then extracts
just that row's audio to voices/refs/<ref>.wav.

Rasa is GATED: accept its terms on the HF page and set HF_TOKEN first.

    pip install huggingface_hub pyarrow soundfile numpy
    export HF_TOKEN=hf_...
    python scripts/fetch_rasa_by_filename.py
    python scripts/fetch_rasa_by_filename.py --max-shards 8   # cap downloads/config

Note: a pick that lives in a train shard means downloading large (~200-580MB)
shards until it's found. Test-split picks are cheap. Edit SELECTIONS to taste.
"""
from __future__ import annotations

import argparse
import io
import os
import sys

# voice ref file -> exact Rasa `filename` (config/language derived from prefix).
SELECTIONS = [
    ("aditi.wav", "HIN_F_WIKI_02836"),
    ("rahul.wav", "HIN_M_CONV_02967"),
    ("meera.wav", "TAM_F_CONV_02300"),
    ("ananya.wav", "BEN_F_CONV_00727"),
    ("arjun.wav", "TEL_M_CONV_02312"),
]

PREFIX_CONFIG = {
    "ASM": "Assamese", "BEN": "Bengali", "BRX": "Bodo", "DOI": "Dogri",
    "GUJ": "Gujarati", "HIN": "Hindi", "KAN": "Kannada", "KAS": "Kashmiri",
    "KOK": "Konkani", "MAI": "Maithili", "MAL": "Malayalam", "MNI": "Manipuri",
    "MAR": "Marathi", "NEP": "Nepali", "ORI": "Odia", "ODI": "Odia",
    "PAN": "Punjabi", "SAN": "Sanskrit", "SAT": "Santali", "SND": "Sindhi",
    "TAM": "Tamil", "TEL": "Telugu", "URD": "Urdu", "ENG": "English",
}

RASA_REPO = "ai4bharat/Rasa"
FILENAME_COL = "filename"
AUDIO_KEYS = ("audio", "wav", "speech")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _norm(name: str) -> str:
    """Strip a trailing extension so 'X.wav' and 'X' compare equal."""
    return str(name).rsplit(".", 1)[0]


def _config_for(fname: str) -> str | None:
    return PREFIX_CONFIG.get(fname.split("_", 1)[0].upper())


def _audio_bytes(row: dict):
    for k in AUDIO_KEYS:
        v = row.get(k)
        if isinstance(v, dict) and v.get("bytes"):
            return v["bytes"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch specific Rasa clips by filename")
    parser.add_argument("--max-shards", type=int, default=12,
                        help="Max shards to download per language before giving up")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        path = os.path.expanduser("~/.cache/huggingface/token")
        token = open(path).read().strip() if os.path.exists(path) else None
    if not token:
        sys.exit("Set HF_TOKEN (and accept the Rasa terms on its HF page) first.")

    try:
        import pyarrow.parquet as pq
        import soundfile as sf
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. Run: pip install huggingface_hub pyarrow soundfile numpy")

    api = HfApi()
    all_files = api.list_repo_files(RASA_REPO, repo_type="dataset", token=token)
    refs_dir = os.path.join(_ROOT, "voices", "refs")
    os.makedirs(refs_dir, exist_ok=True)

    # Group picks by language so each config's shards are scanned once.
    by_config: dict[str, dict[str, str]] = {}
    for ref, fname in SELECTIONS:
        config = _config_for(fname)
        if not config:
            print(f"[{ref}] cannot map prefix of {fname!r} to a Rasa config; skipping")
            continue
        by_config.setdefault(config, {})[_norm(fname)] = ref

    for config, pending in by_config.items():
        # test-* sorts before train-*, so smaller shards are tried first.
        shards = sorted(
            f for f in all_files
            if f.endswith(".parquet") and config in f.split("/")
        )
        print(f"\n=== {config}: {len(pending)} pick(s), {len(shards)} shards ===")

        for i, shard in enumerate(shards):
            if not pending:
                break
            if i >= args.max_shards:
                print(f"  stopped after {args.max_shards} shards; {len(pending)} unresolved")
                break
            print(f"  [{i + 1}/{len(shards)}] {shard}")
            path = hf_hub_download(RASA_REPO, shard, repo_type="dataset", token=token)

            # Cheap membership test: read only the filename column.
            names = pq.read_table(path, columns=[FILENAME_COL])[FILENAME_COL].to_pylist()
            nameset = {_norm(n) for n in names}
            here = [fn for fn in pending if fn in nameset]
            if not here:
                continue
            print(f"      contains: {', '.join(here)}")

            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=64, columns=[FILENAME_COL] + list(AUDIO_KEYS[:1])):
                for row in batch.to_pylist():
                    fn = _norm(row.get(FILENAME_COL, ""))
                    if fn not in pending:
                        continue
                    data = _audio_bytes(row)
                    ref = pending[fn]
                    if not data:
                        print(f"      [{ref}] no audio bytes for {fn}; skipping")
                        pending.pop(fn, None)
                        continue
                    arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
                    if getattr(arr, "ndim", 1) > 1:
                        arr = arr.mean(axis=1)
                    out = os.path.join(refs_dir, ref)
                    sf.write(out, arr, sr)
                    print(f"      [{ref}] wrote {out} ({arr.size / sr:.1f}s @ {sr}Hz)")
                    pending.pop(fn, None)
                if not pending:
                    break

        for fn, ref in pending.items():
            print(f"  [{ref}] {fn} NOT found in scanned shards (raise --max-shards?)")

    print("\nDone. Audition voices/refs/*.wav, then: python scripts/build_voices.py")


if __name__ == "__main__":
    main()
