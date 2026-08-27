"""Pre-bake preset voice embeddings from their reference clips.

The server derives + caches each voice embedding lazily on first boot, so this
script is optional. Run it to (a) make the first boot instant, (b) verify every
ref clip in voices/manifest.json encodes cleanly, or (c) re-bake after swapping a
clip. Needs the same runtime as the server (MioCodec + a CUDA GPU).

Usage (from the indic_mio_tts_server dir, or inside the mio-tts container):
    python scripts/build_voices.py            # build any missing embeddings
    python scripts/build_voices.py --force    # rebuild all (ignore cache)
    python scripts/build_voices.py --voice Aditi --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow `import config, tts_engine` when run from the scripts/ subdir.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from config import Config  # noqa: E402
from tts_engine import MioTTSEngine  # noqa: E402


def _manifest_names(config: Config) -> list[str]:
    path = os.path.join(config.voices_dir, "manifest.json")
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return [str(v.get("name", "")).strip() for v in manifest.get("voices", []) if v.get("name")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake Indic-Mio preset voice embeddings")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached")
    parser.add_argument("--voice", default=None, help="Only this voice id (default: all)")
    args = parser.parse_args()

    config = Config.from_env()

    if args.force:
        targets = [args.voice] if args.voice else _manifest_names(config)
        for name in targets:
            cached = os.path.join(config.voices_cache_dir, f"{name}.pt")
            if os.path.exists(cached):
                os.remove(cached)
                print(f"removed cache: {cached}")

    # load_codec() derives + caches every voice embedding from its ref clip.
    engine = MioTTSEngine(config)
    engine.load_codec()
    print("Done. Voice embeddings are cached under:", config.voices_cache_dir)


if __name__ == "__main__":
    main()
