#!/usr/bin/env python3
"""Transcribe every corpus clip OFFLINE, to give the streaming runs something to be scored against.

Accuracy is not what this campaign measures. Sanity is: the question is whether streaming output
degrades relative to what the *same weights* produce when they see the whole clip at once. Scoring
streaming against the model's own offline transcript answers that, and removes the dataset's
labelling noise from the comparison -- FLEURS references are a different question entirely, and
mixing the two would make a streaming regression indistinguishable from a model error.

Long clips are skipped by default. They are concatenations of 30-45 s, and this checkpoint trains
at `max_duration: 30`, so its whole-file decoding of them is not a fair reference for anything.
Streaming never shows the encoder more than `left + chunk + right`, which is why those clips are
marked streaming-only in the manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/app")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("/corpus/manifest.json"))
    ap.add_argument("--ckpt", default="/artifacts/indic_transcribe_core.nemo")
    ap.add_argument("--out", type=Path, default=Path("/results/offline_reference.json"))
    ap.add_argument("--buckets", default="short,medium")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    from nemo_patch import canary2_romanized, ensure_installed

    ensure_installed()
    canary2_romanized.apply_all()
    from nemo.collections.asr.models import EncDecMultiTaskModel

    man = json.loads(args.corpus.read_text())
    buckets = {b.strip() for b in args.buckets.split(",") if b.strip()}
    items = [i for i in man["items"] if i["bucket"] in buckets]
    skipped = [i["path"] for i in man["items"] if i["bucket"] not in buckets]
    print(f"[offline] {len(items)} clips ({sorted(buckets)}); "
          f"{len(skipped)} skipped as streaming-only", file=sys.stderr)

    model = EncDecMultiTaskModel.restore_from(str(args.ckpt), map_location="cuda")
    model.eval()

    out: dict[str, str] = {}
    by_lang: dict[str, list] = {}
    for it in items:
        by_lang.setdefault(it["lang"], []).append(it)

    for lang, group in sorted(by_lang.items()):
        paths = [i["path"] for i in group]
        with torch.inference_mode():
            hyps = model.transcribe(paths, batch_size=args.batch_size,
                                    source_lang=lang, target_lang=lang,
                                    pnc="yes", timestamp="no", verbose=False)
        for p, h in zip(paths, hyps):
            # NeMo keeps the leading space the first SentencePiece '_' produces; the HF port's
            # decode() strips it. Strip here so CER is not dominated by that one convention.
            out[Path(p).name] = (h.text if hasattr(h, "text") else str(h)).strip()
        print(f"[offline] {lang}: {len(paths)} clips", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[offline] wrote {len(out)} references -> {args.out}", file=sys.stderr)
    for k, v in list(out.items())[:3]:
        print(f"    {k}: {v[:70]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
