#!/usr/bin/env python3
"""Run F — the latency profile of a single stream, per language, with repeats.

Run A swept geometry and Run B swept concurrency; neither answers the plainest question anyone
asks of a live transcriber: **how long until I see the first word, and how evenly do the rest
arrive?** This measures exactly that at N=1 on the shipped configuration.

Three things it does that the earlier runs did not:

1. **Repeats.** Every language is streamed `--repeats` times and the report quotes the spread.
   `BENCHMARKS.md` states outright that no headline config in the first pass was repeated, so
   its percentiles are indicative rather than settled. These are not.

2. **Per-language.** The corpus spans six languages across five scripts. A single Hindi number
   was being read as if it described the service.

3. **Turn-aware gaps.** Partials carry a `turn` index, and a turn increment is a decoder-state
   rotation. Gaps that span a rotation are a different phenomenon from gaps inside a turn --
   pooling them produces a p99 that describes neither. They are separated here and Run G
   characterises the rotation itself.

**TTFT and TTFP are the same event on this server, and the report says so rather than inventing
a second number.** AlignAtt emits a partial when it commits a token, so the first token IS the
first partial; there is no sub-word streaming layer beneath it whose latency could differ.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, "/app/bench")

import metrics_lib as M  # noqa: E402
from ws_client import read_audio, stream_one  # noqa: E402


split_gaps = M.split_gaps
leading_silence_ms = M.leading_silence_ms


def summarise(runs: list[dict]) -> dict:
    """Pool the repeats of one language.

    Percentiles are computed over the POOLED raw samples, never as a mean of per-run
    percentiles -- averaging percentiles is not a percentile of anything.
    """
    ok = [r for r in runs if not r.get("client_bound")]
    ttfps = [r["ttfp_ms"] for r in ok if r.get("ttfp_ms")]
    tails = [r["tail_ms"] for r in ok if r.get("tail_ms") is not None]

    steady, boundary, ticks = [], [], []
    for r in ok:
        s, b = split_gaps(r.get("partials", []))
        steady += s
        boundary += b
        ticks += [p["latency_ms"] for p in r.get("partials", [])
                  if not p["final"] and p.get("latency_ms") is not None]

    return {
        "n_runs": len(runs),
        "n_client_bound": len(runs) - len(ok),
        "audio_s_mean": round(sum(r["audio_s"] for r in ok) / len(ok), 2) if ok else None,

        # --- first word ---
        "leading_silence_ms": ok[0].get("leading_silence_ms") if ok else None,
        "ttfp_ms_mean": round(sum(ttfps) / len(ttfps), 1) if ttfps else None,
        # TTFP with the clip's own leading silence removed: the server's contribution alone.
        "ttfp_from_speech_ms_mean": (
            round(sum(r["ttfp_from_speech_ms"] for r in ok
                      if r.get("ttfp_from_speech_ms") is not None)
                  / max(1, sum(1 for r in ok if r.get("ttfp_from_speech_ms") is not None)), 1)
            if any(r.get("ttfp_from_speech_ms") is not None for r in ok) else None),
        "ttfp_ms_min": round(min(ttfps), 1) if ttfps else None,
        "ttfp_ms_max": round(max(ttfps), 1) if ttfps else None,
        "ttfp_ms_spread": round(max(ttfps) - min(ttfps), 1) if len(ttfps) > 1 else None,

        # --- per-tick compute, the server-side cost of one chunk ---
        "tick_latency_ms_p50": M.percentile(ticks, 0.50),
        "tick_latency_ms_p95": M.percentile(ticks, 0.95),
        "tick_latency_ms_max": round(max(ticks), 1) if ticks else None,

        # --- smoothness, split by cause ---
        "n_gaps_steady": len(steady),
        "gap_steady_ms_p50": M.percentile(steady, 0.50),
        "gap_steady_ms_p90": M.percentile(steady, 0.90),
        "gap_steady_ms_p99": M.percentile(steady, 0.99),
        "gap_steady_ms_max": round(max(steady), 1) if steady else None,
        "n_gaps_boundary": len(boundary),
        "gap_boundary_ms_p50": M.percentile(boundary, 0.50),
        "gap_boundary_ms_max": round(max(boundary), 1) if boundary else None,

        # --- how far behind real time the stream ran ---
        "delta_lag_p50_ms": M.percentile(
            [r["delta_lag_p50_ms"] for r in ok if r.get("delta_lag_p50_ms") is not None], 0.50),
        "delta_lag_p95_ms": M.percentile(
            [r["delta_lag_p95_ms"] for r in ok if r.get("delta_lag_p95_ms") is not None], 0.95),

        # --- audio ends -> transcript settles ---
        "tail_ms_mean": round(sum(tails) / len(tails), 1) if tails else None,
        "tail_ms_max": round(max(tails), 1) if tails else None,

        "words_per_partial_mean": round(
            sum(r["words_per_partial"] for r in ok) / len(ok), 2) if ok else None,
        "n_partials_mean": round(sum(r["n_partials"] for r in ok) / len(ok), 1) if ok else None,
    }


async def profile_clip(url: str, clip: dict, repeats: int, cooldown: float) -> list[dict]:
    wav = read_audio(Path(clip["path"]))
    lead = leading_silence_ms(wav)
    runs = []
    for k in range(repeats):
        r = await stream_one(url, wav, clip["lang"], rate=1.0, block_ms=100,
                             label=f"{Path(clip['path']).stem}#{k}", detail=True)
        r["clip"] = Path(clip["path"]).name
        r["lang"] = clip["lang"]
        r["repeat"] = k
        r["leading_silence_ms"] = lead
        r["ttfp_from_speech_ms"] = (round(r["ttfp_ms"] - lead, 1)
                                    if r.get("ttfp_ms") is not None else None)
        runs.append(r)
        await asyncio.sleep(cooldown)
    return runs


async def main_async(args) -> int:
    manifests = [Path(p) for p in args.corpus.split(",") if p.strip()]
    items = []
    for man in manifests:
        items += json.loads(man.read_text())["items"]

    buckets = {b.strip() for b in args.buckets.split(",") if b.strip()}
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    # One clip per language, chosen deterministically (first by filename in the wanted bucket)
    # so a re-run measures the same audio and the spread is the server's, not the corpus's.
    chosen: dict[str, dict] = {}
    for lang in langs:
        cands = sorted((i for i in items if i["lang"] == lang and i["bucket"] in buckets),
                       key=lambda i: i["path"])
        if not cands:
            print(f"[runF] no clip for lang={lang} buckets={sorted(buckets)}; skipping",
                  file=sys.stderr, flush=True)
            continue
        chosen[lang] = cands[0]

    if not chosen:
        raise SystemExit("no clips matched")

    refs = {}
    for rp in args.refs.split(","):
        rp = rp.strip()
        if rp and Path(rp).exists():
            refs.update(json.loads(Path(rp).read_text()))

    out = {
        "config": {
            "url": args.url, "repeats": args.repeats, "buckets": sorted(buckets),
            "langs": list(chosen), "block_ms": 100, "rate": 1.0,
        },
        "server": {},
        "by_lang": {},
        "runs": [],
    }
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            out["server"]["metrics_before"] = (await c.get(f"{args.base}/metrics")).json()
        except Exception as e:
            out["server"]["metrics_before"] = {"error": repr(e)}

    for lang, clip in chosen.items():
        print(f"[runF] {lang} {Path(clip['path']).name} x{args.repeats} ...",
              file=sys.stderr, flush=True)
        try:
            runs = await profile_clip(args.url, clip, args.repeats, args.cooldown)
        except Exception as e:
            # A cell that fails is a RESULT, not a reason to lose the other five. This is not
            # hypothetical: `en` took the server down mid-campaign with a CUDA illegal memory
            # access, and the first version of this script died with it, discarding every
            # language already measured.
            print(f"[runF]   ERROR: {e!r}", file=sys.stderr, flush=True)
            out["by_lang"][lang] = {
                "clip": Path(clip["path"]).name,
                "error": repr(e),
                "note": "cell failed; the server may have restarted mid-run",
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
            # Give the service time to come back before the next language.
            await asyncio.sleep(args.error_cooldown)
            continue

        ref = refs.get(Path(clip["path"]).name)
        for r in runs:
            # CER against the model's OWN offline transcript: a sanity check that streaming did
            # not degrade the text, not a claim about the model's accuracy on the audio.
            r["cer_vs_offline"] = M.cer(ref, r["text"]) if ref else None

        # Summarise BEFORE dropping the raw partials -- every gap statistic is derived from them.
        summ = summarise(runs)
        if not args.keep_partials:
            for r in runs:
                r.pop("partials", None)
        cers = [r["cer_vs_offline"] for r in runs if r["cer_vs_offline"] is not None]
        summ["cer_vs_offline_mean"] = round(sum(cers) / len(cers), 4) if cers else None
        summ["cer_reference"] = "offline" if ref else "NOT MEASURED (no offline reference)"
        summ["clip"] = Path(clip["path"]).name
        out["by_lang"][lang] = summ
        out["runs"] += runs

        print(f"[runF]   ttfp {summ['ttfp_ms_mean']} ms "
              f"(from speech {summ['ttfp_from_speech_ms_mean']}, "
              f"lead {summ['leading_silence_ms']}, spread {summ['ttfp_ms_spread']}) "
              f"gap p50 {summ['gap_steady_ms_p50']} p99 {summ['gap_steady_ms_p99']} "
              f"boundary {summ['n_gaps_boundary']}x p50 {summ['gap_boundary_ms_p50']} "
              f"tail {summ['tail_ms_mean']} cer {summ['cer_vs_offline_mean']}",
              file=sys.stderr, flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    # Pooled across languages, over raw samples again rather than over the per-language means.
    failed = {k: v["error"] for k, v in out["by_lang"].items() if "error" in v}
    all_ok = [r for r in out["runs"] if not r.get("client_bound")]
    out["pooled"] = {
        "n_runs": len(out["runs"]),
        "n_langs": len(out["by_lang"]) - len(failed),
        "n_langs_failed": len(failed),
        "failed": failed,
        "ttfp_ms_p50": M.percentile([r["ttfp_ms"] for r in all_ok if r.get("ttfp_ms")], 0.50),
        "ttfp_ms_p95": M.percentile([r["ttfp_ms"] for r in all_ok if r.get("ttfp_ms")], 0.95),
        "ttfp_from_speech_ms_p50": M.percentile(
            [r["ttfp_from_speech_ms"] for r in all_ok
             if r.get("ttfp_from_speech_ms") is not None], 0.50),
        "ttfp_from_speech_ms_p95": M.percentile(
            [r["ttfp_from_speech_ms"] for r in all_ok
             if r.get("ttfp_from_speech_ms") is not None], 0.95),
        "tail_ms_p50": M.percentile([r["tail_ms"] for r in all_ok
                                     if r.get("tail_ms") is not None], 0.50),
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[runF] wrote {args.out}", file=sys.stderr, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="/corpus/manifest.json,/corpus/multi/manifest.json")
    ap.add_argument("--refs", default="/results/offline_reference.json,"
                                      "/results/offline_reference_multi.json")
    ap.add_argument("--url", default="ws://localhost:9002/v1/asr/ws")
    ap.add_argument("--base", default="http://localhost:9002")
    ap.add_argument("--langs", default="hi,bn,ta,te,mr,en")
    ap.add_argument("--buckets", default="medium")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=2.0)
    ap.add_argument("--error-cooldown", type=float, default=240.0,
                    help="wait after a failed cell; a fatal CUDA error restarts the container "
                         "and the model reload takes minutes")
    ap.add_argument("--keep-partials", action="store_true",
                    help="keep the raw per-partial list in the output (large)")
    ap.add_argument("--out", type=Path, default=Path("/results/runF_latency.json"))
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
