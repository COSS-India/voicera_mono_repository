#!/usr/bin/env python3
"""Run G — the flush-and-pause, characterised.

The most visible behaviour in the live demo is that transcription stops for a couple of seconds
every ~14 s and then resumes. It is decoder-state rotation, and it is deliberate: `decoder_mems_list`
grows one position per decode step against a 1024-position limit, so a stream that never rotates
stalls outright and never recovers. Rotation trades a periodic pause for not dying.

Until now that trade was quantified from a single 43 s run with three rotations. This run
characterises it properly:

* **Gap-free speech.** Silence is trimmed (same method as `test_continuous.py`), because natural
  pauses let the SOFT trigger fire early and quietly shorten every turn. A steady speaker gives
  no such gap and drives every turn to the HARD cap -- the worse case, and the real one.
* **Long enough to see many rotations**, repeated, so "how often" and "how long" are
  distributions rather than three anecdotes.
* **Gaps partitioned by cause.** A `turn` increment in the partial stream is a rotation. Gaps
  that span one are separated from gaps inside a turn; pooling them yields a p99 that describes
  neither.
* **The causal claim is tested, not asserted.** If the rotation gap equals the turn-0 cold start,
  then the pause IS time-to-first-partial being re-paid on an empty audio window -- not decoder
  reload, not GPU work. `--server-log` parses the server's own per-turn warm-up lines
  (`app.py` logs `turn N first commit after X.XXs`) so the claim rests on both sides of the wire.

Arms are labelled and merged into one file, so the rotation-off counterfactual
(`CORE_ROLL_SOFT_SECS=45 CORE_ROLL_HARD_SECS=60`, which needs a container restart) lands beside
the shipped configuration in the same JSON.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/app/bench")

import metrics_lib as M  # noqa: E402
from ws_client import stream_one  # noqa: E402

SR = 16000
TURN_LOG = re.compile(r"turn (\d+) first commit after ([\d.]+)s")


def trim_silence(w, thresh=0.01, win=1600):
    """Drop near-silent windows so the stream offers the soft trigger no gap to fire on."""
    if len(w) < win:
        return w
    n = len(w) // win
    fr = w[:n * win].reshape(n, win)
    keep = np.abs(fr).max(axis=1) > thresh
    return fr[keep].reshape(-1) if keep.any() else w


def build_continuous(manifest: Path, lang: str, seconds: float) -> np.ndarray:
    man = json.loads(manifest.read_text())
    clips = [i for i in man["items"] if i["lang"] == lang and i["bucket"] in ("medium", "long")]
    if not clips:
        raise SystemExit(f"no medium/long clips for {lang}")
    parts = [trim_silence(sf.read(c["path"], dtype="float32")[0]) for c in sorted(
        clips, key=lambda c: c["path"])]
    wav = np.concatenate(parts)
    while len(wav) / SR < seconds:
        wav = np.concatenate([wav, wav])
    return wav[:int(seconds * SR)]


def analyse(run: dict) -> dict:
    """Turn structure and gap statistics for one stream.

    Gaps are computed over EVERY commit, `turn_final` included. A turn_final updates the
    transcript on screen exactly as a partial does, so excluding it does not measure what a
    user sees -- it measures a gap that spans the real last word of the outgoing turn and
    reports the wait as longer than it was. That distinction matters most at precisely the
    boundary this run exists to characterise.
    """
    body = list(run.get("partials", []))

    steady, boundary = [], []
    for i in range(1, len(body)):
        gap = body[i]["t_ms"] - body[i - 1]["t_ms"]
        if body[i].get("turn") != body[i - 1].get("turn"):
            boundary.append(gap)
        else:
            steady.append(gap)

    # Per turn: when its first partial landed, and how many partials it produced.
    turns: dict[int, list] = {}
    for p in body:
        turns.setdefault(p.get("turn"), []).append(p)
    turn_rows = [{
        "turn": t,
        "n_partials": len(ps),
        "first_partial_t_ms": round(ps[0]["t_ms"], 1),
        "last_partial_t_ms": round(ps[-1]["t_ms"], 1),
        "span_s": round((ps[-1]["t_ms"] - ps[0]["t_ms"]) / 1000, 2),
    } for t, ps in sorted(turns.items(), key=lambda kv: (kv[0] is None, kv[0]))]

    audio_s = run["audio_s"]
    n_rot = max(0, len(turn_rows) - 1)
    return {
        "audio_s": audio_s,
        "n_partials": len(body),
        "n_turns": len(turn_rows),
        "n_rotations": n_rot,
        "rotations_per_min": round(n_rot / (audio_s / 60), 2) if audio_s else None,
        "mean_seconds_between_rotations": round(audio_s / n_rot, 2) if n_rot else None,

        "ttfp_ms": run.get("ttfp_ms"),
        "tail_ms": run.get("tail_ms"),

        "gap_steady_ms_p50": M.percentile(steady, 0.50),
        "gap_steady_ms_p90": M.percentile(steady, 0.90),
        "gap_steady_ms_p99": M.percentile(steady, 0.99),
        "gap_steady_ms_max": round(max(steady), 1) if steady else None,
        "n_gaps_steady": len(steady),

        "gap_boundary_ms_p50": M.percentile(boundary, 0.50),
        "gap_boundary_ms_max": round(max(boundary), 1) if boundary else None,
        "gap_boundary_ms_all": [round(b, 1) for b in boundary],
        "n_gaps_boundary": len(boundary),
        # Raw samples retained so the pooled percentiles below are percentiles of the samples,
        # not an average of per-run percentiles.
        "gap_steady_ms_all": [round(g, 1) for g in steady],

        "turns": turn_rows,
        "words_per_partial": run.get("words_per_partial"),
        "client_bound": run.get("client_bound"),
    }


def parse_server_log(path: Path) -> dict:
    """The server's own per-turn warm-up, from `turn N first commit after X.XXs`.

    Turn 0 is a cold start with no prior state; every later turn is a rotation. If the two
    are the same size, the rotation pause is the cold start and nothing else.
    """
    if not path or not path.exists():
        return {"available": False, "note": "NOT MEASURED (no server log supplied)"}
    cold, rot = [], []
    for line in path.read_text(errors="replace").splitlines():
        m = TURN_LOG.search(line)
        if not m:
            continue
        (cold if int(m.group(1)) == 0 else rot).append(float(m.group(2)) * 1000)
    return {
        "available": bool(cold or rot),
        "n_cold_starts": len(cold),
        "cold_start_ms_p50": M.percentile(cold, 0.50),
        "n_rotation_warmups": len(rot),
        "rotation_warmup_ms_p50": M.percentile(rot, 0.50),
        "rotation_warmup_ms_max": round(max(rot), 1) if rot else None,
        # The whole point: if this ratio is ~1.0 the pause is a re-paid cold start.
        "warmup_over_cold_start": (
            round(M.percentile(rot, 0.50) / M.percentile(cold, 0.50), 3)
            if cold and rot and M.percentile(cold, 0.50) else None),
    }


async def main_async(args) -> int:
    wav = build_continuous(Path(args.corpus), args.lang, args.seconds)
    print(f"[runG] arm={args.arm} {len(wav)/SR:.0f}s gap-free speech x{args.repeats}",
          file=sys.stderr, flush=True)

    runs = []
    for k in range(args.repeats):
        try:
            r = await stream_one(args.url, wav, args.lang, rate=1.0, block_ms=100,
                                 label=f"{args.arm}#{k}", detail=True)
        except Exception as e:
            # A repeat that kills the server is a result, not a reason to discard the repeats
            # that succeeded. This is not hypothetical: it happened twice while this report was
            # being produced, and the first version of this script threw both runs away.
            print(f"[runG]   #{k}: ERROR {e!r}", file=sys.stderr, flush=True)
            runs.append({"repeat": k, "error": repr(e), "client_bound": True,
                         "note": "stream failed; the server may have restarted"})
            await asyncio.sleep(args.error_cooldown)
            continue
        a = analyse(r)
        a["repeat"] = k
        a["text_chars"] = len(r.get("text", ""))
        runs.append(a)
        print(f"[runG]   #{k}: {a['n_partials']} partials, {a['n_rotations']} rotations "
              f"({a['rotations_per_min']}/min), steady p50 {a['gap_steady_ms_p50']} ms, "
              f"boundary p50 {a['gap_boundary_ms_p50']} ms / max {a['gap_boundary_ms_max']} ms, "
              f"ttfp {a['ttfp_ms']} ms, tail {a['tail_ms']} ms",
              file=sys.stderr, flush=True)
        await asyncio.sleep(args.cooldown)

    ok = [r for r in runs if not r.get("client_bound") and "error" not in r]
    steady_all = [g for r in ok for g in r["gap_steady_ms_all"]]
    boundary_all = [g for r in ok for g in r["gap_boundary_ms_all"]]
    ttfps = [r["ttfp_ms"] for r in ok if r.get("ttfp_ms")]

    arm = {
        "arm": args.arm,
        "config": {"lang": args.lang, "seconds": args.seconds, "repeats": args.repeats,
                   "roll_soft_secs": args.roll_soft, "roll_hard_secs": args.roll_hard},
        "runs": runs,
        "pooled": {
            "n_runs": len(runs),
            "n_failed": sum(1 for r in runs if "error" in r),
            "n_client_bound": len(runs) - len(ok),
            "rotations_per_min_mean": round(
                sum(r["rotations_per_min"] for r in ok) / len(ok), 2) if ok else None,
            "seconds_between_rotations_mean": round(
                sum(r["mean_seconds_between_rotations"] for r in ok
                    if r["mean_seconds_between_rotations"]) / len(ok), 2) if ok else None,
            # Pooled over every raw boundary gap across repeats, not a mean of per-run p50s.
            "gap_boundary_ms_p50": M.percentile(boundary_all, 0.50),
            "gap_boundary_ms_p95": M.percentile(boundary_all, 0.95),
            "gap_boundary_ms_max": round(max(boundary_all), 1) if boundary_all else None,
            "n_boundary_gaps": len(boundary_all),
            "gap_steady_ms_p50": M.percentile(steady_all, 0.50),
            "gap_steady_ms_p90": M.percentile(steady_all, 0.90),
            "gap_steady_ms_p99": M.percentile(steady_all, 0.99),
            "n_steady_gaps": len(steady_all),
            "ttfp_ms_mean": round(sum(ttfps) / len(ttfps), 1) if ttfps else None,
            "tail_ms_max": max((r["tail_ms"] for r in ok if r.get("tail_ms")), default=None),
            "partials_per_min_mean": round(
                sum(r["n_partials"] / (r["audio_s"] / 60) for r in ok) / len(ok), 1) if ok else None,
        },
        "server_log": parse_server_log(Path(args.server_log) if args.server_log else None),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(args.out.read_text()) if args.out.exists() else {"arms": {}}
    doc.setdefault("arms", {})[args.arm] = arm
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    print(f"[runG] wrote arm '{args.arm}' to {args.out}", file=sys.stderr, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://localhost:9002/v1/asr/ws")
    ap.add_argument("--corpus", default="/corpus/manifest.json")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=4.0)
    ap.add_argument("--error-cooldown", type=float, default=240.0,
                    help="wait after a failed repeat; a fatal CUDA error restarts the "
                         "container and the model reload takes minutes")
    ap.add_argument("--arm", default="shipped",
                    help="label for this configuration, e.g. 'shipped' or 'rotation_off'")
    ap.add_argument("--roll-soft", type=float, default=12.0, help="recorded, not applied")
    ap.add_argument("--roll-hard", type=float, default=20.0, help="recorded, not applied")
    ap.add_argument("--server-log", default=None,
                    help="file of server log lines, for per-turn warm-up attribution")
    ap.add_argument("--out", type=Path, default=Path("/results/runG_rotation.json"))
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
