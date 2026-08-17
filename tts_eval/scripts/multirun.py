#!/usr/bin/env python3
"""Run a suite N times and aggregate mean +/- stdev ACROSS runs.

Parler (and any ``determinism: best_effort`` model) synthesises different audio
each run, so a single run is one noisy sample: run-scalar metrics come back n=1
and a per-language verdict can flip across the accept threshold between runs. The
report's bootstrap CI captures variance *within* a run (across utterances); it
does NOT capture run-to-run synthesis noise. This wrapper measures that noise
directly -- the between-run standard deviation is the error bar you should judge a
go/no-go decision against.

Two modes:

    # Execute N fresh runs, then aggregate:
    ./venv/bin/python scripts/multirun.py --model ai4bharat-parler \
        --suite indic-full --runs 5 --label parler-multi

    # Aggregate runs you already have (no synthesis):
    ./venv/bin/python scripts/multirun.py --from runs/20260817T10*

For each metric it reports across-run mean, sample stdev, min, max, and CV%
(stdev/mean), plus how many of the runs produced the metric. Writes a
``multirun_summary.csv`` next to the first run (or to --out).
"""
from __future__ import annotations

import argparse
import csv
import glob
import statistics
import subprocess
import sys
from pathlib import Path


def _venv_cli() -> str:
    """The tts-eval console script next to THIS interpreter.

    Calling the bare name would risk a ~/.local shim on a different interpreter;
    the sibling of sys.executable is guaranteed to be this venv's CLI.
    """
    cli = Path(sys.executable).with_name("tts-eval")
    if not cli.exists():
        sys.exit(f"tts-eval not found next to {sys.executable}; run from the venv")
    return str(cli)


def execute_runs(model: str, suite: str, runs: int, label: str) -> list[Path]:
    """Run the suite `runs` times, returning each run directory."""
    cli = _venv_cli()
    dirs: list[Path] = []
    for i in range(1, runs + 1):
        run_label = f"{label}-{i:02d}"
        print(f"\n=== run {i}/{runs}  (label {run_label}) ===", flush=True)
        proc = subprocess.run(
            [cli, "run", "--model", model, "--suite", suite, "--label", run_label],
            capture_output=True,
            text=True,
        )
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            sys.exit(f"run {i} failed (exit {proc.returncode}); aborting")
        run_dir = _parse_run_dir(proc.stdout)
        if run_dir is None:
            sys.exit(f"run {i}: could not find the run directory in output")
        dirs.append(run_dir)
    return dirs


def _parse_run_dir(stdout: str) -> Path | None:
    """Pull the `dir: runs/...` line the CLI prints on success."""
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("dir:"):
            return Path(stripped.split(":", 1)[1].strip())
    return None


def read_aggregates(run_dir: Path) -> dict[str, dict]:
    """metric -> {mean, n, unit, direction} from a run's aggregates.csv."""
    path = run_dir / "aggregates.csv"
    if not path.is_file():
        print(f"  ! {run_dir}: no aggregates.csv, skipping", file=sys.stderr)
        return {}
    out: dict[str, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            mean = row.get("mean")
            if mean in (None, ""):
                continue  # not computed this run
            try:
                out[row["metric"]] = {
                    "mean": float(mean),
                    "n": int(row["n"]) if row.get("n") else 0,
                    "unit": row.get("unit", ""),
                    "direction": row.get("direction", ""),
                }
            except ValueError:
                continue
    return out


def aggregate(run_dirs: list[Path]) -> list[dict]:
    """Collapse per-run means into across-run statistics per metric."""
    per_metric: dict[str, list[float]] = {}
    meta: dict[str, dict] = {}
    for d in run_dirs:
        for metric, info in read_aggregates(d).items():
            per_metric.setdefault(metric, []).append(info["mean"])
            meta.setdefault(metric, {"unit": info["unit"], "direction": info["direction"]})

    rows: list[dict] = []
    for metric in sorted(per_metric):
        values = per_metric[metric]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        cv = (std / abs(mean) * 100.0) if mean else 0.0
        rows.append(
            {
                "metric": metric,
                "unit": meta[metric]["unit"],
                "direction": meta[metric]["direction"],
                "runs": len(values),
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
                "cv_pct": cv,
            }
        )
    return rows


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def print_table(rows: list[dict], n_runs: int) -> None:
    print(f"\nAcross-run summary  ({n_runs} runs)")
    print(f"{'metric':<24} {'runs':>4} {'mean':>10} {'std':>10} "
          f"{'min':>10} {'max':>10} {'cv%':>7}  unit")
    print("-" * 96)
    for r in rows:
        flag = "  <-- high spread" if r["cv_pct"] >= 10 and r["runs"] > 1 else ""
        print(f"{r['metric']:<24} {r['runs']:>4} {_fmt(r['mean']):>10} {_fmt(r['std']):>10} "
              f"{_fmt(r['min']):>10} {_fmt(r['max']):>10} {r['cv_pct']:>6.1f}%  {r['unit']}{flag}")
    print("\ncv% = stdev/mean across runs = the run-to-run synthesis noise. Judge a")
    print("decision against [mean +/- std] and [min, max], not a single run's number.")


def write_csv(rows: list[dict], out: Path) -> None:
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["metric", "unit", "direction", "runs", "mean", "std", "min", "max", "cv_pct"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--suite")
    ap.add_argument("--runs", type=int, default=5, help="number of runs to execute (default 5)")
    ap.add_argument("--label", default="multirun", help="label prefix for executed runs")
    ap.add_argument("--from", dest="from_globs", nargs="+",
                    help="aggregate existing run dirs (globs) instead of executing")
    ap.add_argument("--out", type=Path, help="summary CSV path (default: <first run>/multirun_summary.csv)")
    args = ap.parse_args()

    if args.from_globs:
        run_dirs = [Path(p) for g in args.from_globs for p in sorted(glob.glob(g)) if Path(p).is_dir()]
        if not run_dirs:
            sys.exit("no run directories matched --from")
    else:
        if not (args.model and args.suite):
            sys.exit("provide --model and --suite to execute runs, or --from to aggregate existing")
        run_dirs = execute_runs(args.model, args.suite, args.runs, args.label)

    print(f"\naggregating {len(run_dirs)} run(s):")
    for d in run_dirs:
        print(f"  {d}")

    rows = aggregate(run_dirs)
    if not rows:
        sys.exit("no metrics found across the given runs")
    print_table(rows, len(run_dirs))
    out = args.out or (run_dirs[0] / "multirun_summary.csv")
    write_csv(rows, out)


if __name__ == "__main__":
    main()
