"""Command-line interface: ``tts-eval <command>``.

A thin, dependency-free wrapper over the Python API. Every command maps to a few
public calls (``build_plan`` → ``run_sync`` → ``store.save`` → ``write_run_report``
and so on); the CLI adds argument parsing, a progress line, human-readable tables
and clean error messages, and *no* evaluation logic of its own. Anything the CLI
can do is reproducible from the API, and vice versa.

Design notes:

*   Heavy modules (adapters, metrics, numpy) are imported inside the handlers, not
    at module top level, so ``tts-eval --help`` and tab-completion stay instant and
    a missing optional backend never breaks ``--help``.
*   Handlers raise :class:`~tts_eval.errors.TTSEvalError` for expected failures;
    :func:`main` turns those into a one-line stderr message and a non-zero exit,
    reserving tracebacks for genuine bugs (surface them with ``--debug``).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .errors import TTSEvalError

# Exit codes: 0 ok, 1 expected failure (bad config, unreachable server, …),
# 2 a comparison/verification that ran fine but reports "not comparable" / drift.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NONCOMPARABLE = 2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _progress(completed: int, total: int, last: str) -> None:
    """Single rewritten line on stderr, so piping stdout stays clean."""
    bar_w = 24
    filled = int(bar_w * completed / total) if total else bar_w
    bar = "#" * filled + "-" * (bar_w - filled)
    end = "\n" if completed >= total else ""
    print(f"\r  [{bar}] {completed}/{total}  {last:<24.24}", end=end, file=sys.stderr, flush=True)


def _fmt(value: float | None, ndigits: int = 2) -> str:
    return "—" if value is None else f"{value:.{ndigits}f}"


def _headline(record: Any, metrics: Sequence[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for name in metrics:
        agg = record.aggregates.get(name)
        if agg is None or agg.n == 0:
            continue
        unit = f" {agg.unit}" if agg.unit and agg.unit not in ("ratio", "score") else ""
        rows.append((name, f"{_fmt(agg.mean)}{unit}  (n={agg.n})"))
    return rows


_HEADLINE_METRICS = (
    "success_rate",
    "coverage_ratio",
    "ttfb_ms",
    "first_audible_ms",
    "rtf",
    "cer",
    "wer",
    "slot_accuracy",
    "utmos",
    "audio_quality_score",
    "degenerate_rate",
    "voice_consistency",
)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    from .config import load_model_card, load_suite
    from .report import write_run_report
    from .runner import build_plan, run_sync
    from .store import RunStore

    store = RunStore(args.runs)
    card = load_model_card(args.model)
    suite = load_suite(args.suite)
    plan = build_plan(
        card,
        suite,
        label=args.label,
        output_dir=store.root,
        save_audio=(False if args.no_audio else None),
    )

    _eprint(f"running {card.display_name} / {suite.suite_id}  ({len(plan.dataset)} utterances)")
    record = run_sync(plan, progress=None if args.quiet else _progress)
    run_dir = store.save(record)
    write_run_report(record, run_dir)

    if args.json:
        print(json.dumps({"run_id": record.run_id, "run_dir": str(run_dir),
                          "fingerprint": record.fingerprint,
                          "success_rate": record.success_rate}, indent=2))
        return EXIT_OK

    print(f"\nrun {record.run_id}")
    print(f"  dir:         {run_dir}")
    print(f"  report:      {run_dir / 'report.html'}")
    print(f"  fingerprint: {record.fingerprint}")
    print(f"  success:     {record.n_ok}/{len(record.utterances)}")
    for name, value in _headline(record, _HEADLINE_METRICS):
        print(f"  {name:<20} {value}")
    for warning in record.warnings:
        _eprint(f"  ! {warning}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
def cmd_list(args: argparse.Namespace) -> int:
    from .store import RunStore

    store = RunStore(args.runs)
    runs = store.list_runs(model_id=args.model, limit=args.limit)
    if args.json:
        print(json.dumps([{
            "run_id": r.run_id, "model": r.display_name, "created_at": r.created_at,
            "n_ok": r.n_ok, "n_utterances": r.n_utterances, "success_rate": r.success_rate,
            "concurrency": r.concurrency, "fingerprint": r.fingerprint,
        } for r in runs], indent=2))
        return EXIT_OK

    if not runs:
        _eprint(f"no runs in {store.root}")
        return EXIT_OK
    print(f"{'RUN ID':<28} {'MODEL':<24} {'WHEN':<20} {'OK':>7}  {'CONC':>4}")
    for r in runs:
        ok = f"{r.n_ok}/{r.n_utterances}"
        print(f"{r.run_id:<28} {r.display_name:<24.24} {r.created_at[:19]:<20} {ok:>7}  {r.concurrency:>4}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# report (regenerate)
# ---------------------------------------------------------------------------
def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_run_report
    from .store import RunStore

    store = RunStore(args.runs)
    record = store.load(args.run_id)
    run_dir = store.root / record.run_id
    written = write_run_report(record, run_dir)
    for name, path in written.items():
        print(f"  {name:<18} {path}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare_runs
    from .report import write_comparison_report
    from .store import RunStore

    store = RunStore(args.runs)
    baseline = store.load(args.baseline)
    candidate = store.load(args.candidate)
    comparison = compare_runs(baseline, candidate)

    if args.json:
        print(json.dumps(comparison.to_dict(), indent=2, default=str))
    else:
        print(comparison.summary_line())
        for blocker in comparison.blockers:
            _eprint(f"  BLOCKED: {blocker}")
        for warning in comparison.warnings:
            _eprint(f"  ! {warning}")
        if comparison.comparable:
            for mc in comparison.regressions() + comparison.improvements():
                print(f"  {mc.verdict:<12} {mc.metric:<24} "
                      f"{_fmt(mc.baseline_mean)} -> {_fmt(mc.candidate_mean)}")

    if args.out:
        written = write_comparison_report(comparison, args.out)
        for path in written.values():
            _eprint(f"  wrote {path}")

    return EXIT_OK if comparison.comparable else EXIT_NONCOMPARABLE


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def cmd_dataset(args: argparse.Namespace) -> int:
    from .datasets.loader import list_builtin, load_dataset

    if args.dataset_cmd == "list":
        for name in list_builtin():
            print(name)
        return EXIT_OK

    # show
    dataset = load_dataset(args.name)
    summary = dataset.summary()
    if args.json:
        print(json.dumps(summary, indent=2))
        return EXIT_OK
    print(f"{summary['id']} @ {summary['version']}  ({summary['n_cases']} cases)")
    print(f"  content_hash:  {summary['content_hash']}")
    print(f"  languages:     {', '.join(summary['languages'])}")
    print(f"  categories:    {', '.join(summary['categories'])}")
    print("  per language:")
    for lang, count in summary["per_language_counts"].items():
        print(f"    {lang:<6} {count}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# subjective
# ---------------------------------------------------------------------------
def cmd_subjective_build(args: argparse.Namespace) -> int:
    from .store import RunStore
    from .subjective import TestSpec, build_test

    store = RunStore(args.runs)
    records = [store.load(rid) for rid in args.run_ids]
    spec = TestSpec(scale=args.scale, n_raters=args.raters, n_trials=args.trials)
    build_test(records, args.out, spec)

    out = Path(args.out)
    print(f"listening test written to {out}")
    print(f"  send to raters:   {out / 'index.html'}  (+ the audio/ dir)")
    print(f"  KEEP PRIVATE:     {out / 'ANSWER_KEY.json'}")
    return EXIT_OK


def cmd_subjective_ingest(args: argparse.Namespace) -> int:
    from .store import RunStore
    from .subjective import ingest_sheets, merge_into_run

    sheets = sorted(Path(p) for pat in args.sheets for p in glob.glob(pat))
    if not sheets:
        raise TTSEvalError(f"no sheets matched: {' '.join(args.sheets)}")
    report = ingest_sheets(sheets, args.key)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(f"scale {report.scale}: {report.n_rows_used}/{report.n_rows_read} rows used")
        if report.excluded_raters:
            print(f"  excluded raters: {', '.join(report.excluded_raters)}")
        for system, stats in report.per_system.items():
            mean = stats.get("mean")
            print(f"  {system:<32} {_fmt(mean)}")
        if report.agreement:
            print(f"  agreement: {report.agreement}")

    if args.merge:
        store = RunStore(args.runs)
        for run_id, scores in report.scores_by_run.items():
            store.add_subjective(run_id, scores)
            _eprint(f"  merged {len(scores)} scores into {run_id}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# verify (reproducibility check)
# ---------------------------------------------------------------------------
def cmd_verify(args: argparse.Namespace) -> int:
    from .config import ModelCard, SuiteConfig, load_model_card
    from .compare import compare_runs
    from .runner import build_plan, compute_fingerprint, run_sync
    from .store import RunStore

    store = RunStore(args.runs)
    original = store.load(args.run_id)

    # Reconstruct the exact protocol from the stored record (so a drifted suite file
    # cannot mask a real regression), but take the model card live from configs so
    # real credentials — redacted out of the stored record — are available again.
    suite_dict = original.environment.get("suite") or {}
    if not suite_dict:
        raise TTSEvalError(f"run {original.run_id} has no stored suite; cannot verify")
    suite = SuiteConfig.from_dict(suite_dict)
    try:
        card = load_model_card(args.model or original.model_id)
    except TTSEvalError:
        # Fall back to the stored (redacted) card — fine for mock/replay/offline.
        card = ModelCard.from_dict(original.model_card)

    plan = build_plan(card, suite, output_dir=store.root)
    new_fp, new_inputs = compute_fingerprint(plan)

    if new_fp != original.fingerprint:
        _eprint(f"fingerprint CHANGED: {original.fingerprint} -> {new_fp}")
        old_inputs = original.environment.get("fingerprint_inputs") or {}
        for key in sorted(set(old_inputs) | set(new_inputs)):
            if old_inputs.get(key) != new_inputs.get(key):
                _eprint(f"  {key}: {old_inputs.get(key)!r} -> {new_inputs.get(key)!r}")
        _eprint("inputs differ, so results are not expected to reproduce.")
        return EXIT_NONCOMPARABLE

    print(f"fingerprint reproduces: {new_fp}")
    if args.check_only:
        return EXIT_OK

    _eprint("re-running to measure drift ...")
    replay = run_sync(plan, progress=None if args.quiet else _progress)
    store.save(replay)
    comparison = compare_runs(original, replay)
    print(comparison.summary_line())
    moved = comparison.regressions() + comparison.improvements()
    for mc in moved:
        print(f"  DRIFT {mc.verdict:<10} {mc.metric:<24} "
              f"{_fmt(mc.baseline_mean)} -> {_fmt(mc.candidate_mean)}")
    if moved:
        _eprint(f"{len(moved)} metric(s) moved beyond noise; run is not bit-reproducible "
                "(expected when the model samples stochastically).")
    else:
        print("  no metric moved beyond its noise floor.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
def cmd_serve(args: argparse.Namespace) -> int:
    from .ui.server import serve_forever

    serve_forever(args.runs, host=args.host, port=args.port)
    return EXIT_OK


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts-eval",
        description="Standardised, reproducible evaluation harness for TTS models.",
    )
    parser.add_argument("--version", action="version", version=f"tts-eval {__version__}")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on error")
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    def add_runs(p: argparse.ArgumentParser) -> None:
        p.add_argument("--runs", default="runs", metavar="DIR",
                       help="run store directory (default: ./runs)")

    # run
    p_run = sub.add_parser("run", help="evaluate a model against a suite")
    p_run.add_argument("-m", "--model", required=True, help="model card name or path")
    p_run.add_argument("-s", "--suite", default="smoke", help="suite name or path (default: smoke)")
    add_runs(p_run)
    p_run.add_argument("--label", help="human label for this run")
    p_run.add_argument("--no-audio", action="store_true", help="do not persist synthesised audio")
    p_run.add_argument("--quiet", action="store_true", help="suppress the progress bar")
    p_run.add_argument("--json", action="store_true", help="machine-readable summary on stdout")
    p_run.set_defaults(func=cmd_run)

    # list
    p_list = sub.add_parser("list", help="list stored runs")
    add_runs(p_list)
    p_list.add_argument("-m", "--model", help="filter by model id")
    p_list.add_argument("-n", "--limit", type=int, default=20, help="max rows (default: 20)")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    # report
    p_report = sub.add_parser("report", help="(re)generate reports for a stored run")
    p_report.add_argument("run_id", help="run id (or unique prefix)")
    add_runs(p_report)
    p_report.set_defaults(func=cmd_report)

    # compare
    p_cmp = sub.add_parser("compare", help="compare two runs (is B better than A?)")
    p_cmp.add_argument("baseline", help="baseline run id (or prefix)")
    p_cmp.add_argument("candidate", help="candidate run id (or prefix)")
    add_runs(p_cmp)
    p_cmp.add_argument("--out", metavar="DIR", help="also write comparison.html/md here")
    p_cmp.add_argument("--json", action="store_true")
    p_cmp.set_defaults(func=cmd_compare)

    # dataset
    p_ds = sub.add_parser("dataset", help="inspect built-in test sets")
    ds_sub = p_ds.add_subparsers(dest="dataset_cmd", metavar="<list|show>", required=True)
    ds_sub.add_parser("list", help="list built-in datasets").set_defaults(func=cmd_dataset)
    p_ds_show = ds_sub.add_parser("show", help="summarise a dataset")
    p_ds_show.add_argument("name", help="dataset name or path")
    p_ds_show.add_argument("--json", action="store_true")
    p_ds_show.set_defaults(func=cmd_dataset)

    # subjective
    p_subj = sub.add_parser("subjective", help="human listening tests (build / ingest)")
    subj_sub = p_subj.add_subparsers(dest="subjective_cmd", metavar="<build|ingest>", required=True)
    p_build = subj_sub.add_parser("build", help="build a blinded listening test from runs")
    p_build.add_argument("run_ids", nargs="+", help="run ids (1 for MOS, 2+ for MUSHRA/CMOS)")
    add_runs(p_build)
    p_build.add_argument("--out", required=True, metavar="DIR", help="output directory")
    p_build.add_argument("--scale", default="mushra", choices=["mos", "mushra", "cmos", "smos"])
    p_build.add_argument("--raters", type=int, default=5)
    p_build.add_argument("--trials", type=int, default=20)
    p_build.set_defaults(func=cmd_subjective_build)
    p_ing = subj_sub.add_parser("ingest", help="ingest rater sheets and screen raters")
    p_ing.add_argument("--sheets", nargs="+", required=True, metavar="GLOB",
                       help="sheet CSV path(s) or glob(s)")
    p_ing.add_argument("--key", required=True, metavar="FILE", help="ANSWER_KEY.json path")
    p_ing.add_argument("--merge", action="store_true", help="merge screened scores into the runs")
    add_runs(p_ing)
    p_ing.add_argument("--json", action="store_true")
    p_ing.set_defaults(func=cmd_subjective_ingest)

    # verify
    p_vf = sub.add_parser("verify", help="re-run a stored run and report reproducibility drift")
    p_vf.add_argument("run_id", help="run id (or prefix) to verify")
    add_runs(p_vf)
    p_vf.add_argument("-m", "--model", help="model card to use (default: the run's own model id)")
    p_vf.add_argument("--check-only", action="store_true",
                      help="only confirm the fingerprint reproduces; do not re-run")
    p_vf.add_argument("--quiet", action="store_true")
    p_vf.set_defaults(func=cmd_verify)

    # serve
    p_serve = sub.add_parser("serve", help="browse runs in a local web UI")
    add_runs(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _eprint("\ninterrupted.")
        return 130
    except BrokenPipeError:
        return EXIT_OK
    except TTSEvalError as e:
        if getattr(args, "debug", False):
            raise
        _eprint(f"error: {e}")
        return EXIT_ERROR
    except FileNotFoundError as e:
        if getattr(args, "debug", False):
            raise
        _eprint(f"error: {e}")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
