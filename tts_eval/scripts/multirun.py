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
import json
import re
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


def _esc(s: object) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


_PRETTY = {
    "cer": "Character error rate", "wer": "Word error rate", "slot_accuracy": "Slot accuracy",
    "ttfb_ms": "Time to first byte", "first_audible_ms": "Time to first audible",
    "rtf": "Real-time factor", "inference_time_ms": "Inference time",
    "throughput_utt_per_min": "Throughput", "utmos": "UTMOS (naturalness)",
    "dnsmos_ovrl": "DNSMOS overall", "dnsmos_sig": "DNSMOS signal", "dnsmos_bak": "DNSMOS background",
    "audio_quality_score": "Audio-quality score", "snr_db": "SNR", "clipping_pct": "Clipping",
    "loudness_dbfs": "Loudness", "dynamic_range_db": "Dynamic range", "spectral_flatness": "Spectral flatness",
    "speaker_consistency": "Speaker consistency", "speaker_similarity": "Speaker similarity",
    "voice_consistency": "Voice consistency", "intra_utterance_f0_cv": "Intra-utterance F0 CV",
    "success_rate": "Success rate", "degenerate_rate": "Degenerate rate",
    "coverage_ratio": "Coverage ratio", "languages_verified": "Languages verified",
    "stream_starvation_ms": "Stream starvation", "stream_chunk_gap_p95_ms": "Chunk gap p95",
}
_KPIS = ("success_rate", "cer", "wer", "utmos", "dnsmos_ovrl", "coverage_ratio")


def _pretty(name: str) -> str:
    return _PRETTY.get(name, name.replace("_", " "))


def load_run_meta(run_dir: Path) -> dict:
    """Best-effort header metadata from a run's run.json."""
    try:
        r = json.loads((run_dir / "run.json").read_text())
    except Exception:
        return {}
    env = r.get("environment") or {}
    asr = env.get("asr") or {}
    suite = env.get("suite")
    suite_id = suite.get("suite_id", "") if isinstance(suite, dict) else (suite or "")
    concurrency = suite.get("concurrency", "") if isinstance(suite, dict) else r.get("concurrency", "")
    return {
        "run_id": r.get("run_id", run_dir.name), "label": r.get("label", ""),
        "model_id": r.get("model_id", ""), "model_version": r.get("model_version", ""),
        "provider": r.get("provider", ""), "suite": suite_id,
        "concurrency": str(concurrency) if concurrency != "" else "",
        "dataset_id": r.get("dataset_id", ""), "dataset_version": r.get("dataset_version", ""),
        "dataset_size": r.get("dataset_size", ""), "dataset_hash": r.get("dataset_hash", ""),
        "fingerprint": r.get("fingerprint", ""), "determinism": r.get("determinism", ""),
        "framework_version": r.get("framework_version", ""),
        "asr": (asr.get("endpoint") or asr.get("backend") or ""),
    }


def render_html(rows: list[dict], run_dirs: list[Path], out: Path) -> None:
    """Presentable, sectioned across-run report styled like the per-run report."""
    try:
        html = _rich_report(rows, run_dirs)
    except Exception as e:  # never fail the run over a report; fall back to a plain table
        print(f"  (rich report unavailable: {type(e).__name__}: {e}; writing plain table)", file=sys.stderr)
        html = _plain_report(rows, run_dirs)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")


def _rich_report(rows: list[dict], run_dirs: list[Path]) -> str:
    from tts_eval.report.style import CSS
    from tts_eval.report.scoring import score_value
    from tts_eval.metrics.catalog import by_criterion, criteria_order, spec
    from tts_eval.types import Direction

    metas = [load_run_meta(d) for d in run_dirs]
    m0 = next((m for m in metas if m), {})
    rowmap = {r["metric"]: r for r in rows}
    n_runs = len(run_dirs)
    fps = {m.get("fingerprint") for m in metas if m.get("fingerprint")}
    comparable = len(fps) <= 1

    def arrow(name: str) -> str:
        d = spec(name).direction
        return {Direction.HIGHER_IS_BETTER: "↑", Direction.LOWER_IS_BETTER: "↓"}.get(d, "")

    def badge(name: str, mean: float) -> str:
        s = score_value(name, mean)
        if s.fraction is None:
            return '<span class="small muted">—</span>'
        label = {"good": "good", "warn": "watch", "bad": "poor"}.get(s.band, s.band)
        return f'<span class="badge {s.band}">{label}</span>'

    def value_cell(r: dict) -> str:
        # mean, then std — every row is a value, so bolding every one of them
        # bolds nothing; the table's own structure carries that weight instead.
        return (f"{_fmt(r['mean'])} "
                f"<span class='small muted'>± {_fmt(r['std'])} {_esc(r['unit'])}</span>")

    # ---- headline KPIs ----
    tiles = []
    for name in _KPIS:
        if name not in rowmap:
            continue
        r = rowmap[name]
        s = score_value(name, r["mean"])
        band = s.band if s.fraction is not None else "neutral"
        tiles.append(
            f'<div class="kpi kpi-{band}"><div class="kpi-label">{_esc(_pretty(name))} {arrow(name)}</div>'
            f'<div class="kpi-value">{_fmt(r["mean"])}<span class="kpi-unit"> {_esc(r["unit"])}</span></div>'
            f'<div class="kpi-sub">± {_fmt(r["std"])} · {r["cv_pct"]:.1f}% cv · {_fmt(r["min"])}–{_fmt(r["max"])}</div></div>'
        )
    kpi_block = f'<div class="kpi-row">{"".join(tiles)}</div>' if tiles else ""

    # ---- sectioned tables by criterion ----
    def slug(text: str) -> str:
        return "sec-" + re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

    grouped = by_criterion()
    order = criteria_order() + [c for c in sorted(grouped) if c not in criteria_order()]
    used: set[str] = set()
    sections = []
    toc: list[tuple[str, str]] = []   # (criterion label, anchor) for the overview index
    glossary: list = []               # metric specs to define in the reference at the end
    for crit in order:
        present = [s for s in grouped.get(crit, []) if s.name in rowmap]
        if not present:
            continue
        used |= {s.name for s in present}
        glossary.extend(present)
        sid = slug(crit)
        toc.append((crit, sid))
        body = []
        for s in present:
            r = rowmap[s.name]
            high = r["cv_pct"] >= 10 and r["runs"] > 1
            body.append(
                f'<tr><td><span class="mname">{_esc(_pretty(s.name))}</span> '
                f'<span class="muted pol">{arrow(s.name)}</span></td>'
                f'<td class="num">{value_cell(r)}</td>'
                f'<td class="num">{_fmt(r["min"])} – {_fmt(r["max"])}</td>'
                f'<td class="num {"cv-hi" if high else ""}">{r["cv_pct"]:.1f}%</td>'
                f'<td class="num">{r["runs"]}</td>'
                f'<td>{badge(s.name, r["mean"])}</td></tr>'
            )
        sections.append(
            f"<h2 id='{sid}'>{_esc(crit)}</h2><table class='report'><thead><tr>"
            "<th>Metric</th><th class='num'>Mean ± Std</th><th class='num'>Min – Max</th>"
            "<th class='num'>CV%</th><th class='num'>Runs</th><th>Rating</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>"
        )
    leftover = [n for n in rowmap if n not in used]
    if leftover:
        toc.append(("Other", "sec-other"))
        glossary.extend(spec(n) for n in sorted(leftover))
        body = []
        for n in sorted(leftover):
            r = rowmap[n]
            high = r["cv_pct"] >= 10 and r["runs"] > 1
            body.append(
                f'<tr><td>{_esc(_pretty(n))}</td><td class="num">{value_cell(r)}</td>'
                f'<td class="num">{_fmt(r["min"])} – {_fmt(r["max"])}</td>'
                f'<td class="num {"cv-hi" if high else ""}">{r["cv_pct"]:.1f}%</td>'
                f'<td class="num">{r["runs"]}</td><td></td></tr>'
            )
        sections.append(
            "<h2 id='sec-other'>Other</h2><table class='report'><thead><tr><th>Metric</th>"
            "<th class='num'>Mean ± Std</th><th class='num'>Min – Max</th><th class='num'>CV%</th>"
            f"<th class='num'>Runs</th><th></th></tr></thead><tbody>{''.join(body)}</tbody></table>"
        )

    # ---- header meta + provenance ----
    def meta_item(label: str, value: str) -> str:
        return f"<div><dt>{_esc(label)}</dt><dd>{_esc(value) or '—'}</dd></div>" if value else ""

    ds = f'{m0.get("dataset_id","")} v{m0.get("dataset_version","")} ({m0.get("dataset_size","")} utt)'
    meta = "".join([
        meta_item("Model", f'{m0.get("model_id","")} · {m0.get("model_version","")}'),
        meta_item("Provider", m0.get("provider", "")),
        meta_item("Suite", m0.get("suite", "")),
        meta_item("Concurrency", m0.get("concurrency", "")),
        meta_item("Dataset", ds if m0.get("dataset_id") else ""),
        meta_item("ASR (intelligibility)", m0.get("asr", "")),
        meta_item("Runs aggregated", str(n_runs)),
        meta_item("Determinism", m0.get("determinism", "")),
        meta_item("Fingerprint", (next(iter(fps)) if fps else "")[:16]),
    ])

    comparability = "" if comparable else (
        '<div class="callout bad"><strong>Runs are not directly comparable:</strong> '
        f'they span {len(fps)} different fingerprints (different dataset/config). '
        "Aggregated numbers below mix non-identical inputs.</div>"
    )
    run_rows = "".join(
        f"<tr><td class='num'>{i+1}</td><td><code>{_esc(m.get('run_id',''))}</code></td>"
        f"<td>{_esc(m.get('label',''))}</td></tr>"
        for i, m in enumerate(metas)
    )

    # ---- narrative: model-agnostic determinism note, overview index, glossary ----
    det_raw = m0.get("determinism", "")
    det = (det_raw or "").lower()
    det_code = f" (determinism: <code>{_esc(det_raw)}</code>)" if det_raw else ""
    if det in ("best_effort", "stochastic", "nondeterministic", "non_deterministic", "random", "none", ""):
        det_note = (f"This model is <strong>non-deterministic</strong>{det_code} — it can "
                    "synthesise different audio each run, so a single run is one noisy sample.")
    elif det in ("deterministic", "exact", "reproducible", "bitexact"):
        det_note = (f"This model reports <strong>deterministic</strong> output{det_code} — the "
                    "run-to-run spread below reflects measurement and environment noise, not "
                    "synthesis variation.")
    else:
        det_note = (f"Model determinism{det_code}. The spread below is the observed run-to-run "
                    "variation, whatever its source.")

    toc_items = "".join(f'<li><a href="#{sid}">{_esc(label)}</a></li>' for label, sid in toc)
    overview = f"""
<section class="overview">
<h2 id="overview">Overview</h2>
<p>This report aggregates {n_runs} independent evaluation run(s) of
<strong>{_esc(m0.get('model_id','the model'))}</strong> on the
{_esc(m0.get('suite','') or 'configured')} test set. For every metric it reports the
mean and the run-to-run spread across those runs, grouped by evaluation criterion.</p>
<p>{det_note} Judge a decision against <em>mean ± std</em> and the observed range, not a
single run. Ratings (good / watch / poor) are a reading aid from the metric catalogue's
reference points, not a formal pass/fail gate.</p>
<div class="ov-cols">
  <div><h3>Contents</h3><ol class="toc">{toc_items}
    <li><a href="#runs-included">Runs included</a></li>
    <li><a href="#metric-reference">Metric reference</a></li></ol></div>
  <div><h3>How to read a row</h3><ul class="legend">
    <li><strong>Mean ± Std</strong> — average across runs and its run-to-run standard deviation.</li>
    <li><strong>Min – Max</strong> — the range observed over the runs.</li>
    <li><strong>CV%</strong> — std ÷ mean; <span class="cv-hi">red ≥ 10%</span> marks noisy metrics.</li>
    <li><strong>Runs</strong> — how many runs produced the metric (absent backends excluded).</li>
    <li><strong>Rating</strong> — good / watch / poor vs the catalogue reference points.</li>
  </ul></div>
</div>
<p class="ref-pointer">Each metric is defined in the
<a href="#metric-reference">Metric reference</a> at the end of this report.</p>
</section>"""

    gseen: set[str] = set()
    gloss_rows = []
    for s in glossary:
        if s.name in gseen:
            continue
        gseen.add(s.name)
        better = {Direction.HIGHER_IS_BETTER: "higher", Direction.LOWER_IS_BETTER: "lower"}.get(
            spec(s.name).direction, "—")
        gloss_rows.append(
            f"<tr><td class='mname'>{_esc(_pretty(s.name))}</td>"
            f"<td class='small muted'><code>{_esc(s.name)}</code></td>"
            f"<td class='small'>{_esc(s.unit) or '—'}</td><td class='small'>{better}</td>"
            f"<td>{_esc(s.summary)}</td></tr>"
        )
    glossary_html = (
        "<h2 id='metric-reference'>Metric reference</h2>"
        "<p class='small muted'>Definitions for every metric shown above. "
        "&lsquo;Better&rsquo; is the direction that indicates higher quality.</p>"
        "<table class='report glossary'><thead><tr><th>Metric</th><th>Key</th><th>Unit</th>"
        "<th>Better</th><th>Definition</th></tr></thead><tbody>"
        + "".join(gloss_rows) + "</tbody></table>"
    )

    extra_css = """
.subtitle{color:var(--text-secondary);margin-top:-.4rem}
.overview{background:var(--surface-subtle);border:1px solid var(--border);
  border-radius:var(--radius-lg);padding:6px 22px 18px;margin:18px 0 26px}
.ov-cols{display:flex;flex-wrap:wrap;gap:36px}
.ov-cols>div{flex:1 1 260px}
.overview h3{font-size:.9rem;margin:.7rem 0 .4rem}
.toc{margin:0;padding-left:1.2rem}.toc li{margin:2px 0}
.legend{list-style:none;margin:0;padding:0;font-size:.85rem}
.legend li{margin:3px 0}
.ref-pointer{margin:.9rem 0 0;font-size:.85rem}
.kpi-row{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0 26px}
.kpi{flex:1 1 150px;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius-md);padding:14px 16px;box-shadow:var(--shadow-sm);
  border-top:3px solid var(--muted)}
.kpi-good{border-top-color:var(--good)}.kpi-warn{border-top-color:var(--warn)}
.kpi-bad{border-top-color:var(--bad)}
.kpi-label{font-size:.8rem;color:var(--text-secondary);margin-bottom:6px}
.kpi-value{font-size:1.5rem;font-weight:650;font-variant-numeric:tabular-nums}
.kpi-unit{font-size:.9rem;color:var(--muted);font-weight:400}
.kpi-sub{font-size:.72rem;color:var(--muted);margin-top:4px;font-variant-numeric:tabular-nums}
table.report{width:100%;border-collapse:collapse;margin:.4rem 0 1.6rem}
table.report th,table.report td{border-bottom:1px solid var(--border);padding:7px 10px;text-align:left;vertical-align:top}
table.report th.num,table.report td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.mname{font-weight:550}.pol{font-size:.85em}
.cv-hi{color:var(--bad);font-weight:600}
.glossary td{vertical-align:top}.glossary td:last-child{max-width:52ch}
.report-foot{margin-top:2rem;color:var(--text-secondary);font-size:.85rem}
@media print{.kpi,.overview{break-inside:avoid}table.report{break-inside:auto}
  h2{break-after:avoid;break-before:auto}#metric-reference{break-before:page}}
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-run evaluation — {_esc(m0.get('model_id','model'))}</title>
<style>{CSS}{extra_css}</style></head><body><main class="page">
<h1>Multi-run evaluation report</h1>
<p class="subtitle">Mean ± standard deviation across {n_runs} independent run(s).</p>
<dl class="meta-grid">{meta}</dl>
{comparability}
{overview}
{kpi_block}
{''.join(sections)}
{glossary_html}
<div class="report-foot">
<h2 id="runs-included">Runs included</h2>
<table class="report"><thead><tr><th class="num">#</th><th>Run ID</th><th>Label</th></tr></thead>
<tbody>{run_rows}</tbody></table>
<p>CV% = std ÷ mean across runs. Rows flagged in red (CV ≥ 10%) are the noisy metrics
that most benefit from more runs. Metrics absent from a run (not-computed backends)
are aggregated only over the runs that produced them — see each row's <em>Runs</em>.</p>
</div>
</main></body></html>"""


def _plain_report(rows: list[dict], run_dirs: list[Path]) -> str:
    head = ("<style>body{font-family:system-ui,sans-serif;margin:2rem}"
            "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}"
            ".num{text-align:right}</style>")
    body = "".join(
        f"<tr><td>{_esc(r['metric'])}</td><td class='num'>{r['runs']}</td>"
        f"<td class='num'>{_fmt(r['mean'])}</td><td class='num'>{_fmt(r['std'])}</td>"
        f"<td class='num'>{_fmt(r['min'])} – {_fmt(r['max'])}</td>"
        f"<td class='num'>{r['cv_pct']:.1f}%</td><td>{_esc(r['unit'])}</td></tr>"
        for r in rows
    )
    return (f"<!doctype html><meta charset='utf-8'>{head}<h1>Across-run summary "
            f"({len(run_dirs)} runs)</h1><table><thead><tr><th>Metric</th><th>Runs</th>"
            "<th>Mean</th><th>Std</th><th>Min – Max</th><th>CV%</th><th>Unit</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    """Print the HTML to PDF via headless Chrome, if available."""
    import shutil

    chrome = next((shutil.which(b) for b in
                   ("google-chrome", "chromium", "chromium-browser", "chrome")
                   if shutil.which(b)), None)
    if not chrome:
        print("no Chrome/Chromium found; open the HTML and Print → Save as PDF instead", file=sys.stderr)
        return
    proc = subprocess.run(
        [chrome, "--headless", "--no-sandbox", "--disable-gpu",
         f"--print-to-pdf={pdf_path.resolve()}", "--no-pdf-header-footer",
         html_path.resolve().as_uri()],
        capture_output=True, text=True,
    )
    if proc.returncode == 0 and pdf_path.is_file():
        print(f"wrote {pdf_path}")
    else:
        print(f"PDF export failed: {proc.stderr.strip()[:200]}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--suite")
    ap.add_argument("--runs", type=int, default=5, help="number of runs to execute (default 5)")
    ap.add_argument("--label", default="multirun", help="label prefix for executed runs")
    ap.add_argument("--from", dest="from_globs", nargs="+",
                    help="aggregate existing run dirs (globs) instead of executing")
    ap.add_argument("--out", type=Path, help="summary CSV path (default: <first run>/multirun_summary.csv)")
    ap.add_argument("--pdf", action="store_true", help="also export a PDF (needs Chrome/Chromium)")
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
    html_out = out.with_suffix(".html")
    render_html(rows, run_dirs, html_out)
    if args.pdf:
        render_pdf(html_out, out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
