"""Standalone HTML report rendering.

"Standalone" is a hard requirement, not a nicety: a benchmark report has to be
attachable to a ticket, opened from a shared drive, or archived next to the run —
none of which come with a running server. So every report is one `.html` file
with its CSS inlined (:mod:`.style`) and no external JS dependency. The
lightweight UI (:mod:`tts_eval.ui`) reuses the exact same render functions and
serves the CSS once instead of inlining it; the HTML markup a browser sees is
otherwise identical in both modes, which is what keeps them from drifting apart.

Audio links are parameterised via ``audio_base`` for the same reason: a report
written next to its run's ``audio/`` folder links with a relative path, while the
UI serves audio from a per-run route.
"""
from __future__ import annotations

import html as _html
from typing import Sequence

from ..compare import Comparison, MetricComparison
from ..metrics.catalog import ac_matrix, criteria_order
from ..types import Aggregate, LanguageCoverage, MetricStatus, RunRecord, UtteranceRecord
from .scoring import score_aggregate, verdict_band
from .style import CSS

_VERDICT_LABEL = {
    "better": "better",
    "worse": "worse",
    "negligible": "negligible",
    "inconclusive": "inconclusive",
    "single_observation": "changed",
    "insufficient_data": "n/a",
}

# Utterances beyond this in the "all utterances" table are still exported (CSV
# has everything); the HTML table caps here so a 500-utterance run does not ship
# a multi-megabyte report. The flagged/failed tables have no such cap since they
# are exactly the rows a reviewer needs.
_MAX_FULL_TABLE_ROWS = 300


def _e(text: object) -> str:
    return _html.escape(str(text))


def _fmt(value: float | None, ndigits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:,.{ndigits}f}" if abs(value) >= 1 else f"{value:.{ndigits}f}"


def _badge(label: str, band: str) -> str:
    return f'<span class="badge {band}">{_e(label)}</span>'


def _bar(fraction: float | None, band: str) -> str:
    if fraction is None:
        return ""
    pct = round(fraction * 100)
    return (
        f'<span class="bar-track"><span class="bar-fill {band}" '
        f'style="width:{pct}%"></span></span>'
    )


def _page(title: str, body: str, *, inline_css: bool = True, extra_head: str = "") -> str:
    style = f"<style>{CSS}</style>" if inline_css else '<link rel="stylesheet" href="/static/style.css">'
    theme_script = """
<script>
(function () {
  const root = document.documentElement;
  const storageKey = "tts_eval_theme";

  function getStoredTheme() {
    try {
      const value = localStorage.getItem(storageKey);
      return value === "dark" || value === "light" ? value : null;
    } catch (_) {
      return null;
    }
  }

  function applyTheme(theme) {
    root.dataset.theme = theme;
    const toggle = document.getElementById("theme-toggle");
    if (toggle) {
      const isDark = theme === "dark";
      toggle.setAttribute("aria-pressed", String(isDark));
      toggle.textContent = isDark ? "☀ Light" : "☾ Dark";
      toggle.setAttribute(
        "aria-label",
        isDark ? "Switch to light mode" : "Switch to dark mode"
      );
      toggle.title = isDark ? "Switch to light mode" : "Switch to dark mode";
    }
  }

  const initialTheme = getStoredTheme() || "light";
  applyTheme(initialTheme);

  document.addEventListener("click", function (event) {
    const toggle = event.target.closest("#theme-toggle");
    if (!toggle) return;

    const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
    try {
      localStorage.setItem(storageKey, nextTheme);
    } catch (_) {}
  });
})();
</script>
"""
    theme_toggle = """
<button id="theme-toggle" class="theme-toggle" type="button"
  aria-pressed="false" aria-label="Switch to dark mode"
  title="Switch to dark mode">☾ Dark</button>
"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
{style}
{theme_script}
{extra_head}
</head>
<body>
{theme_toggle}
{body}
</body>
</html>
"""



# ---------------------------------------------------------------------------
# run report
# ---------------------------------------------------------------------------
def render_run_html(
    record: RunRecord,
    *,
    title: str | None = None,
    inline_css: bool = True,
    audio_base: str = "audio/",
    nav_html: str = "",
) -> str:
    """Full single-run report. See module docstring for the standalone contract."""
    parts: list[str] = []

    parts.append(_run_header(record, nav_html))
    if record.warnings:
        parts.append(_warnings_callout(record.warnings))
    parts.append(_absent_backends_callout(record))

    for criterion in criteria_order():
        rows = [
            r for r in ac_matrix() if r["criterion"] == criterion and r["metric"] in record.aggregates
        ]
        if rows:
            parts.append(_criterion_section(criterion, rows, record.aggregates))

    if record.coverage:
        parts.append(_coverage_section(record.coverage))

    per_category = (record.environment or {}).get("per_category") or {}
    if per_category:
        parts.append(_category_section(per_category))

    if record.subjective:
        parts.append(_subjective_section(record))

    parts.append(_reliability_section(record, audio_base))
    parts.append(_signoff_section(record))
    parts.append(_utterances_section(record, audio_base))
    parts.append(_footer(record))

    return _page(title or record.label or record.display_name, "\n".join(parts), inline_css=inline_css)


def _run_header(record: RunRecord, nav_html: str) -> str:
    success = f"{record.success_rate:.1%}" if record.success_rate is not None else "—"
    determinism_note = (
        "" if record.determinism.value == "deterministic"
        else '<span class="badge warn">audio not bit-reproducible</span>'
    )
    return f"""
<div class="topbar">
  <div>
    <h1>{_e(record.label or record.display_name)}</h1>
    <p class="subtitle">{_e(record.display_name)} &middot; {_e(record.provider)}
      &middot; run <code>{_e(record.run_id)}</code></p>
  </div>
  <nav>{nav_html}</nav>
</div>
<dl class="meta-grid">
  <div><dt>Dataset</dt><dd>{_e(record.dataset_id)}@{_e(record.dataset_version)} ({record.dataset_size} utterances)</dd></div>
  <div><dt>Fingerprint</dt><dd>{_e(record.fingerprint)}</dd></div>
  <div><dt>Concurrency</dt><dd>{record.concurrency}</dd></div>
  <div><dt>Determinism</dt><dd>{_e(record.determinism.value)} {determinism_note}</dd></div>
  <div><dt>Started</dt><dd>{_e(record.created_at)}</dd></div>
  <div><dt>Success rate</dt><dd>{success} ({record.n_ok}/{len(record.utterances)})</dd></div>
</dl>
"""


def _warnings_callout(warnings: Sequence[str]) -> str:
    items = "".join(f"<li>{_e(w)}</li>" for w in warnings)
    return f'<div class="callout"><strong>Warnings</strong><ul>{items}</ul></div>'


def _absent_backends_callout(record: RunRecord) -> str:
    absent = {n: s for n, s in record.metric_backends.items() if s.startswith("absent")}
    if not absent:
        return ""
    items = "".join(
        f"<li><code>{_e(name)}</code>: {_e(reason[len('absent: '):])}</li>"
        for name, reason in sorted(absent.items())
    )
    return (
        '<div class="callout"><strong>Not computed in this run</strong>'
        f"<ul>{items}</ul></div>"
    )


def _criterion_section(criterion: str, rows: list[dict], aggregates: dict[str, Aggregate]) -> str:
    body_rows = []
    for row in rows:
        aggregate = aggregates.get(row["metric"])
        body_rows.append(_aggregate_row(row["metric"], row["summary"], aggregate))
    return f"""
<h2>{_e(criterion)}</h2>
<table>
  <thead><tr><th>Metric</th><th class="num">n</th><th class="num">Mean</th>
    <th class="num">p95</th><th class="num">95% CI</th><th></th></tr></thead>
  <tbody>{''.join(body_rows)}</tbody>
</table>
"""


def _aggregate_row(name: str, summary: str, aggregate: Aggregate | None) -> str:
    if aggregate is None or aggregate.n == 0:
        reason = (aggregate.missing_reason if aggregate else None) or "not computed"
        return (
            f'<tr><td title="{_e(summary)}">{_e(name)}</td><td class="num">0</td>'
            f'<td colspan="3" class="muted small">{_e(reason)}</td></tr>'
        )
    scored = score_aggregate(name, aggregate)
    ci = (
        f"[{_fmt(aggregate.ci_low)}, {_fmt(aggregate.ci_high)}]"
        if aggregate.ci_low is not None
        else '<span class="muted">—</span>'
    )
    return (
        f'<tr><td title="{_e(summary)}">{_e(name)}</td>'
        f'<td class="num">{aggregate.n}</td>'
        f'<td class="num">{_fmt(aggregate.mean)} {_e(aggregate.unit)}</td>'
        f'<td class="num">{_fmt(aggregate.p95)} {_e(aggregate.unit)}</td>'
        f'<td class="num">{ci}</td>'
        f'<td>{_bar(scored.fraction, scored.band)}</td></tr>'
    )


def _coverage_section(coverage: Sequence[LanguageCoverage]) -> str:
    rows = "".join(
        f"<tr><td>{_e(c.language)}</td>"
        f'<td class="num">{"✓" if c.claimed else ""}</td>'
        f'<td class="num">{c.attempted}</td>'
        f'<td class="num">{c.succeeded}</td>'
        f'<td>{_badge("verified", "good") if c.verified else _badge("unverified", "bad")}</td>'
        f'<td class="small muted">{_e(c.notes or "")}</td></tr>'
        for c in sorted(coverage, key=lambda x: x.language)
    )
    return f"""
<h2>Language Coverage</h2>
<table>
  <thead><tr><th>Language</th><th class="num">Claimed</th><th class="num">Attempted</th>
    <th class="num">Succeeded</th><th>Verified</th><th>Notes</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def _category_section(per_category: dict) -> str:
    rows = []
    for category, aggs in sorted(per_category.items()):
        cer = aggs.get("cer", {})
        quality = aggs.get("audio_quality_score", {})
        ttfb = aggs.get("ttfb_ms", {})
        n = max((aggs.get(k, {}).get("n") or 0) for k in ("cer", "audio_quality_score", "ttfb_ms"))
        rows.append(
            f"<tr><td>{_e(category)}</td><td class='num'>{n}</td>"
            f"<td class='num'>{_fmt(cer.get('mean'))}</td>"
            f"<td class='num'>{_fmt(quality.get('mean'))}</td>"
            f"<td class='num'>{_fmt(ttfb.get('mean'), 1)}</td></tr>"
        )
    return f"""
<h2>By Category</h2>
<table>
  <thead><tr><th>Category</th><th class="num">n</th><th class="num">CER</th>
    <th class="num">Audio Quality</th><th class="num">TTFB (ms)</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
"""


def _subjective_section(record: RunRecord) -> str:
    items = []
    for name in ("subjective_mos", "subjective_mushra", "subjective_cmos", "subjective_smos"):
        aggregate = record.aggregates.get(name)
        if aggregate and aggregate.n:
            items.append(
                f"<li><strong>{_e(name)}</strong>: mean {_fmt(aggregate.mean)} {_e(aggregate.unit)} "
                f"(n={aggregate.n}, 95% CI [{_fmt(aggregate.ci_low)}, {_fmt(aggregate.ci_high)}])</li>"
            )
    n_raters = len({s.rater_id for s in record.subjective})
    return f"""
<h2>Human Listening Test</h2>
<ul>{''.join(items)}</ul>
<p class="small muted">{n_raters} rater(s), {len(record.subjective)} ratings.</p>
"""


def _reliability_section(record: RunRecord, audio_base: str) -> str:
    failures = [u for u in record.utterances if not u.result.ok]
    flagged = [
        u for u in record.utterances
        if u.result.ok
        and (m := u.metrics.get("degeneracy_score")) is not None
        and m.status is MetricStatus.OK
        and (m.value or 0) > 0.5
    ]
    parts = ["<h2>Reliability</h2>"]

    if failures:
        rows = "".join(
            f"<tr><td class='mono small'>{_e(u.utterance_id)}</td><td>{_e(u.language)}</td>"
            f"<td class='small'>{_e((u.result.error or '')[:160])}</td></tr>"
            for u in failures[:100]
        )
        parts.append(
            f"<h3>Failed utterances ({len(failures)})</h3>"
            f"<table><thead><tr><th>ID</th><th>Lang</th><th>Error</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    else:
        parts.append('<p class="muted small">No synthesis failures.</p>')

    if flagged:
        rows = "".join(
            f"<tr><td class='mono small'>{_e(u.utterance_id)}</td><td>{_e(u.language)}</td>"
            f"<td class='num'>{_fmt(u.metrics['degeneracy_score'].value)}</td>"
            f"<td class='small'>{_e(u.metrics['degeneracy_score'].detail or '')}</td>"
            f"<td class='audio-cell'>{_audio_tag(u, audio_base)}</td></tr>"
            for u in sorted(flagged, key=lambda x: -(x.metrics['degeneracy_score'].value or 0))[:50]
        )
        parts.append(
            f"<h3>Flagged as degenerate ({len(flagged)})</h3>"
            "<p class=\"small muted\">Returned audio, but scored as a likely loop, "
            "truncation or buzz — listen before trusting.</p>"
            f"<table><thead><tr><th>ID</th><th>Lang</th><th class='num'>Score</th>"
            f"<th>Reason</th><th>Audio</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    return "\n".join(parts)


def _signoff_section(record: RunRecord) -> str:
    if not record.signoffs:
        return (
            "<h2>Review Sign-off</h2>"
            '<p class="callout small">Not yet reviewed.</p>'
        )
    rows = "".join(
        f"<li>{_badge(s.verdict, 'good' if s.verdict == 'approved' else 'bad')} "
        f"by {_e(s.reviewer)} on {_e(s.reviewed_at)}"
        + (f" — {_e(s.notes)}" if s.notes else "")
        + "</li>"
        for s in record.signoffs
    )
    return f"<h2>Review Sign-off</h2><ul>{rows}</ul>"


def _audio_tag(utterance: UtteranceRecord, audio_base: str) -> str:
    filename = f"{utterance.utterance_id}.wav"
    return f'<audio controls preload="none" src="{_e(audio_base)}{_e(filename)}"></audio>'


def _utterances_section(record: RunRecord, audio_base: str) -> str:
    successful = [u for u in record.utterances if u.result.ok]
    shown = successful[:_MAX_FULL_TABLE_ROWS]
    rows = []
    for u in shown:
        cer = u.value("cer")
        quality = u.value("audio_quality_score")
        ttfb = u.value("ttfb_ms")
        rows.append(
            f"<tr><td class='mono small'>{_e(u.utterance_id)}</td><td>{_e(u.language)}</td>"
            f"<td class='num'>{_fmt(ttfb, 1) if ttfb is not None else '—'}</td>"
            f"<td class='num'>{_fmt(cer) if cer is not None else '—'}</td>"
            f"<td class='num'>{_fmt(quality) if quality is not None else '—'}</td>"
            f"<td class='audio-cell'>{_audio_tag(u, audio_base)}</td></tr>"
        )
    truncation_note = (
        f'<p class="small muted">Showing {len(shown)} of {len(successful)} successful '
        "utterances. Full data is in the run's CSV export.</p>"
        if len(successful) > _MAX_FULL_TABLE_ROWS
        else ""
    )
    return f"""
<h2>All Utterances</h2>
{truncation_note}
<details><summary class="small muted">Expand ({len(shown)} rows)</summary>
<table>
  <thead><tr><th>ID</th><th>Lang</th><th class="num">TTFB (ms)</th>
    <th class="num">CER</th><th class="num">Audio Quality</th><th>Listen</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
</details>
"""


def _footer(record: RunRecord) -> str:
    return (
        f'<footer>tts_eval {_e(record.framework_version)}, schema v{record.schema_version} '
        f"&middot; generated from run <code>{_e(record.run_id)}</code></footer>"
    )


# ---------------------------------------------------------------------------
# comparison report
# ---------------------------------------------------------------------------
def render_comparison_html(
    comparison: Comparison, *, inline_css: bool = True, nav_html: str = ""
) -> str:
    title = f"{comparison.candidate.display_name} vs {comparison.baseline.display_name}"
    parts: list[str] = [f"""
<div class="topbar">
  <div>
    <h1>{_e(title)}</h1>
    <p class="subtitle">baseline <code>{_e(comparison.baseline.run_id)}</code>
      &rarr; candidate <code>{_e(comparison.candidate.run_id)}</code></p>
  </div>
  <nav>{nav_html}</nav>
</div>
<p>{_e(comparison.summary_line())}</p>
"""]

    if comparison.blockers:
        items = "".join(f"<li>{_e(b)}</li>" for b in comparison.blockers)
        parts.append(f'<div class="callout bad"><strong>Not comparable</strong><ul>{items}</ul></div>')
        return _page(title, "\n".join(parts), inline_css=inline_css)

    if comparison.warnings:
        items = "".join(f"<li>{_e(w)}</li>" for w in comparison.warnings)
        parts.append(f'<div class="callout"><strong>Caveats</strong><ul>{items}</ul></div>')

    regressions = comparison.regressions()
    if regressions:
        parts.append(_comparison_list("Regressions", regressions, "bad"))
    improvements = comparison.improvements()
    if improvements:
        parts.append(_comparison_list("Improvements", improvements, "good"))
    moved = comparison.moved()
    if moved:
        parts.append(
            '<h2>Changed, No Confidence Interval</h2>'
            '<p class="small muted">Run-level metrics: one value per run, so the change '
            "is exact but untested for significance.</p>"
            + _comparison_table(moved)
        )

    for criterion in criteria_order():
        rows = comparison.by_criterion().get(criterion, [])
        if rows:
            parts.append(f"<h2>{_e(criterion)}</h2>" + _comparison_table(rows))

    parts.append(
        f'<footer>Paired on {comparison.common_utterances} shared utterance ids.</footer>'
    )
    return _page(title, "\n".join(parts), inline_css=inline_css)


def _comparison_list(heading: str, items: Sequence[MetricComparison], band: str) -> str:
    rows = "".join(
        f"<li><strong>{_e(c.metric)}</strong>: {_fmt(c.baseline_mean)} &rarr; "
        f"{_fmt(c.candidate_mean)} ({_fmt(c.delta)} {_e(c.unit)}, "
        f"95% CI [{_fmt(c.ci_low)}, {_fmt(c.ci_high)}])</li>"
        for c in items
    )
    return f'<h2>{_e(heading)}</h2><ul class="{band}">{rows}</ul>'


def _comparison_table(rows: Sequence[MetricComparison]) -> str:
    body = []
    for comp in rows:
        band = verdict_band(comp.verdict)
        ci = (
            f"[{_fmt(comp.ci_low)}, {_fmt(comp.ci_high)}]"
            if comp.ci_low is not None
            else '<span class="muted">—</span>'
        )
        note = f'<div class="small muted">{_e(comp.note)}</div>' if comp.note else ""
        body.append(
            f"<tr><td>{_e(comp.metric)}{note}</td>"
            f"<td class='num'>{_fmt(comp.baseline_mean)}</td>"
            f"<td class='num'>{_fmt(comp.candidate_mean)}</td>"
            f"<td class='num'>{_fmt(comp.delta)} {_e(comp.unit)}</td>"
            f"<td class='num'>{ci}</td>"
            f"<td>{_badge(_VERDICT_LABEL[comp.verdict], band)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Metric</th><th class='num'>Baseline</th>"
        "<th class='num'>Candidate</th><th class='num'>&Delta;</th>"
        "<th class='num'>95% CI</th><th>Verdict</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


__all__ = ["render_comparison_html", "render_run_html"]
