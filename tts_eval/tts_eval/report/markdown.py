"""Markdown report rendering.

Markdown is the format for the artefact that gets pasted into a PR description, a
ticket, or a Slack message — places an HTML file cannot go. It carries the same
numbers as the HTML report; only the presentation differs, so nobody has to read
two renderers to know a metric is covered.
"""
from __future__ import annotations

from typing import Any

from ..compare import Comparison, MetricComparison
from ..metrics.catalog import ac_matrix, criteria_order
from ..types import Aggregate, RunRecord
from .scoring import score_aggregate

_VERDICT_WORD = {
    "better": "**better**",
    "worse": "**worse**",
    "negligible": "negligible",
    "inconclusive": "inconclusive",
    "single_observation": "changed (no CI)",
    "insufficient_data": "insufficient data",
}


def _fmt(value: float | None, ndigits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:,.{ndigits}f}" if abs(value) >= 1 else f"{value:.{ndigits}f}"


def _band_word(band: str) -> str:
    return {"good": "🟢", "warn": "🟡", "bad": "🔴", "neutral": ""}[band]


def _aggregate_row(name: str, aggregate: Aggregate | None) -> str:
    if aggregate is None or aggregate.n == 0:
        reason = (aggregate.missing_reason if aggregate else None) or "not computed"
        return f"| {name} | — | — | — | — | *{reason}* |"
    scored = score_aggregate(name, aggregate)
    ci = (
        f"[{_fmt(aggregate.ci_low)}, {_fmt(aggregate.ci_high)}]"
        if aggregate.ci_low is not None
        else "—"
    )
    return (
        f"| {_band_word(scored.band)} {name} | {aggregate.n} | {_fmt(aggregate.mean)} "
        f"{aggregate.unit} | {_fmt(aggregate.p95)} {aggregate.unit} | {ci} |"
    )


def render_run_markdown(record: RunRecord, *, title: str | None = None) -> str:
    """One run, one report. Self-contained: no external files referenced."""
    lines: list[str] = []
    add = lines.append

    add(f"# {title or record.label or record.display_name}")
    add("")
    add(f"Model **{record.display_name}** ({record.provider}) — run `{record.run_id}`")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Dataset | {record.dataset_id}@{record.dataset_version} ({record.dataset_size} utterances) |")
    add(f"| Fingerprint | `{record.fingerprint}` |")
    add(f"| Concurrency | {record.concurrency} |")
    add(f"| Determinism | {record.determinism.value} |")
    add(f"| Started | {record.created_at} |")
    add(f"| Metric backends | {', '.join(sorted(record.metric_backends)) or 'none'} |")
    add(f"| Success rate | {record.success_rate:.1%} ({record.n_ok}/{len(record.utterances)}) |"
        if record.success_rate is not None else "| Success rate | — |")
    add("")

    if record.warnings:
        add("> **Warnings**")
        for warning in record.warnings:
            add(f"> - {warning}")
        add("")

    absent = [n for n, s in record.metric_backends.items() if s.startswith("absent")]
    if absent:
        add("**Not computed in this run:**")
        for name in sorted(absent):
            add(f"- `{name}`: {record.metric_backends[name][len('absent: '):]}")
        add("")

    for criterion in criteria_order():
        rows = [
            r for r in ac_matrix() if r["criterion"] == criterion and r["metric"] in record.aggregates
        ]
        if not rows:
            continue
        add(f"## {criterion}")
        add("")
        add("| Metric | n | Mean | p95 | 95% CI |")
        add("|---|---:|---:|---:|---:|")
        for row in rows:
            add(_aggregate_row(row["metric"], record.aggregates.get(row["metric"])))
        add("")

    if record.coverage:
        add("## Language Coverage")
        add("")
        add("| Language | Claimed | Attempted | Succeeded | Verified | Notes |")
        add("|---|:---:|---:|---:|:---:|---|")
        for cov in sorted(record.coverage, key=lambda c: c.language):
            add(
                f"| {cov.language} | {'✓' if cov.claimed else ''} | {cov.attempted} | "
                f"{cov.succeeded} | {'✓' if cov.verified else '✗'} | {cov.notes or ''} |"
            )
        add("")

    per_category = (record.environment or {}).get("per_category") or {}
    if per_category:
        add("## By Category")
        add("")
        # Each mean carries its own n: a single max-n column overstated the
        # sample count behind the lower-n metrics.
        add("| Category | CER | Audio quality | TTFB (ms) |")
        add("|---|---:|---:|---:|")

        def _cell(agg: dict, digits: int = 3) -> str:
            return f"{_fmt(agg.get('mean'), digits)} (n={agg.get('n') or 0})"

        for category, aggs in sorted(per_category.items()):
            cer = aggs.get("cer", {})
            quality = aggs.get("audio_quality_score", {})
            ttfb = aggs.get("ttfb_ms", {})
            add(
                f"| {category} | {_cell(cer)} | {_cell(quality)} | {_cell(ttfb, 1)} |"
            )
        add("")

    if record.subjective:
        add("## Human Listening Test")
        add("")
        for name in ("subjective_mos", "subjective_mushra", "subjective_cmos", "subjective_smos"):
            aggregate = record.aggregates.get(name)
            if aggregate and aggregate.n:
                add(f"- **{name}**: mean {_fmt(aggregate.mean)} {aggregate.unit} "
                    f"(n={aggregate.n}, 95% CI [{_fmt(aggregate.ci_low)}, {_fmt(aggregate.ci_high)}])")
        add(f"- Raters: {len({s.rater_id for s in record.subjective})}, "
            f"ratings: {len(record.subjective)}")
        add("")

    if record.signoffs:
        add("## Review Sign-off")
        add("")
        for signoff in record.signoffs:
            add(f"- **{signoff.verdict}** by {signoff.reviewer} on {signoff.reviewed_at}"
                + (f" — {signoff.notes}" if signoff.notes else ""))
        add("")
    else:
        add("## Review Sign-off")
        add("")
        add("*Not yet reviewed. See `tts-eval review` / `store.add_signoff(...)`.*")
        add("")

    add("---")
    add(f"*tts_eval {record.framework_version}, schema v{record.schema_version}, "
        f"generated from run `{record.run_id}`.*")
    return "\n".join(lines) + "\n"


def _comparison_metric_row(comparison: MetricComparison) -> str:
    ci = (
        f"[{_fmt(comparison.ci_low)}, {_fmt(comparison.ci_high)}]"
        if comparison.ci_low is not None
        else ("*no CI*" if comparison.verdict == "single_observation" else "—")
    )
    delta = _fmt(comparison.delta) if comparison.delta is not None else "—"
    note = f" *{comparison.note}*" if comparison.note and comparison.verdict != "negligible" else ""
    return (
        f"| {comparison.metric} | {_fmt(comparison.baseline_mean)} | {_fmt(comparison.candidate_mean)} "
        f"| {delta} {comparison.unit} | {ci} | {_VERDICT_WORD[comparison.verdict]}{note} |"
    )


def render_comparison_markdown(comparison: Comparison) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"# {comparison.candidate.display_name} vs {comparison.baseline.display_name}")
    add("")
    add(f"Baseline run `{comparison.baseline.run_id}` -> candidate run `{comparison.candidate.run_id}`")
    add("")
    add(f"**{comparison.summary_line()}**")
    add("")

    if comparison.blockers:
        add("> **Not comparable**")
        for blocker in comparison.blockers:
            add(f"> - {blocker}")
        add("")
        return "\n".join(lines) + "\n"

    if comparison.warnings:
        add("> **Caveats**")
        for warning in comparison.warnings:
            add(f"> - {warning}")
        add("")

    regressions = comparison.regressions()
    if regressions:
        add("## Regressions")
        add("")
        for comp in regressions:
            add(f"- **{comp.metric}**: {_fmt(comp.baseline_mean)} → {_fmt(comp.candidate_mean)} "
                f"({_fmt(comp.delta)} {comp.unit}, 95% CI [{_fmt(comp.ci_low)}, {_fmt(comp.ci_high)}])")
        add("")

    improvements = comparison.improvements()
    if improvements:
        add("## Improvements")
        add("")
        for comp in improvements:
            add(f"- **{comp.metric}**: {_fmt(comp.baseline_mean)} → {_fmt(comp.candidate_mean)} "
                f"({_fmt(comp.delta)} {comp.unit}, 95% CI [{_fmt(comp.ci_low)}, {_fmt(comp.ci_high)}])")
        add("")

    moved = comparison.moved()
    if moved:
        add("## Changed, No Confidence Interval")
        add("")
        add("*Run-level metrics: one value per run. The change is exact, but has no variance estimate.*")
        add("")
        for comp in moved:
            add(f"- **{comp.metric}**: {_fmt(comp.baseline_mean)} → {_fmt(comp.candidate_mean)}")
        add("")

    for criterion in criteria_order():
        rows = [c for c in comparison.by_criterion().get(criterion, [])]
        if not rows:
            continue
        add(f"## {criterion}")
        add("")
        add("| Metric | Baseline | Candidate | Δ | 95% CI | Verdict |")
        add("|---|---:|---:|---:|---:|---|")
        for comp in rows:
            add(_comparison_metric_row(comp))
        add("")

    add("---")
    add(f"*Paired on {comparison.common_utterances} shared utterance ids.*")
    return "\n".join(lines) + "\n"


__all__ = ["render_comparison_markdown", "render_run_markdown"]
