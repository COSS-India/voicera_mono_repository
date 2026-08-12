"""Benchmark reports: Markdown, standalone HTML, and CSV — one data path, three
outputs so each can go where the others cannot (a PR description, a shared link,
a spreadsheet).
"""
from __future__ import annotations

from pathlib import Path

from ..compare import Comparison
from ..types import RunRecord
from .csv_export import aggregates_csv, coverage_csv, utterances_csv, write_csv_bundle
from .html import render_comparison_html, render_run_html
from .markdown import render_comparison_markdown, render_run_markdown
from .scoring import Scored, score_aggregate, score_value, verdict_band

# Report artefacts are written INTO the run directory, alongside its audio and
# run.json, rather than to some separate reports/ tree: a run and its report are
# one archival unit, and copying the run directory should be enough to bring the
# report with it. Audio links in report.html are therefore relative ("audio/..").
REPORT_HTML_NAME = "report.html"
REPORT_MARKDOWN_NAME = "report.md"


def write_run_report(record: RunRecord, run_dir: str | Path) -> dict[str, Path]:
    """Write report.html, report.md and the CSV bundle into ``run_dir``.

    ``run_dir`` is expected to be the directory a :class:`~tts_eval.store.RunStore`
    already created for this run (it contains ``run.json`` and ``audio/``), so the
    HTML report's relative ``audio/`` links resolve without configuration.
    """
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    html_path = directory / REPORT_HTML_NAME
    html_path.write_text(render_run_html(record), encoding="utf-8")
    written[REPORT_HTML_NAME] = html_path

    md_path = directory / REPORT_MARKDOWN_NAME
    md_path.write_text(render_run_markdown(record), encoding="utf-8")
    written[REPORT_MARKDOWN_NAME] = md_path

    written.update(write_csv_bundle(record, directory))
    return written


def write_comparison_report(comparison: Comparison, out_dir: str | Path) -> dict[str, Path]:
    """Write comparison.html and comparison.md into ``out_dir``.

    Unlike a run report, a comparison has no natural "home" directory (it spans
    two runs), so the caller picks ``out_dir`` explicitly.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    html_path = directory / "comparison.html"
    html_path.write_text(render_comparison_html(comparison), encoding="utf-8")
    md_path = directory / "comparison.md"
    md_path.write_text(render_comparison_markdown(comparison), encoding="utf-8")
    return {"comparison.html": html_path, "comparison.md": md_path}


__all__ = [
    "REPORT_HTML_NAME",
    "REPORT_MARKDOWN_NAME",
    "Scored",
    "aggregates_csv",
    "coverage_csv",
    "render_comparison_html",
    "render_comparison_markdown",
    "render_run_html",
    "render_run_markdown",
    "score_aggregate",
    "score_value",
    "utterances_csv",
    "verdict_band",
    "write_comparison_report",
    "write_csv_bundle",
    "write_run_report",
]
