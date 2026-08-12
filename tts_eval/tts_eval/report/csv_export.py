"""CSV export: the format for a spreadsheet, a pandas notebook, or a data team
that does not want to parse `run.json`.

Two files rather than one, because they are different shapes of data and forcing
them into one table means either repeating run-level numbers on every utterance
row or losing them entirely:

* ``utterances.csv``  — one row per utterance, one column per metric. The unit of
  analysis for anything a data scientist wants to slice by language or category.
* ``aggregates.csv``  — one row per metric, with n/mean/percentiles/CI. The unit
  of analysis for "what does this run's headline number look like".

Column set is a union across all utterances/metrics actually present in the run,
not the full catalogue — an ``all``-tier run and a ``core``-tier run produce
different column sets, which is correct: a column of universal ``not_computed``
strings would be pure noise.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from ..types import RunRecord

# Fixed leading columns, before the metric columns. Kept in one place so
# utterances.csv and any future consumer agree on identity-column order.
_IDENTITY_COLUMNS = ("utterance_id", "language", "voice", "text", "ok", "error", "audio_path")


def utterances_csv(record: RunRecord) -> str:
    metric_names: list[str] = []
    seen: set[str] = set()
    for utterance in record.utterances:
        for name in utterance.metrics:
            if name not in seen:
                seen.add(name)
                metric_names.append(name)
    metric_names.sort()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(list(_IDENTITY_COLUMNS) + metric_names)

    for utterance in record.utterances:
        request = utterance.result.request
        row = [
            utterance.utterance_id,
            utterance.language,
            request.voice or "",
            request.text,
            "1" if utterance.result.ok else "0",
            utterance.result.error or "",
            utterance.result.audio_path or "",
        ]
        for name in metric_names:
            metric = utterance.metrics.get(name)
            row.append("" if metric is None or metric.value is None else metric.value)
        writer.writerow(row)
    return buffer.getvalue()


_AGGREGATE_COLUMNS = (
    "metric", "unit", "direction", "n", "n_missing", "mean", "std", "median",
    "p90", "p95", "p99", "min", "max", "ci_low", "ci_high", "missing_reason",
)


def aggregates_csv(record: RunRecord) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_AGGREGATE_COLUMNS)
    for name, aggregate in sorted(record.aggregates.items()):
        writer.writerow(
            [
                name, aggregate.unit, aggregate.direction.value, aggregate.n, aggregate.n_missing,
                aggregate.mean, aggregate.std, aggregate.median, aggregate.p90, aggregate.p95,
                aggregate.p99, aggregate.minimum, aggregate.maximum, aggregate.ci_low,
                aggregate.ci_high, aggregate.missing_reason or "",
            ]
        )
    return buffer.getvalue()


def coverage_csv(record: RunRecord) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["language", "claimed", "attempted", "succeeded", "intelligible", "verified", "notes"])
    for coverage in sorted(record.coverage, key=lambda c: c.language):
        writer.writerow(
            [
                coverage.language, int(coverage.claimed), coverage.attempted, coverage.succeeded,
                coverage.intelligible if coverage.intelligible is not None else "",
                int(coverage.verified), coverage.notes or "",
            ]
        )
    return buffer.getvalue()


def write_csv_bundle(record: RunRecord, out_dir: str | Path) -> dict[str, Path]:
    """Write all three CSVs into ``out_dir``. Returns name -> path."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "utterances.csv": utterances_csv(record),
        "aggregates.csv": aggregates_csv(record),
        "coverage.csv": coverage_csv(record),
    }
    written: dict[str, Path] = {}
    for filename, content in files.items():
        path = directory / filename
        path.write_text(content, encoding="utf-8", newline="")
        written[filename] = path
    return written


__all__ = ["aggregates_csv", "coverage_csv", "utterances_csv", "write_csv_bundle"]
