"""Turning per-utterance values into defensible run-level summaries.

Two decisions here carry most of the weight:

*   **Percentiles are first-class.** For a conversational agent the mean TTFB is
    nearly useless; p95 is what decides whether callers hear dead air. Reporting
    only means is how a model with a heavy tail passes a benchmark and then fails
    in production.
*   **Uncertainty is reported, not implied.** Every mean carries a bootstrap 95%
    confidence interval. Without it, a 40 ms difference between two models on 69
    utterances reads as a win when it is noise, and the story's requirement to
    "compare across model versions" would produce confident nonsense.

Bootstrap rather than a t-interval because none of these distributions are
normal — latency is right-skewed with a hard floor, CER is bounded at 0 and
piles up there, MOS is bounded at both ends.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ..types import Aggregate, Direction, MetricStatus, MetricValue, UtteranceRecord
from .catalog import direction as metric_direction
from .catalog import unit as metric_unit

# Resamples for the bootstrap CI. 2000 is enough for a stable 95% interval at
# these sample sizes and stays fast enough to run inline on every aggregate.
BOOTSTRAP_RESAMPLES = 2000


def _percentile(sorted_values: np.ndarray, q: float) -> float:
    return float(np.percentile(sorted_values, q, method="linear"))


def bootstrap_mean_ci(
    values: Sequence[float], *, confidence: float = 0.95, seed: int = 12345
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI for the mean.

    Seeded so the same data yields the same interval on every machine — an
    unseeded CI would make two identical runs report different uncertainty and
    undermine the reproducibility claim.
    """
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    if arr.size < 3:
        # Below three points a bootstrap interval is theatre; say nothing instead.
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(BOOTSTRAP_RESAMPLES, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def summarise(
    name: str,
    values: Sequence[float | None],
    *,
    n_missing: int = 0,
    missing_reason: str | None = None,
    unit: str | None = None,
    direction: Direction | None = None,
) -> Aggregate:
    """Build an :class:`Aggregate` from raw per-utterance values."""
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    agg_unit = unit if unit is not None else metric_unit(name)
    agg_dir = direction if direction is not None else metric_direction(name)

    if not clean:
        return Aggregate(
            name=name,
            unit=agg_unit,
            direction=agg_dir,
            n=0,
            n_missing=n_missing,
            missing_reason=missing_reason,
        )

    arr = np.sort(np.asarray(clean, dtype=np.float64))
    ci_low, ci_high = bootstrap_mean_ci(clean)
    return Aggregate(
        name=name,
        unit=agg_unit,
        direction=agg_dir,
        n=len(clean),
        n_missing=n_missing,
        mean=float(arr.mean()),
        # ddof=1 for a sample std; with n=1 numpy would emit a warning and NaN,
        # so fall back to 0.0 which is the honest answer for a single point.
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        median=_percentile(arr, 50),
        p90=_percentile(arr, 90),
        p95=_percentile(arr, 95),
        p99=_percentile(arr, 99),
        minimum=float(arr[0]),
        maximum=float(arr[-1]),
        ci_low=ci_low,
        ci_high=ci_high,
    )


def aggregate_records(
    records: Sequence[UtteranceRecord], metric_names: Sequence[str] | None = None
) -> dict[str, Aggregate]:
    """Aggregate every per-utterance metric across a set of records.

    ``metric_names`` pins the output key set. Passing it (the runner does) means
    a metric that was expected but produced no values anywhere still appears with
    n=0 and a reason, instead of vanishing from the report.
    """
    names: list[str] = list(metric_names or [])
    if not names:
        seen: dict[str, None] = {}
        for rec in records:
            for key in rec.metrics:
                seen.setdefault(key, None)
        names = list(seen)

    out: dict[str, Aggregate] = {}
    for name in names:
        values: list[float | None] = []
        n_missing = 0
        reason: str | None = None
        for rec in records:
            mv = rec.metrics.get(name)
            if mv is None:
                n_missing += 1
                continue
            if mv.status is MetricStatus.OK and mv.value is not None:
                values.append(float(mv.value))
            else:
                n_missing += 1
                if reason is None and mv.detail:
                    reason = mv.detail
        out[name] = summarise(name, values, n_missing=n_missing, missing_reason=reason)
    return out


def aggregate_run_values(values: Mapping[str, MetricValue]) -> dict[str, Aggregate]:
    """Wrap run-scalar metrics as degenerate aggregates (n=1).

    Uniformity is the point: reports, CSV export and comparison all read
    ``run.aggregates[name]`` without branching on scope.
    """
    out: dict[str, Aggregate] = {}
    for name, mv in values.items():
        if mv.status is MetricStatus.OK and mv.value is not None:
            v = float(mv.value)
            out[name] = Aggregate(
                name=name,
                unit=mv.unit or metric_unit(name),
                direction=mv.direction,
                n=1,
                n_missing=0,
                mean=v,
                std=0.0,
                median=v,
                p90=v,
                p95=v,
                p99=v,
                minimum=v,
                maximum=v,
            )
        else:
            out[name] = Aggregate(
                name=name,
                unit=mv.unit or metric_unit(name),
                direction=mv.direction,
                n=0,
                n_missing=1,
                missing_reason=mv.detail,
            )
    return out


def aggregate_per_language(
    records: Sequence[UtteranceRecord], metric_names: Sequence[str] | None = None
) -> dict[str, dict[str, Aggregate]]:
    """Per-language breakdown.

    Required by the criterion "metrics are successfully captured for every
    supported language" — a single global mean cannot demonstrate that, and
    routinely hides one language being unusable.
    """
    groups: dict[str, list[UtteranceRecord]] = {}
    for rec in records:
        groups.setdefault(rec.language, []).append(rec)
    return {
        lang: aggregate_records(recs, metric_names) for lang, recs in sorted(groups.items())
    }


def aggregate_per_category(
    records: Sequence[UtteranceRecord],
    cases_by_id: Mapping[str, str],
    metric_names: Sequence[str] | None = None,
) -> dict[str, dict[str, Aggregate]]:
    """Per-category breakdown (``greeting``, ``numeric``, ``code_switch``, ...).

    Category is the axis that exposes text-normalisation failures: a model can be
    fine on prose and mangle every currency amount, which any language- or
    run-level mean averages away.
    """
    groups: dict[str, list[UtteranceRecord]] = {}
    for rec in records:
        cat = cases_by_id.get(rec.utterance_id, "general")
        groups.setdefault(cat, []).append(rec)
    return {cat: aggregate_records(recs, metric_names) for cat, recs in sorted(groups.items())}


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "aggregate_per_category",
    "aggregate_per_language",
    "aggregate_records",
    "aggregate_run_values",
    "bootstrap_mean_ci",
    "summarise",
]
