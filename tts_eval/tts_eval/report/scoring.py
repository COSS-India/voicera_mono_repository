"""Shared good/warn/bad classification for aggregate values.

One function, used by Markdown, HTML and the UI, so a metric never gets three
different colour rules in three renderers. Classification is a straight-line
interpolation between the catalogue's ``good``/``bad`` reference points — it is a
**reading aid**, not a pass/fail gate. The actual pass/fail logic for language
coverage and comparison verdicts lives in ``coverage.py`` / ``compare.py``; this
module only decides how a number is painted.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..metrics.catalog import MetricSpec
from ..metrics.catalog import spec as metric_spec
from ..types import Aggregate, Direction

Band = str  # "good" | "warn" | "bad" | "neutral"


@dataclass(frozen=True)
class Scored:
    band: Band
    # 0..1, 1 = best. None when the metric has no good/bad reference points, so a
    # renderer can fall back to plain text instead of drawing a meaningless bar.
    fraction: float | None


def score_value(name: str, value: float | None) -> Scored:
    """Classify one value against its metric's good/bad reference points."""
    if value is None:
        return Scored("neutral", None)
    spec: MetricSpec = metric_spec(name)
    if spec.good is None or spec.bad is None or spec.direction is Direction.NEUTRAL:
        return Scored("neutral", None)

    good, bad = float(spec.good), float(spec.bad)
    if good == bad:
        return Scored("neutral", None)
    fraction = (value - bad) / (good - bad)
    fraction = max(0.0, min(1.0, fraction))

    if fraction >= 0.7:
        band = "good"
    elif fraction >= 0.35:
        band = "warn"
    else:
        band = "bad"
    return Scored(band, fraction)


def score_aggregate(name: str, aggregate: Aggregate | None) -> Scored:
    if aggregate is None or aggregate.n == 0:
        return Scored("neutral", None)
    return score_value(name, aggregate.mean)


def verdict_band(verdict: str) -> Band:
    """Map a comparison verdict onto the same three-colour vocabulary."""
    return {
        "better": "good",
        "worse": "bad",
        "negligible": "neutral",
        "inconclusive": "neutral",
        "single_observation": "warn",
        "insufficient_data": "neutral",
    }.get(verdict, "neutral")


__all__ = ["Band", "Scored", "score_aggregate", "score_value", "verdict_band"]
