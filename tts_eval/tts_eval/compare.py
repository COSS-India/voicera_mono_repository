"""Cross-run comparison: is model B actually better than model A?

Three things separate this from subtracting two means, and each one exists because
its absence produces a confidently wrong answer:

1.  **Paired on utterance id.** Utterances differ enormously in difficulty — a
    four-word greeting and a 180-character long-form sentence are not
    interchangeable. Comparing run means lets a difference in *which* utterances
    succeeded masquerade as a difference in model quality. Pairing removes
    per-utterance difficulty from the comparison entirely, and it is only valid
    because both runs used the same test set (enforced via ``dataset_hash``).

2.  **Bootstrap CI on the paired difference, with a verdict.** A delta whose
    interval straddles zero is reported as ``inconclusive``, not as a win. This is
    the single most common failure in model bake-offs: 69 utterances is not enough
    to resolve a 20 ms latency difference, and saying so is more useful than a
    ranking that reverses next week.

3.  **Comparability checks before any arithmetic.** Different dataset, different
    concurrency, different ASR backend, or different metric definitions make a
    delta meaningless. Those are surfaced as blocking or warning conditions rather
    than silently absorbed — a CER comparison across two different ASR models is
    measuring the ASRs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from .metrics.aggregate import BOOTSTRAP_RESAMPLES
from .metrics.catalog import criteria_order
from .metrics.catalog import spec as metric_spec
from .types import Direction, MetricStatus, RunRecord

#: Verdict values, and why each exists:
#:
#: * ``better`` / ``worse`` — significant AND large enough to act on.
#: * ``negligible`` — statistically significant but below the metric's effect floor.
#:   With enough paired samples a bootstrap certifies a 0.001 ms chunk-gap
#:   difference as real, and it *is* real; calling it a win would make the
#:   benchmark a noise amplifier.
#: * ``inconclusive`` — the confidence interval straddles zero.
#: * ``single_observation`` — a run-level metric such as ``coverage_ratio`` has one
#:   value per run, so a drop from 1.0 to 0.77 is *exact* (three languages lost
#:   verification) but has no variance estimate. Labelling that "inconclusive"
#:   would read as "probably noise", the opposite of the truth.
#: * ``insufficient_data`` — too few paired observations, or the metric is missing.
Verdict = Literal[
    "better", "worse", "negligible", "inconclusive", "single_observation", "insufficient_data"
]

# Minimum paired observations before a verdict is offered at all. Below this the
# bootstrap is not informative and the honest answer is "not enough data".
MIN_PAIRS = 5

# Fallback effect floor, as a fraction of the baseline, for metrics whose catalogue
# entry sets no absolute ``min_effect``. 2% is small enough not to hide real
# regressions and large enough to filter measurement jitter.
MIN_RELATIVE_EFFECT = 0.02


@dataclass
class MetricComparison:
    """One metric, compared between a baseline run and a candidate run."""

    metric: str
    unit: str
    direction: Direction
    baseline_mean: float | None
    candidate_mean: float | None
    delta: float | None                 # candidate - baseline
    relative_delta: float | None        # delta / |baseline|
    n_pairs: int
    ci_low: float | None                # CI on the paired mean difference
    ci_high: float | None
    verdict: Verdict
    # Present when unpaired comparison was used as a fallback (run-level metrics
    # have exactly one value each and cannot be paired).
    paired: bool = True
    note: str | None = None

    @property
    def improved(self) -> bool:
        return self.verdict == "better"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "unit": self.unit,
            "direction": self.direction.value,
            "baseline_mean": _r(self.baseline_mean),
            "candidate_mean": _r(self.candidate_mean),
            "delta": _r(self.delta),
            "relative_delta": _r(self.relative_delta, 4),
            "n_pairs": self.n_pairs,
            "ci_low": _r(self.ci_low),
            "ci_high": _r(self.ci_high),
            "verdict": self.verdict,
            "paired": self.paired,
            "note": self.note,
        }


@dataclass
class Comparison:
    """Full comparison of two runs, plus the caveats a reader needs."""

    baseline: RunRecord
    candidate: RunRecord
    metrics: dict[str, MetricComparison] = field(default_factory=dict)
    # Conditions that make the numbers non-comparable at all.
    blockers: list[str] = field(default_factory=list)
    # Conditions that qualify interpretation but do not invalidate it.
    warnings: list[str] = field(default_factory=list)
    common_utterances: int = 0

    @property
    def comparable(self) -> bool:
        return not self.blockers

    def by_criterion(self) -> dict[str, list[MetricComparison]]:
        grouped: dict[str, list[MetricComparison]] = {}
        for comparison in self.metrics.values():
            grouped.setdefault(metric_spec(comparison.metric).criterion, []).append(comparison)
        order = criteria_order()
        return {
            criterion: sorted(grouped[criterion], key=lambda c: c.metric)
            for criterion in order + [c for c in sorted(grouped) if c not in order]
            if criterion in grouped
        }

    def regressions(self) -> list[MetricComparison]:
        return sorted(
            (c for c in self.metrics.values() if c.verdict == "worse"),
            key=lambda c: -abs(c.relative_delta or 0.0),
        )

    def improvements(self) -> list[MetricComparison]:
        return sorted(
            (c for c in self.metrics.values() if c.verdict == "better"),
            key=lambda c: -abs(c.relative_delta or 0.0),
        )

    def moved(self) -> list[MetricComparison]:
        """Run-level metrics that changed exactly, without a CI to test against.

        Surfaced separately from ``improvements``/``regressions`` so a reader sees a
        coverage or consistency shift that is real but statistically untestable.
        """
        return sorted(
            (
                c
                for c in self.metrics.values()
                if c.verdict == "single_observation" and direction_of_change(c) != "flat"
            ),
            key=lambda c: -abs(c.relative_delta or 0.0),
        )

    def negligible(self) -> list[MetricComparison]:
        """Significant but below the effect floor — real, and not worth acting on."""
        return sorted(
            (c for c in self.metrics.values() if c.verdict == "negligible"),
            key=lambda c: c.metric,
        )

    def summary_line(self) -> str:
        better, worse = len(self.improvements()), len(self.regressions())
        inconclusive = sum(1 for c in self.metrics.values() if c.verdict == "inconclusive")
        moved = len(self.moved())
        parts = [f"{better} better", f"{worse} worse", f"{inconclusive} inconclusive"]
        if self.negligible():
            parts.append(f"{len(self.negligible())} negligible")
        if moved:
            parts.append(f"{moved} changed (no CI)")
        return (
            f"{self.candidate.display_name} vs {self.baseline.display_name}: "
            f"{', '.join(parts)} ({self.common_utterances} paired utterances)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": {
                "run_id": self.baseline.run_id,
                "model": self.baseline.display_name,
                "fingerprint": self.baseline.fingerprint,
            },
            "candidate": {
                "run_id": self.candidate.run_id,
                "model": self.candidate.display_name,
                "fingerprint": self.candidate.fingerprint,
            },
            "comparable": self.comparable,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "common_utterances": self.common_utterances,
            "metrics": {k: v.to_dict() for k, v in sorted(self.metrics.items())},
            "summary": self.summary_line(),
        }


# ---------------------------------------------------------------------------
def check_comparability(baseline: RunRecord, candidate: RunRecord) -> tuple[list[str], list[str]]:
    """Return ``(blockers, warnings)`` for comparing these two runs."""
    blockers: list[str] = []
    warnings: list[str] = []

    if baseline.dataset_hash != candidate.dataset_hash:
        blockers.append(
            f"different test sets: baseline used {baseline.dataset_id}@"
            f"{baseline.dataset_version} (content {baseline.dataset_hash[:12]}), candidate used "
            f"{candidate.dataset_id}@{candidate.dataset_version} (content "
            f"{candidate.dataset_hash[:12]}). Re-run both on one dataset."
        )

    if baseline.concurrency != candidate.concurrency:
        blockers.append(
            f"different concurrency ({baseline.concurrency} vs {candidate.concurrency}): "
            "latency and throughput are load-dependent, so these numbers measure different "
            "things. Re-run at matched concurrency."
        )

    base_major = ".".join(baseline.framework_version.split(".")[:2])
    cand_major = ".".join(candidate.framework_version.split(".")[:2])
    if base_major != cand_major:
        warnings.append(
            f"runs were produced by different framework minor versions "
            f"({baseline.framework_version} vs {candidate.framework_version}); metric "
            "definitions may have changed"
        )

    base_asr = (baseline.environment.get("asr") or {})
    cand_asr = (candidate.environment.get("asr") or {})
    if base_asr.get("backend") != cand_asr.get("backend") or base_asr.get("version") != cand_asr.get("version"):
        if any(m in baseline.aggregates for m in ("cer", "wer")):
            warnings.append(
                f"different ASR backends ({base_asr.get('version') or 'none'} vs "
                f"{cand_asr.get('version') or 'none'}): CER/WER deltas partly reflect the ASR, "
                "not the TTS models. Treat intelligibility deltas as unreliable."
            )

    if baseline.model_version == candidate.model_version and baseline.model_id == candidate.model_id:
        warnings.append(
            f"both runs are {baseline.display_name}: this is a repeatability check "
            "(run-to-run variance), not a model comparison"
        )

    if baseline.environment.get("processor") != candidate.environment.get("processor"):
        warnings.append(
            "runs were measured on different hardware; latency and throughput deltas may "
            "reflect the machines rather than the models"
        )

    for record, role in ((baseline, "baseline"), (candidate, "candidate")):
        if (record.success_rate or 0.0) < 0.9:
            warnings.append(
                f"{role} run only succeeded on {(record.success_rate or 0):.0%} of utterances; "
                "its aggregates are conditioned on the subset that worked"
            )
    return blockers, warnings


def paired_bootstrap(
    diffs: Sequence[float], *, confidence: float = 0.95, seed: int = 20260811
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI on the mean paired difference."""
    arr = np.asarray([d for d in diffs if math.isfinite(d)], dtype=np.float64)
    if arr.size < MIN_PAIRS:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(BOOTSTRAP_RESAMPLES, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def effect_floor(metric: str, baseline_mean: float | None) -> float:
    """Smallest absolute change worth reporting for this metric.

    Prefers the catalogue's absolute ``min_effect``; otherwise falls back to a
    fraction of the baseline. Returns 0.0 only when neither is available, in which
    case any significant change counts.
    """
    absolute = metric_spec(metric).min_effect
    if absolute is not None:
        return abs(float(absolute))
    if baseline_mean is not None and math.isfinite(baseline_mean):
        return abs(float(baseline_mean)) * MIN_RELATIVE_EFFECT
    return 0.0


def _verdict(
    delta: float | None,
    ci_low: float | None,
    ci_high: float | None,
    direction: Direction,
    n: int,
    *,
    metric: str = "",
    baseline_mean: float | None = None,
) -> Verdict:
    """Turn a delta plus its interval into a decision.

    Two gates, both required:

    1.  The whole confidence interval must sit on one side of zero. Strict on
        purpose — reporting a coin-flip as an improvement destroys the benchmark's
        usefulness.
    2.  The change must clear the metric's effect floor. Significance without an
        effect size is how a benchmark ends up certifying microsecond differences
        as wins; those are reported as ``negligible`` instead.
    """
    if delta is None or n < MIN_PAIRS:
        return "insufficient_data"
    if direction is Direction.NEUTRAL:
        return "inconclusive"
    if ci_low is None or ci_high is None:
        return "inconclusive"
    if ci_low <= 0.0 <= ci_high:
        return "inconclusive"
    if abs(delta) < effect_floor(metric, baseline_mean):
        return "negligible"
    improved = (delta > 0) if direction is Direction.HIGHER_IS_BETTER else (delta < 0)
    return "better" if improved else "worse"


def compare_runs(baseline: RunRecord, candidate: RunRecord) -> Comparison:
    """Compare two runs metric by metric."""
    blockers, warnings = check_comparability(baseline, candidate)

    base_by_id = {u.utterance_id: u for u in baseline.utterances}
    cand_by_id = {u.utterance_id: u for u in candidate.utterances}
    common = sorted(set(base_by_id) & set(cand_by_id))

    comparison = Comparison(
        baseline=baseline,
        candidate=candidate,
        blockers=blockers,
        warnings=warnings,
        common_utterances=len(common),
    )
    if not common and not blockers:
        comparison.blockers.append("no utterance ids are shared between the two runs")

    metric_names = sorted(set(baseline.aggregates) | set(candidate.aggregates))
    for name in metric_names:
        spec = metric_spec(name)
        base_agg = baseline.aggregates.get(name)
        cand_agg = candidate.aggregates.get(name)

        if spec.scope == "run" or not common:
            comparison.metrics[name] = _unpaired(name, base_agg, cand_agg)
            continue

        pairs: list[tuple[float, float]] = []
        for uid in common:
            b = base_by_id[uid].metrics.get(name)
            c = cand_by_id[uid].metrics.get(name)
            if (
                b is not None
                and c is not None
                and b.status is MetricStatus.OK
                and c.status is MetricStatus.OK
                and b.value is not None
                and c.value is not None
            ):
                pairs.append((float(b.value), float(c.value)))

        if len(pairs) < MIN_PAIRS:
            fallback = _unpaired(name, base_agg, cand_agg)
            fallback.note = (
                f"only {len(pairs)} utterance(s) have this metric in both runs; compared on "
                "run means instead of paired differences" + (f" — {fallback.note}" if fallback.note else "")
            )
            fallback.paired = False
            comparison.metrics[name] = fallback
            continue

        base_values = np.asarray([p[0] for p in pairs], dtype=np.float64)
        cand_values = np.asarray([p[1] for p in pairs], dtype=np.float64)
        diffs = cand_values - base_values
        delta = float(diffs.mean())
        base_mean = float(base_values.mean())
        ci_low, ci_high = paired_bootstrap(diffs.tolist())

        verdict = _verdict(
            delta,
            ci_low,
            ci_high,
            spec.direction,
            len(pairs),
            metric=name,
            baseline_mean=base_mean,
        )
        comparison.metrics[name] = MetricComparison(
            metric=name,
            unit=spec.unit,
            direction=spec.direction,
            baseline_mean=base_mean,
            candidate_mean=float(cand_values.mean()),
            delta=delta,
            relative_delta=(delta / abs(base_mean)) if abs(base_mean) > 1e-12 else None,
            n_pairs=len(pairs),
            ci_low=ci_low,
            ci_high=ci_high,
            verdict=verdict,
            note=(
                f"change is statistically real but below the {effect_floor(name, base_mean):g} "
                f"{spec.unit or 'unit'} effect floor for this metric"
                if verdict == "negligible"
                else None
            ),
        )
    return comparison


def _unpaired(name: str, base_agg: Any, cand_agg: Any) -> MetricComparison:
    """Compare on means only — for run-scoped metrics, which have one value each."""
    spec = metric_spec(name)
    base_mean = getattr(base_agg, "mean", None) if base_agg else None
    cand_mean = getattr(cand_agg, "mean", None) if cand_agg else None

    note: str | None = None
    if base_mean is None or cand_mean is None:
        missing = []
        if base_mean is None:
            missing.append(f"baseline ({getattr(base_agg, 'missing_reason', None) or 'absent'})")
        if cand_mean is None:
            missing.append(f"candidate ({getattr(cand_agg, 'missing_reason', None) or 'absent'})")
        return MetricComparison(
            metric=name,
            unit=spec.unit,
            direction=spec.direction,
            baseline_mean=base_mean,
            candidate_mean=cand_mean,
            delta=None,
            relative_delta=None,
            n_pairs=0,
            ci_low=None,
            ci_high=None,
            verdict="insufficient_data",
            paired=False,
            note="not computed in " + " and ".join(missing),
        )

    delta = cand_mean - base_mean
    verdict: Verdict = "inconclusive"
    if spec.scope == "run":
        # Exactly one observation per run, so the delta is exact but has no
        # uncertainty attached. Flagged as its own verdict so a report can show the
        # direction (and colour it) without implying statistical significance.
        note = (
            "run-level metric: one value per run, so the change is exact but has no "
            "confidence interval"
        )
        verdict = "single_observation"
    elif abs(delta) < 1e-12:
        note = "no change"
    return MetricComparison(
        metric=name,
        unit=spec.unit,
        direction=spec.direction,
        baseline_mean=base_mean,
        candidate_mean=cand_mean,
        delta=delta,
        relative_delta=(delta / abs(base_mean)) if abs(base_mean) > 1e-12 else None,
        n_pairs=0,
        ci_low=None,
        ci_high=None,
        verdict=verdict,
        paired=False,
        note=note,
    )


def direction_of_change(comparison: MetricComparison) -> Literal["good", "bad", "flat"]:
    """Which way a delta points, independent of whether it is significant.

    Used by reports to colour single-observation and inconclusive rows: the reader
    still needs to see that coverage fell, even where no CI can be offered.
    """
    if comparison.delta is None or abs(comparison.delta) < 1e-12:
        return "flat"
    if comparison.direction is Direction.NEUTRAL:
        return "flat"
    improved = (
        comparison.delta > 0
        if comparison.direction is Direction.HIGHER_IS_BETTER
        else comparison.delta < 0
    )
    return "good" if improved else "bad"


def compare_many(baseline: RunRecord, candidates: Sequence[RunRecord]) -> list[Comparison]:
    return [compare_runs(baseline, candidate) for candidate in candidates]


def repeatability(runs: Sequence[RunRecord]) -> dict[str, Any]:
    """Run-to-run variance for repeats of the SAME fingerprint.

    This is the noise floor of the whole benchmark. Without it there is no way to
    know whether a 5% inter-model difference is meaningful, and it is the direct
    evidence for "multiple evaluation runs can be performed using the same test
    dataset" producing consistent results.
    """
    if len(runs) < 2:
        return {"error": "need at least two runs to measure repeatability"}

    fingerprints = {r.fingerprint for r in runs}
    out: dict[str, Any] = {
        "n_runs": len(runs),
        "run_ids": [r.run_id for r in runs],
        "fingerprints": sorted(fingerprints),
        "same_fingerprint": len(fingerprints) == 1,
        "metrics": {},
    }
    if len(fingerprints) > 1:
        out["warning"] = (
            "these runs do not share a fingerprint, so the spread below mixes genuine "
            "run-to-run noise with configuration differences"
        )

    names = sorted({name for r in runs for name in r.aggregates})
    for name in names:
        means = [
            r.aggregates[name].mean
            for r in runs
            if name in r.aggregates and r.aggregates[name].mean is not None
        ]
        if len(means) < 2:
            continue
        arr = np.asarray(means, dtype=np.float64)
        mean_of_means = float(arr.mean())
        out["metrics"][name] = {
            "n": int(arr.size),
            "mean": round(mean_of_means, 4),
            "std": round(float(arr.std(ddof=1)), 4),
            # CV is the headline: it says "repeat runs of this metric agree to
            # within X%", which is exactly the threshold a reader needs before
            # believing any inter-model delta.
            "cv": (
                round(float(arr.std(ddof=1) / abs(mean_of_means)), 4)
                if abs(mean_of_means) > 1e-12
                else None
            ),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "spread_pct": (
                round(float((arr.max() - arr.min()) / abs(mean_of_means)) * 100.0, 2)
                if abs(mean_of_means) > 1e-12
                else None
            ),
        }
    return out


def _r(v: float | None, ndigits: int = 3) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), ndigits)


__all__ = [
    "MIN_PAIRS",
    "Comparison",
    "MetricComparison",
    "Verdict",
    "MIN_RELATIVE_EFFECT",
    "check_comparability",
    "compare_many",
    "compare_runs",
    "direction_of_change",
    "effect_floor",
    "paired_bootstrap",
    "repeatability",
]
