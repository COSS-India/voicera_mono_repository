"""Ingest rater scores, de-blind them, screen raters, merge into run records.

Ingestion is where a listening test either becomes evidence or becomes noise. Four
things happen here, each because skipping it produces a wrong number:

1.  **De-blinding via the answer key.** Sheets carry opaque tokens; without the key
    the scores cannot be attributed, and with the wrong key they would be attributed
    to the wrong system — worse than having no data.

2.  **Rater screening.** A rater who scored the low-pass anchor as highly as the
    systems under test was not listening, and their rows are dropped with the reason
    recorded. Panels reliably contain one or two such raters, and unscreened they
    compress the differences being measured toward zero.

3.  **MUSHRA per-trial normalisation.** Raters use wildly different portions of the
    0-100 range, so raw cross-rater means measure rater temperament as much as
    system quality. Normalised-within-rater scores are reported alongside raw.

4.  **Inter-rater agreement.** A mean without an agreement figure cannot be
    interpreted: 3.8 from a panel that agrees is a result, 3.8 from a panel that
    does not is an average of disagreement.

Merged scores land on the run record as ``subjective_mos`` / ``subjective_mushra``
per-utterance metrics plus re-derived aggregates, so they appear in the normal
Naturalness section next to the predicted MOS they exist to check.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..errors import ConfigError
from ..metrics.aggregate import aggregate_per_language, aggregate_records, summarise
from ..metrics.base import make_value
from ..types import RunRecord, SubjectiveScore
from .listening_test import SCALES

# A rater is excluded when their mean anchor score is within this fraction of the
# scale of their mean system score. The anchor is a 3.5 kHz low-pass version and is
# audibly worse; rating it comparably means the rater was not attending.
_ANCHOR_TOLERANCE = 0.15

# Metric name written onto utterance records, per scale.
_SCALE_METRIC = {
    "mos": "subjective_mos",
    "mushra": "subjective_mushra",
    "cmos": "subjective_cmos",
    "smos": "subjective_smos",
}


@dataclass
class RaterReport:
    rater_id: str
    n_scores: int
    mean_score: float | None
    anchor_mean: float | None
    system_mean: float | None
    excluded: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rater_id": self.rater_id,
            "n_scores": self.n_scores,
            "mean_score": _r(self.mean_score),
            "anchor_mean": _r(self.anchor_mean),
            "system_mean": _r(self.system_mean),
            "excluded": self.excluded,
            "reason": self.reason,
        }


@dataclass
class IngestReport:
    """Everything a reviewer needs to trust (or reject) the panel's numbers."""

    scale: str
    n_rows_read: int
    n_rows_used: int
    raters: list[RaterReport] = field(default_factory=list)
    # system label -> stats
    per_system: dict[str, dict[str, Any]] = field(default_factory=dict)
    agreement: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # run_id -> scores attributed to it
    scores_by_run: dict[str, list[SubjectiveScore]] = field(default_factory=dict)

    @property
    def excluded_raters(self) -> list[str]:
        return [r.rater_id for r in self.raters if r.excluded]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "n_rows_read": self.n_rows_read,
            "n_rows_used": self.n_rows_used,
            "raters": [r.to_dict() for r in self.raters],
            "excluded_raters": self.excluded_raters,
            "per_system": self.per_system,
            "agreement": self.agreement,
            "warnings": list(self.warnings),
            "n_scores_by_run": {k: len(v) for k, v in self.scores_by_run.items()},
        }


# ---------------------------------------------------------------------------
def load_answer_key(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"answer key not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"{p}: invalid JSON: {e}") from e
    if "trials" not in data:
        raise ConfigError(f"{p}: not an answer key (no `trials` field)")
    return data


def _read_sheets(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise ConfigError(f"score sheet not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            missing = {"trial_id", "system_token", "score"} - set(reader.fieldnames or ())
            if missing:
                raise ConfigError(
                    f"{path}: sheet is missing column(s) {', '.join(sorted(missing))}; "
                    f"found {', '.join(reader.fieldnames or [])}"
                )
            for row in reader:
                rows.append({k: (v or "").strip() for k, v in row.items() if k})
    return rows


def ingest_sheets(
    sheet_paths: Sequence[str | Path],
    answer_key_path: str | Path,
    *,
    anchor_tolerance: float = _ANCHOR_TOLERANCE,
) -> IngestReport:
    """Parse sheets, de-blind, screen raters, and group scores by run."""
    key = load_answer_key(answer_key_path)
    trials: Mapping[str, Any] = key["trials"]
    run_names: Mapping[str, str] = key.get("runs") or {}
    # Reverse the display-name -> run_id mapping so a de-blinded label can be
    # attributed back to the run whose record must carry the score.
    label_to_run = {f"{name} ({rid})": rid for rid, name in run_names.items()}

    rows = _read_sheets([Path(p) for p in sheet_paths])
    scale = str(key.get("scale") or "mushra")
    if scale not in SCALES:
        scale = "mushra"
    lo, hi = SCALES[scale]["min"], SCALES[scale]["max"]
    span = float(hi - lo) or 1.0

    report = IngestReport(scale=scale, n_rows_read=len(rows), n_rows_used=0)

    # (rater, trial, token) -> (score, system_label, utterance_id, comment)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("score", "")
        if raw == "":
            continue  # unscored row: a rater skipping a trial is normal
        try:
            score = float(raw)
        except ValueError:
            report.warnings.append(
                f"non-numeric score {raw!r} for trial {row.get('trial_id')} "
                f"token {row.get('system_token')} — row skipped"
            )
            continue
        if not (lo <= score <= hi):
            report.warnings.append(
                f"score {score} for trial {row.get('trial_id')} is outside the {scale} "
                f"range [{lo}, {hi}] — row skipped"
            )
            continue

        trial_id = row.get("trial_id", "")
        token = row.get("system_token", "")
        trial = trials.get(trial_id)
        if trial is None:
            report.warnings.append(f"unknown trial_id {trial_id!r} — not in the answer key")
            continue
        system_label = (trial.get("systems") or {}).get(token)
        if system_label is None:
            report.warnings.append(
                f"token {token!r} in trial {trial_id} is not in the answer key — wrong key file?"
            )
            continue

        parsed.append(
            {
                "rater_id": row.get("rater_id") or "rater_unknown",
                "trial_id": trial_id,
                "token": token,
                "score": score,
                "system": system_label,
                "utterance_id": trial.get("utterance_id") or trial_id,
                "comment": row.get("comment") or None,
                "is_anchor": "ANCHOR" in system_label,
            }
        )

    if not parsed:
        report.warnings.append("no usable scores found in the supplied sheets")
        return report

    # --- rater screening ---------------------------------------------------
    by_rater: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in parsed:
        by_rater[item["rater_id"]].append(item)

    excluded: set[str] = set()
    for rater_id, items in sorted(by_rater.items()):
        anchors = [i["score"] for i in items if i["is_anchor"]]
        systems = [i["score"] for i in items if not i["is_anchor"]]
        anchor_mean = statistics.fmean(anchors) if anchors else None
        system_mean = statistics.fmean(systems) if systems else None

        drop, reason = False, None
        if anchor_mean is not None and system_mean is not None:
            # Anchor must sit clearly below the systems. Compared as a fraction of
            # the scale so the same rule works for 1-5 MOS and 0-100 MUSHRA.
            if (system_mean - anchor_mean) / span < anchor_tolerance:
                drop = True
                reason = (
                    f"failed anchor check: rated the low-pass anchor {anchor_mean:.1f} vs "
                    f"{system_mean:.1f} for real systems (gap {(system_mean - anchor_mean) / span:.0%} "
                    f"of scale, needs >= {anchor_tolerance:.0%})"
                )
        if not drop and len(systems) >= 4 and len(set(systems)) == 1:
            drop = True
            reason = f"gave the identical score ({systems[0]:g}) to every clip"

        if drop:
            excluded.add(rater_id)
        report.raters.append(
            RaterReport(
                rater_id=rater_id,
                n_scores=len(items),
                mean_score=statistics.fmean([i["score"] for i in items]),
                anchor_mean=anchor_mean,
                system_mean=system_mean,
                excluded=drop,
                reason=reason,
            )
        )

    if excluded:
        report.warnings.append(
            f"excluded {len(excluded)} of {len(by_rater)} rater(s) by screening: "
            f"{', '.join(sorted(excluded))}"
        )
    surviving = [r for r in report.raters if not r.excluded]
    if len(surviving) < 3:
        report.warnings.append(
            f"only {len(surviving)} rater(s) survived screening; subjective means from fewer "
            "than 3 raters are weak evidence and should not decide a model choice"
        )

    # --- normalisation and grouping ---------------------------------------
    used = [i for i in parsed if i["rater_id"] not in excluded and not i["is_anchor"]]
    report.n_rows_used = len(used)

    # Per-rater z-ish normalisation onto the scale, removing each rater's own
    # offset and spread. Reported alongside raw, never instead of it.
    per_rater_stats: dict[str, tuple[float, float]] = {}
    for rater_id in {i["rater_id"] for i in used}:
        vals = [i["score"] for i in used if i["rater_id"] == rater_id]
        mean = statistics.fmean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        per_rater_stats[rater_id] = (mean, sd)

    for item in used:
        mean, sd = per_rater_stats[item["rater_id"]]
        item["normalised"] = ((item["score"] - mean) / sd) if sd > 1e-9 else 0.0

    for item in used:
        run_id = label_to_run.get(item["system"])
        if run_id is None:
            report.warnings.append(
                f"de-blinded system {item['system']!r} does not map to a run id — score dropped"
            )
            continue
        report.scores_by_run.setdefault(run_id, []).append(
            SubjectiveScore(
                utterance_id=item["utterance_id"],
                rater_id=item["rater_id"],
                scale=scale,
                score=item["score"],
                system=item["system"],
                comment=item["comment"],
            )
        )

    # --- per-system summary and agreement ---------------------------------
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in used:
        by_system[item["system"]].append(item)
    for system, items in sorted(by_system.items()):
        raw = [i["score"] for i in items]
        norm = [i["normalised"] for i in items]
        report.per_system[system] = {
            "n_scores": len(raw),
            "n_raters": len({i["rater_id"] for i in items}),
            "n_utterances": len({i["utterance_id"] for i in items}),
            "mean": _r(statistics.fmean(raw)),
            "median": _r(statistics.median(raw)),
            "std": _r(statistics.pstdev(raw) if len(raw) > 1 else 0.0),
            "ci95": [_r(v) for v in _bootstrap_ci(raw)],
            "mean_rater_normalised": _r(statistics.fmean(norm), 4),
        }

    report.agreement = _agreement(used, span)
    return report


def _bootstrap_ci(values: Sequence[float], *, seed: int = 991) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 3:
        return None, None
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(2000, arr.size))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _agreement(items: Sequence[Mapping[str, Any]], span: float) -> dict[str, Any]:
    """Inter-rater agreement on the items that more than one rater scored.

    Computed as ``1 - (mean within-item variance / total variance)`` — an
    intraclass-correlation-style ratio. 1.0 means raters agree perfectly on the
    ranking; 0.0 means their disagreement is as large as the spread between clips,
    i.e. the panel carries no signal and a bigger panel is needed before the means
    mean anything.
    """
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item in items:
        groups[(item["utterance_id"], item["system"])].append(float(item["score"]))

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if len(multi) < 2:
        return {
            "n_items_multi_rated": len(multi),
            "agreement": None,
            "note": (
                "fewer than two items were scored by more than one rater, so agreement "
                "cannot be estimated — increase overlap between raters' sheets"
            ),
        }

    all_scores = [s for v in groups.values() for s in v]
    total_var = float(np.var(np.asarray(all_scores, dtype=np.float64)))
    within = float(np.mean([np.var(np.asarray(v, dtype=np.float64)) for v in multi.values()]))
    agreement = 1.0 - (within / total_var) if total_var > 1e-12 else None

    return {
        "n_items_multi_rated": len(multi),
        "mean_within_item_std": _r(float(np.sqrt(within))),
        "mean_within_item_std_pct_of_scale": _r(100.0 * float(np.sqrt(within)) / span, 1),
        "total_std": _r(float(np.sqrt(total_var))),
        "agreement": _r(agreement, 4),
        "interpretation": _interpret_agreement(agreement),
    }


def _interpret_agreement(value: float | None) -> str:
    if value is None:
        return "not estimable"
    if value >= 0.75:
        return "strong: raters largely agree, means are trustworthy"
    if value >= 0.5:
        return "moderate: means are usable but treat small differences with caution"
    if value >= 0.2:
        return "weak: rater disagreement is comparable to the differences between systems"
    return (
        "none: the panel carries essentially no shared signal — do not decide a model "
        "choice on these means"
    )


# ---------------------------------------------------------------------------
def merge_into_run(record: RunRecord, scores: Sequence[SubjectiveScore]) -> RunRecord:
    """Attach scores to a run and re-derive its subjective aggregates.

    Mutates and returns ``record``. Per-utterance means (not individual ratings)
    become the metric value, because the metric's unit of analysis is the utterance;
    the individual ratings remain on ``record.subjective`` for audit.
    """
    if not scores:
        return record

    existing = {(s.utterance_id, s.rater_id, s.scale) for s in record.subjective}
    for score in scores:
        k = (score.utterance_id, score.rater_id, score.scale)
        if k not in existing:
            record.subjective.append(score)
            existing.add(k)

    by_scale: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for score in record.subjective:
        by_scale[score.scale][score.utterance_id].append(score.score)

    records_by_id = {u.utterance_id: u for u in record.utterances}
    touched: list[str] = []

    for scale, per_utterance in by_scale.items():
        metric_name = _SCALE_METRIC.get(scale, f"subjective_{scale}")
        touched.append(metric_name)
        for utterance_id, values in per_utterance.items():
            target = records_by_id.get(utterance_id)
            if target is None:
                record.warnings.append(
                    f"subjective score references utterance {utterance_id!r}, which is not in "
                    "this run — score kept but not merged into a metric"
                )
                continue
            target.metrics[metric_name] = make_value(
                metric_name,
                statistics.fmean(values),
                extra={
                    "n_ratings": len(values),
                    "raters": sorted(
                        {s.rater_id for s in record.subjective
                         if s.utterance_id == utterance_id and s.scale == scale}
                    ),
                    "scores": [round(v, 3) for v in values],
                    "std": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
                },
            )

    # Re-aggregate only the subjective metrics: everything else is untouched, and
    # recomputing all aggregates would risk changing numbers that were signed off.
    for metric_name in touched:
        record.aggregates[metric_name] = aggregate_records(record.utterances, [metric_name])[metric_name]
        for language, aggs in aggregate_per_language(record.utterances, [metric_name]).items():
            record.per_language.setdefault(language, {})[metric_name] = aggs[metric_name]

    # A signoff attests to a specific report; adding ratings changes the report, so
    # prior signoffs no longer describe what is on disk.
    if record.signoffs:
        record.warnings.append(
            f"subjective scores were added after {len(record.signoffs)} signoff(s); those "
            "signoffs refer to an earlier report and should be renewed"
        )
    return record


def divergence_report(record: RunRecord) -> dict[str, Any]:
    """Compare human ratings against predicted MOS on the same utterances.

    This is the check the naturalness literature demands: UTMOS and DNSMOS are
    documented to rank-invert against listeners. Reporting the correlation makes
    that visible for *this* model and test set instead of assuming the proxy holds.
    """
    out: dict[str, Any] = {}
    subjective_metrics = [m for m in ("subjective_mos", "subjective_mushra") if m in record.aggregates]
    predicted_metrics = [m for m in ("utmos", "dnsmos_ovrl") if m in record.aggregates]
    if not subjective_metrics or not predicted_metrics:
        return {
            "available": False,
            "reason": (
                "needs both a human scale and a predicted MOS on the same run "
                f"(human: {subjective_metrics or 'none'}, predicted: {predicted_metrics or 'none'})"
            ),
        }

    for human in subjective_metrics:
        for predicted in predicted_metrics:
            pairs = [
                (u.value(human), u.value(predicted))
                for u in record.utterances
                if u.value(human) is not None and u.value(predicted) is not None
            ]
            if len(pairs) < 5:
                out[f"{human}_vs_{predicted}"] = {
                    "n": len(pairs),
                    "correlation": None,
                    "note": "fewer than 5 utterances have both metrics",
                }
                continue
            h = np.asarray([p[0] for p in pairs], dtype=np.float64)
            p_ = np.asarray([p[1] for p in pairs], dtype=np.float64)
            corr = (
                float(np.corrcoef(h, p_)[0, 1]) if h.std() > 1e-9 and p_.std() > 1e-9 else None
            )
            out[f"{human}_vs_{predicted}"] = {
                "n": len(pairs),
                "pearson": _r(corr, 4),
                "interpretation": (
                    "not estimable" if corr is None
                    else "predicted MOS tracks the listeners here" if corr >= 0.6
                    else "weak alignment: trust the human scores over the predictor"
                    if corr >= 0.3
                    else "predicted MOS disagrees with the listeners on this set — do not use "
                         "it as a naturalness substitute"
                ),
            }
    out["available"] = True
    return out


def _r(v: float | None, ndigits: int = 3) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, ndigits) if np.isfinite(f) else None


__all__ = [
    "IngestReport",
    "RaterReport",
    "divergence_report",
    "ingest_sheets",
    "load_answer_key",
    "merge_into_run",
]
