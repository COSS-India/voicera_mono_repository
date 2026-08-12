"""Language coverage: which languages the model *actually* serves.

A model card's language list is a claim, not evidence. The acceptance criterion
"Metrics are successfully captured for every supported language" is only
meaningful if the framework distinguishes the two, so a language is marked
``verified`` here only when all of the following hold:

1.  its utterances synthesised at or above ``success_verified_min``;
2.  round-trip intelligibility passed — median CER at or below
    ``cer_verified_max`` — *when an ASR backend was available*;
3.  fewer than half its utterances were flagged degenerate.

When no ASR is available, condition 2 cannot be tested. The language is then
reported as ``verified`` on the evidence obtained, with an explicit note that
intelligibility was unverified, and ``coverage_ratio`` carries the same caveat.
That is deliberate: silently treating "synthesised without error" as "supported"
is exactly how a model ends up shipping for a language it renders as noise.

Coverage also flags the inverse mismatch — a language present in the test set but
absent from the model card — because that is usually a stale card rather than a
model limitation, and it is worth surfacing before someone routes traffic to it.
"""
from __future__ import annotations

import statistics
from typing import Mapping, Sequence

from ..types import LanguageCoverage, MetricStatus, MetricValue, UtteranceRecord
from .base import MetricContext, RunBackend, make_value, missing_value, register_backend


@register_backend
class CoverageBackend(RunBackend):
    name = "coverage"
    provides = ("coverage_ratio", "languages_verified", "languages_attempted")

    def compute(
        self, records: Sequence[UtteranceRecord], ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        rows = build_coverage(records, ctx)
        if not rows:
            return {
                name: missing_value(name, "no utterances in this run") for name in self.provides
            }

        claimed = set(ctx.capabilities.languages)
        verified = [r for r in rows if r.verified]
        attempted = [r for r in rows if r.attempted > 0]

        # Denominator is the claimed set, not the tested set: a run that only
        # tested 3 of 22 claimed languages must not report 100% coverage.
        if claimed:
            covered_claimed = [r for r in verified if r.language in claimed]
            ratio = len(covered_claimed) / len(claimed)
            ratio_detail = (
                f"{len(covered_claimed)}/{len(claimed)} languages claimed by the model card "
                "were verified"
            )
            untested = sorted(claimed - {r.language for r in attempted})
            if untested:
                ctx.warn(
                    f"model claims {len(untested)} language(s) not present in this test set: "
                    f"{', '.join(untested)}"
                )
        else:
            ratio = len(verified) / len(attempted) if attempted else 0.0
            ratio_detail = (
                "model card declares no languages, so coverage is measured against the "
                "test set instead of the claim"
            )
            ctx.warn(
                "model card declares no `languages`; coverage_ratio is relative to the test "
                "set and cannot detect unsupported-but-claimed languages"
            )

        unclaimed = sorted({r.language for r in attempted} - claimed) if claimed else []
        if unclaimed:
            ctx.warn(
                f"test set contains language(s) absent from the model card: {', '.join(unclaimed)}"
            )

        return {
            "coverage_ratio": make_value(
                "coverage_ratio",
                ratio,
                detail=ratio_detail,
                extra={
                    "claimed": sorted(claimed),
                    "verified": sorted(r.language for r in verified),
                    "failed_verification": sorted(
                        r.language for r in attempted if not r.verified
                    ),
                    "claimed_but_untested": sorted(claimed - {r.language for r in attempted}),
                    "tested_but_unclaimed": unclaimed,
                    "per_language": [r.to_dict() for r in rows],
                },
            ),
            "languages_verified": make_value("languages_verified", float(len(verified))),
            "languages_attempted": make_value("languages_attempted", float(len(attempted))),
        }


def build_coverage(
    records: Sequence[UtteranceRecord], ctx: MetricContext
) -> list[LanguageCoverage]:
    """Per-language verification verdicts, also stored on the run record itself."""
    groups: dict[str, list[UtteranceRecord]] = {}
    for rec in records:
        groups.setdefault(rec.language, []).append(rec)

    claimed = set(ctx.capabilities.languages)
    thresholds = ctx.thresholds
    out: list[LanguageCoverage] = []

    for language, recs in sorted(groups.items()):
        attempted = len(recs)
        ok = [r for r in recs if r.result.ok]
        succeeded = len(ok)
        success_rate = succeeded / attempted if attempted else 0.0

        cers = [
            float(m.value)
            for r in ok
            if (m := r.metrics.get("cer")) is not None
            and m.status is MetricStatus.OK
            and m.value is not None
        ]
        intelligible: int | None
        median_cer: float | None
        if cers:
            median_cer = statistics.median(cers)
            intelligible = sum(1 for c in cers if c <= thresholds.cer_verified_max)
        else:
            median_cer = None
            intelligible = None

        degenerate = [
            r
            for r in ok
            if (m := r.metrics.get("degeneracy_score")) is not None
            and m.status is MetricStatus.OK
            and (m.value or 0.0) > thresholds.degeneracy_max
        ]
        degenerate_rate = len(degenerate) / succeeded if succeeded else 1.0

        notes: list[str] = []
        verified = True

        if success_rate < thresholds.success_verified_min:
            verified = False
            notes.append(
                f"success rate {success_rate:.0%} below the {thresholds.success_verified_min:.0%} "
                "threshold"
            )
        if median_cer is None:
            notes.append(
                "intelligibility unverified: no CER computed — either no `asr:` block in "
                "the suite, or every ASR call failed (check the Pronunciation Accuracy "
                "section for the error)"
            )
        elif median_cer > thresholds.cer_verified_max:
            verified = False
            notes.append(
                f"median CER {median_cer:.3f} above the {thresholds.cer_verified_max:.2f} threshold"
            )
        if degenerate_rate > 0.5:
            verified = False
            notes.append(f"{degenerate_rate:.0%} of successful utterances flagged degenerate")
        if attempted < 2:
            notes.append(f"only {attempted} utterance(s) tested; verdict is weak evidence")

        out.append(
            LanguageCoverage(
                language=language,
                claimed=language in claimed,
                attempted=attempted,
                succeeded=succeeded,
                intelligible=intelligible,
                verified=verified,
                notes="; ".join(notes) or None,
            )
        )
    return out


__all__ = ["CoverageBackend", "build_coverage"]
