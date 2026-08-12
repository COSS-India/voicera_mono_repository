"""Reliability: success rate and the rate of nominal successes that are not.

Reported before anything else in every report, because every other aggregate is
conditioned on it. A model that fails 30% of requests will show excellent p95
latency on the 70% that returned — the failures have no latency at all — so
reading latency without reading success rate inverts the ranking.

``degenerate_rate`` is the second-order version of the same trap: requests that
returned audio the caller would nonetheless hear as broken (looping, buzz, half a
sentence). Those inflate the success rate while degrading the product.
"""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from ..types import MetricStatus, MetricValue, UtteranceRecord
from .base import MetricContext, RunBackend, make_value, missing_value, register_backend


@register_backend
class ReliabilityBackend(RunBackend):
    name = "reliability"
    provides = ("success_rate", "degenerate_rate")

    def compute(
        self, records: Sequence[UtteranceRecord], ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        if not records:
            return {
                name: missing_value(name, "no utterances in this run") for name in self.provides
            }

        ok = [r for r in records if r.result.ok]
        failed = [r for r in records if not r.result.ok]

        # Group failure reasons so a report can name the dominant cause instead of
        # only showing a count. Truncated to the message prefix because provider
        # errors often embed request ids that would fragment the grouping.
        reasons = Counter(
            (r.result.error or "unknown").split("(")[0].strip()[:120] for r in failed
        )
        by_language = Counter(r.language for r in failed)

        success = make_value(
            "success_rate",
            len(ok) / len(records),
            extra={
                "n_total": len(records),
                "n_ok": len(ok),
                "n_failed": len(failed),
                "failure_reasons": dict(reasons.most_common(10)),
                "failures_by_language": dict(sorted(by_language.items())),
            },
        )

        # Only utterances that *returned* audio can be degenerate; dividing by the
        # total would let outright failures suppress this rate.
        threshold = ctx.thresholds.degeneracy_max
        scored = [
            r
            for r in ok
            if (m := r.metrics.get("degeneracy_score")) is not None
            and m.status is MetricStatus.OK
            and m.value is not None
        ]
        if not scored:
            degenerate = missing_value(
                "degenerate_rate",
                "degeneracy_score was not computed (audio_quality backend did not run)",
            )
        else:
            flagged = [r for r in scored if (r.metrics["degeneracy_score"].value or 0.0) > threshold]
            degenerate = make_value(
                "degenerate_rate",
                len(flagged) / len(scored),
                extra={
                    "threshold": threshold,
                    "n_scored": len(scored),
                    "n_flagged": len(flagged),
                    # Named so a reviewer can go straight to the audio.
                    "flagged_utterances": [
                        {
                            "utterance_id": r.utterance_id,
                            "language": r.language,
                            "score": r.metrics["degeneracy_score"].value,
                            "reason": r.metrics["degeneracy_score"].detail,
                        }
                        for r in sorted(
                            flagged,
                            key=lambda x: -(x.metrics["degeneracy_score"].value or 0.0),
                        )[:25]
                    ],
                },
            )

        return {"success_rate": success, "degenerate_rate": degenerate}


__all__ = ["ReliabilityBackend"]
