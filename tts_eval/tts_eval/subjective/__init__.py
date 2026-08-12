"""Human listening tests: generation, ingestion, and merge into run records.

Objective naturalness metrics are predictors and are documented to rank-invert
against listeners, so this package supplies the actual measurement rather than
letting a proxy stand in for it. See :mod:`~tts_eval.subjective.listening_test` for
the blinding design and :mod:`~tts_eval.subjective.ingest` for rater screening.
"""
from __future__ import annotations

from .ingest import (  # noqa: F401
    IngestReport,
    RaterReport,
    divergence_report,
    ingest_sheets,
    load_answer_key,
    merge_into_run,
)
from .listening_test import SCALES, TestSpec, Trial, blind_token, build_test, iter_scales  # noqa: F401

__all__ = [
    "SCALES",
    "IngestReport",
    "RaterReport",
    "TestSpec",
    "Trial",
    "blind_token",
    "build_test",
    "divergence_report",
    "ingest_sheets",
    "iter_scales",
    "load_answer_key",
    "merge_into_run",
]
