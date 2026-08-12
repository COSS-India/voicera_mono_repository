"""Exception hierarchy.

The split matters operationally: a run must survive per-utterance synthesis
failures (they become data — see reliability metrics) but must fail fast on
configuration errors, because a misconfigured run produces numbers that look
valid and are not.
"""
from __future__ import annotations


class TTSEvalError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(TTSEvalError):
    """Malformed model card, suite config, or CLI argument combination.

    Always fatal: we refuse to produce a report from a config we do not
    understand.
    """


class DatasetError(TTSEvalError):
    """Test dataset is missing, malformed, or fails its integrity check."""


class AdapterUnavailable(TTSEvalError):
    """An adapter cannot be constructed at all (missing transport dependency,
    unreachable endpoint at probe time, unknown adapter name).

    Fatal for the run, because every utterance would fail identically.
    """


class SynthesisFailed(TTSEvalError):
    """A single utterance failed to synthesise.

    NOT fatal. Recorded against the utterance and folded into the reliability
    and language-coverage metrics.
    """


class MetricUnavailable(TTSEvalError):
    """A metric backend's dependencies or weights are absent.

    Never fatal: converted into a ``not_computed`` MetricValue carrying the
    reason, so reports state honestly what was and was not measured.
    """


class StoreError(TTSEvalError):
    """Run registry could not be read or written."""
