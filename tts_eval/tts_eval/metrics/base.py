"""Metric backend contract, registry and engine.

The layering rule this file enforces: **a missing dependency is data, not an
error**. Every backend declares what it can produce and whether it is currently
available; when it is not, the engine emits ``not_computed`` values carrying the
reason for exactly those metric names. A report therefore always has a complete
row set, and says "UTMOS: not computed — torch not installed" rather than
omitting naturalness and letting a reader assume it was fine.

Two scopes, because the metrics genuinely differ in shape:

*   ``utterance`` — one value per synthesised sentence, later aggregated with
    percentiles and a bootstrap CI.
*   ``run`` — one value for the whole run (voice consistency across utterances,
    language coverage, throughput). Stored as a degenerate aggregate (n=1) so
    reports and comparison have a single uniform place to look.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..datasets.loader import TestCase, TestDataset
from ..types import (
    Capabilities,
    MetricStatus,
    MetricValue,
    SynthesisResult,
    UtteranceRecord,
)
from .catalog import direction as metric_direction
from .catalog import spec as metric_spec
from .catalog import unit as metric_unit


@dataclass
class Thresholds:
    """Pass/fail lines that turn measurements into verdicts.

    Defaults are deliberately conservative and documented rather than tuned to
    make any particular model look good. Override per suite in YAML; the resolved
    values are stored in the run record so a report can be re-read years later
    and still be interpretable.
    """

    # A language counts as intelligible only if its median CER is at or below
    # this. 0.30 is permissive on purpose: round-trip CER also carries the ASR
    # model's own error rate, which for low-resource Indic languages is
    # substantial, so a tighter line would fail languages the TTS renders fine.
    cer_verified_max: float = 0.30
    # Fraction of a language's utterances that must synthesise successfully.
    success_verified_min: float = 0.90
    # Above this, an utterance is counted as degenerate (looping/buzz/truncated).
    degeneracy_max: float = 0.50
    # Language-specific characters-per-second used to predict expected duration
    # for length_ratio. Indic scripts encode more phonemes per character.
    chars_per_second_default: float = 11.0
    chars_per_second_latin: float = 15.0
    # Fixed duration overhead per utterance, independent of text length: onset,
    # offset, and the short silence pad most servers emit. Without this intercept,
    # expected duration for a 4-character utterance is ~0.27 s, the padding alone
    # is ~0.06 s, and a HALVED clip still scores length_ratio 0.69 — so truncation
    # of short utterances goes undetected. Negligible for long utterances.
    duration_overhead_s: float = 0.15
    # length_ratio outside this band contributes to degeneracy.
    length_ratio_min: float = 0.55
    length_ratio_max: float = 1.90

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "Thresholds":
        if not d:
            return cls()
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class MetricContext:
    """Everything a backend may need that is not the audio itself."""

    dataset: TestDataset
    capabilities: Capabilities
    thresholds: Thresholds = field(default_factory=Thresholds)
    # Where a backend may cache models or intermediate files.
    workdir: Path = field(default_factory=lambda: Path(".tts_eval_cache"))
    # Round-trip ASR, injected rather than constructed so one model instance is
    # shared by every metric that needs it instead of being loaded per metric.
    asr: Any | None = None
    # Directory the dataset was loaded from, for resolving relative reference audio.
    dataset_dir: Path | None = None
    # Non-fatal problems worth surfacing in the report header.
    warnings: list[str] = field(default_factory=list)
    # Free-form options from the suite config, passed through to backends.
    options: dict[str, Any] = field(default_factory=dict)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def resolve_reference(self, case: TestCase) -> Path | None:
        if not case.reference_audio:
            return None
        p = Path(case.reference_audio)
        if p.is_absolute():
            return p if p.is_file() else None
        base = self.dataset_dir or Path.cwd()
        candidate = base / p
        return candidate if candidate.is_file() else None

    def chars_per_second(self, language: str) -> float:
        return (
            self.thresholds.chars_per_second_latin
            if language.startswith("en")
            else self.thresholds.chars_per_second_default
        )


class MetricBackend(abc.ABC):
    """Base class for anything that produces metrics."""

    #: Registry key, referenced in a suite's ``metrics:`` list.
    name: str = ""
    #: Metric names this backend can emit. Used to synthesise ``not_computed``
    #: values when the backend is unavailable, which is why it must be exhaustive.
    provides: tuple[str, ...] = ()
    #: "utterance" or "run".
    scope: str = "utterance"

    def __init__(self, options: Mapping[str, Any] | None = None):
        self.options = dict(options or {})

    def available(self) -> tuple[bool, str]:
        """Return ``(is_available, reason_or_version)``.

        Called once per run before synthesis so the report header can state
        exactly which backends were live. Must not raise and must not load heavy
        models — that belongs in :meth:`prepare`.
        """
        return True, "core"

    def prepare(self, ctx: MetricContext) -> None:
        """Load models/weights. Called once, only if :meth:`available` passed."""

    def teardown(self) -> None:
        """Release models. Always called after a run."""

    def version(self) -> str:
        return self.available()[1]


class UtteranceBackend(MetricBackend):
    scope = "utterance"

    @abc.abstractmethod
    def compute(
        self, case: TestCase, result: SynthesisResult, ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        """Score one utterance. ``result.audio`` is guaranteed non-None and
        non-empty; the engine handles failed utterances itself."""


class RunBackend(MetricBackend):
    scope = "run"

    @abc.abstractmethod
    def compute(
        self, records: Sequence[UtteranceRecord], ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        """Score the run. ``records`` includes failures, because several run-level
        metrics (success rate, coverage) are *about* the failures."""


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, type[MetricBackend]] = {}

#: ``--metrics core``: dependency-free, always computable, no network, no GPU.
#: This tier alone satisfies Response Latency, Inference Time, Audio Quality,
#: Voice Consistency and Language Coverage (the last without an intelligibility
#: check, which it says so in the coverage notes).
CORE_BACKENDS = (
    "latency",
    "throughput",
    "audio_quality",
    "voice_consistency",
    "voice_consistency_run",
    "reliability",
    "coverage",
)
#: ``--metrics standard``: core plus round-trip ASR, which needs either a local
#: Whisper or a reachable ASR server. Adds Pronunciation Accuracy and upgrades
#: Language Coverage to a verified intelligibility check.
STANDARD_BACKENDS = CORE_BACKENDS + ("intelligibility",)
#: ``--metrics all``: everything registered, including the heavy perceptual
#: backends (UTMOS, DNSMOS, speaker embeddings, VERSA, TTSDS2). Any that are
#: unavailable report why, and the run still completes.


def register_backend(cls: type[MetricBackend]) -> type[MetricBackend]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`")
    if not cls.provides:
        raise ValueError(f"{cls.__name__} must declare `provides`")
    existing = _REGISTRY.get(cls.name)
    if existing is not None and existing is not cls:
        raise ValueError(f"metric backend {cls.name!r} already registered by {existing.__name__}")
    _REGISTRY[cls.name] = cls
    return cls


def get_backend_class(name: str) -> type[MetricBackend]:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown metric backend {name!r}; registered: {known}")
    return _REGISTRY[name]


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def resolve_backend_names(selection: str | Sequence[str]) -> list[str]:
    """Expand ``core`` / ``standard`` / ``all`` or pass through an explicit list."""
    if isinstance(selection, str):
        key = selection.strip().lower()
        if key == "core":
            return [n for n in CORE_BACKENDS if n in _REGISTRY]
        if key == "standard":
            return [n for n in STANDARD_BACKENDS if n in _REGISTRY]
        if key == "all":
            return available_backends()
        selection = [s.strip() for s in selection.split(",") if s.strip()]
    out: list[str] = []
    for name in selection:
        if name in ("core", "standard", "all"):
            for expanded in resolve_backend_names(name):
                if expanded not in out:
                    out.append(expanded)
            continue
        get_backend_class(name)  # raises on typo, before an expensive run starts
        if name not in out:
            out.append(name)
    return out


def build_backends(
    names: Sequence[str], options: Mapping[str, Mapping[str, Any]] | None = None
) -> list[MetricBackend]:
    opts = dict(options or {})
    return [get_backend_class(n)(opts.get(n, {})) for n in names]


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
def make_value(name: str, value: float | None, **kw: Any) -> MetricValue:
    """Build a MetricValue with unit/direction filled in from the catalogue.

    Backends call this instead of constructing MetricValue directly so polarity
    is never accidentally omitted — a metric without polarity silently becomes
    uncomparable in reports.
    """
    return MetricValue(
        name=name,
        value=value,
        unit=kw.pop("unit", metric_unit(name)),
        direction=kw.pop("direction", metric_direction(name)),
        **kw,
    )


def missing_value(name: str, reason: str, status: MetricStatus = MetricStatus.NOT_COMPUTED) -> MetricValue:
    return MetricValue(
        name=name,
        value=None,
        unit=metric_unit(name),
        direction=metric_direction(name),
        status=status,
        detail=reason,
    )


class MetricEngine:
    """Runs the selected backends over a run's results.

    Owns availability resolution, so the decision "can this metric be computed"
    is made once and recorded, not re-derived per utterance.
    """

    def __init__(self, backends: Sequence[MetricBackend], ctx: MetricContext):
        self.ctx = ctx
        self._all = list(backends)
        self._live: list[MetricBackend] = []
        #: backend name -> version string, or "absent: <reason>"
        self.backend_status: dict[str, str] = {}
        #: metric name -> reason, for everything that cannot be computed this run
        self._unavailable: dict[str, str] = {}

        for backend in self._all:
            try:
                ok, info = backend.available()
            except Exception as e:  # noqa: BLE001 - availability must never break a run
                ok, info = False, f"{type(e).__name__}: {e}"
            if ok:
                self._live.append(backend)
                self.backend_status[backend.name] = info
            else:
                self.backend_status[backend.name] = f"absent: {info}"
                for metric in backend.provides:
                    self._unavailable[metric] = f"{backend.name} unavailable: {info}"
                ctx.warn(f"metric backend {backend.name!r} unavailable: {info}")

    # -- lifecycle ---------------------------------------------------------
    def prepare(self) -> None:
        """Load backend models, demoting any that fails to load.

        A backend that passes ``available()`` but throws in ``prepare()`` is
        downgraded here rather than at the first utterance, so its metrics are
        reported as missing once with a clear reason instead of producing N
        identical per-utterance errors.
        """
        still_live: list[MetricBackend] = []
        for backend in self._live:
            try:
                backend.prepare(self.ctx)
                still_live.append(backend)
            except Exception as e:  # noqa: BLE001
                reason = f"failed to load: {type(e).__name__}: {e}"
                self.backend_status[backend.name] = f"absent: {reason}"
                for metric in backend.provides:
                    self._unavailable[metric] = f"{backend.name} {reason}"
                self.ctx.warn(f"metric backend {backend.name!r} {reason}")
        self._live = still_live

    def teardown(self) -> None:
        for backend in self._live:
            try:
                backend.teardown()
            except Exception:  # noqa: BLE001 - teardown must not mask run results
                pass

    @property
    def live_backend_names(self) -> list[str]:
        return [b.name for b in self._live]

    @property
    def expected_metrics(self) -> list[str]:
        out: list[str] = []
        for backend in self._all:
            for metric in backend.provides:
                if metric not in out:
                    out.append(metric)
        return out

    # -- computation -------------------------------------------------------
    def score_utterance(self, case: TestCase, result: SynthesisResult) -> dict[str, MetricValue]:
        """Score one utterance, filling in reasons for everything not computed."""
        values: dict[str, MetricValue] = {}

        if not result.ok or result.audio is None or result.audio.is_empty():
            reason = result.error or "no audio returned"
            for backend in self._all:
                if backend.scope != "utterance":
                    continue
                for metric in backend.provides:
                    values[metric] = missing_value(
                        metric, f"utterance failed: {reason}", MetricStatus.NOT_APPLICABLE
                    )
            return values

        for backend in self._live:
            if backend.scope != "utterance":
                continue
            try:
                produced = backend.compute(case, result, self.ctx)
            except Exception as e:  # noqa: BLE001 - one metric must not kill the run
                for metric in backend.provides:
                    values[metric] = MetricValue(
                        name=metric,
                        value=None,
                        unit=metric_unit(metric),
                        direction=metric_direction(metric),
                        status=MetricStatus.ERROR,
                        detail=f"{type(e).__name__}: {e}",
                    )
                continue
            for metric_name, metric_value in produced.items():
                values[metric_name] = metric_value
            # A live backend that silently omits one of its declared metrics
            # still gets an explicit row, so the report never has holes.
            for metric in backend.provides:
                values.setdefault(metric, missing_value(metric, "backend produced no value"))

        for metric, reason in self._unavailable.items():
            if metric_spec(metric).scope == "utterance":
                values.setdefault(metric, missing_value(metric, reason))
        return values

    def score_run(self, records: Sequence[UtteranceRecord]) -> dict[str, MetricValue]:
        values: dict[str, MetricValue] = {}
        for backend in self._live:
            if backend.scope != "run":
                continue
            try:
                produced = backend.compute(records, self.ctx)
            except Exception as e:  # noqa: BLE001
                for metric in backend.provides:
                    values[metric] = MetricValue(
                        name=metric,
                        value=None,
                        unit=metric_unit(metric),
                        direction=metric_direction(metric),
                        status=MetricStatus.ERROR,
                        detail=f"{type(e).__name__}: {e}",
                    )
                continue
            values.update(produced)
            for metric in backend.provides:
                values.setdefault(metric, missing_value(metric, "backend produced no value"))

        for metric, reason in self._unavailable.items():
            if metric_spec(metric).scope == "run":
                values.setdefault(metric, missing_value(metric, reason))
        return values


def iter_registered() -> Iterable[tuple[str, type[MetricBackend]]]:
    return sorted(_REGISTRY.items())


__all__ = [
    "CORE_BACKENDS",
    "STANDARD_BACKENDS",
    "MetricBackend",
    "MetricContext",
    "MetricEngine",
    "RunBackend",
    "Thresholds",
    "UtteranceBackend",
    "available_backends",
    "build_backends",
    "get_backend_class",
    "iter_registered",
    "make_value",
    "missing_value",
    "register_backend",
    "resolve_backend_names",
]
