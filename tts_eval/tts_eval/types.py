"""Core data types shared by every layer.

These types are the framework's real interface. Adapters produce
:class:`SynthesisResult`; metrics consume it and produce :class:`MetricValue`;
the store persists :class:`RunRecord`. A new model or metric plugs in by
speaking these types and nothing else, which is what keeps the framework
generic across providers.

Everything that is persisted has an explicit ``to_dict``/``from_dict`` pair
rather than relying on ``dataclasses.asdict``, so the on-disk schema is a
deliberate, reviewable contract instead of an accident of field ordering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioBuffer:
    """Mono float32 audio in [-1, 1].

    Float32 is the canonical in-memory form because that is what the WebSocket
    TTS servers in this monorepo emit; int16 conversion happens only at WAV
    write time. Keeping one canonical form means metrics never have to guess a
    scale.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError(f"AudioBuffer expects mono 1-D samples, got shape {self.samples.shape}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    @property
    def duration_s(self) -> float:
        return float(self.samples.size) / float(self.sample_rate)

    @property
    def n_samples(self) -> int:
        return int(self.samples.size)

    def is_empty(self) -> bool:
        return self.samples.size == 0


class Determinism(str, Enum):
    """How reproducible a given run's audio actually is.

    This is recorded, not assumed. An autoregressive LLM-based TTS such as
    Indic-Mio sampling at temperature 0.9 is *not* bit-reproducible even with
    identical text, so claiming reproducibility without qualification would be
    false. The harness distinguishes:

    * ``SEEDED``      — provider accepted a seed and greedy/seeded decoding was
                        requested; audio is expected to be reproducible.
    * ``BEST_EFFORT`` — inputs are pinned (same text, voice, params, dataset
                        hash) but sampling is stochastic; *metrics* are
                        reproducible within confidence intervals, audio is not.
    * ``DETERMINISTIC`` — provider is inherently deterministic (e.g. replay of
                        stored audio, or a non-sampling vocoder pipeline).
    """

    DETERMINISTIC = "deterministic"
    SEEDED = "seeded"
    BEST_EFFORT = "best_effort"


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SynthesisRequest:
    """One unit of work: render `text` in `language` with `voice`."""

    utterance_id: str
    text: str
    language: str
    voice: str | None = None
    seed: int | None = None
    # Free-form provider generation overrides (temperature, top_p, style, ...).
    # Hashed into the run fingerprint so a params change is never invisible.
    params: Mapping[str, Any] = field(default_factory=dict)
    # Optional ground-truth recording of this sentence, used by reference-based
    # metrics (speaker similarity vs. a target voice, TTSDS2 distributions).
    reference_audio: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "text": self.text,
            "language": self.language,
            "voice": self.voice,
            "seed": self.seed,
            "params": dict(self.params),
            "reference_audio": self.reference_audio,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SynthesisRequest":
        return cls(
            utterance_id=str(d["utterance_id"]),
            text=str(d["text"]),
            language=str(d["language"]),
            voice=d.get("voice"),
            seed=d.get("seed"),
            params=dict(d.get("params") or {}),
            reference_audio=d.get("reference_audio"),
        )


@dataclass(frozen=True)
class ChunkTiming:
    """Arrival of one audio chunk, measured from the moment the request was sent.

    The full sequence is kept (not just first/last) because streaming *jitter*
    is what breaks a conversational pipeline: a model can have an excellent TTFB
    and still starve the playout buffer mid-utterance. See
    ``metrics.latency.stream_stability``.
    """

    offset_ms: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {"offset_ms": round(self.offset_ms, 3), "n_samples": self.n_samples}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ChunkTiming":
        return cls(offset_ms=float(d["offset_ms"]), n_samples=int(d["n_samples"]))


@dataclass
class SynthesisResult:
    """Everything one synthesis call produced, including how it was timed.

    ``audio`` is dropped from the record once persisted to WAV (``audio_path``);
    the in-memory buffer only lives for the duration of metric computation so a
    1000-utterance run does not hold gigabytes.
    """

    request: SynthesisRequest
    audio: AudioBuffer | None = None
    # Wall-clock from request-send to the first byte of audio of any kind.
    ttfb_ms: float | None = None
    # Wall-clock to the first *audible* sample (above the silence floor). Some
    # servers emit a leading pad of zeros; TTFB alone then flatters the model.
    first_audible_ms: float | None = None
    # Wall-clock from request-send to the terminal "done" message.
    total_ms: float = 0.0
    chunk_timings: list[ChunkTiming] = field(default_factory=list)
    # Anything the provider told us about itself (sample_rate, model id, token
    # counts). Kept verbatim for auditability.
    provider_meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    audio_path: str | None = None
    # Count of audio samples the provider produced, captured at synthesis time and
    # persisted. This is the durable record of "did the request succeed": it stays
    # correct after the in-memory buffer is freed AND when audio is deliberately not
    # written to disk (``save_audio: false``), neither of which the buffer or the
    # file path can be trusted for. ``None`` only on a record from before this field
    # existed, where ``ok`` falls back to the buffer/path.
    n_samples: int | None = None

    @property
    def ok(self) -> bool:
        """True when this utterance produced usable audio.

        Empty audio counts as a failure even without an exception: a server that
        returns a clean ``done`` with zero samples has failed the request, and
        silently scoring it would corrupt every aggregate.
        """
        if self.error is not None:
            return False
        # The captured sample count is the source of truth: it is independent of
        # whether the buffer is still in memory or the WAV was ever written, so a
        # not-saved-audio run is not mistaken for a run of total failures.
        if self.n_samples is not None:
            return self.n_samples > 0
        if self.audio is None:
            return self.audio_path is not None
        return not self.audio.is_empty()

    @property
    def duration_s(self) -> float | None:
        return self.audio.duration_s if self.audio is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "ttfb_ms": _round_opt(self.ttfb_ms),
            "first_audible_ms": _round_opt(self.first_audible_ms),
            "total_ms": round(self.total_ms, 3),
            "chunk_timings": [c.to_dict() for c in self.chunk_timings],
            "provider_meta": dict(self.provider_meta),
            "error": self.error,
            "audio_path": self.audio_path,
            "n_samples": self.n_samples,
            "audio_duration_s": _round_opt(self.duration_s, 4),
            "sample_rate": self.audio.sample_rate if self.audio is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SynthesisResult":
        res = cls(
            request=SynthesisRequest.from_dict(d["request"]),
            ttfb_ms=d.get("ttfb_ms"),
            first_audible_ms=d.get("first_audible_ms"),
            total_ms=float(d.get("total_ms") or 0.0),
            chunk_timings=[ChunkTiming.from_dict(c) for c in d.get("chunk_timings") or []],
            provider_meta=dict(d.get("provider_meta") or {}),
            error=d.get("error"),
            audio_path=d.get("audio_path"),
            n_samples=d.get("n_samples"),
        )
        # Duration/sample-rate survive without the buffer so reports and
        # comparisons work on a store that has had its WAVs pruned.
        res.provider_meta.setdefault("_audio_duration_s", d.get("audio_duration_s"))
        res.provider_meta.setdefault("_sample_rate", d.get("sample_rate"))
        return res


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class MetricStatus(str, Enum):
    OK = "ok"
    # Backend dependency/weights absent, or the metric needs an input this run
    # does not have (e.g. no reference audio). Carries a human-readable reason.
    NOT_COMPUTED = "not_computed"
    # Backend was available and raised. Distinguished from NOT_COMPUTED because
    # this one is a bug or an environment fault worth chasing.
    ERROR = "error"
    # Metric is meaningless for this model/run by definition (e.g. streaming
    # jitter for a non-streaming provider).
    NOT_APPLICABLE = "not_applicable"


class Direction(str, Enum):
    """Which way is better. Required for report colouring and regression gates —
    without it, a comparison cannot say whether a delta is an improvement.
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class MetricValue:
    """One measurement.

    ``value`` is ``None`` for every non-OK status; consumers must check
    ``status`` rather than treating ``None`` as zero.
    """

    name: str
    value: float | None
    unit: str = ""
    status: MetricStatus = MetricStatus.OK
    direction: Direction = Direction.NEUTRAL
    detail: str | None = None
    # Supporting numbers a reviewer needs to trust the headline value
    # (e.g. the ASR hypothesis behind a CER, the per-chunk gaps behind jitter).
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def not_computed(cls, name: str, reason: str, **kw: Any) -> "MetricValue":
        return cls(name=name, value=None, status=MetricStatus.NOT_COMPUTED, detail=reason, **kw)

    @classmethod
    def error(cls, name: str, reason: str, **kw: Any) -> "MetricValue":
        return cls(name=name, value=None, status=MetricStatus.ERROR, detail=reason, **kw)

    @classmethod
    def not_applicable(cls, name: str, reason: str, **kw: Any) -> "MetricValue":
        return cls(name=name, value=None, status=MetricStatus.NOT_APPLICABLE, detail=reason, **kw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": _clean_float(self.value),
            "unit": self.unit,
            "status": self.status.value,
            "direction": self.direction.value,
            "detail": self.detail,
            "extra": _json_safe(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MetricValue":
        return cls(
            name=str(d["name"]),
            value=d.get("value"),
            unit=str(d.get("unit") or ""),
            status=MetricStatus(d.get("status", "ok")),
            direction=Direction(d.get("direction", "neutral")),
            detail=d.get("detail"),
            extra=dict(d.get("extra") or {}),
        )


@dataclass(frozen=True)
class Aggregate:
    """Distribution summary for one metric across a run.

    Latency work needs tails, not means — p95 is the number that decides whether
    a model is usable in a live call — so percentiles are first-class rather
    than an afterthought. ``ci_low``/``ci_high`` are a bootstrap 95% CI on the
    mean, which is what makes an A/B claim defensible.
    """

    name: str
    unit: str
    direction: Direction
    n: int
    n_missing: int
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    # Why nothing was computed, when n == 0. Propagated from MetricValue.detail
    # so a report can say "UTMOS: not installed" rather than showing a blank.
    missing_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction.value,
            "n": self.n,
            "n_missing": self.n_missing,
            "mean": _clean_float(self.mean),
            "std": _clean_float(self.std),
            "median": _clean_float(self.median),
            "p90": _clean_float(self.p90),
            "p95": _clean_float(self.p95),
            "p99": _clean_float(self.p99),
            "min": _clean_float(self.minimum),
            "max": _clean_float(self.maximum),
            "ci_low": _clean_float(self.ci_low),
            "ci_high": _clean_float(self.ci_high),
            "missing_reason": self.missing_reason,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Aggregate":
        return cls(
            name=str(d["name"]),
            unit=str(d.get("unit") or ""),
            direction=Direction(d.get("direction", "neutral")),
            n=int(d.get("n") or 0),
            n_missing=int(d.get("n_missing") or 0),
            mean=d.get("mean"),
            std=d.get("std"),
            median=d.get("median"),
            p90=d.get("p90"),
            p95=d.get("p95"),
            p99=d.get("p99"),
            minimum=d.get("min"),
            maximum=d.get("max"),
            ci_low=d.get("ci_low"),
            ci_high=d.get("ci_high"),
            missing_reason=d.get("missing_reason"),
        )


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Capabilities:
    """What an adapter's model can actually do.

    Declared by the adapter (from its model card) so the harness can (a) mark
    metrics ``not_applicable`` instead of reporting a misleading zero, and
    (b) distinguish *claimed* language support from *verified* language support
    in the coverage matrix.
    """

    streaming: bool = False
    voices: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    supports_seed: bool = False
    supports_emotion: bool = False
    native_sample_rate: int | None = None
    determinism: Determinism = Determinism.BEST_EFFORT

    def to_dict(self) -> dict[str, Any]:
        return {
            "streaming": self.streaming,
            "voices": list(self.voices),
            "languages": list(self.languages),
            "supports_seed": self.supports_seed,
            "supports_emotion": self.supports_emotion,
            "native_sample_rate": self.native_sample_rate,
            "determinism": self.determinism.value,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Capabilities":
        return cls(
            streaming=bool(d.get("streaming", False)),
            voices=tuple(d.get("voices") or ()),
            languages=tuple(d.get("languages") or ()),
            supports_seed=bool(d.get("supports_seed", False)),
            supports_emotion=bool(d.get("supports_emotion", False)),
            native_sample_rate=d.get("native_sample_rate"),
            determinism=Determinism(d.get("determinism", "best_effort")),
        )


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass
class UtteranceRecord:
    """Per-utterance row: what was asked, what came back, and how it scored.

    Persisted in full (not just aggregated) for three reasons the acceptance
    criteria require: reproducibility auditing, regenerating reports without
    re-synthesis, and paired statistical comparison across model versions.
    """

    result: SynthesisResult
    metrics: dict[str, MetricValue] = field(default_factory=dict)

    @property
    def utterance_id(self) -> str:
        return self.result.request.utterance_id

    @property
    def language(self) -> str:
        return self.result.request.language

    def metric(self, name: str) -> MetricValue | None:
        return self.metrics.get(name)

    def value(self, name: str) -> float | None:
        m = self.metrics.get(name)
        return m.value if m is not None and m.status is MetricStatus.OK else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result.to_dict(),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "UtteranceRecord":
        return cls(
            result=SynthesisResult.from_dict(d["result"]),
            metrics={k: MetricValue.from_dict(v) for k, v in (d.get("metrics") or {}).items()},
        )


@dataclass
class LanguageCoverage:
    """Verified per-language support.

    "Language Coverage" in the acceptance criteria cannot be answered by a model
    card alone — a card can claim 22 languages while the model emits noise for
    six of them. A language is ``verified`` only if synthesis succeeded *and*
    intelligibility passed the configured threshold *and* the audio was not
    degenerate.
    """

    language: str
    claimed: bool
    attempted: int
    succeeded: int
    intelligible: int | None  # None when no ASR backend was available
    verified: bool
    notes: str | None = None

    @property
    def success_rate(self) -> float | None:
        return (self.succeeded / self.attempted) if self.attempted else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "claimed": self.claimed,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "intelligible": self.intelligible,
            "verified": self.verified,
            "success_rate": _clean_float(self.success_rate, 4),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LanguageCoverage":
        return cls(
            language=str(d["language"]),
            claimed=bool(d.get("claimed", False)),
            attempted=int(d.get("attempted") or 0),
            succeeded=int(d.get("succeeded") or 0),
            intelligible=d.get("intelligible"),
            verified=bool(d.get("verified", False)),
            notes=d.get("notes"),
        )


@dataclass
class SubjectiveScore:
    """One human rating, from an ingested listening-test sheet."""

    utterance_id: str
    rater_id: str
    scale: str  # "mos" | "mushra" | "cmos" | "smos"
    score: float
    system: str | None = None  # blinded system label resolved at ingest time
    comment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "rater_id": self.rater_id,
            "scale": self.scale,
            "score": _clean_float(self.score, 4),
            "system": self.system,
            "comment": self.comment,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SubjectiveScore":
        return cls(
            utterance_id=str(d["utterance_id"]),
            rater_id=str(d["rater_id"]),
            scale=str(d.get("scale") or "mos"),
            score=float(d["score"]),
            system=d.get("system"),
            comment=d.get("comment"),
        )


@dataclass
class RunRecord:
    """The unit of storage and comparison: one evaluation of one model.

    ``fingerprint`` is the reproducibility key. It covers dataset content,
    model card, generation params, seed, adapter identity and framework version
    — everything that could change a number — but deliberately excludes the run
    id, timestamps and machine, so repeating the same evaluation yields the same
    fingerprint and is recognised as a repeat rather than a new experiment.
    """

    run_id: str
    schema_version: int
    framework_version: str
    created_at: str
    finished_at: str | None
    label: str

    # --- identity of what was evaluated ---
    model_id: str
    model_version: str
    provider: str
    adapter: str
    model_card: dict[str, Any]
    capabilities: Capabilities
    generation_params: dict[str, Any]
    seed: int | None
    determinism: Determinism

    # --- identity of how it was evaluated ---
    dataset_id: str
    dataset_version: str
    dataset_hash: str
    dataset_size: int
    metric_backends: dict[str, str]  # backend name -> resolved version/"absent"
    concurrency: int
    fingerprint: str
    environment: dict[str, Any]

    # --- results ---
    utterances: list[UtteranceRecord] = field(default_factory=list)
    aggregates: dict[str, Aggregate] = field(default_factory=dict)
    per_language: dict[str, dict[str, Aggregate]] = field(default_factory=dict)
    coverage: list[LanguageCoverage] = field(default_factory=list)
    subjective: list[SubjectiveScore] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ---- convenience -----------------------------------------------------
    @property
    def n_ok(self) -> int:
        return sum(1 for u in self.utterances if u.result.ok)

    @property
    def n_failed(self) -> int:
        return len(self.utterances) - self.n_ok

    @property
    def success_rate(self) -> float | None:
        return (self.n_ok / len(self.utterances)) if self.utterances else None

    @property
    def display_name(self) -> str:
        return f"{self.model_id}@{self.model_version}"

    def aggregate(self, name: str) -> Aggregate | None:
        return self.aggregates.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "framework_version": self.framework_version,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "label": self.label,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "provider": self.provider,
            "adapter": self.adapter,
            "model_card": self.model_card,
            "capabilities": self.capabilities.to_dict(),
            "generation_params": self.generation_params,
            "seed": self.seed,
            "determinism": self.determinism.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "dataset_size": self.dataset_size,
            "metric_backends": self.metric_backends,
            "concurrency": self.concurrency,
            "fingerprint": self.fingerprint,
            "environment": self.environment,
            "summary": {
                "n_utterances": len(self.utterances),
                "n_ok": self.n_ok,
                "n_failed": self.n_failed,
                "success_rate": _clean_float(self.success_rate, 4),
            },
            "aggregates": {k: v.to_dict() for k, v in self.aggregates.items()},
            "per_language": {
                lang: {k: v.to_dict() for k, v in aggs.items()}
                for lang, aggs in self.per_language.items()
            },
            "coverage": [c.to_dict() for c in self.coverage],
            "subjective": [s.to_dict() for s in self.subjective],
            "warnings": list(self.warnings),
            "utterances": [u.to_dict() for u in self.utterances],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunRecord":
        return cls(
            run_id=str(d["run_id"]),
            schema_version=int(d.get("schema_version") or 1),
            framework_version=str(d.get("framework_version") or "unknown"),
            created_at=str(d.get("created_at") or ""),
            finished_at=d.get("finished_at"),
            label=str(d.get("label") or ""),
            model_id=str(d.get("model_id") or "unknown"),
            model_version=str(d.get("model_version") or "unknown"),
            provider=str(d.get("provider") or "unknown"),
            adapter=str(d.get("adapter") or "unknown"),
            model_card=dict(d.get("model_card") or {}),
            capabilities=Capabilities.from_dict(d.get("capabilities") or {}),
            generation_params=dict(d.get("generation_params") or {}),
            seed=d.get("seed"),
            determinism=Determinism(d.get("determinism", "best_effort")),
            dataset_id=str(d.get("dataset_id") or "unknown"),
            dataset_version=str(d.get("dataset_version") or "0"),
            dataset_hash=str(d.get("dataset_hash") or ""),
            dataset_size=int(d.get("dataset_size") or 0),
            metric_backends=dict(d.get("metric_backends") or {}),
            concurrency=int(d.get("concurrency") or 1),
            fingerprint=str(d.get("fingerprint") or ""),
            environment=dict(d.get("environment") or {}),
            utterances=[UtteranceRecord.from_dict(u) for u in d.get("utterances") or []],
            aggregates={k: Aggregate.from_dict(v) for k, v in (d.get("aggregates") or {}).items()},
            per_language={
                lang: {k: Aggregate.from_dict(v) for k, v in aggs.items()}
                for lang, aggs in (d.get("per_language") or {}).items()
            },
            coverage=[LanguageCoverage.from_dict(c) for c in d.get("coverage") or []],
            subjective=[SubjectiveScore.from_dict(s) for s in d.get("subjective") or []],
            warnings=list(d.get("warnings") or []),
        )


# ---------------------------------------------------------------------------
# json hygiene
# ---------------------------------------------------------------------------
def _clean_float(v: Any, ndigits: int = 3) -> float | None:
    """Round for readable JSON and convert non-finite values to null.

    NaN/Inf are not valid JSON; letting them through produces files that some
    parsers accept and others reject, which is the worst outcome for a format
    meant to be a durable record.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, ndigits)


def _round_opt(v: float | None, ndigits: int = 3) -> float | None:
    return _clean_float(v, ndigits)


def _json_safe(obj: Any) -> Any:
    """Recursively coerce numpy scalars/arrays and non-finite floats to JSON.

    Metric ``extra`` payloads routinely carry numpy values; without this the
    store fails at write time, after the expensive synthesis has already run.
    """
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (str, bytes, bool)) or obj is None:
        return obj.decode("utf-8", "replace") if isinstance(obj, bytes) else obj
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, float):
        return _clean_float(obj, 6)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, Sequence):
        return [_json_safe(x) for x in obj]
    return str(obj)
