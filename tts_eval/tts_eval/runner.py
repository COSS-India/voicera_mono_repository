"""Run orchestration: synthesise, score, aggregate, fingerprint.

The reproducibility contract lives here. ``compute_fingerprint`` hashes exactly the
inputs that can change a number — dataset content, model identity, generation
params, seed, voice, concurrency, metric set, thresholds — and nothing that cannot
(run id, timestamps, hostname, output paths). Two runs with equal fingerprints are
therefore *supposed* to produce equal results, which is what makes "results are
reproducible using identical test inputs" a checkable claim rather than a wish:
``tts-eval verify`` re-runs a fingerprint and reports the drift.

Concurrency is part of the measurement, not a speed knob. Latency at concurrency 1
and at concurrency 8 are different quantities, so the value is recorded and folded
into the fingerprint, and the comparison engine refuses to treat runs at different
concurrency as equivalent.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import SCHEMA_VERSION, __version__
from .adapters import build_adapter, load_adapter_module
from .adapters.base import TTSAdapter
from .asr import build_asr
from .audio import write_wav
from .config import ModelCard, SuiteConfig
from .datasets.loader import TestCase, TestDataset, load_dataset
from .errors import AdapterUnavailable, ConfigError
from .metrics import (
    MetricContext,
    MetricEngine,
    aggregate_per_category,
    aggregate_per_language,
    aggregate_records,
    aggregate_run_values,
    build_backends,
    resolve_backend_names,
)
from .metrics.coverage import build_coverage
from .types import (
    Determinism,
    RunRecord,
    SynthesisRequest,
    UtteranceRecord,
)

# Progress callback: (completed, total, last_utterance_id).
ProgressFn = Callable[[int, int, str], None]


@dataclass
class RunPlan:
    """A fully resolved, ready-to-execute evaluation.

    Resolution happens before any synthesis so a bad config fails in
    milliseconds rather than after a 30-minute run.
    """

    card: ModelCard
    suite: SuiteConfig
    dataset: TestDataset
    backend_names: list[str]
    label: str
    output_dir: Path
    save_audio: bool = True
    adapter: TTSAdapter | None = None

    @property
    def voice(self) -> str | None:
        # Suite choice wins over the card default so one protocol can pin a voice
        # across models; falls back to the card, then to whatever the model does
        # by default.
        return self.suite.voice or self.card.default_voice

    @property
    def generation_params(self) -> dict[str, Any]:
        # Suite overrides card: the protocol decides sampling temperature, so two
        # models are compared under the same generation regime.
        return {**self.card.generation_params, **self.suite.generation_params}


def build_plan(
    card: ModelCard,
    suite: SuiteConfig,
    *,
    label: str | None = None,
    output_dir: str | Path = "runs",
    dataset: TestDataset | None = None,
    save_audio: bool | None = None,
) -> RunPlan:
    """Resolve card + suite into an executable plan, validating as we go."""
    if card.adapter_module:
        load_adapter_module(card.adapter_module)

    ds = dataset if dataset is not None else load_dataset(suite.dataset)
    if suite.languages or suite.categories:
        ds = ds.filter(languages=suite.languages or None, categories=suite.categories or None)
    if suite.sample:
        ds = ds.sample(suite.sample, seed=suite.seed or 0)

    backend_names = resolve_backend_names(suite.metrics)

    # Fail early on a voice the card does not declare. Providers silently fall back
    # to a default voice, which would produce a run labelled with one voice and
    # measured on another — and voice consistency would then look excellent for the
    # wrong reason.
    voice = suite.voice or card.default_voice
    if voice and card.voices and voice not in card.voices:
        raise ConfigError(
            f"voice {voice!r} is not declared by model card {card.display_name} "
            f"(declared: {', '.join(card.voices) or 'none'})"
        )

    return RunPlan(
        card=card,
        suite=suite,
        dataset=ds,
        backend_names=backend_names,
        label=label or f"{card.display_name} / {suite.suite_id}",
        output_dir=Path(output_dir),
        save_audio=suite.save_audio if save_audio is None else save_audio,
    )


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def compute_fingerprint(plan: RunPlan) -> tuple[str, dict[str, Any]]:
    """Return ``(fingerprint, inputs)``.

    ``inputs`` is stored alongside the hash so a mismatch is diagnosable: you can
    diff two runs' inputs and see *which* field changed instead of comparing two
    opaque hex strings.

    Only the framework's MAJOR.MINOR is included. A patch release must not
    invalidate every stored fingerprint, but a change to metric definitions (minor)
    should, because the numbers are no longer the same measurement.
    """
    major_minor = ".".join(__version__.split(".")[:2])
    inputs = {
        "framework": major_minor,
        "schema": SCHEMA_VERSION,
        "dataset_id": plan.dataset.id,
        "dataset_version": plan.dataset.version,
        # Content hash, not manifest hash: editing a comment must not break
        # comparability, editing a sentence must.
        "dataset_content_hash": plan.dataset.content_hash,
        "model_id": plan.card.model_id,
        "model_version": plan.card.model_version,
        "adapter": plan.card.adapter,
        "voice": plan.voice,
        "generation_params": plan.generation_params,
        "seed": plan.suite.seed,
        "concurrency": plan.suite.concurrency,
        "metric_backends": sorted(plan.backend_names),
        "thresholds": plan.suite.thresholds.to_dict(),
        "asr": {k: v for k, v in sorted(plan.suite.asr.items()) if "key" not in k.lower()},
    }
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32], inputs


def _environment() -> dict[str, Any]:
    """Machine facts recorded for auditing but excluded from the fingerprint.

    Latency depends on hardware, so a reader needs to know where a run happened;
    including it in the fingerprint would however mean the same evaluation on two
    machines counted as two different experiments, defeating repeat detection.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "tts_eval_version": __version__,
    }


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def _requests_for(plan: RunPlan) -> list[tuple[TestCase, SynthesisRequest]]:
    voice = plan.voice
    params = plan.generation_params
    out: list[tuple[TestCase, SynthesisRequest]] = []
    for case in plan.dataset:
        out.append(
            (
                case,
                SynthesisRequest(
                    utterance_id=case.id,
                    text=case.text,
                    language=case.language,
                    # A case-pinned voice wins, so multi-voice consistency tests
                    # work inside a single run.
                    voice=case.voice or voice,
                    seed=plan.suite.seed,
                    params={**params, **dict(case.params)},
                    reference_audio=case.reference_audio,
                ),
            )
        )
    return out


async def execute(
    plan: RunPlan, *, progress: ProgressFn | None = None, run_id: str | None = None
) -> RunRecord:
    """Run the evaluation and return a complete, storable record.

    Ordering matters and is deliberate:

    1.  Resolve metric-backend availability **before** synthesis, so a missing
        MOS model is reported in the header rather than discovered on utterance 1.
    2.  Probe the adapter, so an unreachable endpoint aborts in a second.
    3.  Synthesise and score per utterance, freeing each audio buffer as soon as
        it has been written and scored — a 1000-utterance run must not hold every
        waveform in memory.
    4.  Score run-level metrics, which read the per-utterance results.
    """
    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    fingerprint, fingerprint_inputs = compute_fingerprint(plan)
    started_at = datetime.now(timezone.utc)

    run_dir = plan.output_dir / run_id
    audio_dir = run_dir / "audio"
    if plan.save_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    adapter = plan.adapter or build_adapter(plan.card.adapter, plan.card.resolved_adapter_config())

    asr = build_asr(plan.suite.asr)
    ctx = MetricContext(
        dataset=plan.dataset,
        capabilities=adapter.capabilities,
        thresholds=plan.suite.thresholds,
        workdir=run_dir / "cache",
        asr=asr,
        dataset_dir=(plan.dataset.source_path.parent if plan.dataset.source_path else None),
        options={"concurrency": plan.suite.concurrency, "run_id": run_id},
    )
    engine = MetricEngine(build_backends(plan.backend_names, plan.suite.metric_options), ctx)

    warnings: list[str] = []
    if plan.card.model_version == "unversioned":
        warnings.append(
            "model card has no `model_version`; runs of different builds will be "
            "indistinguishable when comparing"
        )
    if plan.suite.seed is None and adapter.capabilities.determinism is not Determinism.DETERMINISTIC:
        warnings.append(
            "no seed set and the provider samples stochastically: metrics are reproducible "
            "only within their confidence intervals, and audio is not reproducible at all"
        )
    if plan.suite.seed is not None and not adapter.capabilities.supports_seed:
        warnings.append(
            f"seed {plan.suite.seed} was requested but the model card does not declare "
            "`supports_seed`; the provider will ignore it and output will vary between runs"
        )

    records: list[UtteranceRecord] = []
    work = _requests_for(plan)
    total = len(work)
    wall_start = time.perf_counter()

    engine.prepare()

    try:
        await adapter.aopen()
        try:
            await adapter.probe()
        except AdapterUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 - probe must not fail for a novel reason
            warnings.append(f"adapter probe raised {type(e).__name__}: {e}")

        semaphore = asyncio.Semaphore(plan.suite.concurrency)
        # Scoring is CPU-bound numpy and (optionally) a blocking ASR call, so it
        # runs in a thread: doing it inline on the event loop would serialise
        # scoring against synthesis and inflate every latency measurement of the
        # utterances still in flight.
        completed = 0
        lock = asyncio.Lock()

        async def one(index: int, case: TestCase, request: SynthesisRequest) -> tuple[int, UtteranceRecord]:
            nonlocal completed
            async with semaphore:
                result = await adapter.synthesize(request)

            if result.ok and result.audio is not None and plan.save_audio:
                path = audio_dir / f"{case.id}.wav"
                await asyncio.to_thread(write_wav, path, result.audio)
                result.audio_path = str(path)

            metrics = await asyncio.to_thread(engine.score_utterance, case, result)
            record = UtteranceRecord(result=result, metrics=dict(metrics))
            # Drop the waveform now that it is persisted and scored. Everything
            # downstream (reports, comparison, subjective bundles) reads the WAV.
            record.result.audio = None

            async with lock:
                completed += 1
                if progress is not None:
                    progress(completed, total, case.id)
            return index, record

        gathered = await asyncio.gather(
            *(one(i, case, request) for i, (case, request) in enumerate(work))
        )
        # Restore dataset order: gather returns in completion order, and a stable
        # record order is what makes two run files diffable.
        records = [rec for _, rec in sorted(gathered, key=lambda pair: pair[0])]
    finally:
        await adapter.aclose()

    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    ctx.options["wall_clock_ms"] = wall_ms

    run_values = engine.score_run(records)
    engine.teardown()

    expected = engine.expected_metrics
    utterance_metrics = [m for m in expected if _is_utterance_metric(m, records)]
    aggregates = aggregate_records(records, utterance_metrics)
    aggregates.update(aggregate_run_values(run_values))

    cases_by_id = {c.id: c.category for c in plan.dataset}
    per_language = aggregate_per_language(records, utterance_metrics)
    per_category = aggregate_per_category(records, cases_by_id, utterance_metrics)

    coverage = build_coverage(records, ctx)
    warnings.extend(w for w in ctx.warnings if w not in warnings)

    record = RunRecord(
        run_id=run_id,
        schema_version=SCHEMA_VERSION,
        framework_version=__version__,
        created_at=started_at.isoformat(),
        finished_at=datetime.now(timezone.utc).isoformat(),
        label=plan.label,
        model_id=plan.card.model_id,
        model_version=plan.card.model_version,
        provider=plan.card.provider,
        adapter=plan.card.adapter,
        model_card=plan.card.to_dict(),
        capabilities=adapter.capabilities,
        generation_params=plan.generation_params,
        seed=plan.suite.seed,
        determinism=adapter.capabilities.determinism,
        dataset_id=plan.dataset.id,
        dataset_version=plan.dataset.version,
        dataset_hash=plan.dataset.content_hash,
        dataset_size=len(plan.dataset),
        metric_backends=dict(engine.backend_status),
        concurrency=plan.suite.concurrency,
        fingerprint=fingerprint,
        environment={
            **_environment(),
            "wall_clock_ms": round(wall_ms, 2),
            "fingerprint_inputs": fingerprint_inputs,
            "suite": plan.suite.to_dict(),
            "dataset_manifest_hash": plan.dataset.manifest_hash,
            "asr": asr.describe() if asr is not None else None,
            "audio_saved": plan.save_audio,
            "per_category": {
                cat: {k: v.to_dict() for k, v in aggs.items()}
                for cat, aggs in per_category.items()
            },
        },
        utterances=records,
        aggregates=aggregates,
        per_language=per_language,
        coverage=coverage,
        warnings=warnings,
    )
    return record


def _is_utterance_metric(name: str, records: Sequence[UtteranceRecord]) -> bool:
    """True when any utterance carries this metric.

    Checked against the data rather than the catalogue so a plugin emitting an
    undeclared per-utterance metric is still aggregated instead of silently
    dropped.
    """
    from .metrics.catalog import spec as metric_spec

    if any(name in rec.metrics for rec in records):
        return True
    return metric_spec(name).scope == "utterance"


def run_sync(plan: RunPlan, *, progress: ProgressFn | None = None) -> RunRecord:
    """Blocking wrapper for the CLI."""
    return asyncio.run(execute(plan, progress=progress))


__all__ = [
    "RunPlan",
    "build_plan",
    "compute_fingerprint",
    "execute",
    "run_sync",
]
