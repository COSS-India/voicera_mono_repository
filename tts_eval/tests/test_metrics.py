"""The metric layer: catalogue coverage, failure detection, honest missing data.

Three properties, one file because they all exercise the metric engine:

* every acceptance criterion has metrics and every metric declares its polarity;
* each silent-failure mode (loop, truncation, silence, clipping) is caught;
* an absent backend degrades to a reason, never a crash or a clean-looking zero.
"""
from __future__ import annotations

import numpy as np
import pytest

from tts_eval.audio import AudioBuffer, write_wav
from tts_eval.config import load_model_card, load_suite
from tts_eval.datasets import load_dataset
from tts_eval.metrics import CATALOG, ac_matrix, available_backends, resolve_backend_names
from tts_eval.metrics.aggregate import summarise
from tts_eval.metrics.catalog import criteria_order
from tts_eval.runner import build_plan, run_sync
from tts_eval.types import MetricStatus

DATASET = "indic_conversational_v1"


def _spec_and_hop(samples, sample_rate, hop_ms=10.0):
    """Spectrogram plus the hop in seconds, as _self_similarity expects."""
    from tts_eval.audio import magnitude_spectrogram

    spec, _freqs = magnitude_spectrogram(samples, sample_rate, hop_ms=hop_ms)
    return spec, hop_ms / 1000.0


class TestCatalogue:
    def test_every_acceptance_criterion_has_metrics(self):
        """Mechanical check of "the evaluation captures, at a minimum, ..."."""
        covered = {spec.criterion for spec in CATALOG.values()}
        for criterion in criteria_order():
            assert criterion in covered, f"no metric serves {criterion!r}"

    def test_every_metric_declares_polarity_and_scope(self):
        for name, spec in CATALOG.items():
            assert spec.scope in ("utterance", "run"), name
            assert spec.direction is not None, name

    def test_ac_matrix_marks_availability(self):
        rows = ac_matrix(available_backends())
        assert rows and all("criterion" in r for r in rows)
        assert any(r["available"] for r in rows)

    def test_tiers_resolve_to_registered_backends(self):
        for tier in ("core", "standard", "all"):
            names = resolve_backend_names(tier)
            assert names and set(names) <= set(available_backends())

    def test_core_is_a_subset_of_standard(self):
        assert set(resolve_backend_names("core")) <= set(resolve_backend_names("standard"))

    def test_unknown_backend_fails_before_a_run_starts(self):
        with pytest.raises(KeyError, match="unknown metric backend"):
            resolve_backend_names(["not_a_backend"])


class TestDegeneracyDetection:
    """The characteristic autoregressive-TTS failures, each detected."""

    def _score(self, tmp_path, **overrides):
        card = load_model_card("mock")
        card.adapter_config.update({"ttfb_ms": 0, **overrides})
        suite = load_suite("smoke")
        record = run_sync(build_plan(card, suite, output_dir=tmp_path / "runs"))
        return record

    def test_clean_output_is_not_flagged(self, tmp_path):
        record = self._score(tmp_path)
        assert record.aggregates["degenerate_rate"].mean == 0.0
        assert record.aggregates["audio_quality_score"].mean > 0.85

    def test_looping_is_detected(self, tmp_path):
        record = self._score(tmp_path, degenerate_rate=1.0)
        offenders = [
            u for u in record.utterances
            if "repetitive" in (u.metrics["degeneracy_score"].detail or "")
        ]
        assert offenders, "loop detector did not fire"

        # Utterances too short to hold two loop periods never received the fault; the
        # mock reports that rather than pretending. Every utterance that DID get one
        # must be flagged.
        with_fault = [
            u for u in record.utterances
            if u.result.provider_meta.get("fault_applied") is True
        ]
        assert with_fault
        for utterance in with_fault:
            assert (utterance.value("degeneracy_score") or 0) > 0.5, (
                f"{utterance.utterance_id} was given a loop and not flagged"
            )

        # And where a check could not run, the score must say so instead of reading
        # as a verified-clean 0.0.
        for utterance in record.utterances:
            if (utterance.value("degeneracy_score") or 0) <= 0.5:
                assert utterance.result.provider_meta.get("fault_applied") is False

    def test_truncation_is_detected(self, tmp_path):
        """Including short utterances, which need the affine duration model."""
        record = self._score(tmp_path, truncate_rate=1.0)
        assert record.aggregates["degenerate_rate"].mean == 1.0
        assert record.aggregates["length_ratio"].mean < 0.6

    def test_short_utterance_truncation_is_detected(self, tmp_path):
        """Regression guard: a purely proportional duration model missed this."""
        record = self._score(tmp_path, truncate_rate=1.0)
        short = next(u for u in record.utterances if u.utterance_id == "en-edge-short-01")
        assert short.value("length_ratio") < 0.55
        assert short.value("degeneracy_score") > 0.5

    def test_silent_output_is_detected(self, tmp_path):
        record = self._score(tmp_path, silent_rate=1.0)
        assert record.aggregates["degenerate_rate"].mean == 1.0
        assert record.aggregates["audio_quality_score"].mean == 0.0

    def test_clipping_is_measured(self, tmp_path):
        record = self._score(tmp_path, clip_rate=1.0)
        assert record.aggregates["clipping_pct"].mean > 0.5

    def test_hard_failures_lower_success_rate(self, tmp_path):
        record = self._score(tmp_path, fail_rate=1.0)
        assert record.aggregates["success_rate"].mean == 0.0
        assert record.n_ok == 0

    def test_loop_detector_on_synthetic_signals(self):
        """The three cases the detector must tell apart.

        A held tone is self-similar at every lag, so a naive measure calls it a loop;
        it has no spectral trajectory at all and must be reported as unassessable.
        White noise has a trajectory but no repetition. A tiled segment is a real loop.
        """
        from tts_eval.metrics.audio_quality import _self_similarity

        sr = 24000
        n = 3 * sr
        t = np.arange(n) / sr

        tone = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        similarity, _lag, _contrast = _self_similarity(*_spec_and_hop(tone, sr))
        assert similarity is None, "a static spectrum must be unassessable, not a loop"

        noise = (0.1 * np.random.default_rng(0).standard_normal(n)).astype(np.float32)
        similarity, _lag, _contrast = _self_similarity(*_spec_and_hop(noise, sr))
        assert similarity is not None and similarity < 0.3

        loop_len = int(0.42 * sr)
        sweep = np.arange(loop_len) / sr
        segment = (
            0.3 * np.sin(2 * np.pi * (200 + 400 * np.sin(2 * np.pi * 3 * sweep)) * sweep)
        ).astype(np.float32)
        tiled = np.tile(segment, 8)[:n]
        similarity, lag, _contrast = _self_similarity(*_spec_and_hop(tiled, sr))
        assert similarity is not None and similarity > 0.9
        assert lag == pytest.approx(0.42, abs=0.02), "loop period must be recovered"


class TestMissingDataIsExplicit:
    def test_absent_backend_yields_reasons_not_crashes(self, tmp_path):
        """`--metrics all` on a machine without torch must still complete."""
        card = load_model_card("mock")
        suite = load_suite("smoke")
        suite.metrics = "all"
        record = run_sync(build_plan(card, suite, output_dir=tmp_path / "runs"))

        assert record.n_ok == len(record.utterances)
        absent = [n for n, s in record.metric_backends.items() if s.startswith("absent")]
        assert absent, "expected at least one optional backend to be unavailable here"
        for name in absent:
            assert record.metric_backends[name].startswith("absent: ")

    def test_missing_metrics_carry_a_reason(self, tmp_path):
        card = load_model_card("mock")
        suite = load_suite("smoke")
        suite.metrics = "all"
        record = run_sync(build_plan(card, suite, output_dir=tmp_path / "runs"))
        empty = [a for a in record.aggregates.values() if a.n == 0]
        for aggregate in empty:
            assert aggregate.missing_reason, f"{aggregate.name} is missing with no reason given"

    def test_intelligibility_without_asr_reports_why(self, tmp_path):
        card = load_model_card("mock")
        suite = load_suite("smoke")
        suite.metrics = ["latency", "intelligibility"]
        record = run_sync(build_plan(card, suite, output_dir=tmp_path / "runs"))
        assert "no ASR backend configured" in record.metric_backends["intelligibility"]
        assert record.aggregates["cer"].n == 0

    def test_coverage_notes_unverified_intelligibility(self, tmp_path):
        """Without ASR, "synthesised fine" must not silently mean "supported"."""
        card = load_model_card("mock")
        record = run_sync(build_plan(card, load_suite("smoke"), output_dir=tmp_path / "runs"))
        assert all("intelligibility unverified" in (c.notes or "") for c in record.coverage)

    def test_non_streaming_marks_jitter_not_applicable(self, tmp_path):
        sr = 24000
        audio_dir = tmp_path / "audio"
        ds = load_dataset(DATASET).sample(13, seed=1)
        for case in ds:
            write_wav(
                audio_dir / f"{case.id}.wav",
                AudioBuffer(
                    samples=(0.3 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32),
                    sample_rate=sr,
                ),
            )
        card = load_model_card("mock")
        card.adapter = "replay"
        card.adapter_config = {"audio_dir": str(audio_dir)}
        record = run_sync(
            build_plan(card, load_suite("smoke"), output_dir=tmp_path / "runs", dataset=ds)
        )
        jitter = record.utterances[0].metrics["stream_chunk_gap_p95_ms"]
        assert jitter.status is MetricStatus.NOT_APPLICABLE

    def test_replayed_latency_is_not_invented(self, tmp_path):
        sr = 24000
        audio_dir = tmp_path / "audio"
        ds = load_dataset(DATASET).sample(13, seed=1)
        for case in ds:
            write_wav(
                audio_dir / f"{case.id}.wav",
                AudioBuffer(samples=np.full(sr, 0.2, dtype=np.float32), sample_rate=sr),
            )
        card = load_model_card("mock")
        card.adapter = "replay"
        card.adapter_config = {"audio_dir": str(audio_dir)}
        record = run_sync(
            build_plan(card, load_suite("smoke"), output_dir=tmp_path / "runs", dataset=ds)
        )
        ttfb = record.utterances[0].metrics["ttfb_ms"]
        assert ttfb.status is MetricStatus.NOT_APPLICABLE
        assert "no original timings" in (ttfb.detail or "")

    def test_aggregate_of_nothing_is_not_zero(self):
        aggregate = summarise("cer", [], n_missing=5, missing_reason="no ASR")
        assert aggregate.n == 0 and aggregate.mean is None
        assert aggregate.missing_reason == "no ASR"
