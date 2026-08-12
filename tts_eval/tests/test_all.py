"""Test suite for the parts of tts_eval built so far.

Organised by the property being defended rather than by module, because the
properties are what the acceptance criteria ask for: reproducibility, honest
missing-data handling, failure detection, and defensible comparison.
"""
from __future__ import annotations

import asyncio
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tts_eval.adapters import available_adapters, build_adapter
from tts_eval.asr.base import (
    character_error_rate,
    normalise_text,
    slot_hits,
    word_error_rate,
)
from tts_eval.audio import AudioBuffer, estimate_f0, read_wav, resample, write_wav
from tts_eval.compare import compare_runs, direction_of_change, repeatability
from tts_eval.config import list_model_cards, list_suites, load_model_card, load_suite
from tts_eval.datasets import dataset_from_cases, load_dataset
from tts_eval.errors import ConfigError, DatasetError
from tts_eval.metrics import CATALOG, ac_matrix, available_backends, resolve_backend_names
from tts_eval.metrics.aggregate import summarise
from tts_eval.metrics.base import MetricContext, MetricEngine, build_backends
from tts_eval.metrics.catalog import criteria_order
from tts_eval.runner import build_plan, compute_fingerprint, run_sync
from tts_eval.store import RunStore
from tts_eval.subjective import TestSpec, build_test, ingest_sheets, merge_into_run
from tts_eval.types import MetricStatus, SynthesisRequest

DATASET = "indic_conversational_v1"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_card():
    return load_model_card("mock")


@pytest.fixture
def smoke_suite():
    return load_suite("smoke")


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "runs")


def _spec_and_hop(samples, sample_rate, hop_ms=10.0):
    """Spectrogram plus the hop in seconds, as _self_similarity expects."""
    from tts_eval.audio import magnitude_spectrogram

    spec, _freqs = magnitude_spectrogram(samples, sample_rate, hop_ms=hop_ms)
    return spec, hop_ms / 1000.0


def _run(card, suite, tmp_path, label="test", **card_overrides):
    card = copy.deepcopy(card)
    card.adapter_config.update(card_overrides)
    return run_sync(build_plan(card, suite, output_dir=tmp_path / "runs", label=label))


# ---------------------------------------------------------------------------
# dataset identity and reproducibility
# ---------------------------------------------------------------------------
class TestDatasetIdentity:
    def test_builtin_loads_with_pinned_hash(self):
        ds = load_dataset(DATASET)
        assert len(ds) == 69
        assert len(ds.languages) == 13
        assert ds.version == "1.0.0"

    def test_content_hash_is_stable_across_loads(self):
        assert load_dataset(DATASET).content_hash == load_dataset(DATASET).content_hash

    def test_annotation_change_moves_manifest_hash_but_not_content_hash(self):
        """The whole point of two hashes: comments must not break comparability."""
        base = dataset_from_cases([{"id": "a", "text": "hello", "language": "en"}])
        annotated = dataset_from_cases(
            [{"id": "a", "text": "hello", "language": "en", "notes": "reviewed", "category": "greeting"}]
        )
        assert base.content_hash == annotated.content_hash
        assert base.manifest_hash != annotated.manifest_hash

    def test_text_change_moves_content_hash(self):
        a = dataset_from_cases([{"id": "a", "text": "hello", "language": "en"}])
        b = dataset_from_cases([{"id": "a", "text": "hello there", "language": "en"}])
        assert a.content_hash != b.content_hash

    def test_expected_transcript_does_not_change_content_hash(self):
        """It affects scoring, not synthesis, so it must not invalidate comparisons."""
        a = dataset_from_cases([{"id": "a", "text": "12,450", "language": "en"}])
        b = dataset_from_cases(
            [{"id": "a", "text": "12,450", "language": "en",
              "expected_transcript": "twelve thousand four hundred fifty"}]
        )
        assert a.content_hash == b.content_hash
        assert b.cases[0].reference_text == "twelve thousand four hundred fifty"
        assert a.cases[0].reference_text == "12,450"

    def test_duplicate_ids_rejected(self):
        """Duplicates would break paired comparison and double-count aggregates."""
        with pytest.raises(DatasetError, match="duplicate case ids"):
            dataset_from_cases(
                [
                    {"id": "dup", "text": "one", "language": "en"},
                    {"id": "dup", "text": "two", "language": "en"},
                ]
            )

    def test_stratified_sample_is_deterministic_and_covers_languages(self):
        ds = load_dataset(DATASET)
        a, b = ds.sample(13, seed=1), ds.sample(13, seed=1)
        assert [c.id for c in a] == [c.id for c in b]
        assert len(a.languages) == 13, "one utterance per language expected"

    def test_sample_records_itself_in_the_version(self):
        ds = load_dataset(DATASET).sample(13, seed=1)
        assert "sample13" in ds.version

    def test_filter_records_itself_in_the_version(self):
        ds = load_dataset(DATASET).filter(languages=["hi"])
        assert "lang=hi" in ds.version
        assert ds.languages == ["hi"]

    def test_empty_filter_raises(self):
        with pytest.raises(DatasetError, match="removed every case"):
            load_dataset(DATASET).filter(languages=["klingon"])


# ---------------------------------------------------------------------------
# text scoring
# ---------------------------------------------------------------------------
class TestTextNormalisation:
    def test_native_digits_fold_to_ascii(self):
        assert "12 450" in normalise_text("আপনার ১২,৪৫০ টাকা")

    def test_indic_punctuation_stripped(self):
        assert normalise_text("नमस्ते, मैं ठीक हूँ।") == "नमस्ते मैं ठीक हूँ"

    def test_urdu_punctuation_stripped(self):
        assert "؟" not in normalise_text("کیا آپ ٹھیک ہیں؟")

    def test_identical_text_scores_zero(self):
        assert character_error_rate("नमस्ते जी", "नमस्ते जी").rate == 0.0
        assert word_error_rate("the quick fox", "the quick fox").rate == 0.0

    def test_cer_ignores_spacing_differences(self):
        """Indic ASR word segmentation is inconsistent; spacing must not count."""
        assert character_error_rate("नमस्ते जी", "नमस्तेजी").rate == 0.0

    def test_cer_counts_real_substitutions(self):
        assert character_error_rate("abcd", "abxd").rate == pytest.approx(0.25)

    def test_empty_reference_reports_nan_not_zero(self):
        """Zero would read as a perfect score for an unscoreable case."""
        assert np.isnan(character_error_rate("", "anything").rate)

    def test_slot_hit_tolerates_spelled_out_letters(self):
        hits, missing = slot_hits("please share the o t p now", ("OTP",))
        assert (hits, missing) == (1, [])

    def test_slot_miss_reported(self):
        hits, missing = slot_hits("please share the code", ("OTP",))
        assert (hits, missing) == (0, ["OTP"])


# ---------------------------------------------------------------------------
# audio helpers
# ---------------------------------------------------------------------------
class TestAudio:
    def test_wav_roundtrip_preserves_signal(self, tmp_path):
        sr = 24000
        samples = (0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
        path = write_wav(tmp_path / "a.wav", AudioBuffer(samples=samples, sample_rate=sr))
        back = read_wav(path)
        assert back.sample_rate == sr
        assert np.allclose(back.samples, samples, atol=1e-4)

    def test_resample_preserves_duration(self):
        buf = AudioBuffer(samples=np.zeros(24000, dtype=np.float32), sample_rate=24000)
        assert resample(buf, 16000).duration_s == pytest.approx(1.0, abs=1e-3)

    def test_f0_tracks_a_known_tone(self):
        sr = 16000
        t = np.arange(sr) / sr
        f0 = estimate_f0((0.6 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), sr)
        voiced = f0[np.isfinite(f0)]
        assert voiced.size > 5
        assert np.median(voiced) == pytest.approx(150, rel=0.05)

    def test_f0_reports_silence_as_unvoiced(self):
        f0 = estimate_f0(np.zeros(16000, dtype=np.float32), 16000)
        assert not np.isfinite(f0).any()

    def test_write_wav_clips_instead_of_wrapping(self, tmp_path):
        """Integer wraparound would turn a loud passage into a click we then score."""
        loud = np.array([3.0, -3.0, 0.0], dtype=np.float32)
        path = write_wav(tmp_path / "loud.wav", AudioBuffer(samples=loud, sample_rate=8000))
        back = read_wav(path).samples
        assert back.max() <= 1.0 and back.min() >= -1.0


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------
class TestAdapters:
    def test_all_builtin_adapters_registered(self):
        assert set(available_adapters()) == {"http_rest", "mock", "replay", "websocket_pcm"}

    def test_mock_is_bit_deterministic(self):
        adapter = build_adapter("mock", {"ttfb_ms": 0})
        req = SynthesisRequest(utterance_id="u1", text="नमस्ते", language="hi", voice="v", seed=7)
        a = asyncio.run(adapter.synthesize(req))
        b = asyncio.run(adapter.synthesize(req))
        assert np.array_equal(a.audio.samples, b.audio.samples)

    def test_different_seed_changes_audio(self):
        adapter = build_adapter("mock", {"ttfb_ms": 0})
        base = dict(utterance_id="u1", text="नमस्ते", language="hi", voice="v")
        a = asyncio.run(adapter.synthesize(SynthesisRequest(**base, seed=1)))
        b = asyncio.run(adapter.synthesize(SynthesisRequest(**base, seed=2)))
        assert not np.array_equal(a.audio.samples, b.audio.samples)

    def test_timings_are_populated_and_ordered(self):
        adapter = build_adapter("mock", {"ttfb_ms": 5})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u", text="hello there", language="en"))
        )
        assert result.ttfb_ms is not None and result.first_audible_ms is not None
        assert result.first_audible_ms >= result.ttfb_ms
        offsets = [c.offset_ms for c in result.chunk_timings]
        assert offsets == sorted(offsets)

    def test_first_audible_is_later_than_ttfb_when_padded(self):
        """The mock pads 30 ms of silence; TTFB alone would flatter it."""
        adapter = build_adapter("mock", {"ttfb_ms": 0})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u", text="hello", language="en"))
        )
        assert result.first_audible_ms > result.ttfb_ms

    def test_injected_failure_is_recorded_not_raised(self):
        adapter = build_adapter("mock", {"ttfb_ms": 0, "fail_rate": 1.0})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u", text="hi", language="en"))
        )
        assert not result.ok
        assert "injected synthesis failure" in result.error

    def test_unsupported_language_fails_that_utterance(self):
        adapter = build_adapter("mock", {"ttfb_ms": 0, "unsupported_languages": ["sat"]})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u", text="hi", language="sat"))
        )
        assert not result.ok and "not supported" in result.error

    def test_faults_are_disjoint(self):
        """Overlapping bands once made every utterance truncated AND clipped."""
        adapter = build_adapter("mock", {"ttfb_ms": 0, "truncate_rate": 1.0, "clip_rate": 0.0})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u", text="hello world", language="en"))
        )
        assert result.provider_meta.get("injected_fault") == "truncate"

    def test_silent_output_counts_as_failure_not_success(self):
        """A clean `done` with no audible samples is a failed request."""
        adapter = build_adapter("mock", {"ttfb_ms": 0, "silent_rate": 1.0})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u", text="hi", language="en"))
        )
        assert result.provider_meta.get("all_silent") is True

    def test_replay_reads_back_written_audio(self, tmp_path):
        sr = 24000
        samples = (0.3 * np.sin(2 * np.pi * 180 * np.arange(sr) / sr)).astype(np.float32)
        write_wav(tmp_path / "u1.wav", AudioBuffer(samples=samples, sample_rate=sr))
        adapter = build_adapter("replay", {"audio_dir": str(tmp_path)})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="u1", text="x", language="en"))
        )
        assert result.ok
        assert result.audio.n_samples == sr
        # No sidecar timings present, so latency must be absent, not invented.
        assert result.ttfb_ms is None

    def test_replay_missing_file_fails_cleanly(self, tmp_path):
        adapter = build_adapter("replay", {"audio_dir": str(tmp_path)})
        result = asyncio.run(
            adapter.synthesize(SynthesisRequest(utterance_id="nope", text="x", language="en"))
        )
        assert not result.ok and "no replay audio" in result.error

    def test_adapter_config_redacts_secrets(self):
        adapter = build_adapter("mock", {"api_key": "super-secret", "ttfb_ms": 0})
        assert adapter.describe()["config"]["api_key"] == "***redacted***"

    def test_http_rest_requires_url(self):
        with pytest.raises(ConfigError, match="requires 'url'"):
            build_adapter("http_rest", {})


# ---------------------------------------------------------------------------
# metric catalogue
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# failure detection
# ---------------------------------------------------------------------------
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
        from tts_eval.audio import magnitude_spectrogram
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


# ---------------------------------------------------------------------------
# honest missing data
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# runner / fingerprint
# ---------------------------------------------------------------------------
class TestFingerprint:
    def test_identical_inputs_give_identical_fingerprints(self, mock_card, smoke_suite, tmp_path):
        a = build_plan(mock_card, smoke_suite, output_dir=tmp_path)
        b = build_plan(mock_card, smoke_suite, output_dir=tmp_path / "elsewhere")
        assert compute_fingerprint(a)[0] == compute_fingerprint(b)[0]

    @pytest.mark.parametrize(
        "field,value",
        [("seed", 999), ("concurrency", 8), ("voice", "mock_male"), ("dataset", "x")],
    )
    def test_protocol_changes_move_the_fingerprint(
        self, mock_card, smoke_suite, tmp_path, field, value
    ):
        base = compute_fingerprint(build_plan(mock_card, smoke_suite, output_dir=tmp_path))[0]
        suite = copy.deepcopy(smoke_suite)
        if field == "dataset":
            plan = build_plan(
                mock_card, suite, output_dir=tmp_path,
                dataset=dataset_from_cases([{"id": "z", "text": "hi", "language": "en"}]),
            )
        else:
            setattr(suite, field, value)
            plan = build_plan(mock_card, suite, output_dir=tmp_path)
        assert compute_fingerprint(plan)[0] != base, f"{field} must affect the fingerprint"

    def test_model_version_moves_the_fingerprint(self, mock_card, smoke_suite, tmp_path):
        base = compute_fingerprint(build_plan(mock_card, smoke_suite, output_dir=tmp_path))[0]
        other = copy.deepcopy(mock_card)
        other.model_version = "99"
        assert compute_fingerprint(build_plan(other, smoke_suite, output_dir=tmp_path))[0] != base

    def test_output_dir_does_not_affect_the_fingerprint(self, mock_card, smoke_suite, tmp_path):
        a = compute_fingerprint(build_plan(mock_card, smoke_suite, output_dir=tmp_path / "a"))[0]
        b = compute_fingerprint(build_plan(mock_card, smoke_suite, output_dir=tmp_path / "b"))[0]
        assert a == b

    def test_fingerprint_inputs_are_stored_for_diffing(self, mock_card, smoke_suite, tmp_path):
        _fp, inputs = compute_fingerprint(build_plan(mock_card, smoke_suite, output_dir=tmp_path))
        assert {"dataset_content_hash", "model_version", "concurrency", "seed"} <= set(inputs)

    def test_undeclared_voice_is_rejected_early(self, mock_card, smoke_suite, tmp_path):
        suite = copy.deepcopy(smoke_suite)
        suite.voice = "no_such_voice"
        with pytest.raises(ConfigError, match="not declared by model card"):
            build_plan(mock_card, suite, output_dir=tmp_path)


class TestRunnerBehaviour:
    def test_run_produces_a_complete_record(self, mock_card, smoke_suite, tmp_path):
        record = run_sync(build_plan(mock_card, smoke_suite, output_dir=tmp_path / "runs"))
        assert len(record.utterances) == 13
        assert record.n_ok == 13
        assert record.success_rate == 1.0
        assert record.fingerprint and record.dataset_hash
        assert record.per_language and len(record.per_language) == 13
        assert len(record.coverage) == 13

    def test_utterance_order_is_stable(self, mock_card, smoke_suite, tmp_path):
        """Concurrency returns out of order; a stable file is what makes it diffable."""
        suite = copy.deepcopy(smoke_suite)
        suite.concurrency = 4
        a = run_sync(build_plan(mock_card, suite, output_dir=tmp_path / "a"))
        b = run_sync(build_plan(mock_card, suite, output_dir=tmp_path / "b"))
        assert [u.utterance_id for u in a.utterances] == [u.utterance_id for u in b.utterances]

    def test_audio_is_written_and_freed(self, mock_card, smoke_suite, tmp_path):
        record = run_sync(build_plan(mock_card, smoke_suite, output_dir=tmp_path / "runs"))
        for utterance in record.utterances:
            assert utterance.result.audio is None, "buffers must not be retained"
            assert Path(utterance.result.audio_path).is_file()

    def test_save_audio_false_writes_nothing(self, mock_card, smoke_suite, tmp_path):
        suite = copy.deepcopy(smoke_suite)
        suite.save_audio = False
        record = run_sync(build_plan(mock_card, suite, output_dir=tmp_path / "runs"))
        assert all(u.result.audio_path is None for u in record.utterances)

    def test_seed_ignored_by_provider_raises_a_warning(self, smoke_suite, tmp_path):
        card = load_model_card("mock")
        card.supports_seed = False
        record = run_sync(build_plan(card, smoke_suite, output_dir=tmp_path / "runs"))
        assert any("does not declare `supports_seed`" in w for w in record.warnings)

    def test_unversioned_card_raises_a_warning(self, smoke_suite, tmp_path):
        card = load_model_card("mock")
        card.model_version = "unversioned"
        record = run_sync(build_plan(card, smoke_suite, output_dir=tmp_path / "runs"))
        assert any("no `model_version`" in w for w in record.warnings)

    def test_claimed_but_untested_languages_are_warned(self, smoke_suite, tmp_path):
        card = load_model_card("mock")
        card.languages = card.languages + ("sat", "brx")
        record = run_sync(build_plan(card, smoke_suite, output_dir=tmp_path / "runs"))
        assert any("not present in this test set" in w for w in record.warnings)
        # And coverage must not credit them.
        assert record.aggregates["coverage_ratio"].mean < 1.0

    def test_concurrency_actually_overlaps_requests(self, mock_card, tmp_path):
        """Guards against the semaphore or gather being accidentally serialised.

        Only the latency backend is enabled: the DSP backends are CPU-bound and
        largely GIL-bound, so including them would measure scoring throughput rather
        than request overlap and the test would be near-meaningless.
        """
        import time

        suite = load_suite("smoke")
        suite.sample = 8
        suite.metrics = ["latency"]
        suite.save_audio = False
        card = copy.deepcopy(mock_card)
        # Pure await time, so overlap is the only thing that can shorten the run.
        card.adapter_config.update({"ttfb_ms": 200, "realtime_factor": 0.0})

        def timed(concurrency: int, tag: str) -> float:
            suite.concurrency = concurrency
            start = time.perf_counter()
            run_sync(build_plan(card, suite, output_dir=tmp_path / tag))
            return time.perf_counter() - start

        sequential = timed(1, "seq")
        parallel = timed(8, "par")
        # 8 x 200 ms serial vs one 200 ms wave. The margin is 2x rather than 8x
        # because per-request rendering and scoring stay partly GIL-bound; if the
        # semaphore or gather were serialised the ratio would sit near 1.0, which
        # this still catches.
        assert parallel < sequential / 2.0, f"{parallel:.2f}s vs {sequential:.2f}s"


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
class TestStore:
    def test_roundtrip_preserves_the_record(self, mock_card, smoke_suite, tmp_path, store):
        record = run_sync(build_plan(mock_card, smoke_suite, output_dir=store.root))
        store.save(record)
        back = store.load(record.run_id)
        assert back.run_id == record.run_id
        assert back.fingerprint == record.fingerprint
        assert len(back.utterances) == len(record.utterances)
        # Persisted values are rounded for readable JSON, so compare at that
        # precision rather than bit-for-bit.
        assert back.aggregates["ttfb_ms"].mean == pytest.approx(
            record.aggregates["ttfb_ms"].mean, abs=1e-3
        )
        assert [c.language for c in back.coverage] == [c.language for c in record.coverage]

    def test_json_is_the_source_of_truth_and_is_readable(self, mock_card, smoke_suite, store):
        record = run_sync(build_plan(mock_card, smoke_suite, output_dir=store.root))
        store.save(record)
        raw = json.loads((store.root / record.run_id / "run.json").read_text(encoding="utf-8"))
        assert raw["summary"]["n_ok"] == record.n_ok
        assert raw["schema_version"] >= 1

    def test_index_is_rebuildable_after_deletion(self, mock_card, smoke_suite, store):
        for _ in range(2):
            store.save(run_sync(build_plan(mock_card, smoke_suite, output_dir=store.root)))
        (store.root / "index.sqlite3").unlink()
        rebuilt = RunStore(store.root)
        assert rebuilt.reindex() == 2
        assert len(rebuilt.list_runs()) == 2

    def test_prefix_lookup(self, mock_card, smoke_suite, store):
        record = run_sync(build_plan(mock_card, smoke_suite, output_dir=store.root))
        store.save(record)
        assert store.load(record.run_id[:12]).run_id == record.run_id

    def test_repeats_are_found_by_fingerprint(self, mock_card, smoke_suite, store):
        a = run_sync(build_plan(mock_card, smoke_suite, output_dir=store.root))
        b = run_sync(build_plan(mock_card, smoke_suite, output_dir=store.root))
        store.save(a)
        store.save(b)
        assert a.fingerprint == b.fingerprint
        assert len(store.find_repeats(a.fingerprint)) == 2

    def test_secrets_never_reach_disk(self, smoke_suite, store):
        card = load_model_card("mock")
        card.adapter_config["api_key"] = "leak-me"
        store.save(run_sync(build_plan(card, smoke_suite, output_dir=store.root)))
        text = (store.root / next(iter(store.list_runs())).run_id / "run.json").read_text()
        assert "leak-me" not in text
        assert "***redacted***" in text

    def test_timings_sidecar_written_for_replay(self, mock_card, smoke_suite, store):
        record = run_sync(build_plan(mock_card, smoke_suite, output_dir=store.root))
        store.save(record)
        sidecar = store.root / record.run_id / "audio" / "timings.json"
        assert sidecar.is_file()
        timings = json.loads(sidecar.read_text())
        assert len(timings) == record.n_ok
        assert all("ttfb_ms" in v for v in timings.values())

    def test_replay_reuses_original_timings(self, mock_card, smoke_suite, store, tmp_path):
        """The point of the sidecar: re-scoring keeps the real latencies."""
        original = run_sync(build_plan(mock_card, smoke_suite, output_dir=store.root))
        store.save(original)

        card = copy.deepcopy(mock_card)
        card.adapter = "replay"
        card.adapter_config = {"audio_dir": str(store.audio_dir(original.run_id))}
        suite = copy.deepcopy(smoke_suite)
        suite.metrics = "core"
        rescored = run_sync(
            build_plan(card, suite, output_dir=tmp_path / "re", dataset=None)
        )
        assert rescored.aggregates["ttfb_ms"].n > 0
        assert rescored.aggregates["ttfb_ms"].mean == pytest.approx(
            original.aggregates["ttfb_ms"].mean, rel=0.01
        )


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
class TestComparison:
    def test_regression_is_detected_with_a_ci(self, mock_card, smoke_suite, tmp_path):
        baseline = _run(mock_card, smoke_suite, tmp_path / "a", ttfb_ms=20)
        candidate_card = copy.deepcopy(mock_card)
        candidate_card.model_version = "2"
        candidate = _run(candidate_card, smoke_suite, tmp_path / "b", ttfb_ms=300)

        comparison = compare_runs(baseline, candidate)
        assert comparison.comparable
        ttfb = comparison.metrics["ttfb_ms"]
        assert ttfb.verdict == "worse"
        assert ttfb.delta > 200
        assert ttfb.ci_low > 0, "CI must exclude zero for a verdict"
        assert ttfb.paired and ttfb.n_pairs == 13

    def test_improvement_is_detected(self, mock_card, smoke_suite, tmp_path):
        baseline = _run(mock_card, smoke_suite, tmp_path / "a", ttfb_ms=300)
        candidate_card = copy.deepcopy(mock_card)
        candidate_card.model_version = "2"
        candidate = _run(candidate_card, smoke_suite, tmp_path / "b", ttfb_ms=20)
        assert compare_runs(baseline, candidate).metrics["ttfb_ms"].verdict == "better"

    def test_identical_runs_produce_no_winners(self, mock_card, smoke_suite, tmp_path):
        a = _run(mock_card, smoke_suite, tmp_path / "a")
        b = _run(mock_card, smoke_suite, tmp_path / "b")
        comparison = compare_runs(a, b)
        assert comparison.improvements() == []
        assert comparison.regressions() == []

    def test_different_dataset_blocks_comparison(self, mock_card, smoke_suite, tmp_path):
        baseline = _run(mock_card, smoke_suite, tmp_path / "a")
        other = copy.deepcopy(smoke_suite)
        other.sample = 20
        candidate = run_sync(build_plan(mock_card, other, output_dir=tmp_path / "b"))
        comparison = compare_runs(baseline, candidate)
        assert not comparison.comparable
        assert any("different test sets" in b for b in comparison.blockers)

    def test_different_concurrency_blocks_comparison(self, mock_card, smoke_suite, tmp_path):
        baseline = _run(mock_card, smoke_suite, tmp_path / "a")
        other = copy.deepcopy(smoke_suite)
        other.concurrency = 8
        candidate = run_sync(build_plan(mock_card, other, output_dir=tmp_path / "b"))
        assert any("different concurrency" in b for b in compare_runs(baseline, candidate).blockers)

    def test_same_model_flagged_as_a_repeatability_check(self, mock_card, smoke_suite, tmp_path):
        a = _run(mock_card, smoke_suite, tmp_path / "a")
        b = _run(mock_card, smoke_suite, tmp_path / "b")
        assert any("repeatability check" in w for w in compare_runs(a, b).warnings)

    def test_run_level_change_is_exact_not_inconclusive(self, mock_card, smoke_suite, tmp_path):
        """A coverage drop is exact; calling it "inconclusive" would read as noise."""
        baseline = _run(mock_card, smoke_suite, tmp_path / "a")
        card = copy.deepcopy(mock_card)
        card.model_version = "2"
        card.adapter_config["unsupported_languages"] = ["ta", "te", "ml"]
        candidate = _run(card, smoke_suite, tmp_path / "b")

        coverage = compare_runs(baseline, candidate).metrics["coverage_ratio"]
        assert coverage.verdict == "single_observation"
        assert coverage.delta < 0
        assert direction_of_change(coverage) == "bad"

    def test_repeatability_reports_the_noise_floor(self, mock_card, smoke_suite, tmp_path):
        runs = [_run(mock_card, smoke_suite, tmp_path / f"r{i}") for i in range(3)]
        report = repeatability(runs)
        assert report["same_fingerprint"] is True
        # Deterministic adapter: signal metrics must not move at all.
        assert report["metrics"]["snr_db"]["cv"] == 0.0
        assert report["metrics"]["audio_quality_score"]["cv"] == 0.0
        # Timing does move; that is the floor below which deltas mean nothing.
        assert "ttfb_ms" in report["metrics"]

    def test_repeatability_needs_two_runs(self, mock_card, smoke_suite, tmp_path):
        assert "error" in repeatability([_run(mock_card, smoke_suite, tmp_path)])


# ---------------------------------------------------------------------------
# subjective
# ---------------------------------------------------------------------------
class TestSubjective:
    def _two_runs(self, tmp_path):
        card_a = load_model_card("mock")
        suite = load_suite("smoke")
        a = run_sync(build_plan(card_a, suite, output_dir=tmp_path / "runs", label="A"))
        card_b = copy.deepcopy(card_a)
        card_b.model_version = "2"
        b = run_sync(build_plan(card_b, suite, output_dir=tmp_path / "runs", label="B"))
        return a, b

    def test_build_test_produces_a_shippable_bundle(self, tmp_path):
        a, b = self._two_runs(tmp_path)
        out = tmp_path / "test"
        manifest = build_test([a, b], out, TestSpec(scale="mushra", n_raters=3, n_trials=6, seed=7))

        assert manifest["n_trials"] == 6
        assert (out / "index.html").is_file()
        assert (out / "manifest.json").is_file()
        assert len(list(out.glob("sheet_*.csv"))) == 3
        # 2 systems + 1 anchor per trial.
        assert len(list((out / "audio").glob("*.wav"))) == 6 * 3

    def test_answer_key_is_separate_from_the_rater_bundle(self, tmp_path):
        a, b = self._two_runs(tmp_path)
        out = tmp_path / "test"
        build_test([a, b], out, TestSpec(n_raters=2, n_trials=4, seed=1))

        key = json.loads((out / "ANSWER_KEY.json").read_text())
        assert "DO NOT SEND" in key["warning"]
        # Rater-facing files must not name the systems.
        manifest_text = (out / "manifest.json").read_text()
        sheet_text = next(out.glob("sheet_*.csv")).read_text()
        for identifier in (a.run_id, b.run_id):
            assert identifier not in sheet_text
        assert "sys_" in manifest_text

    def test_tokens_differ_per_system(self, tmp_path):
        a, b = self._two_runs(tmp_path)
        out = tmp_path / "test"
        manifest = build_test([a, b], out, TestSpec(n_raters=1, n_trials=3, seed=2))
        for trial in manifest["trials"]:
            assert len(set(trial["clips"])) == len(trial["clips"]) >= 2

    def test_rater_order_differs_between_raters(self, tmp_path):
        a, b = self._two_runs(tmp_path)
        out = tmp_path / "test"
        build_test([a, b], out, TestSpec(n_raters=2, n_trials=8, seed=3))
        sheets = sorted(out.glob("sheet_*.csv"))
        orders = []
        for sheet in sheets:
            rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
            seen: list[str] = []
            for row in rows:
                if row["trial_id"] not in seen:
                    seen.append(row["trial_id"])
            orders.append(seen)
        assert orders[0] != orders[1], "presentation order must be shuffled per rater"

    def test_cmos_requires_exactly_two_systems(self, tmp_path):
        a, _b = self._two_runs(tmp_path)
        with pytest.raises(ConfigError, match="exactly two systems"):
            build_test([a], tmp_path / "t", TestSpec(scale="cmos"))

    def test_mos_rejects_multiple_systems(self, tmp_path):
        a, b = self._two_runs(tmp_path)
        with pytest.raises(ConfigError, match="rates one system at a time"):
            build_test([a, b], tmp_path / "t", TestSpec(scale="mos"))

    def _fill_sheets(self, out, good_label_marker="@1", lazy_index=None):
        key = json.loads((out / "ANSWER_KEY.json").read_text())
        import random

        rng = random.Random(11)
        for index, sheet in enumerate(sorted(out.glob("sheet_*.csv"))):
            rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
            lazy = index == lazy_index
            for row in rows:
                label = key["trials"][row["trial_id"]]["systems"][row["system_token"]]
                if "ANCHOR" in label:
                    row["score"] = 75 if lazy else rng.randint(5, 20)
                elif good_label_marker in label:
                    row["score"] = rng.randint(75, 90)
                else:
                    row["score"] = rng.randint(50, 65)
            with sheet.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def test_ingest_deblinds_and_separates_systems(self, tmp_path):
        a, b = self._two_runs(tmp_path)
        out = tmp_path / "test"
        build_test([a, b], out, TestSpec(n_raters=4, n_trials=8, seed=5))
        self._fill_sheets(out)

        report = ingest_sheets(sorted(out.glob("sheet_*.csv")), out / "ANSWER_KEY.json")
        assert report.n_rows_used > 0
        assert len(report.per_system) == 2
        means = {k: v["mean"] for k, v in report.per_system.items()}
        better = max(means, key=means.get)
        assert "@1" in better, "the higher-rated system must de-blind to the right run"
        assert report.agreement["agreement"] > 0.5

    def test_inattentive_rater_is_excluded_with_a_reason(self, tmp_path):
        a, b = self._two_runs(tmp_path)
        out = tmp_path / "test"
        build_test([a, b], out, TestSpec(n_raters=4, n_trials=8, seed=6))
        self._fill_sheets(out, lazy_index=3)

        report = ingest_sheets(sorted(out.glob("sheet_*.csv")), out / "ANSWER_KEY.json")
        assert len(report.excluded_raters) == 1
        excluded = next(r for r in report.raters if r.excluded)
        assert "anchor" in (excluded.reason or "")

    def test_scores_merge_into_the_run_record(self, tmp_path):
        a, b = self._two_runs(tmp_path)
        out = tmp_path / "test"
        build_test([a, b], out, TestSpec(n_raters=3, n_trials=8, seed=8))
        self._fill_sheets(out)
        report = ingest_sheets(sorted(out.glob("sheet_*.csv")), out / "ANSWER_KEY.json")

        merge_into_run(a, report.scores_by_run[a.run_id])
        aggregate = a.aggregates["subjective_mushra"]
        assert aggregate.n > 0
        assert aggregate.mean > 50
        assert aggregate.ci_low is not None
        # Per-language breakdown must be refreshed too.
        assert any("subjective_mushra" in per for per in a.per_language.values())

    def test_out_of_range_score_is_rejected(self, tmp_path):
        a, b = self._two_runs(tmp_path)
        out = tmp_path / "test"
        build_test([a, b], out, TestSpec(n_raters=1, n_trials=4, seed=9))
        sheet = next(out.glob("sheet_*.csv"))
        rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
        rows[0]["score"] = "500"
        with sheet.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        report = ingest_sheets([sheet], out / "ANSWER_KEY.json")
        assert any("outside the mushra range" in w for w in report.warnings)

    def test_wrong_answer_key_is_detected(self, tmp_path):
        a, b = self._two_runs(tmp_path)
        out_a, out_b = tmp_path / "ta", tmp_path / "tb"
        build_test([a, b], out_a, TestSpec(n_raters=1, n_trials=4, seed=10))
        build_test([a, b], out_b, TestSpec(n_raters=1, n_trials=4, seed=99))
        self._fill_sheets(out_a)

        report = ingest_sheets(sorted(out_a.glob("sheet_*.csv")), out_b / "ANSWER_KEY.json")
        assert any("wrong key file" in w for w in report.warnings) or report.n_rows_used == 0


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_bundled_cards_and_suites_all_load(self):
        assert set(list_model_cards()) >= {"indic-mio", "mock", "ai4bharat-parler", "sarvam"}
        assert set(list_suites()) >= {"smoke", "indic-full", "latency", "offline-rescore"}
        for name in list_model_cards():
            load_model_card(name)
        for name in list_suites():
            load_suite(name)

    def test_indic_mio_and_parler_share_one_adapter(self):
        """The generalisation claim, asserted: two providers, zero adapter code."""
        mio = load_model_card("indic-mio")
        parler = load_model_card("ai4bharat-parler")
        assert mio.adapter == parler.adapter == "websocket_pcm"
        assert mio.adapter_config["url"] != parler.adapter_config["url"]

    def test_env_expansion_with_default(self, monkeypatch):
        monkeypatch.delenv("INDIC_MIO_SERVER_URL", raising=False)
        assert load_model_card("indic-mio").adapter_config["url"] == "ws://localhost:8003"

    def test_env_expansion_uses_the_environment(self, monkeypatch):
        monkeypatch.setenv("INDIC_MIO_SERVER_URL", "ws://gpu-box:9000")
        assert load_model_card("indic-mio").adapter_config["url"] == "ws://gpu-box:9000"

    def test_card_declarations_reach_the_adapter(self):
        card = load_model_card("indic-mio")
        resolved = card.resolved_adapter_config()
        assert resolved["voices"] == list(card.voices)
        assert resolved["sample_rate"] == 24000

    def test_missing_required_field_is_fatal(self):
        from tts_eval.config import ModelCard

        with pytest.raises(ConfigError, match="missing required field 'adapter'"):
            ModelCard.from_dict({"model_id": "x"})

    def test_unknown_card_lists_alternatives(self):
        with pytest.raises(ConfigError, match="bundled options"):
            load_model_card("no-such-model")

    def test_secrets_redacted_in_serialised_card(self):
        """Redaction is per-key and recursive: the header name survives, its value dies."""
        card = load_model_card("sarvam")
        headers = card.to_dict()["adapter_config"]["headers"]
        assert headers["api-subscription-key"] == "***redacted***"
        assert "SARVAM_API_KEY" not in json.dumps(card.to_dict())
