"""Runner behaviour and the reproducibility fingerprint.

Fingerprint: only inputs that can change a number move the hash. Runner: stable
ordering, freed buffers, honest warnings, and real request overlap under concurrency.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tts_eval.config import load_model_card, load_suite
from tts_eval.datasets import dataset_from_cases
from tts_eval.errors import ConfigError
from tts_eval.runner import build_plan, compute_fingerprint, run_sync


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
        # Not saving audio must NOT turn every utterance into a failure: success is
        # measured by what the provider produced, not by whether a WAV was kept.
        # (The `latency` suite runs with save_audio=false, so this is its success
        # path too.) Regression guard: previously `ok` read audio_path and reported
        # 0/13 here, corrupting success_rate, throughput and every aggregate.
        assert record.n_ok == 13
        assert record.success_rate == 1.0
        assert all(u.result.ok and u.result.n_samples for u in record.utterances)

    def test_ok_survives_reload_without_audio(self, mock_card, smoke_suite, tmp_path, store):
        """A stored record with no WAVs on disk still reads back as successful."""
        suite = copy.deepcopy(smoke_suite)
        suite.save_audio = False
        record = run_sync(build_plan(mock_card, suite, output_dir=store.root))
        run_id = store.save(record).name
        reloaded = store.load(run_id)
        assert reloaded.n_ok == 13 and reloaded.success_rate == 1.0

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
