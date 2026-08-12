"""Defensible cross-run verdicts: paired CIs, effect floors, blocked comparisons."""
from __future__ import annotations

import copy

from tts_eval.compare import compare_runs, direction_of_change, repeatability
from tts_eval.runner import build_plan, run_sync


def _run(card, suite, tmp_path, label="test", **card_overrides):
    card = copy.deepcopy(card)
    card.adapter_config.update(card_overrides)
    return run_sync(build_plan(card, suite, output_dir=tmp_path / "runs", label=label))


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
