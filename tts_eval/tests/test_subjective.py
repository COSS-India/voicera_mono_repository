"""Blinded listening tests: shuffling, rater screening, correct de-blinding."""
from __future__ import annotations

import copy
import csv
import json

import pytest

from tts_eval.config import load_model_card, load_suite
from tts_eval.errors import ConfigError
from tts_eval.runner import build_plan, run_sync
from tts_eval.subjective import TestSpec, build_test, ingest_sheets, merge_into_run


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
