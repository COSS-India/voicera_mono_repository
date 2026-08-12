"""Durable, rebuildable, secret-free run storage (JSON source of truth + index)."""
from __future__ import annotations

import copy
import json

import pytest

from tts_eval.config import load_model_card
from tts_eval.runner import build_plan, run_sync
from tts_eval.store import RunStore


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
