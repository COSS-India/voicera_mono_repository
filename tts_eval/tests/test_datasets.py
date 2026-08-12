"""Dataset identity and reproducibility: tamper-evident, comparable test inputs."""
from __future__ import annotations

import pytest

from tts_eval.datasets import dataset_from_cases, load_dataset
from tts_eval.errors import DatasetError

DATASET = "indic_conversational_v1"


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
