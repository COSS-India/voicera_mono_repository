"""Symmetric, script-aware CER/WER/slot scoring for round-trip intelligibility."""
from __future__ import annotations

import numpy as np
import pytest

from tts_eval.asr.base import (
    character_error_rate,
    normalise_text,
    slot_hits,
    word_error_rate,
)


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
