"""Config-only model onboarding: cards, suites, env expansion, secret redaction.

The example provider cards that once doubled as fixtures here were removed to keep
the shipped configs generic. Each test now builds exactly the card it needs —
either inline via :meth:`ModelCard.from_dict`, or as a temp YAML loaded through the
real loader so env expansion and redaction exercise the same path a bundled card
would.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tts_eval.config import (
    ModelCard,
    list_model_cards,
    list_suites,
    load_model_card,
    load_suite,
)
from tts_eval.errors import ConfigError


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "card.yaml"
    p.write_text(text, encoding="utf-8")
    return p


class TestConfig:
    def test_bundled_cards_and_suites_all_load(self):
        # Only the offline fixtures are guaranteed to ship; whatever else is
        # present must still parse cleanly.
        assert "mock" in set(list_model_cards())
        assert {"smoke", "latency"} <= set(list_suites())
        for name in list_model_cards():
            load_model_card(name)
        for name in list_suites():
            load_suite(name)

    def test_two_providers_share_one_adapter(self):
        """The generalisation claim, asserted: two providers, zero adapter code."""
        a = ModelCard.from_dict(
            {"model_id": "prov-a", "adapter": "websocket_pcm",
             "adapter_config": {"url": "ws://a:8003"}}
        )
        b = ModelCard.from_dict(
            {"model_id": "prov-b", "adapter": "websocket_pcm",
             "adapter_config": {"url": "ws://b:9000"}}
        )
        assert a.adapter == b.adapter == "websocket_pcm"
        assert a.adapter_config["url"] != b.adapter_config["url"]

    def test_env_expansion_with_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TTS_EVAL_TEST_URL", raising=False)
        p = _write(
            tmp_path,
            "model_id: x\nadapter: websocket_pcm\n"
            "adapter_config:\n  url: ${TTS_EVAL_TEST_URL:-ws://localhost:8003}\n",
        )
        assert load_model_card(str(p)).adapter_config["url"] == "ws://localhost:8003"

    def test_env_expansion_uses_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TTS_EVAL_TEST_URL", "ws://gpu-box:9000")
        p = _write(
            tmp_path,
            "model_id: x\nadapter: websocket_pcm\n"
            "adapter_config:\n  url: ${TTS_EVAL_TEST_URL:-ws://localhost:8003}\n",
        )
        assert load_model_card(str(p)).adapter_config["url"] == "ws://gpu-box:9000"

    def test_card_declarations_reach_the_adapter(self):
        card = ModelCard.from_dict(
            {"model_id": "x", "adapter": "websocket_pcm",
             "sample_rate": 24000, "voices": ["a", "b"]}
        )
        resolved = card.resolved_adapter_config()
        assert resolved["voices"] == list(card.voices)
        assert resolved["sample_rate"] == 24000

    def test_missing_required_field_is_fatal(self):
        with pytest.raises(ConfigError, match="missing required field 'adapter'"):
            ModelCard.from_dict({"model_id": "x"})

    def test_unknown_card_lists_alternatives(self):
        with pytest.raises(ConfigError, match="bundled options"):
            load_model_card("no-such-model")

    def test_secrets_redacted_in_serialised_card(self, tmp_path, monkeypatch):
        """Redaction is per-key and recursive: the header name survives, its value dies."""
        monkeypatch.setenv("TTS_EVAL_TEST_KEY", "super-secret-value")
        p = _write(
            tmp_path,
            "model_id: x\nadapter: http_rest\n"
            "adapter_config:\n  headers:\n    api-subscription-key: ${TTS_EVAL_TEST_KEY}\n",
        )
        card = load_model_card(str(p))
        headers = card.to_dict()["adapter_config"]["headers"]
        assert headers["api-subscription-key"] == "***redacted***"
        assert "super-secret-value" not in json.dumps(card.to_dict())
