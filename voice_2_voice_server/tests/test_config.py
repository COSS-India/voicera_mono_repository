"""Tests for the config package (config/__init__.py, config/llm_mappings.py)."""

import pytest
from config import get_llm_model
from config.llm_mappings import LLM_DEFAULT_MODELS


class TestLlmDefaultModels:
    def test_default_models_is_dict(self):
        assert isinstance(LLM_DEFAULT_MODELS, dict)

    def test_openai_default_model(self):
        assert LLM_DEFAULT_MODELS["OpenAI"] == "gpt-4o"

    def test_kenpath_default_model_is_none(self):
        assert LLM_DEFAULT_MODELS["Kenpath"] is None

    def test_grok_default_model(self):
        assert LLM_DEFAULT_MODELS["Grok"] == "grok-3-beta"

    def test_qwen_default_model(self):
        assert LLM_DEFAULT_MODELS["qwen"] == "Qwen/Qwen3-8B"

    def test_custom_llm_default_model_is_none(self):
        assert LLM_DEFAULT_MODELS["custom_llm"] is None


class TestGetLlmModel:
    def test_returns_override_when_provided(self):
        """When a model is supplied explicitly, it takes priority over defaults."""
        result = get_llm_model("openai", model="gpt-3.5-turbo")
        assert result == "gpt-3.5-turbo"

    def test_returns_default_for_openai(self):
        result = get_llm_model("openai")
        assert result == "gpt-4o"

    def test_returns_default_for_grok(self):
        result = get_llm_model("grok")
        assert result == "grok-3-beta"

    def test_returns_default_for_qwen(self):
        result = get_llm_model("qwen")
        assert result == "Qwen/Qwen3-8B"

    def test_unknown_provider_falls_back_to_gpt4o(self):
        """Unknown providers fall back to gpt-4o."""
        result = get_llm_model("unknown_provider")
        assert result == "gpt-4o"

    def test_provider_name_is_lowercased(self):
        """get_llm_model lowercases the provider before lookup."""
        result = get_llm_model("QWEN")
        assert result == "Qwen/Qwen3-8B"

    def test_custom_llm_returns_empty_string_when_no_model(self):
        """custom_llm with no model and None default returns empty string."""
        result = get_llm_model("custom_llm")
        assert result == ""

    def test_custom_llm_returns_override_model(self):
        result = get_llm_model("custom_llm", model="my-custom-model")
        assert result == "my-custom-model"
