import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock

from api.services import (
    _normalize_provider_name,
    _normalize_elevenlabs_stt_model,
    _is_multilingual_elevenlabs_tts_model,
    _extract_elevenlabs_tts_voice_id,
    create_llm_service,
    create_stt_service,
    create_tts_service,
    ServiceCreationError,
)

# ---------------------------------------------------------------------------
# Provider name constants — use these instead of repeating string literals.
# ---------------------------------------------------------------------------
PROVIDER_OPENAI = "OpenAI"
PROVIDER_DEEPGRAM = "Deepgram"
PROVIDER_ELEVENLABS = "elevenlabs"
PROVIDER_ANTHROPIC = "Anthropic"
PROVIDER_GROK = "Grok"
PROVIDER_KENPATH = "Kenpath"
PROVIDER_QWEN = "qwen"
PROVIDER_CUSTOM_LLM = "custom_llm"
PROVIDER_GOOGLE = "google"
PROVIDER_OPENAI_STT = "openai"
PROVIDER_AI4BHARAT = "ai4bharat"
PROVIDER_BHASHINI = "bhashini"
PROVIDER_SARVAM = "sarvam"
PROVIDER_CARTESIA = "cartesia"

class TestNormalizeProviderName:
    def test_normalize_success(self):
        provider_map = {
            "deepgram": "Deepgram",
            "google": "Google",
            "openai": "OpenAI",
        }
        assert _normalize_provider_name("deepgram", provider_map, "STT") == "Deepgram"
        assert _normalize_provider_name("  GOOGLE  ", provider_map, "STT") == "Google"
        assert _normalize_provider_name("other", provider_map, "STT") == "other"

    def test_normalize_empty_raises(self):
        provider_map = {"openai": "OpenAI"}
        with pytest.raises(ServiceCreationError) as exc_info:
            _normalize_provider_name(None, provider_map, "STT")
        assert "STT provider is missing" in str(exc_info.value)

        with pytest.raises(ServiceCreationError):
            _normalize_provider_name("   ", provider_map, "STT")


class TestNormalizeElevenlabsSttModel:
    def test_default_model(self):
        assert _normalize_elevenlabs_stt_model(None) == "scribe_v2_realtime"

    @patch("api.services.logger")
    def test_for_realtime_scribe_v2(self, mock_logger):
        assert _normalize_elevenlabs_stt_model("scribe-v2", for_realtime=True) == "scribe_v2_realtime"
        mock_logger.warning.assert_called_once()

    def test_for_realtime_scribe_v2_realtime(self):
        assert _normalize_elevenlabs_stt_model("scribe-v2-realtime", for_realtime=True) == "scribe_v2_realtime"

    def test_for_realtime_invalid(self):
        with pytest.raises(ServiceCreationError) as exc_info:
            _normalize_elevenlabs_stt_model("invalid_model", for_realtime=True)
        assert "Invalid ElevenLabs realtime STT model" in str(exc_info.value)

    def test_not_for_realtime_allowed(self):
        assert _normalize_elevenlabs_stt_model("scribe-v2", for_realtime=False) == "scribe_v2"
        assert _normalize_elevenlabs_stt_model("scribe-v2-realtime", for_realtime=False) == "scribe_v2_realtime"

    def test_not_for_realtime_invalid(self):
        with pytest.raises(ServiceCreationError) as exc_info:
            _normalize_elevenlabs_stt_model("invalid_model", for_realtime=False)
        assert "Allowed models: scribe_v2, scribe_v2_realtime" in str(exc_info.value)


class TestIsMultilingualElevenlabsTtsModel:
    def test_is_multilingual(self):
        assert _is_multilingual_elevenlabs_tts_model("eleven_multilingual_v2") is True
        assert _is_multilingual_elevenlabs_tts_model("eleven_v3") is True
        assert _is_multilingual_elevenlabs_tts_model("eleven_turbo_v2") is False
        assert _is_multilingual_elevenlabs_tts_model("") is False
        assert _is_multilingual_elevenlabs_tts_model(None) is False


class TestExtractElevenlabsTtsVoiceId:
    def test_extract_voice_id_priority(self):
        # args.voice_id
        assert _extract_elevenlabs_tts_voice_id({}, {"voice_id": "v1"}) == ("v1", "args.voice_id")
        # args.voice
        assert _extract_elevenlabs_tts_voice_id({}, {"voice": "v2"}) == ("v2", "args.voice")
        # tts_config.voice_id
        assert _extract_elevenlabs_tts_voice_id({"voice_id": "v3"}, {}) == ("v3", "tts_model.voice_id")
        # tts_config.speaker
        assert _extract_elevenlabs_tts_voice_id({"speaker": "v4"}, {}) == ("v4", "tts_model.speaker")

    def test_extract_voice_id_missing_raises(self):
        with pytest.raises(ServiceCreationError) as exc_info:
            _extract_elevenlabs_tts_voice_id({}, {})
        assert "ElevenLabs TTS requires a non-empty voice ID" in str(exc_info.value)


@patch("api.services.fetch_integration_key")
@patch("api.services.OpenAIKnowledgeLLMService")
@patch("api.services.KenpathLLM")
@patch("api.services.AnthropicLLMService")
@patch("api.services.GrokLLMService")
@patch("api.services.create_voice_llm")
@patch("api.services.create_custom_llm")
@patch("api.services.fetch_custom_llm_config")
class TestCreateLlmService:
    def test_openai_service_with_org_id(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                         mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        mock_fetch_key.return_value = "mock_key"
        config = {
            "name": "OpenAI",
            "args": {"model": "gpt-4o", "aggregation_timeout": 0.1},
            "knowledge_base_enabled": True,
            "knowledge_document_ids": ["doc_1"],
            "knowledge_top_k": 5
        }
        service = create_llm_service(config, org_id="org_123")
        assert service is not None
        mock_fetch_key.assert_called_once_with("org_123", "OpenAI")
        mock_openai.assert_called_once_with(
            api_key="mock_key",
            model="gpt-4o",
            org_id="org_123",
            knowledge_enabled=True,
            knowledge_document_ids=["doc_1"],
            knowledge_top_k=5
        )

    def test_openai_service_missing_org_key(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                             mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        mock_fetch_key.return_value = None
        config = {"name": "OpenAI", "args": {"model": "gpt-4o"}}
        with pytest.raises(ServiceCreationError) as exc_info:
            create_llm_service(config, org_id="org_123")
        assert "OpenAI API key must be configured" in str(exc_info.value)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env_openai_key"})
    def test_openai_service_no_org_id(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                       mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        config = {"name": "OpenAI", "args": {"model": "gpt-4o"}}
        service = create_llm_service(config)
        assert service is not None
        mock_openai.assert_called_once_with(
            api_key="env_openai_key",
            model="gpt-4o",
            org_id=None,
            knowledge_enabled=False,
            knowledge_document_ids=[],
            knowledge_top_k=3
        )

    def test_kenpath_service(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                             mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        config = {
            "name": "Kenpath",
            "args": {"vistaar_environment": "staging"},
            "hold_messages": ["please wait"],
        }
        service = create_llm_service(
            config,
            vistaar_session_id="session_123",
            language="hindi",
            hold_messages=config["hold_messages"],
            hold_message_timeout_seconds=0.5
        )
        assert service is not None
        mock_kenpath.assert_called_once_with(
            vistaar_session_id="session_123",
            language="hindi",
            vistaar_environment="staging",
            hold_messages=["please wait"],
            response_timeout=0.5
        )

    def test_anthropic_service_success(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                        mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        mock_fetch_key.return_value = "anthropic_key"
        config = {
            "name": "Anthropic",
            "args": {
                "model": "claude-3-opus",
                "max_tokens": 1000,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 5,
                "enable_prompt_caching": True,
            }
        }
        
        # Mock Settings
        settings_mock = MagicMock()
        mock_settings_class = MagicMock(return_value=settings_mock)
        mock_anthropic.Settings = mock_settings_class

        service = create_llm_service(config, org_id="org_123")
        assert service is not None
        mock_settings_class.assert_called_once_with(
            model="claude-3-opus",
            max_tokens=1000,
            temperature=0.5,
            top_p=0.9,
            top_k=5,
            enable_prompt_caching=True
        )
        mock_anthropic.assert_called_once_with(api_key="anthropic_key", settings=settings_mock)

    def test_anthropic_service_missing_org(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                            mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        config = {"name": "Anthropic"}
        with pytest.raises(ServiceCreationError) as exc_info:
            create_llm_service(config, org_id=None)
        assert "Anthropic requires an organization context" in str(exc_info.value)

    def test_anthropic_service_missing_key(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                            mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        mock_fetch_key.return_value = None
        config = {"name": "Anthropic"}
        with pytest.raises(ServiceCreationError) as exc_info:
            create_llm_service(config, org_id="org_123")
        assert "Anthropic API key must be configured" in str(exc_info.value)

    def test_grok_service_with_org_id(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                       mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        mock_fetch_key.return_value = "grok_key"
        config = {"name": "Grok", "args": {"model": "grok-beta"}}
        service = create_llm_service(config, org_id="org_123")
        assert service is not None
        mock_grok.assert_called_once_with(api_key="grok_key", model="grok-beta")

    @patch.dict(os.environ, {"GROK_API_KEY": "env_grok_key"})
    def test_grok_service_no_org_id(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                     mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        config = {"name": "Grok", "args": {"model": "grok-beta"}}
        service = create_llm_service(config)
        assert service is not None
        mock_grok.assert_called_once_with(api_key="env_grok_key", model="grok-beta")

    def test_grok_service_missing_key(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                       mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        with patch.dict(os.environ, {}, clear=True):
            config = {"name": "Grok"}
            with pytest.raises(ServiceCreationError) as exc_info:
                create_llm_service(config)
            assert "XAI_API_KEY or GROK_API_KEY is required" in str(exc_info.value)

    @patch("api.services.BaseOpenAILLMService")
    def test_qwen_service(self, mock_base_llm, mock_fetch_custom, mock_create_custom, mock_create_voice,
                          mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        params_mock = MagicMock()
        mock_base_llm.InputParams.return_value = params_mock
        config = {
            "name": "qwen",
            "args": {
                "model": "qwen-7b",
                "temperature": 0.5,
                "top_p": 0.9,
                "max_tokens": 150,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
                "seed": 42,
                "extra": {"chat_template_kwargs": {"enable_thinking": True}},
                "api_key": "vllm_key",
                "base_url": "vllm_url"
            }
        }
        service = create_llm_service(config)
        assert service is not None
        mock_base_llm.InputParams.assert_called_once_with(
            temperature=0.5,
            top_p=0.9,
            max_tokens=150,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            seed=42,
            extra={
                "chat_template_kwargs": {"enable_thinking": True},
                "top_k": 20,
                "repetition_penalty": 1.0
            }
        )
        mock_create_voice.assert_called_once_with(
            model="qwen-7b",
            api_key="vllm_key",
            base_url="vllm_url",
            params=params_mock,
            retry_timeout_secs=20.0,
            retry_on_timeout=False
        )

    def test_custom_llm_service_success(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                        mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        mock_fetch_custom.return_value = {"api_key": "c_key", "base_url": "c_url", "model": "c_model"}
        config = {
            "name": "custom_llm",
            "custom_llm_id": "llm_123",
            "args": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_tokens": 300,
                "frequency_penalty": 0.05,
                "presence_penalty": 0.05,
            }
        }
        service = create_llm_service(config, org_id="org_123")
        assert service is not None
        mock_fetch_custom.assert_called_once_with("org_123", "llm_123")
        mock_create_custom.assert_called_once()

    def test_custom_llm_service_missing_org(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                            mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        config = {"name": "custom_llm"}
        with pytest.raises(ServiceCreationError) as exc_info:
            create_llm_service(config, org_id=None)
        assert "Custom LLM requires an organization context" in str(exc_info.value)

    def test_custom_llm_service_missing_id(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                            mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        config = {"name": "custom_llm"}
        with pytest.raises(ServiceCreationError) as exc_info:
            create_llm_service(config, org_id="org_123")
        assert "Custom LLM integration id is missing" in str(exc_info.value)

    def test_custom_llm_service_not_found(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                           mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        mock_fetch_custom.return_value = None
        config = {"name": "custom_llm", "custom_llm_id": "llm_123"}
        with pytest.raises(ServiceCreationError) as exc_info:
            create_llm_service(config, org_id="org_123")
        assert "Custom LLM integration not found" in str(exc_info.value)

    def test_custom_llm_service_missing_model(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                                               mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        mock_fetch_custom.return_value = {"api_key": "c_key", "base_url": "c_url", "model": None}
        config = {"name": "custom_llm", "custom_llm_id": "llm_123"}
        with pytest.raises(ServiceCreationError) as exc_info:
            create_llm_service(config, org_id="org_123")
        assert "Custom LLM model is not configured" in str(exc_info.value)

    def test_unknown_provider(self, mock_fetch_custom, mock_create_custom, mock_create_voice,
                              mock_grok, mock_anthropic, mock_kenpath, mock_openai, mock_fetch_key):
        config = {"name": "Unknown"}
        with pytest.raises(ServiceCreationError) as exc_info:
            create_llm_service(config)
        assert "Unknown LLM provider" in str(exc_info.value)


@patch("api.services.fetch_integration_key")
@patch("api.services.ElevenLabsRealtimeSTTService")
@patch("api.services.DeepgramSTTService")
@patch("api.services.GoogleSTTService")
@patch("api.services.OpenAISTTService")
@patch("api.services.IndicConformerRESTSTTService")
@patch("api.services.BhashiniBhiliSTTService")
@patch("api.services.BhashiniSTTService")
@patch("api.services.SarvamSTTService")
class TestCreateSttService:
    def test_elevenlabs_stt_org_key(self, mock_sarvam, mock_bhashini, mock_bhashini_bhili,
                                     mock_indic, mock_openai, mock_google, mock_deepgram, mock_eleven, mock_fetch_key):
        mock_fetch_key.return_value = "el_key"
        config = {"name": "elevenlabs", "language": "English", "args": {"model": "scribe-v2-realtime"}}
        service = create_stt_service(config, sample_rate=16000, org_id="org_1")
        assert service is not None
        mock_eleven.assert_called_once()
        args, kwargs = mock_eleven.call_args
        assert kwargs["api_key"] == "el_key"
        assert kwargs["sample_rate"] == 16000
        assert kwargs["model"] == "scribe_v2_realtime"

    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "env_el_key"})
    def test_elevenlabs_stt_env_key(self, mock_sarvam, mock_bhashini, mock_bhashini_bhili,
                                     mock_indic, mock_openai, mock_google, mock_deepgram, mock_eleven, mock_fetch_key):
        config = {"name": "elevenlabs", "language": "English"}
        service = create_stt_service(config, sample_rate=16000)
        assert service is not None
        args, kwargs = mock_eleven.call_args
        assert kwargs["api_key"] == "env_el_key"

    def test_elevenlabs_stt_missing_key(self, mock_sarvam, mock_bhashini, mock_bhashini_bhili,
                                         mock_indic, mock_openai, mock_google, mock_deepgram, mock_eleven, mock_fetch_key):
        with patch.dict(os.environ, {}, clear=True):
            config = {"name": "elevenlabs"}
            with pytest.raises(ServiceCreationError) as exc_info:
                create_stt_service(config, sample_rate=16000)
            assert "ELEVENLABS_API_KEY is required" in str(exc_info.value)

    def test_deepgram_stt_org_key(self, mock_sarvam, mock_bhashini, mock_bhashini_bhili,
                                   mock_indic, mock_openai, mock_google, mock_deepgram, mock_eleven, mock_fetch_key):
        mock_fetch_key.return_value = "dg_key"
        config = {"name": "deepgram", "language": "English", "args": {"model": "nova-2"}}
        service = create_stt_service(config, sample_rate=8000, org_id="org_1")
        assert service is not None
        mock_deepgram.assert_called_once()
        args, kwargs = mock_deepgram.call_args
        assert kwargs["api_key"] == "dg_key"
        assert kwargs["sample_rate"] == 8000
        assert kwargs["live_options"].model == "nova-2"
        assert kwargs["live_options"].language == "en-US"

    @patch.dict(os.environ, {"DEEPGRAM_API_KEY": "env_dg_key"})
    def test_deepgram_stt_env_key(self, mock_sarvam, mock_bhashini, mock_bhashini_bhili,
                                   mock_indic, mock_openai, mock_google, mock_deepgram, mock_eleven, mock_fetch_key):
        config = {"name": "deepgram", "language": "Hindi"}
        service = create_stt_service(config, sample_rate=8000)
        assert service is not None
        args, kwargs = mock_deepgram.call_args
        assert kwargs["api_key"] == "env_dg_key"

    @patch.dict(os.environ, {"GOOGLE_STT_CREDENTIALS_PATH": "my_path.json"})
    def test_google_stt(self, mock_sarvam, mock_bhashini, mock_bhashini_bhili,
                        mock_indic, mock_openai, mock_google, mock_deepgram, mock_eleven, mock_fetch_key):
        config = {"name": "google", "language": "Hindi"}
        service = create_stt_service(config, sample_rate=8000)
        assert service is not None
        mock_google.assert_called_once()
        args, kwargs = mock_google.call_args
        assert kwargs["credentials_path"] == "my_path.json"
        assert kwargs["sample_rate"] == 8000
        mock_google.InputParams.assert_called_once_with(languages=["hi-IN"])
        assert kwargs["params"] == mock_google.InputParams.return_value

    def test_openai_stt(self, mock_sarvam, mock_bhashini, mock_bhashini_bhili,
                        mock_indic, mock_openai, mock_google, mock_deepgram, mock_eleven, mock_fetch_key):
        mock_fetch_key.return_value = "o_key"
        config = {"name": "openai", "language": "Hindi"}
        service = create_stt_service(config, sample_rate=8000, org_id="org_1")
        assert service is not None
        mock_openai.assert_called_once_with(api_key="o_key", language="hi")

    def test_ai4bharat_stt_success(self, mock_sarvam, mock_bhashini, mock_bhashini_bhili,
                                   mock_indic, mock_openai, mock_google, mock_deepgram, mock_eleven, mock_fetch_key):
        config = {"name": "ai4bharat", "language": "Hindi", "args": {"model": "indic-conformer-stt"}}
        vad = MagicMock()
        service = create_stt_service(config, sample_rate=8000, vad_analyzer=vad)
        assert service is not None
        mock_indic.assert_called_once_with(
            language_id="hi",
            sample_rate=16000,
            input_sample_rate=8000,
            vad_analyzer=vad
        )

    def test_ai4bharat_stt_invalid_model(self, mock_sarvam, mock_bhashini, mock_bhashini_bhili,
                                         mock_indic, mock_openai, mock_google, mock_deepgram, mock_eleven, mock_fetch_key):
        config = {"name": "ai4bharat", "language": "Hindi", "args": {"model": "invalid-model"}}
        with pytest.raises(ServiceCreationError) as exc_info:
            create_stt_service(config, sample_rate=8000)
        assert "Unknown ai4bharat STT model" in str(exc_info.value)

    @patch.dict(os.environ, {"BHASHINI_API_KEY": "bh_key"})
    def test_bhashini_stt_bhili(self, mock_sarvam, mock_bhashini, mock_bhashini_bhili,
                                mock_indic, mock_openai, mock_google, mock_deepgram, mock_eleven, mock_fetch_key):
        config = {"name": "bhashini", "language": "bhb"}
        service = create_stt_service(config, sample_rate=16000, vad_analyzer=MagicMock())
        assert service is not None
        mock_bhashini_bhili.assert_called_once_with(
            model="asr_streaming",
            language="bhb",
            sample_rate=16000,
            input_sample_rate=16000,
            suppress_vad_frames=True
        )

    @patch.dict(os.environ, {"BHASHINI_API_KEY": "bh_key"})
    def test_bhashini_stt_general(self, mock_sarvam, mock_bhashini, mock_bhashini_bhili,
                                  mock_indic, mock_openai, mock_google, mock_deepgram, mock_eleven, mock_fetch_key):
        config = {"name": "bhashini", "language": "Hindi"}
        service = create_stt_service(config, sample_rate=16000, vad_analyzer=None)
        assert service is not None
        mock_bhashini.assert_called_once_with(
            api_key="bh_key",
            language="hi",
            service_id="bhashini/ai4b/indic-conformer/grpc",
            sample_rate=16000,
            input_sample_rate=16000,
            suppress_vad_frames=False
        )

    def test_sarvam_stt(self, mock_sarvam, mock_bhashini, mock_bhashini_bhili,
                       mock_indic, mock_openai, mock_google, mock_deepgram, mock_eleven, mock_fetch_key):
        mock_fetch_key.return_value = "sar_key"
        config = {"name": "sarvam", "language": "Hindi", "args": {"model": "saarika:v2.5"}}
        service = create_stt_service(config, sample_rate=16000, org_id="org_1")
        assert service is not None
        mock_sarvam.assert_called_once_with(
            api_key="sar_key",
            language="hi-IN",
            model="saarika:v2.5",
            sample_rate=16000
        )

    def test_unknown_stt_provider(self, mock_sarvam, mock_bhashini, mock_bhashini_bhili,
                                  mock_indic, mock_openai, mock_google, mock_deepgram, mock_eleven, mock_fetch_key):
        config = {"name": "invalid_stt"}
        with pytest.raises(ServiceCreationError) as exc_info:
            create_stt_service(config, sample_rate=8000)
        assert "Unknown STT provider" in str(exc_info.value)


@patch("api.services.fetch_integration_key")
@patch("api.services.ElevenLabsTTSService")
@patch("api.services.CartesiaTTSService")
@patch("api.services.GoogleTTSService")
@patch("api.services.OpenAITTSService")
@patch("api.services.IndicParlerRESTTTSService")
@patch("api.services.BhashiniTTSService")
@patch("api.services.SarvamTTSService")
@patch("api.services.DeepgramTTSService")
class TestCreateTtsService:
    def test_elevenlabs_tts_success(self, mock_deepgram, mock_sarvam, mock_bhashini, mock_indic,
                                     mock_openai, mock_google, mock_cartesia, mock_eleven, mock_fetch_key):
        mock_fetch_key.return_value = "el_key"
        config = {
            "name": "elevenlabs",
            "language": "Hindi",
            "voice_id": "mock_voice_id",
            "model": "eleven_multilingual_v2"
        }
        service = create_tts_service(config, sample_rate=16000, org_id="org_1")
        assert service is not None
        mock_eleven.assert_called_once()
        args, kwargs = mock_eleven.call_args
        assert kwargs["api_key"] == "el_key"
        assert kwargs["voice_id"] == "mock_voice_id"
        assert kwargs["model"] == "eleven_multilingual_v2"
        mock_eleven.InputParams.assert_called_once_with(language="hi")
        assert kwargs["params"] == mock_eleven.InputParams.return_value

    def test_cartesia_tts(self, mock_deepgram, mock_sarvam, mock_bhashini, mock_indic,
                          mock_openai, mock_google, mock_cartesia, mock_eleven, mock_fetch_key):
        mock_fetch_key.return_value = "cart_key"
        config = {
            "name": "cartesia",
            "args": {"model": "sonic-english", "voice_id": "c_voice"}
        }
        service = create_tts_service(config, sample_rate=16000, org_id="org_1")
        assert service is not None
        mock_cartesia.assert_called_once_with(
            api_key="cart_key",
            model="sonic-english",
            encoding="pcm_s16le",
            voice_id="c_voice"
        )

    def test_google_tts(self, mock_deepgram, mock_sarvam, mock_bhashini, mock_indic,
                        mock_openai, mock_google, mock_cartesia, mock_eleven, mock_fetch_key):
        config = {
            "name": "google",
            "language": "Hindi",
            "voice_id": "g_voice"
        }
        with patch.dict(os.environ, {"GOOGLE_TTS_CREDENTIALS_PATH": "g_path.json"}):
            service = create_tts_service(config, sample_rate=16000)
        assert service is not None
        mock_google.assert_called_once()
        args, kwargs = mock_google.call_args
        assert kwargs["credentials_path"] == "g_path.json"
        assert kwargs["voice_id"] == "g_voice"
        mock_google.InputParams.assert_called_once_with(language="hi-IN")
        assert kwargs["params"] == mock_google.InputParams.return_value

    def test_openai_tts(self, mock_deepgram, mock_sarvam, mock_bhashini, mock_indic,
                        mock_openai, mock_google, mock_cartesia, mock_eleven, mock_fetch_key):
        mock_fetch_key.return_value = "o_key"
        config = {
            "name": "openai",
            "args": {"voice": "alloy"}
        }
        service = create_tts_service(config, sample_rate=16000, org_id="org_1")
        assert service is not None
        mock_openai.assert_called_once_with(api_key="o_key", voice="alloy")

    def test_ai4bharat_tts(self, mock_deepgram, mock_sarvam, mock_bhashini, mock_indic,
                           mock_openai, mock_google, mock_cartesia, mock_eleven, mock_fetch_key):
        config = {
            "name": "ai4bharat",
            "language": "Hindi",
            "model": "indic-parler-tts",
            "args": {"speaker": "spk_1", "description": "desc_1"}
        }
        service = create_tts_service(config, sample_rate=22050)
        assert service is not None
        mock_indic.assert_called_once_with(
            speaker="spk_1",
            description="desc_1",
            language_id="hi",
            sample_rate=22050
        )

    def test_ai4bharat_tts_invalid_model(self, mock_deepgram, mock_sarvam, mock_bhashini, mock_indic,
                                         mock_openai, mock_google, mock_cartesia, mock_eleven, mock_fetch_key):
        config = {
            "name": "ai4bharat",
            "model": "invalid-model"
        }
        with pytest.raises(ServiceCreationError) as exc_info:
            create_tts_service(config, sample_rate=22050)
        assert "Unknown ai4bharat TTS model" in str(exc_info.value)

    def test_bhashini_tts(self, mock_deepgram, mock_sarvam, mock_bhashini, mock_indic,
                          mock_openai, mock_google, mock_cartesia, mock_eleven, mock_fetch_key):
        config = {
            "name": "bhashini",
            "language": "Hindi",
            "args": {"speaker": "spk_2", "description": "desc_2"}
        }
        service = create_tts_service(config, sample_rate=16000)
        assert service is not None
        mock_bhashini.assert_called_once_with(
            speaker="spk_2",
            description="desc_2",
            language="hi",
            sample_rate=44100
        )

    def test_sarvam_tts(self, mock_deepgram, mock_sarvam, mock_bhashini, mock_indic,
                        mock_openai, mock_google, mock_cartesia, mock_eleven, mock_fetch_key):
        mock_fetch_key.return_value = "sar_key"
        config = {
            "name": "sarvam",
            "language": "Hindi",
            "model": "bulbul:v3",
            "args": {"speaker": "aditya", "pace": 1.1}
        }
        service = create_tts_service(config, sample_rate=16000, org_id="org_1")
        assert service is not None
        mock_sarvam.assert_called_once_with(
            api_key="sar_key",
            target_language_code="hi-IN",
            model="bulbul:v3",
            voice_id="aditya",
            pitch=None,
            pace=1.1,
            loudness=None
        )

    def test_deepgram_tts(self, mock_deepgram, mock_sarvam, mock_bhashini, mock_indic,
                          mock_openai, mock_google, mock_cartesia, mock_eleven, mock_fetch_key):
        mock_fetch_key.return_value = "dg_key"
        config = {
            "name": "deepgram",
            "args": {"voice": "helena"}
        }
        service = create_tts_service(config, sample_rate=16000, org_id="org_1")
        assert service is not None
        mock_deepgram.assert_called_once_with(
            api_key="dg_key",
            voice="aura-2-helena-en",
            sample_rate=16000,
            encoding="linear16"
        )

    def test_unknown_tts_provider(self, mock_deepgram, mock_sarvam, mock_bhashini, mock_indic,
                                  mock_openai, mock_google, mock_cartesia, mock_eleven, mock_fetch_key):
        config = {"name": "invalid_tts"}
        with pytest.raises(ServiceCreationError) as exc_info:
            create_tts_service(config, sample_rate=8000)
        assert "Unknown TTS provider" in str(exc_info.value)
