"""Service factory functions for creating LLM, STT, and TTS services."""

import os
from typing import Any, Optional

from loguru import logger
from deepgram import LiveOptions

# Pipecat services
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams

# Local services
from services.kenpath_llm.llm import KenpathLLM
from services.ai4bharat.tts import IndicParlerRESTTTSService
from services.ai4bharat.stt import IndicConformerRESTSTTService
from services.bhashini.stt import BhashiniSTTService
from services.bhashini.tts import BhashiniTTSService
from config import get_llm_model
from config.stt_mappings import STT_LANGUAGE_MAP
from config.tts_mappings import TTS_LANGUAGE_MAP

from .backend_utils import fetch_integration_key


class ServiceCreationError(Exception):
    """Raised when a service cannot be created."""
    pass


def create_llm_service(
    llm_config: dict,
    vistaar_session_id: Optional[str] = None,
    language: Optional[str] = None,
    org_id: Optional[str] = None,
) -> Any:
    """Create an LLM service based on configuration.

    Args:
        llm_config: LLM configuration dict with 'name' and optional 'args'
        vistaar_session_id: Optional session ID for Kenpath/Vistaar
        language: Optional agent language (e.g. "hindi", "marathi") for Kenpath
        org_id: Optional organization ID to fetch integration API key from backend

    Returns:
        Configured LLM service instance

    Raises:
        ServiceCreationError: If the LLM provider is unknown
    """
    provider = llm_config.get("name") or llm_config.get("provider")
    args = llm_config.get("args", {})
    model = args.get("model") or llm_config.get("model")

    if provider == "OpenAI":
        if org_id:
            api_key = fetch_integration_key(org_id, "OpenAI")
            if not api_key:
                raise ServiceCreationError(
                    "OpenAI API key must be configured in Integrations for this organization."
                )
        else:
            api_key = os.getenv("OPENAI_API_KEY")

        # Extract user aggregator params from config, with defaults
        user_aggregator_params = LLMUserAggregatorParams(
            aggregation_timeout=args.get("aggregation_timeout", 0.05)
        )

        service = OpenAILLMService(
            api_key=api_key,
            model=get_llm_model(provider, model),
        )

        # Store user aggregator params on the service instance for later use
        service._user_aggregator_params = user_aggregator_params

        return service
    elif provider == "Kenpath":
        return KenpathLLM(
            vistaar_session_id=vistaar_session_id,
            language=language,
        )
    else:
        raise ServiceCreationError(f"Unknown LLM provider: {provider}")


def create_stt_service(
    stt_config: dict,
    sample_rate: int,
    vad_analyzer: Any = None,
    org_id: Optional[str] = None,
) -> Any:
    """Create an STT service based on configuration.
    
    Args:
        stt_config: STT configuration dict with 'name', 'language', and optional 'args'
        sample_rate: Audio sample rate in Hz
        vad_analyzer: Optional VAD analyzer instance for direct state monitoring
        org_id: Optional organization ID to fetch integration API key from backend
        
    Returns:
        Configured STT service instance
        
    Raises:
        ServiceCreationError: If the STT provider is unknown
    """
    provider = stt_config.get("name")
    language = stt_config.get("language")
    args = stt_config.get("args", {})
    
    # Normalize provider name to match map keys (capitalize first letter, handle special cases)
    provider_map = {
        "deepgram": "Deepgram",
        "google": "Google",
        "openai": "OpenAI",
        "sarvam": "Sarvam",
        "ai4bharat": "AI4Bharat",
        "bhashini": "Bhashini",
        "elevenlabs": "ElevenLabs",
    }
    provider = provider_map.get(provider.lower(), provider)
    
    if provider == "ElevenLabs":
        if org_id:
            api_key = fetch_integration_key(org_id, "ElevenLabs")
            if not api_key:
                raise ServiceCreationError(
                    "ElevenLabs API key must be configured in Integrations for this organization."
                )
        else:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                raise ServiceCreationError("ELEVENLABS_API_KEY is required for ElevenLabs STT")
        raw_model = args.get("model") or stt_config.get("model")
        # ElevenLabs API expects model_id with underscores (e.g. scribe_v2_realtime), not hyphens
        model = (
            raw_model.replace("-", "_") if isinstance(raw_model, str) and raw_model else raw_model
        )
        lang_code = STT_LANGUAGE_MAP[provider].get(language) if language else None
        # Pipecat < 0.0.105 uses params=InputParams(language_code=...); >= 0.0.105 uses settings=
        try:
            from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTSettings
            settings = ElevenLabsRealtimeSTTSettings(
                language=lang_code,
                **({"model": model} if model else {}),
            )
            return ElevenLabsRealtimeSTTService(
                api_key=api_key,
                sample_rate=sample_rate,
                settings=settings,
            )
        except ImportError:
            params = ElevenLabsRealtimeSTTService.InputParams(
                language_code=lang_code,
            )
            return ElevenLabsRealtimeSTTService(
                api_key=api_key,
                sample_rate=sample_rate,
                model=model,
                params=params,
            )
    
    if provider == "Deepgram":
        model = args.get("model")
        if org_id:
            api_key = fetch_integration_key(org_id, "Deepgram")
            if not api_key:
                raise ServiceCreationError(
                    "Deepgram API key must be configured in Integrations for this organization."
                )
        else:
            api_key = os.getenv("DEEPGRAM_API_KEY")
        return DeepgramSTTService(
            api_key=api_key,
            sample_rate=sample_rate,
            live_options=LiveOptions(
                model=model,
                language=STT_LANGUAGE_MAP[provider][language],
                channels=1,
                encoding="linear16",
                sample_rate=sample_rate,
                interim_results=True,
                endpointing=150,
                smart_format=True,
                punctuate=True
            )
        )
    
    elif provider == "Google":
        return GoogleSTTService(
            credentials_path=os.getenv("GOOGLE_STT_CREDENTIALS_PATH", "credentials/google_stt.json"),
            sample_rate=sample_rate,
            params=GoogleSTTService.InputParams(
                languages=[STT_LANGUAGE_MAP[provider][language]]
            )
        )
    
    elif provider == "OpenAI":
        if org_id:
            api_key = fetch_integration_key(org_id, "OpenAI")
            if not api_key:
                raise ServiceCreationError(
                    "OpenAI API key must be configured in Integrations for this organization."
                )
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        return OpenAISTTService(
            api_key=api_key,
            language=STT_LANGUAGE_MAP[provider][language]
        )
    
    elif provider == "AI4Bharat":
        model = args.get("model") or stt_config.get("model")
        if model == "indic-conformer-stt":
            return IndicConformerRESTSTTService(
                language_id=STT_LANGUAGE_MAP[provider][language],
                sample_rate=16000,
                input_sample_rate=sample_rate,
                vad_analyzer=vad_analyzer
            )
        else:
            raise ServiceCreationError(f"Unknown ai4bharat STT model: {model}. Expected 'indic-conformer-stt'")
    
    elif provider == "Bhashini":
        if org_id:
            api_key = fetch_integration_key(org_id, "Bhashini")
            if not api_key:
                raise ServiceCreationError(
                    "Bhashini API key must be configured in Integrations for this organization."
                )
        else:
            api_key = os.getenv("BHASHINI_API_KEY")
        return BhashiniSTTService(
            api_key=api_key,
            language=STT_LANGUAGE_MAP[provider][language],
            service_id=args.get("model", "bhashini/ai4bharat/conformer-multilingual-asr"),
            sample_rate=sample_rate,
        )
    
    elif provider == "Sarvam":
        # Model can be at stt_config.model (from agent config) or stt_config.args.model
        model = args.get("model") or stt_config.get("model") or "saarika:v2.5"
        if org_id:
            api_key = fetch_integration_key(org_id, "Sarvam")
            if not api_key:
                raise ServiceCreationError(
                    "Sarvam API key must be configured in Integrations for this organization."
                )
        else:
            api_key = os.getenv("SARVAM_API_KEY")
        return SarvamSTTService(
            api_key=api_key,
            language=STT_LANGUAGE_MAP[provider][language],
            model=model,
            sample_rate=sample_rate
        )

    else:
        raise ServiceCreationError(f"Unknown STT provider: {provider}")


def create_tts_service(
    tts_config: dict,
    sample_rate: int,
    org_id: Optional[str] = None,
) -> Any:
    """Create a TTS service based on configuration.
    
    Args:
        tts_config: TTS configuration dict with 'name', 'language', and optional 'args'
        sample_rate: Audio sample rate in Hz (used for some services)
        org_id: Optional organization ID to fetch integration API key from backend
        
    Returns:
        Configured TTS service instance
        
    Raises:
        ServiceCreationError: If the TTS provider is unknown
    """
    provider = tts_config.get("name")
    language = tts_config.get("language")
    args = tts_config.get("args", {})
    
    # Normalize provider name to match map keys (capitalize first letter, handle special cases)
    provider_map = {
        "cartesia": "Cartesia",
        "deepgram": "Deepgram",
        "google": "Google",
        "openai": "OpenAI",
        "sarvam": "Sarvam",
        "ai4bharat": "AI4Bharat",
        "bhashini": "Bhashini",
        "elevenlabs": "ElevenLabs",
    }
    provider = provider_map.get(provider.lower(), provider)
    
    if provider == "ElevenLabs":
        if org_id:
            api_key = fetch_integration_key(org_id, "ElevenLabs")
            if not api_key:
                raise ServiceCreationError(
                    "ElevenLabs API key must be configured in Integrations for this organization."
                )
        else:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                raise ServiceCreationError("ELEVENLABS_API_KEY is required for ElevenLabs TTS")
        voice_id = (
            args.get("voice_id")
            or args.get("voice")
            or tts_config.get("voice_id")
            or tts_config.get("speaker")
        )
        if not voice_id:
            raise ServiceCreationError(
                "ElevenLabs TTS requires a voice ID (from ElevenLabs voice library)"
            )
        model = args.get("model") or tts_config.get("model") or "eleven_turbo_v2_5"
        lang_code = TTS_LANGUAGE_MAP[provider].get(language) if language else None
        
        # Consistent with Deepgram/Cartesia: use standard kwargs
        try:
            from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
            from pipecat.transcriptions.language import Language
            
            params = None
            if lang_code:
                try:
                    params = ElevenLabsTTSService.InputParams(language=Language(lang_code))
                except Exception:
                    logger.warning(f"Could not initialize ElevenLabs params with language {lang_code}")

            return ElevenLabsTTSService(
                api_key=api_key,
                voice_id=voice_id,
                model=model,
                params=params,
            )
        except (TypeError, AttributeError, ImportError):
            # Fallback for very old pipecat if needed
            return ElevenLabsTTSService(
                api_key=api_key,
                voice_id=voice_id,
                model=model,
            )
    
    if provider == "Cartesia":
        model = args.get("model")
        voice_id = args.get("voice_id")
        if org_id:
            api_key = fetch_integration_key(org_id, "Cartesia")
            if not api_key:
                raise ServiceCreationError(
                    "Cartesia API key must be configured in Integrations for this organization."
                )
        else:
            api_key = os.getenv("CARTESIA_API_KEY")
        return CartesiaTTSService(
            api_key=api_key,
            model=model,
            encoding="pcm_s16le",
            voice_id=voice_id
        )
    
    elif provider == "Google":
        lang_code = TTS_LANGUAGE_MAP[provider][language]
        voice_id = args.get("voice_id") or tts_config.get("voice_id")
        return GoogleTTSService(
            credentials_path=os.getenv("GOOGLE_TTS_CREDENTIALS_PATH", "credentials/google_tts.json"),
            voice_id=voice_id,
            params=GoogleTTSService.InputParams(language=lang_code)
        )
    
    elif provider == "OpenAI":
        voice = args.get("voice") or tts_config.get("voice_id")
        if org_id:
            api_key = fetch_integration_key(org_id, "OpenAI")
            if not api_key:
                raise ServiceCreationError(
                    "OpenAI API key must be configured in Integrations for this organization."
                )
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        return OpenAITTSService(
            api_key=api_key,
            voice=voice
        )
    
    elif provider == "AI4Bharat":
        model = args.get("model") or tts_config.get("model")
        if model == "indic-parler-tts":
            speaker = tts_config.get("speaker") or args.get("speaker")
            description = tts_config.get("description") or args.get("description")
            return IndicParlerRESTTTSService(
                speaker=speaker,
                description=description,
                sample_rate=sample_rate
            )
        else:
            raise ServiceCreationError(f"Unknown ai4bharat TTS model: {model}. Expected 'indic-parler-tts'")
    
    elif provider == "Bhashini":
        speaker = tts_config.get("speaker") or args.get("speaker")
        description = tts_config.get("description") or args.get("description")
        return BhashiniTTSService(
            speaker=speaker,
            description=description,
            sample_rate=44100
        )
    
    elif provider == "Sarvam":
        model = args.get("model") or tts_config.get("model") or "bulbul:v3"
        speaker = args.get("speaker") or tts_config.get("speaker")
        # Default speaker by model: bulbul:v2 -> anushka, bulbul:v3 -> aditya (per Sarvam docs)
        if not speaker:
            speaker = "anushka" if model == "bulbul:v2" else "aditya"
        pitch = args.get("pitch")
        pace = args.get("pace")
        loudness = args.get("loudness")
        # bulbul:v3 does not support pitch/loudness; do not pass them
        if model == "bulbul:v3":
            pitch = None
            loudness = None
        if org_id:
            api_key = fetch_integration_key(org_id, "Sarvam")
            if not api_key:
                raise ServiceCreationError(
                    "Sarvam API key must be configured in Integrations for this organization."
                )
        else:
            api_key = os.getenv("SARVAM_API_KEY")
            if not api_key:
                raise ServiceCreationError(
                    "SARVAM_API_KEY is required for Sarvam TTS when not using org integration."
                )
        return SarvamTTSService(
            api_key=api_key,
            target_language_code=TTS_LANGUAGE_MAP[provider][language],
            model=model,
            voice_id=speaker,
            pitch=pitch,
            pace=pace,
            loudness=loudness
        )

    elif provider == "Deepgram":
        if org_id:
            api_key = fetch_integration_key(org_id, "Deepgram")
            if not api_key:
                raise ServiceCreationError(
                    "Deepgram API key must be configured in Integrations for this organization."
                )
        else:
            api_key = os.getenv("DEEPGRAM_API_KEY")
            if not api_key:
                raise ServiceCreationError("DEEPGRAM_API_KEY is required for Deepgram TTS")
        raw_voice = (
            args.get("voice")
            or args.get("voice_id")
            or tts_config.get("voice_id")
            or tts_config.get("speaker")
            or "aura-2-helena-en"
        )
        # Frontend sends short names (e.g. thalia); Deepgram expects full model id (aura-2-thalia-en)
        if isinstance(raw_voice, str) and raw_voice.strip():
            v = raw_voice.strip()
            if v.startswith("aura-"):
                voice = v
            else:
                voice = f"aura-2-{v.lower()}-en"
        else:
            voice = "aura-2-helena-en"
        encoding = args.get("encoding", "linear16")
        if encoding not in ("linear16", "mulaw", "alaw"):
            logger.warning(
                f"Deepgram TTS encoding {encoding!r} may be unsupported; using linear16"
            )
            encoding = "linear16"
        return DeepgramTTSService(
            api_key=api_key,
            voice=voice,
            sample_rate=sample_rate,
            encoding=encoding,
        )

    else:
        raise ServiceCreationError(f"Unknown TTS provider: {provider}")
