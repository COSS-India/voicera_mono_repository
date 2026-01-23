"""Service factory functions for creating LLM, STT, and TTS services."""

import os
from typing import Any

from loguru import logger
from deepgram import LiveOptions

# Pipecat services
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
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


class ServiceCreationError(Exception):
    """Raised when a service cannot be created."""
    pass


def create_llm_service(llm_config: dict) -> Any:
    """Create an LLM service based on configuration.
    
    Args:
        llm_config: LLM configuration dict with 'name' and optional 'args'
        
    Returns:
        Configured LLM service instance
        
    Raises:
        ServiceCreationError: If the LLM provider is unknown
    """
    provider = llm_config.get("name") or llm_config.get("provider")
    args = llm_config.get("args", {})
    model = args.get("model") or llm_config.get("model")
    
    if provider == "OpenAI":
        # Extract user aggregator params from config, with defaults
        user_aggregator_params = LLMUserAggregatorParams(
            aggregation_timeout=args.get("aggregation_timeout", 0.1)
        )
        
        service = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=get_llm_model(provider, model),
        )
        
        # Store user aggregator params on the service instance for later use
        service._user_aggregator_params = user_aggregator_params
        
        return service
    elif provider == "Kenpath":
        return KenpathLLM()
    else:
        raise ServiceCreationError(f"Unknown LLM provider: {provider}")


def create_stt_service(stt_config: dict, sample_rate: int, vad_analyzer: Any = None) -> Any:
    """Create an STT service based on configuration.
    
    Args:
        stt_config: STT configuration dict with 'name', 'language', and optional 'args'
        sample_rate: Audio sample rate in Hz
        vad_analyzer: Optional VAD analyzer instance for direct state monitoring
        
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
    }
    provider = provider_map.get(provider.lower(), provider)
    
    if provider == "Deepgram":
        model = args.get("model")
        return DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
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
        return OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            language=STT_LANGUAGE_MAP[provider][language]
        )
    
    elif provider == "AI4Bharat":
        model = args.get("model") or stt_config.get("model")
        if model == "indic-conformer-stt":
            return IndicConformerRESTSTTService(
                language_id=STT_LANGUAGE_MAP[provider][language],
                sample_rate=16000,
                input_sample_rate=8000,
                vad_analyzer=vad_analyzer
            )
        else:
            raise ServiceCreationError(f"Unknown ai4bharat STT model: {model}. Expected 'indic-conformer-stt'")
    
    elif provider == "Bhashini":
        return BhashiniSTTService(
            api_key=os.getenv("BHASHINI_API_KEY"),
            language=STT_LANGUAGE_MAP[provider][language],
            service_id=args.get("model", "bhashini/ai4bharat/conformer-multilingual-asr"),
            sample_rate=sample_rate,
        )
    
    elif provider == "Sarvam":
        model = args.get("model")
        return SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY"),
            language=STT_LANGUAGE_MAP[provider][language],
            model=model,
            sample_rate=sample_rate
        )

    else:
        raise ServiceCreationError(f"Unknown STT provider: {provider}")


def create_tts_service(tts_config: dict, sample_rate: int) -> Any:
    """Create a TTS service based on configuration.
    
    Args:
        tts_config: TTS configuration dict with 'name', 'language', and optional 'args'
        sample_rate: Audio sample rate in Hz (used for some services)
        
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
        "google": "Google",
        "openai": "OpenAI",
        "sarvam": "Sarvam",
        "ai4bharat": "AI4Bharat",
        "bhashini": "Bhashini"
    }
    provider = provider_map.get(provider.lower(), provider)
    
    if provider == "Cartesia":
        model = args.get("model")
        voice_id = args.get("voice_id")
        return CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
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
        return OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
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
                sample_rate=44100,
                play_steps_in_s=0.5
            )
        else:
            raise ServiceCreationError(f"Unknown ai4bharat TTS model: {model}. Expected 'indic-parler-tts'")
    
    elif provider == "Bhashini":
        speaker = tts_config.get("speaker") or args.get("speaker")
        description = tts_config.get("description") or args.get("description")
        return BhashiniTTSService(
            speaker=speaker,
            description=description,
            sample_rate=44100,
            play_steps_in_s=0.5
        )
    
    elif provider == "Sarvam":
        model = args.get("model")
        speaker = args.get("speaker") or tts_config.get("speaker")
        pitch = args.get("pitch")
        pace = args.get("pace")
        loudness = args.get("loudness")
        return SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            target_language_code=TTS_LANGUAGE_MAP[provider][language],
            model=model,
            speaker=speaker,
            pitch=pitch,
            pace=pace,
            loudness=loudness
        )
    
    else:
        raise ServiceCreationError(f"Unknown TTS provider: {provider}")
