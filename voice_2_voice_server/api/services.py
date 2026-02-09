"""Service factory functions for creating LLM, STT, and TTS services."""

import os
from typing import Any

from loguru import logger
from deepgram import LiveOptions

# Pipecat services
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService, Language

# NOTE: LLMUserAggregatorParams is no longer needed here.
# In Pipecat 0.0.101, user aggregator params (aggregation_timeout, etc.)
# are configured directly in bot.py via LLMContextAggregatorPair.

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


# Map string language codes (from our config) to Pipecat Language enum
_SARVAM_LANGUAGE_ENUM_MAP = {
    "hi-IN": Language.HI_IN,
    "en-IN": Language.EN_IN,
    "en-US": Language.EN_US,
    "bn-IN": Language.BN_IN,
    "ta-IN": Language.TA_IN,
    "te-IN": Language.TE_IN,
    "gu-IN": Language.GU_IN,
    "kn-IN": Language.KN_IN,
    "ml-IN": Language.ML_IN,
    "mr-IN": Language.MR_IN,
    "pa-IN": Language.PA_IN,
    "od-IN": Language.OR_IN,
    "as-IN": Language.AS_IN,
}


def _get_sarvam_language(language_code: str) -> Language:
    """Convert a string language code to Pipecat Language enum for Sarvam services."""
    lang = _SARVAM_LANGUAGE_ENUM_MAP.get(language_code)
    if lang is None:
        logger.warning(
            f"Unknown Sarvam language code: {language_code}, defaulting to en-IN"
        )
        return Language.EN_IN
    return lang


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

    # Normalize provider name
    provider_map = {
        "openai": "OpenAI",
        "gemini": "Gemini",
        "google": "Gemini",
        "kenpath": "Kenpath",
    }
    provider = provider_map.get(provider.lower(), provider) if provider else provider

    if provider == "OpenAI":
        resolved_model = get_llm_model(provider, model)
        logger.info(f"OpenAI LLM: model={resolved_model}")

        return OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=resolved_model,
        )

    elif provider == "Gemini":
        resolved_model = get_llm_model(provider, model)
        logger.info(f"Gemini LLM: model={resolved_model}")

        return GoogleLLMService(
            api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            model=resolved_model,
        )

    elif provider == "Kenpath":
        return KenpathLLM()

    else:
        raise ServiceCreationError(f"Unknown LLM provider: {provider}")


def create_stt_service(
    stt_config: dict, sample_rate: int, vad_analyzer: Any = None
) -> Any:
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
        # Model is at top level for Deepgram (not in args)
        model = stt_config.get("model") or args.get("model") or "nova-2"
        logger.info(f"Deepgram STT: model={model}, language={language}")
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
                punctuate=True,
            ),
        )

    elif provider == "Google":
        return GoogleSTTService(
            credentials_path=os.getenv(
                "GOOGLE_STT_CREDENTIALS_PATH", "credentials/google_stt.json"
            ),
            sample_rate=sample_rate,
            params=GoogleSTTService.InputParams(
                languages=[STT_LANGUAGE_MAP[provider][language]]
            ),
        )

    elif provider == "OpenAI":
        return OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            language=STT_LANGUAGE_MAP[provider][language],
        )

    elif provider == "AI4Bharat":
        model = args.get("model") or stt_config.get("model")
        if model == "indic-conformer-stt":
            return IndicConformerRESTSTTService(
                language_id=STT_LANGUAGE_MAP[provider][language],
                sample_rate=16000,
                input_sample_rate=sample_rate,
                vad_analyzer=vad_analyzer,
            )
        else:
            raise ServiceCreationError(
                f"Unknown ai4bharat STT model: {model}. Expected 'indic-conformer-stt'"
            )

    elif provider == "Bhashini":
        return BhashiniSTTService(
            api_key=os.getenv("BHASHINI_API_KEY"),
            language=STT_LANGUAGE_MAP[provider][language],
            service_id=args.get(
                "model", "bhashini/ai4bharat/conformer-multilingual-asr"
            ),
            sample_rate=sample_rate,
        )

    elif provider == "Sarvam":
        # Model is at top level for Sarvam (not in args)
        model = stt_config.get("model") or args.get("model") or "saarika:v2.5"
        lang_code = STT_LANGUAGE_MAP[provider][language]
        sarvam_lang = _get_sarvam_language(lang_code)
        logger.info(f"Sarvam STT: model={model}, language={language} ({lang_code})")
        return SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY"),
            model=model,
            sample_rate=sample_rate,
            params=SarvamSTTService.InputParams(
                language=sarvam_lang,
            ),
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
        "bhashini": "Bhashini",
        "deepgram": "Deepgram",
    }
    provider = provider_map.get(provider.lower(), provider)

    if provider == "Deepgram":
        # Deepgram voice format depends on model version:
        # - Aura-2: aura-2-{voice}-en (e.g., aura-2-thalia-en, aura-2-arcas-en)
        # - Aura-1: aura-{voice}-en (e.g., aura-asteria-en, aura-arcas-en)
        speaker = args.get("speaker") or tts_config.get("speaker") or "asteria"
        model = args.get("model") or tts_config.get("model") or "aura-2"

        # If the voice already has the full format, use it as-is
        if speaker.startswith("aura-"):
            voice = speaker
        else:
            # Construct voice based on model version
            if model == "aura-2":
                voice = f"aura-2-{speaker}-en"
            else:
                # Aura-1 or other versions
                voice = f"aura-{speaker}-en"

        logger.info(f"Deepgram TTS: model={model}, speaker={speaker}, voice={voice}")
        return DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            voice=voice,
        )

    elif provider == "Cartesia":
        model = args.get("model")
        voice_id = args.get("voice_id")
        return CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            model=model,
            encoding="pcm_s16le",
            voice_id=voice_id,
        )

    elif provider == "Google":
        lang_code = TTS_LANGUAGE_MAP[provider][language]
        voice_id = args.get("voice_id") or tts_config.get("voice_id")
        return GoogleTTSService(
            credentials_path=os.getenv(
                "GOOGLE_TTS_CREDENTIALS_PATH", "credentials/google_tts.json"
            ),
            voice_id=voice_id,
            params=GoogleTTSService.InputParams(language=lang_code),
        )

    elif provider == "OpenAI":
        # OpenAI TTS models: tts-1, tts-1-hd, gpt-4o-mini-tts
        # Voices: alloy, echo, fable, onyx, nova, shimmer
        model = tts_config.get("model") or args.get("model") or "tts-1"
        voice = (
            tts_config.get("speaker")
            or args.get("voice")
            or tts_config.get("voice_id")
            or "alloy"
        )
        logger.info(f"OpenAI TTS: model={model}, voice={voice}")
        return OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            voice=voice,
        )

    elif provider == "AI4Bharat":
        model = args.get("model") or tts_config.get("model")
        if model == "indic-parler-tts":
            speaker = tts_config.get("speaker") or args.get("speaker")
            description = tts_config.get("description") or args.get("description")
            return IndicParlerRESTTTSService(
                speaker=speaker, description=description, sample_rate=sample_rate
            )
        else:
            raise ServiceCreationError(
                f"Unknown ai4bharat TTS model: {model}. Expected 'indic-parler-tts'"
            )

    elif provider == "Bhashini":
        speaker = tts_config.get("speaker") or args.get("speaker")
        description = tts_config.get("description") or args.get("description")
        return BhashiniTTSService(
            speaker=speaker, description=description, sample_rate=44100
        )

    elif provider == "Sarvam":
        # Sarvam config is at top level (not in args)
        model = tts_config.get("model") or args.get("model") or "bulbul:v2"
        voice_id = tts_config.get("speaker") or args.get("speaker") or "anushka"
        lang_code = TTS_LANGUAGE_MAP[provider][language]
        sarvam_lang = _get_sarvam_language(lang_code)

        # Build InputParams based on model version:
        # - bulbul:v2: supports pitch, pace, loudness
        # - bulbul:v3 / bulbul:v3-beta: supports temperature (no pace/loudness)
        if model in ("bulbul:v3", "bulbul:v3-beta"):
            temperature = (
                tts_config.get("temperature") or args.get("temperature") or 0.6
            )
            logger.info(
                f"Sarvam TTS: model={model}, voice={voice_id}, "
                f"language={lang_code}, temperature={temperature}"
            )
            params = SarvamTTSService.InputParams(
                language=sarvam_lang,
                temperature=float(temperature),
            )
        else:
            pitch = tts_config.get("pitch") or args.get("pitch")
            pace = tts_config.get("pace") or args.get("pace") or tts_config.get("speed")
            loudness = tts_config.get("loudness") or args.get("loudness")
            logger.info(
                f"Sarvam TTS: model={model}, voice={voice_id}, "
                f"language={lang_code}, pitch={pitch}, pace={pace}, loudness={loudness}"
            )
            params = SarvamTTSService.InputParams(
                language=sarvam_lang,
                pitch=float(pitch) if pitch else 0.0,
                pace=float(pace) if pace else 1.0,
                loudness=float(loudness) if loudness else 1.0,
            )

        return SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            model=model,
            voice_id=voice_id,
            params=params,
        )

    else:
        raise ServiceCreationError(f"Unknown TTS provider: {provider}")
