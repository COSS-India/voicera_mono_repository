"""API module for voice bot server.

This module provides:
- FastAPI application for handling telephony webhooks
- Voice bot pipeline implementation
- Service factories for LLM, STT, and TTS providers
"""

from .bot import bot, run_bot
from .server import app
from .services import (
    create_llm_service,
    create_stt_service,
    create_tts_service,
    ServiceCreationError,
)

__all__ = [
    # Bot
    "bot",
    "run_bot",
    # Server
    "app",
    # Services
    "create_llm_service",
    "create_stt_service",
    "create_tts_service",
    "ServiceCreationError",
]
