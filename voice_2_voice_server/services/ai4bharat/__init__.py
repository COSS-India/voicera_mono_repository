"""Clients for the model-server slots.

Both classes are named for the slot they speak to, not for a model. The STT slot
takes a WAV and a language; the TTS slot returns audio that says what format it
is in. Which model sits behind either is a deploy-time choice the client never
sees, so `ModelServerSTTService` serves Indic-Conformer and Indic-Transcribe
alike, and `ModelServerTTSService` serves Indic Parler, Orpheus and Indic-Mio.

The `Indic*REST*` names are the originals, kept as aliases: they are what
existing configs and imports say, and renaming a class is not worth breaking a
running deployment over.
"""

from .stt import IndicConformerRESTSTTService, ModelServerSTTService
from .tts import IndicParlerRESTTTSService, ModelServerTTSService

__all__ = [
    "ModelServerSTTService",
    "ModelServerTTSService",
    # Aliases, for callers that predate the rename.
    "IndicConformerRESTSTTService",
    "IndicParlerRESTTTSService",
]
