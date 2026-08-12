"""tts_eval — a standardised, reproducible evaluation harness for TTS models.

Design contract (why this package exists in this shape):

*   **Standalone.** Nothing here imports from `voice_2_voice_server`,
    `voicera_backend` or any other service in this monorepo. It talks to TTS
    servers over the wire only. You can copy this directory out and it still
    works.
*   **Provider-agnostic.** A model is described by a YAML *model card* plus an
    *adapter* naming a wire protocol. Adding a new TTS model is a new YAML file;
    adding a genuinely new protocol is one ~100-line adapter class. The
    framework, metrics, storage and reports never change.
*   **Layered metrics.** A dependency-light core (latency, inference time,
    audio-quality DSP, voice consistency, language coverage) always runs. Heavy
    perceptual metrics (predicted MOS, speaker embeddings, VERSA, TTSDS2) are
    optional backends that report ``not_computed`` with a reason when their
    dependencies are absent — they never crash a run.
*   **Reproducible by construction.** Every run stores the dataset hash, the
    resolved model card, generation params, seed, framework version and a
    ``fingerprint`` derived from all of them. Two runs with equal fingerprints
    are directly comparable; a fingerprint mismatch is surfaced in reports
    instead of being silently averaged over.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Bumped whenever the persisted run-record JSON schema changes in a way that
# older readers cannot handle. Stored inside every run so the loader can refuse
# or migrate rather than misinterpret.
SCHEMA_VERSION = 1

__all__ = ["__version__", "SCHEMA_VERSION"]
