"""Runtime configuration for the Indic-Mio TTS server.

All knobs are environment-driven so the same image runs unchanged across dev and
prod (mirrors the AI4Bharat server's env-first approach). Nothing here loads a
model; that happens in tts_engine.MioTTSEngine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Speech tokens are emitted by the LLM as the literal strings "<|s_1234|>".
# The captured integer IS the MioCodec content-token index (i.e. it already has
# the LLM vocab offset removed). This mirrors the official MioTTS-Inference
# token_parser and is the authoritative extraction path when serving via vLLM.
SPEECH_TOKEN_PATTERN = r"<\|s_(\d+)\|>"

# Fallback output sample rate. The real value is read from the codec at load
# time (codec.config.sample_rate); this is only used if the codec does not
# expose it. The SPRINGLab Indic-Mio card writes WAVs at 44100 Hz.
DEFAULT_SAMPLE_RATE = 44100


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value.strip() else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # WebSocket front (voicera contract, identical framing to ai4bharat tts).
    host: str = "0.0.0.0"
    port: int = 8003

    # vLLM OpenAI-compatible backend that runs Indic-Mio token generation.
    # base_url must include the /v1 suffix, e.g. http://vllm-mio:8100/v1
    llm_base_url: str = "http://localhost:8100/v1"
    llm_model: str = "SPRINGLab/Indic-Mio"
    llm_api_key: str = "EMPTY"  # vLLM ignores the value but the header must exist
    llm_timeout: float = 300.0

    # Generation params (SPRINGLab card defaults).
    temperature: float = 0.9
    top_p: float = 0.9
    max_tokens: int = 1024
    repetition_penalty: float = 1.0

    # MioCodec that decodes content tokens -> waveform. Colocated on the same GPU
    # as vLLM (vLLM caps itself at --gpu-memory-utilization 0.5, leaving room).
    codec_model_id: str = "Aratako/MioCodec-25Hz-24kHz"
    device: str = "cuda"

    # Bound concurrent GPU decodes so many in-flight WS requests cannot thrash
    # VRAM. vLLM already batches the (heavier) token-gen stage server-side.
    decode_concurrency: int = 2

    # --- streaming decode (low TTFB) -------------------------------------
    # Generation streams token-by-token from vLLM; we decode incrementally and
    # push PCM as it is produced instead of waiting for the whole utterance.
    stream_decode: bool = True
    # Decode cadence: run the codec once per this many newly generated tokens.
    # MioCodec is 25 Hz, so 32 tokens ~= 1.3 s of audio per flush.
    flush_tokens: int = 32
    # Look-ahead hold-back. Each incremental decode is over the full token prefix
    # (full left context), but the final tokens lack right context. We hold back
    # this many tokens' worth of tail samples until more tokens arrive, so emitted
    # audio always had >= this much right context -> no flush-boundary artifacts.
    lookahead_tokens: int = 6
    # MioCodec content-token frame rate (Hz), used to size the hold-back in samples.
    token_rate_hz: int = 25

    # Size of each binary PCM frame streamed over the socket (float32 samples).
    frame_samples: int = 8192

    # Reject absurd inputs early.
    max_text_length: int = 2000

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            host=_env("MIO_HOST", cls.host),
            port=_env_int("MIO_PORT", cls.port),
            llm_base_url=_env("INDIC_MIO_VLLM_URL", cls.llm_base_url).rstrip("/"),
            llm_model=_env("MIO_LLM_MODEL", cls.llm_model),
            llm_api_key=_env("MIO_LLM_API_KEY", cls.llm_api_key),
            llm_timeout=_env_float("MIO_LLM_TIMEOUT", cls.llm_timeout),
            temperature=_env_float("MIO_TEMPERATURE", cls.temperature),
            top_p=_env_float("MIO_TOP_P", cls.top_p),
            max_tokens=_env_int("MIO_MAX_TOKENS", cls.max_tokens),
            repetition_penalty=_env_float("MIO_REPETITION_PENALTY", cls.repetition_penalty),
            codec_model_id=_env("MIO_CODEC_MODEL_ID", cls.codec_model_id),
            device=_env("MIO_DEVICE", cls.device),
            decode_concurrency=_env_int("MIO_DECODE_CONCURRENCY", cls.decode_concurrency),
            stream_decode=_env("MIO_STREAM_DECODE", str(cls.stream_decode)).lower()
            in ("1", "true", "yes"),
            flush_tokens=_env_int("MIO_FLUSH_TOKENS", cls.flush_tokens),
            lookahead_tokens=_env_int("MIO_LOOKAHEAD_TOKENS", cls.lookahead_tokens),
            token_rate_hz=_env_int("MIO_TOKEN_RATE_HZ", cls.token_rate_hz),
            frame_samples=_env_int("MIO_FRAME_SAMPLES", cls.frame_samples),
            max_text_length=_env_int("MIO_MAX_TEXT_LENGTH", cls.max_text_length),
        )
