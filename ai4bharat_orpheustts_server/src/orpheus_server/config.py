"""Configuration.

One YAML file is the source of truth; every key can be overridden by a flat
``ORPHEUS_*`` environment variable (so Docker/Compose can tune the service
without editing files or rebuilding the image).

Resolution order, lowest priority first:

  1. the defaults declared on the models below
  2. ``config.yaml`` (path from ``ORPHEUS_CONFIG``, default ``<repo>/config.yaml``)
  3. ``ORPHEUS_*`` environment variables

Nothing here is GPU-model-specific. ``check_hardware()`` only *warns*; it never
changes behaviour, so a config that is wrong for the installed GPU produces a
readable message instead of a vLLM stack trace.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]

_NULLISH = {"", "none", "null", "nil", "~"}


class ModelConfig(BaseModel):
    """Which checkpoint to serve and in what precision."""

    model_config = ConfigDict(protected_namespaces=())

    path: str = Field(
        "models/orpheus-indic-5679",
        description="Local directory or HuggingFace repo id. Relative paths resolve against the repo root.",
    )
    dtype: str = Field(
        "auto",
        description="'auto' lets vLLM pick (bfloat16 on Ampere+, float16 below). Or set bfloat16/float16.",
    )
    quantization: Optional[str] = Field(
        "fp8",
        description="'fp8' for online FP8 (needs compute capability >= 8.9). null = serve at dtype.",
    )
    max_model_len: int = Field(8192, description="Context window: prompt + generated audio tokens.")
    trust_remote_code: bool = False


class EngineConfig(BaseModel):
    """vLLM engine capacity and admission settings."""

    gpu_memory_utilization: float = Field(
        0.90, gt=0.0, le=1.0,
        description="Fraction of GPU memory vLLM reserves for weights + KV cache.",
    )
    max_num_seqs: int = Field(
        256, ge=1,
        description="Concurrent-sequence admission limit. This is an ADMISSION policy, not a "
                    "throughput control - see the tuning table in config.yaml and the README.",
    )
    enforce_eager: bool = Field(
        False, description="true skips CUDA-graph capture: faster boot, less VRAM, slower decode.",
    )
    tensor_parallel_size: int = Field(1, ge=1, description="Number of GPUs to shard the model across.")
    max_tokens_default: int = Field(
        8192, ge=64,
        description="Default generation cap when a request omits max_tokens. ~85 ms of audio per token.",
    )
    max_tokens_limit: int = Field(8192, ge=64, description="Hard server-side ceiling on max_tokens.")


class DecoderConfig(BaseModel):
    """SNAC audio-codec decoder."""

    device: str = Field("cuda", description="Torch device for the SNAC codec ('cuda', 'cuda:1', 'cpu').")
    max_batch: int = Field(
        256, ge=1,
        description="Max SNAC windows coalesced into one decode call. Keep >= engine.max_num_seqs, "
                    "or decode batches less than it could and becomes a bottleneck under load.",
    )
    model_id: str = Field("hubertsiuzdak/snac_24khz", description="SNAC codec weights (~76 MB, from HF).")


class SamplingConfig(BaseModel):
    """Generation sampling. These are stack-verified for Orpheus - changing them degrades audio."""

    temperature: float = 0.6
    top_p: float = 0.8
    repetition_penalty: float = 1.3
    min_tokens: int = Field(
        28, ge=0,
        description="Floor before the stop token is honoured; 28 = one full decode window (~0.34 s).",
    )


class WarmupConfig(BaseModel):
    """Pre-compile Triton kernels at boot so real traffic never pays a JIT spike."""

    enabled: bool = True
    concurrency_widths: list[int] = Field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256],
        description="Batch widths to pre-compile. Should reach engine.max_num_seqs. Each width costs "
                    "boot time; trimming the list shifts that cost onto the first real burst instead.",
    )
    max_tokens: int = Field(
        64, ge=28,
        description="Tokens per warmup request. Kernels compile per batch WIDTH, not per length, so "
                    "this only needs to be long enough to run the sampler a few times.",
    )


class ServerConfig(BaseModel):
    """HTTP surface."""

    host: str = "0.0.0.0"
    port: int = 9000
    model_name: str = Field("orpheus-indic", description="Id reported by GET /v1/models.")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"], description="Allowed CORS origins.")


class Settings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: ModelConfig = Field(default_factory=ModelConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    voices_file: str = Field("voices.json", description="Roster of languages, speakers and styles.")

    def resolved_model_path(self) -> str:
        """Absolute path for a local checkpoint; an HF repo id is returned unchanged."""
        p = Path(self.model.path)
        if p.is_absolute():
            return str(p)
        candidate = (REPO_ROOT / p).resolve()
        if candidate.exists():
            return str(candidate)
        # Not on disk: treat as a HuggingFace repo id (contains a '/', no local dir).
        return self.model.path

    def resolved_voices_file(self) -> Path:
        p = Path(self.voices_file)
        return p if p.is_absolute() else (REPO_ROOT / p)


# ---------------------------------------------------------------------------
# Flat environment overrides. Keeping this map explicit (rather than deriving
# ORPHEUS_ENGINE__MAX_NUM_SEQS-style nested names) buys short, documentable
# variable names -- these are what docker-compose.yml and .env.example use.
# ---------------------------------------------------------------------------
_ENV_MAP: dict[str, tuple[str, ...]] = {
    "ORPHEUS_MODEL_PATH": ("model", "path"),
    "ORPHEUS_DTYPE": ("model", "dtype"),
    "ORPHEUS_QUANTIZATION": ("model", "quantization"),
    "ORPHEUS_MAX_MODEL_LEN": ("model", "max_model_len"),
    "ORPHEUS_TRUST_REMOTE_CODE": ("model", "trust_remote_code"),
    "ORPHEUS_GPU_MEMORY_UTILIZATION": ("engine", "gpu_memory_utilization"),
    "ORPHEUS_MAX_NUM_SEQS": ("engine", "max_num_seqs"),
    "ORPHEUS_ENFORCE_EAGER": ("engine", "enforce_eager"),
    "ORPHEUS_TENSOR_PARALLEL_SIZE": ("engine", "tensor_parallel_size"),
    "ORPHEUS_MAX_TOKENS_DEFAULT": ("engine", "max_tokens_default"),
    "ORPHEUS_MAX_TOKENS_LIMIT": ("engine", "max_tokens_limit"),
    "ORPHEUS_DECODER_DEVICE": ("decoder", "device"),
    "ORPHEUS_DECODER_MAX_BATCH": ("decoder", "max_batch"),
    "ORPHEUS_SNAC_MODEL_ID": ("decoder", "model_id"),
    "ORPHEUS_TEMPERATURE": ("sampling", "temperature"),
    "ORPHEUS_TOP_P": ("sampling", "top_p"),
    "ORPHEUS_REPETITION_PENALTY": ("sampling", "repetition_penalty"),
    "ORPHEUS_MIN_TOKENS": ("sampling", "min_tokens"),
    "ORPHEUS_WARMUP_ENABLED": ("warmup", "enabled"),
    "ORPHEUS_WARMUP_WIDTHS": ("warmup", "concurrency_widths"),
    "ORPHEUS_WARMUP_MAX_TOKENS": ("warmup", "max_tokens"),
    "ORPHEUS_HOST": ("server", "host"),
    "ORPHEUS_PORT": ("server", "port"),
    "ORPHEUS_MODEL_NAME": ("server", "model_name"),
    "ORPHEUS_CORS_ORIGINS": ("server", "cors_origins"),
    "ORPHEUS_VOICES_FILE": ("voices_file",),
}

# Values these env vars carry as comma-separated lists.
_LIST_ENVS = {"ORPHEUS_WARMUP_WIDTHS", "ORPHEUS_CORS_ORIGINS"}
# Values where an empty/"none" string means "unset this option".
_NULLABLE_ENVS = {"ORPHEUS_QUANTIZATION"}


def _assign(tree: dict, path: tuple[str, ...], value: object) -> None:
    node = tree
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _apply_env(tree: dict) -> list[str]:
    """Overlay ORPHEUS_* variables onto the config tree. Returns the names applied."""
    applied = []
    for env_name, path in _ENV_MAP.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        if env_name in _NULLABLE_ENVS and raw.strip().lower() in _NULLISH:
            value: object = None
        elif env_name in _LIST_ENVS:
            value = [part.strip() for part in raw.split(",") if part.strip()]
        else:
            value = raw
        _assign(tree, path, value)
        applied.append(env_name)
    return applied


def load_settings(config_path: Optional[str] = None) -> tuple[Settings, list[str]]:
    """Build Settings from YAML + environment. Returns (settings, notes_for_logging)."""
    notes: list[str] = []
    path = Path(config_path or os.environ.get("ORPHEUS_CONFIG") or (REPO_ROOT / "config.yaml"))
    tree: dict = {}
    if path.is_file():
        tree = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        notes.append(f"config file: {path}")
    else:
        notes.append(f"config file not found at {path} - using defaults + environment")
    applied = _apply_env(tree)
    if applied:
        notes.append("env overrides: " + ", ".join(sorted(applied)))
    return Settings.model_validate(tree), notes


# ---------------------------------------------------------------------------
# Hardware advisory. Warns only -- never branches on the GPU model.
# ---------------------------------------------------------------------------
FP8_MIN_CAPABILITY = 89  # SM 8.9: Ada Lovelace, Hopper, Blackwell and newer


def check_hardware(settings: Settings) -> list[tuple[str, str]]:
    """Compare the config against the installed GPU.

    Returns ``(level, message)`` pairs where level is ``"info"`` or ``"warning"``,
    so a description of the hardware does not get logged as though it were a
    problem. Nothing here changes behaviour - it only reports.
    """
    notes: list[tuple[str, str]] = []
    warnings: list[str] = []
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency in practice
        return [("warning", f"could not import torch to check hardware: {exc}")]

    if not torch.cuda.is_available():
        return [("warning",
                 "no CUDA device visible. With Docker, check that nvidia-container-toolkit is "
                 "installed and the compose 'devices' reservation is present.")]

    count = torch.cuda.device_count()
    major, minor = torch.cuda.get_device_capability(0)
    capability = major * 10 + minor
    name = torch.cuda.get_device_name(0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    notes.append((
        "info",
        f"GPU 0: {name}, compute capability {major}.{minor}, {total_gib:.1f} GiB, "
        f"{count} device(s) visible",
    ))

    if settings.model.quantization == "fp8" and capability < FP8_MIN_CAPABILITY:
        warnings.append(
            f"config sets model.quantization=fp8 but this GPU is compute capability {major}.{minor}; "
            f"native FP8 needs >= 8.9 (Ada/Hopper/Blackwell). Set quantization: null "
            f"(or ORPHEUS_QUANTIZATION=none) or expect vLLM to fail or fall back to slow emulation."
        )
    if settings.engine.tensor_parallel_size > count:
        warnings.append(
            f"config sets engine.tensor_parallel_size={settings.engine.tensor_parallel_size} but only "
            f"{count} GPU(s) are visible; vLLM will fail to start."
        )
    if settings.decoder.max_batch < settings.engine.max_num_seqs:
        warnings.append(
            f"decoder.max_batch ({settings.decoder.max_batch}) is below engine.max_num_seqs "
            f"({settings.engine.max_num_seqs}); SNAC decode will batch less than it could under load."
        )
    if settings.warmup.enabled and settings.warmup.concurrency_widths:
        widest = max(settings.warmup.concurrency_widths)
        if widest < settings.engine.max_num_seqs:
            warnings.append(
                f"widest warmup width ({widest}) is below engine.max_num_seqs "
                f"({settings.engine.max_num_seqs}); the first burst at full width will pay a one-time "
                f"Triton JIT cost mid-stream."
            )
    return notes + [("warning", w) for w in warnings]
