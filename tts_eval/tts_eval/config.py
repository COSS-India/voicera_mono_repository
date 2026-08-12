"""Model cards and suite configs — the layer that makes new models config-only.

Two YAML documents, with deliberately separate jobs:

*   **Model card** (``configs/models/<id>.yaml``) — describes *what is being
    evaluated*: identity, wire protocol, endpoint, declared languages and voices,
    generation params. One file per model version. This is the file you add when a
    new TTS model arrives.
*   **Suite** (``configs/suites/<id>.yaml``) — describes *how it is evaluated*:
    test set, metric tier, ASR backend, thresholds, concurrency, seed. One file
    per benchmark protocol, shared by every model so comparisons are apples to
    apples.

Keeping them apart is what makes the framework generalise. A new model reuses
every suite unchanged; a new benchmark protocol applies to every existing model
unchanged. If the two were one file, adding a model would mean copying and
diverging the protocol, and cross-model comparison would quietly stop being valid.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError
from .metrics.base import Thresholds
from .types import Determinism

# Repo-relative default search paths, checked before treating a name as a path.
CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"
MODELS_DIR = CONFIG_ROOT / "models"
SUITES_DIR = CONFIG_ROOT / "suites"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in string values.

    Endpoints and credentials belong in the environment, not in a committed YAML.
    Unset variables without a default are left verbatim so the resulting error
    names the missing variable instead of failing as "cannot connect to ''".
    """
    if isinstance(value, str):
        def sub(m: re.Match[str]) -> str:
            got = os.getenv(m.group(1))
            if got is not None:
                return got
            return m.group(2) if m.group(2) is not None else m.group(0)

        return _ENV_PATTERN.sub(sub, value)
    if isinstance(value, Mapping):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"{path}: could not parse YAML: {e}") from e
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")
    return dict(_expand(dict(data)))


def _resolve(spec: str | Path, search_dir: Path, kind: str) -> Path:
    """Resolve a bare name against the bundled config dir, else treat as a path."""
    if isinstance(spec, str) and not any(sep in spec for sep in ("/", "\\")):
        for suffix in (".yaml", ".yml"):
            candidate = search_dir / f"{spec}{suffix}"
            if candidate.is_file():
                return candidate
    path = Path(spec).expanduser()
    if path.is_file():
        return path
    available = ", ".join(sorted(p.stem for p in search_dir.glob("*.y*ml"))) or "(none)"
    raise ConfigError(f"{kind} {spec!r} not found; bundled options: {available}")


# ---------------------------------------------------------------------------
# model card
# ---------------------------------------------------------------------------
@dataclass
class ModelCard:
    """Identity and wire configuration for one evaluatable model version."""

    model_id: str
    model_version: str
    provider: str
    adapter: str
    adapter_config: dict[str, Any] = field(default_factory=dict)
    # Languages the model *claims* to support. Coverage measures the claim.
    languages: tuple[str, ...] = ()
    voices: tuple[str, ...] = ()
    default_voice: str | None = None
    generation_params: dict[str, Any] = field(default_factory=dict)
    supports_seed: bool = False
    supports_emotion: bool = False
    sample_rate: int | None = None
    determinism: Determinism = Determinism.BEST_EFFORT
    # Dotted module path registering a custom adapter, for out-of-tree providers.
    adapter_module: str | None = None
    description: str = ""
    source_path: Path | None = None
    # Anything else in the YAML, preserved so a card can carry provenance
    # (checkpoint hash, HF revision, deployment notes) into the run record.
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN = {
        "model_id",
        "model_version",
        "provider",
        "adapter",
        "adapter_config",
        "languages",
        "voices",
        "default_voice",
        "generation_params",
        "supports_seed",
        "supports_emotion",
        "sample_rate",
        "determinism",
        "adapter_module",
        "description",
    }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: Path | None = None) -> "ModelCard":
        where = str(source or "<inline>")
        for required in ("model_id", "adapter"):
            if not str(data.get(required) or "").strip():
                raise ConfigError(f"{where}: model card is missing required field {required!r}")

        determinism_raw = str(data.get("determinism") or Determinism.BEST_EFFORT.value)
        try:
            determinism = Determinism(determinism_raw)
        except ValueError as e:
            valid = ", ".join(d.value for d in Determinism)
            raise ConfigError(f"{where}: determinism must be one of {valid}, got {determinism_raw!r}") from e

        return cls(
            model_id=str(data["model_id"]).strip(),
            # Unversioned cards are accepted but flagged: comparing two runs of
            # "unversioned" is how a regression gets attributed to the wrong build.
            model_version=str(data.get("model_version") or "unversioned"),
            provider=str(data.get("provider") or "unknown"),
            adapter=str(data["adapter"]).strip(),
            adapter_config=dict(data.get("adapter_config") or {}),
            languages=tuple(str(l) for l in (data.get("languages") or ())),
            voices=tuple(str(v) for v in (data.get("voices") or ())),
            default_voice=(str(data["default_voice"]) if data.get("default_voice") else None),
            generation_params=dict(data.get("generation_params") or {}),
            supports_seed=bool(data.get("supports_seed", False)),
            supports_emotion=bool(data.get("supports_emotion", False)),
            sample_rate=(int(data["sample_rate"]) if data.get("sample_rate") else None),
            determinism=determinism,
            adapter_module=(str(data["adapter_module"]) if data.get("adapter_module") else None),
            description=str(data.get("description") or ""),
            source_path=source,
            extra={k: v for k, v in data.items() if k not in cls._KNOWN},
        )

    def resolved_adapter_config(self) -> dict[str, Any]:
        """Adapter config with card-level declarations folded in.

        Adapters read capabilities from their config, so languages/voices/sample
        rate declared once at card level reach the adapter without being repeated
        under ``adapter_config``. Explicit ``adapter_config`` keys win, so a card
        can still override for an adapter that needs something different.
        """
        merged: dict[str, Any] = {
            "languages": list(self.languages),
            "voices": list(self.voices),
            "supports_seed": self.supports_seed,
            "supports_emotion": self.supports_emotion,
            "determinism": self.determinism.value,
        }
        if self.sample_rate:
            merged["sample_rate"] = self.sample_rate
        merged.update(self.adapter_config)
        return merged

    def to_dict(self) -> dict[str, Any]:
        """Card as stored in the run record.

        Credentials are stripped: the run record is the artefact that gets shared
        and archived, so a token must never reach it even indirectly.
        """
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "provider": self.provider,
            "adapter": self.adapter,
            "adapter_config": _redact(self.adapter_config),
            "languages": list(self.languages),
            "voices": list(self.voices),
            "default_voice": self.default_voice,
            "generation_params": dict(self.generation_params),
            "supports_seed": self.supports_seed,
            "supports_emotion": self.supports_emotion,
            "sample_rate": self.sample_rate,
            "determinism": self.determinism.value,
            "description": self.description,
            "source_path": str(self.source_path) if self.source_path else None,
            "extra": _redact(self.extra),
        }

    @property
    def display_name(self) -> str:
        return f"{self.model_id}@{self.model_version}"


def load_model_card(spec: str | Path) -> ModelCard:
    path = _resolve(spec, MODELS_DIR, "model card")
    return ModelCard.from_dict(_load_yaml(path), source=path)


def list_model_cards() -> list[str]:
    return sorted(p.stem for p in MODELS_DIR.glob("*.y*ml")) if MODELS_DIR.is_dir() else []


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------
@dataclass
class SuiteConfig:
    """The evaluation protocol: identical across every model being compared."""

    suite_id: str = "default"
    description: str = ""
    dataset: str = "indic_conversational_v1"
    # Metric tier ("core" | "standard" | "all") or an explicit backend list.
    metrics: Any = "standard"
    metric_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    asr: dict[str, Any] = field(default_factory=dict)
    thresholds: Thresholds = field(default_factory=Thresholds)
    # Concurrency is part of the protocol, not a convenience knob: latency and
    # throughput are only comparable between runs measured at the same load.
    concurrency: int = 1
    # Fixed by default so repeat runs are reproducible without being asked for.
    seed: int | None = 1234
    voice: str | None = None
    languages: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    sample: int | None = None
    generation_params: dict[str, Any] = field(default_factory=dict)
    save_audio: bool = True
    source_path: Path | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: Path | None = None) -> "SuiteConfig":
        where = str(source or "<inline>")
        concurrency = int(data.get("concurrency") or 1)
        if concurrency < 1:
            raise ConfigError(f"{where}: concurrency must be >= 1, got {concurrency}")
        sample = data.get("sample")
        if sample is not None and int(sample) < 1:
            raise ConfigError(f"{where}: sample must be >= 1, got {sample}")

        return cls(
            suite_id=str(data.get("suite_id") or (source.stem if source else "default")),
            description=str(data.get("description") or ""),
            dataset=str(data.get("dataset") or "indic_conversational_v1"),
            metrics=data.get("metrics") or "standard",
            metric_options={
                str(k): dict(v or {}) for k, v in (data.get("metric_options") or {}).items()
            },
            asr=dict(data.get("asr") or {}),
            thresholds=Thresholds.from_dict(data.get("thresholds")),
            concurrency=concurrency,
            # `seed: null` means "explicitly unseeded"; a missing key means default.
            seed=(data.get("seed") if "seed" in data else 1234),
            voice=(str(data["voice"]) if data.get("voice") else None),
            languages=tuple(str(l) for l in (data.get("languages") or ())),
            categories=tuple(str(c) for c in (data.get("categories") or ())),
            sample=(int(sample) if sample is not None else None),
            generation_params=dict(data.get("generation_params") or {}),
            save_audio=bool(data.get("save_audio", True)),
            source_path=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "description": self.description,
            "dataset": self.dataset,
            "metrics": self.metrics,
            "metric_options": _redact(self.metric_options),
            "asr": _redact(self.asr),
            "thresholds": self.thresholds.to_dict(),
            "concurrency": self.concurrency,
            "seed": self.seed,
            "voice": self.voice,
            "languages": list(self.languages),
            "categories": list(self.categories),
            "sample": self.sample,
            "generation_params": dict(self.generation_params),
            "save_audio": self.save_audio,
            "source_path": str(self.source_path) if self.source_path else None,
        }


def load_suite(spec: str | Path) -> SuiteConfig:
    path = _resolve(spec, SUITES_DIR, "suite config")
    return SuiteConfig.from_dict(_load_yaml(path), source=path)


def list_suites() -> list[str]:
    return sorted(p.stem for p in SUITES_DIR.glob("*.y*ml")) if SUITES_DIR.is_dir() else []


# ---------------------------------------------------------------------------
def _redact(obj: Any) -> Any:
    """Recursively mask anything that looks like a credential."""
    secret_markers = ("key", "token", "secret", "password", "auth", "credential")
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if any(marker in str(k).lower() for marker in secret_markers):
                out[str(k)] = "***redacted***"
            else:
                out[str(k)] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


__all__ = [
    "CONFIG_ROOT",
    "MODELS_DIR",
    "SUITES_DIR",
    "ModelCard",
    "SuiteConfig",
    "list_model_cards",
    "list_suites",
    "load_model_card",
    "load_suite",
]
