"""The adapter contract — the seam that makes this framework model-agnostic.

An adapter is the *only* thing that knows a provider's wire protocol. It turns a
:class:`~tts_eval.types.SynthesisRequest` into a
:class:`~tts_eval.types.SynthesisResult` with timings attached. Metrics, storage,
reporting and comparison never learn which provider produced the audio.

Adding a new TTS model therefore costs:

*   **New model, protocol already supported** — one YAML model card. Zero code.
    (Indic-Mio and AI4Bharat Parler share ``websocket_pcm``; Sarvam/ElevenLabs/
    Cartesia/OpenAI all share ``http_rest``.)
*   **New model, new protocol** — one subclass implementing ``_synthesise``,
    typically ~100 lines, registered with ``@register_adapter``.
*   **Out-of-tree adapter** — no fork needed: point ``--adapter-module`` at any
    importable module that registers itself.

Timing discipline lives in the base class, not in subclasses. Subclasses report
*events* (``on_chunk``, ``on_meta``) and the base class converts them into TTFB,
first-audible and per-chunk offsets. If each adapter timed itself, cross-provider
latency numbers would not be comparable — which is the whole point of the suite.
"""
from __future__ import annotations

import abc
import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from ..audio import SILENCE_FLOOR
from ..errors import AdapterUnavailable, ConfigError, SynthesisFailed
from ..types import (
    AudioBuffer,
    Capabilities,
    ChunkTiming,
    Determinism,
    SynthesisRequest,
    SynthesisResult,
)


@dataclass
class _Capture:
    """Mutable timing accumulator handed to subclasses during one synthesis.

    Subclasses call ``chunk()`` the instant bytes arrive and ``meta()`` when the
    server describes itself. Everything derived (TTFB, first-audible, jitter) is
    computed here from a single monotonic clock so no adapter can accidentally
    measure something subtly different.
    """

    started_at: float
    sample_rate: int | None = None
    parts: list[np.ndarray] = field(default_factory=list)
    timings: list[ChunkTiming] = field(default_factory=list)
    meta_payload: dict[str, Any] = field(default_factory=dict)
    ttfb_ms: float | None = None
    first_audible_ms: float | None = None
    _samples_before_audible: int = 0

    def meta(self, **kw: Any) -> None:
        sr = kw.pop("sample_rate", None)
        if sr:
            self.sample_rate = int(sr)
        self.meta_payload.update(kw)

    def chunk(self, samples: np.ndarray) -> None:
        """Record an arriving audio chunk. Must be called with no work in between.

        ``samples`` must be float32 mono in [-1, 1]; adapters convert at the
        boundary so every downstream consumer sees one representation.
        """
        now = time.perf_counter()
        if samples.size == 0:
            return
        offset_ms = (now - self.started_at) * 1000.0

        if self.ttfb_ms is None:
            self.ttfb_ms = offset_ms

        if self.first_audible_ms is None:
            # Locate the first audible sample *within this chunk* and interpolate
            # its arrival time across the chunk. A server that pads with silence
            # would otherwise get credit for speaking earlier than it does.
            above = np.flatnonzero(np.abs(samples) > SILENCE_FLOOR)
            if above.size:
                sr = self.sample_rate or 0
                if sr > 0:
                    self.first_audible_ms = offset_ms + (int(above[0]) / sr) * 1000.0
                else:
                    self.first_audible_ms = offset_ms
            else:
                self._samples_before_audible += int(samples.size)

        self.parts.append(samples)
        self.timings.append(ChunkTiming(offset_ms=offset_ms, n_samples=int(samples.size)))

    def audio(self, fallback_rate: int) -> AudioBuffer:
        rate = self.sample_rate or fallback_rate
        if not self.parts:
            return AudioBuffer(samples=np.zeros(0, dtype=np.float32), sample_rate=rate)
        return AudioBuffer(
            samples=np.concatenate(self.parts).astype(np.float32, copy=False),
            sample_rate=rate,
        )


class TTSAdapter(abc.ABC):
    """Base class for every provider adapter.

    Subclasses implement :meth:`_synthesise` and (optionally) :meth:`probe`,
    :meth:`aopen` and :meth:`aclose`. They must not time themselves and must not
    raise for a single failed utterance — raising :class:`SynthesisFailed` is
    fine and is converted into a recorded failure by :meth:`synthesize`.
    """

    #: Registry key, referenced by ``adapter:`` in a model card.
    name: str = ""
    #: pip extra needed for this adapter's transport, quoted in error messages.
    requires: tuple[str, ...] = ()

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self._capabilities = self._build_capabilities(self.config)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TTSAdapter":
        """Build from the ``adapter_config`` block of a model card.

        Override only if a subclass needs to validate or normalise config before
        ``__init__``; most do their validation in ``__init__``.
        """
        return cls(config)

    def _build_capabilities(self, config: Mapping[str, Any]) -> Capabilities:
        """Derive declared capabilities from config.

        Defaults read straight from the model card so a card can describe a new
        model's languages/voices without touching code. Subclasses narrow this
        when the protocol itself constrains something (e.g. a non-streaming REST
        endpoint can never be ``streaming=True``).
        """
        det = config.get("determinism")
        return Capabilities(
            streaming=bool(config.get("streaming", False)),
            voices=tuple(config.get("voices") or ()),
            languages=tuple(config.get("languages") or ()),
            supports_seed=bool(config.get("supports_seed", False)),
            supports_emotion=bool(config.get("supports_emotion", False)),
            native_sample_rate=config.get("sample_rate"),
            determinism=Determinism(det) if det else Determinism.BEST_EFFORT,
        )

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def describe(self) -> dict[str, Any]:
        """Config safe to persist in a run record.

        Secrets are redacted here rather than at write time: the record is the
        artefact that gets shared, so it must never carry an API key even if
        someone adds a new store backend later.
        """
        redacted = {}
        for k, v in self.config.items():
            if any(s in k.lower() for s in ("key", "token", "secret", "password", "auth")):
                redacted[k] = "***redacted***"
            else:
                redacted[k] = v
        return {"adapter": self.name, "config": redacted}

    # -- lifecycle ---------------------------------------------------------
    async def aopen(self) -> None:
        """Create shared connections/sessions. Called once before a run.

        Session reuse matters for latency measurement: paying TCP+TLS setup on
        every utterance would inflate TTFB by tens of milliseconds and make an
        on-prem model look worse than it is.
        """

    async def aclose(self) -> None:
        """Release resources. Always called, even if the run failed."""

    async def probe(self) -> None:
        """Optional pre-flight reachability check.

        Raise :class:`AdapterUnavailable` to abort the run before synthesising
        anything. Failing fast here saves a 30-minute run that produces a report
        full of connection errors.
        """

    # -- the one method subclasses must write ------------------------------
    @abc.abstractmethod
    async def _synthesise(self, request: SynthesisRequest, capture: _Capture) -> None:
        """Perform one synthesis, calling ``capture.chunk()`` as audio arrives.

        Raise :class:`SynthesisFailed` for a per-utterance failure. Any other
        exception is also caught and recorded, but ``SynthesisFailed`` documents
        intent.
        """

    # -- public entry point (timed, never raises for one utterance) --------
    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        capture = _Capture(started_at=time.perf_counter())
        error: str | None = None
        try:
            await self._synthesise(request, capture)
        except SynthesisFailed as e:
            error = str(e)
        except Exception as e:  # noqa: BLE001 - one bad utterance must not kill a run
            error = f"{type(e).__name__}: {e}"

        total_ms = (time.perf_counter() - capture.started_at) * 1000.0
        fallback_rate = self.capabilities.native_sample_rate or 24000
        audio = capture.audio(fallback_rate)

        meta = dict(capture.meta_payload)
        if capture._samples_before_audible and capture.first_audible_ms is None:
            # Audio arrived but every sample was below the silence floor: a
            # degenerate-output signal that audio_quality will flag.
            meta["all_silent"] = True

        return SynthesisResult(
            request=request,
            audio=audio if audio.n_samples else None,
            ttfb_ms=capture.ttfb_ms,
            first_audible_ms=capture.first_audible_ms,
            total_ms=total_ms,
            chunk_timings=capture.timings,
            provider_meta=meta,
            error=error,
        )

    # -- helpers for subclasses -------------------------------------------
    @staticmethod
    def _require(module: str, extra: str) -> Any:
        """Import a transport dependency or explain exactly how to install it."""
        try:
            return importlib.import_module(module)
        except ImportError as e:
            raise AdapterUnavailable(
                f"adapter needs '{module}' which is not installed "
                f"(pip install 'tts-eval[{extra}]')"
            ) from e

    @staticmethod
    def to_float32(raw: bytes | np.ndarray, encoding: str) -> np.ndarray:
        """Normalise a provider's PCM encoding to float32 mono in [-1, 1].

        Centralised so a new adapter cannot introduce a private scaling
        convention; a 2x amplitude difference would silently shift every
        audio-quality and loudness metric.
        """
        if isinstance(raw, np.ndarray):
            arr = raw
        elif encoding in ("float32", "f32le", "pcm_f32le"):
            arr = np.frombuffer(raw, dtype="<f4")
        elif encoding in ("int16", "s16le", "pcm_s16le", "pcm_16"):
            arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif encoding in ("int32", "s32le", "pcm_s32le"):
            arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif encoding in ("uint8", "u8"):
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ConfigError(
                f"unsupported audio encoding {encoding!r}; expected one of "
                "float32, int16, int32, uint8"
            )
        return np.ascontiguousarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, type[TTSAdapter]] = {}


def register_adapter(cls: type[TTSAdapter]) -> type[TTSAdapter]:
    """Class decorator adding an adapter to the registry.

    Duplicate names are rejected rather than overwritten: a silent override
    would mean a run reports one adapter while using another's timing semantics.
    """
    if not cls.name:
        raise ConfigError(f"{cls.__name__} must set a non-empty `name`")
    existing = _REGISTRY.get(cls.name)
    if existing is not None and existing is not cls:
        raise ConfigError(
            f"adapter name {cls.name!r} already registered by {existing.__module__}."
            f"{existing.__name__}"
        )
    _REGISTRY[cls.name] = cls
    return cls


def get_adapter_class(name: str) -> type[TTSAdapter]:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none loaded)"
        raise AdapterUnavailable(f"unknown adapter {name!r}; registered: {known}")
    return _REGISTRY[name]


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)


def load_adapter_module(dotted: str) -> None:
    """Import a module so its ``@register_adapter`` classes become available.

    This is the no-fork extension point: a team can keep a proprietary adapter in
    its own package and pass ``--adapter-module my_pkg.tts_adapter``.
    """
    try:
        importlib.import_module(dotted)
    except ImportError as e:
        raise AdapterUnavailable(f"could not import adapter module {dotted!r}: {e}") from e


def build_adapter(name: str, config: Mapping[str, Any]) -> TTSAdapter:
    return get_adapter_class(name).from_config(config)


def resolve_voice(
    requested: str | None, capabilities: Capabilities, default: str | None = None
) -> str | None:
    """Pick a voice, preferring the request, then the card default, then the first
    declared voice. Returns None when the model has no notion of voices."""
    for candidate in (requested, default):
        if candidate:
            return candidate
    return capabilities.voices[0] if capabilities.voices else None


def iter_registered() -> Iterable[tuple[str, type[TTSAdapter]]]:
    return sorted(_REGISTRY.items())


__all__ = [
    "TTSAdapter",
    "_Capture",
    "register_adapter",
    "get_adapter_class",
    "available_adapters",
    "load_adapter_module",
    "build_adapter",
    "resolve_voice",
    "iter_registered",
]
