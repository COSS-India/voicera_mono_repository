"""The synthesis engine: one vLLM engine plus one shared batched SNAC decoder.

Everything latency-sensitive lives here. Two design points worth knowing before
changing anything:

  * vLLM hands back a *cumulative* token list on every streaming step, and that
    list keeps growing while we await a decode. The loop below snapshots it and
    tracks a cursor over the snapshot; reading the live list instead silently
    drops tokens (audible as clipped syllables).
  * Per-request timings live on a ``StreamStats`` object owned by the caller, not
    in a process-global. With concurrent streams a global ``last_ttfa`` reports
    whichever request happened to finish last, which is how you end up publishing
    another stream's latency in your own response headers.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from . import codec, prompt
from .config import Settings
from .voices import Roster

log = logging.getLogger("orpheus.engine")


@dataclass
class StreamStats:
    """Timings for one synthesis request. Owned by the caller; never shared."""

    ttfa_ms: Optional[float] = None      # time to first audio chunk
    tokens: int = 0                      # audio tokens generated
    frames: int = 0                      # 85.33 ms frames emitted
    pcm_bytes: int = 0
    gen_ms: float = 0.0
    gaps_ms: list[float] = field(default_factory=list)

    @property
    def audio_ms(self) -> float:
        return self.frames * codec.FRAME_MS

    @property
    def rtf(self) -> Optional[float]:
        """Generation time / audio duration. Below 1.0 is faster than real time."""
        return round(self.gen_ms / self.audio_ms, 3) if self.audio_ms else None

    @property
    def tokens_per_s(self) -> Optional[float]:
        return round(self.tokens / (self.gen_ms / 1000.0), 1) if self.gen_ms else None

    def summary(self) -> dict:
        return {
            "ttfa_ms": round(self.ttfa_ms, 1) if self.ttfa_ms is not None else None,
            "audio_ms": round(self.audio_ms, 1),
            "gen_ms": round(self.gen_ms, 1),
            "rtf": self.rtf,
            "tokens": self.tokens,
            "tokens_per_s": self.tokens_per_s,
            "frames": self.frames,
        }


@dataclass
class Metrics:
    """Process-wide counters. Deliberately aggregate only - see StreamStats."""

    requests_total: int = 0
    streams_active: int = 0
    errors_total: int = 0
    audio_seconds_total: float = 0.0

    def snapshot(self) -> dict:
        return {
            "requests_total": self.requests_total,
            "streams_active": self.streams_active,
            "errors_total": self.errors_total,
            "audio_seconds_total": round(self.audio_seconds_total, 2),
        }


class TTSEngine:
    def __init__(self, settings: Settings, roster: Roster) -> None:
        self.settings = settings
        self.roster = roster
        self.metrics = Metrics()
        self.model_path = settings.resolved_model_path()
        self.started_at: Optional[float] = None
        self._engine = None
        self._tokenizer = None
        self._decoder: Optional[codec.SnacDecoder] = None
        self._batcher: Optional[codec.BatchedSnacDecoder] = None
        self._ready = asyncio.Event()

    # -- lifecycle ----------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    async def start(self) -> None:
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        cfg = self.settings
        log.info("loading SNAC codec %s on %s", cfg.decoder.model_id, cfg.decoder.device)
        self._decoder = codec.SnacDecoder(device=cfg.decoder.device, model_id=cfg.decoder.model_id)
        self._batcher = codec.BatchedSnacDecoder(self._decoder, max_batch=cfg.decoder.max_batch)
        self._batcher.start()

        log.info(
            "loading model %s (dtype=%s quantization=%s max_num_seqs=%d gpu_mem=%.2f tp=%d)",
            self.model_path, cfg.model.dtype, cfg.model.quantization,
            cfg.engine.max_num_seqs, cfg.engine.gpu_memory_utilization,
            cfg.engine.tensor_parallel_size,
        )
        args = AsyncEngineArgs(
            model=self.model_path,
            dtype=cfg.model.dtype,
            quantization=cfg.model.quantization,
            max_model_len=cfg.model.max_model_len,
            trust_remote_code=cfg.model.trust_remote_code,
            gpu_memory_utilization=cfg.engine.gpu_memory_utilization,
            max_num_seqs=cfg.engine.max_num_seqs,
            enforce_eager=cfg.engine.enforce_eager,
            tensor_parallel_size=cfg.engine.tensor_parallel_size,
            disable_log_stats=True,
        )
        self._engine = AsyncLLMEngine.from_engine_args(args)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        if cfg.warmup.enabled:
            await self._warmup()
        self.started_at = time.time()
        self._ready.set()
        log.info("ready: serving %s", cfg.server.model_name)

    async def stop(self) -> None:
        self._ready.clear()
        if self._batcher is not None:
            self._batcher.stop()
        if self._engine is not None:
            with contextlib.suppress(Exception):
                self._engine.shutdown()
        self._engine = None

    async def _warmup(self) -> None:
        """Pre-compile Triton kernels for each batch width we expect to serve.

        vLLM's sampling kernels JIT-compile per batch shape on first use. Without
        this, the first burst of N concurrent users pays a multi-second stall
        mid-stream; paying it here costs boot time nobody is waiting on.
        """
        language = self.roster.languages[0]
        voice = language["voices"][0]
        text = self.roster.sample_text(language["code"])

        async def once() -> None:
            stats = StreamStats()
            async for _ in self.stream_pcm(
                text=text, voice=voice, language=language["code"],
                style=self.roster.default_style,
                max_tokens=self.settings.warmup.max_tokens, stats=stats,
            ):
                pass

        for width in sorted(set(self.settings.warmup.concurrency_widths)):
            if width < 1:
                continue
            log.info("warmup: concurrency %d", width)
            with contextlib.suppress(Exception):
                await asyncio.gather(*[once() for _ in range(width)])

    # -- synthesis ----------------------------------------------------------
    def clamp_max_tokens(self, requested: Optional[int]) -> int:
        cfg = self.settings.engine
        value = cfg.max_tokens_default if requested is None else int(requested)
        return max(64, min(value, cfg.max_tokens_limit))

    async def stream_pcm(
        self,
        text: str,
        voice: str,
        language: str,
        style: Optional[str],
        max_tokens: int,
        stats: StreamStats,
    ) -> AsyncIterator[bytes]:
        """Yield 24 kHz mono s16le PCM, one 85.33 ms frame at a time.

        ``stats`` is filled in as the stream progresses so the caller can report
        this request's own latency rather than a shared global's.
        """
        from vllm import SamplingParams, TokensPrompt

        if self._engine is None:
            raise RuntimeError("engine is not loaded")
        if not text or not text.strip():
            raise ValueError("input text is empty")

        sampling = self.settings.sampling
        token_ids = prompt.build_prompt_token_ids(
            self._tokenizer, self.roster.template, text, voice, style
        )
        params = SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            repetition_penalty=sampling.repetition_penalty,
            max_tokens=max_tokens,
            min_tokens=sampling.min_tokens,
            stop_token_ids=prompt.STOP_TOKEN_IDS,
            detokenize=False,          # we never want text back; skip the detokenizer entirely
        )

        buffer = codec.StreamingAudioBuffer()
        cursor = 0
        started = time.perf_counter()
        last_chunk_at: Optional[float] = None

        generator = self._engine.generate(
            TokensPrompt(prompt_token_ids=token_ids), params, str(uuid.uuid4())
        )
        async for output in generator:
            # Snapshot: this list keeps growing while we await the decode below.
            tokens = list(output.outputs[0].token_ids)
            for token_id in tokens[cursor:]:
                stats.tokens += 1
                window = buffer.push_token(token_id)
                if window is None:
                    continue
                pcm = await self._batcher.decode(window)
                if not pcm:
                    continue
                now = time.perf_counter()
                if stats.ttfa_ms is None:
                    stats.ttfa_ms = (now - started) * 1000.0
                elif last_chunk_at is not None:
                    stats.gaps_ms.append((now - last_chunk_at) * 1000.0)
                last_chunk_at = now
                stats.frames += 1
                stats.pcm_bytes += len(pcm)
                yield pcm
            cursor = len(tokens)

        stats.gen_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.audio_seconds_total += stats.audio_ms / 1000.0
