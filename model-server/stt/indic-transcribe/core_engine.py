"""
Live AlignAtt streaming engine for `indic-transcribe-core` (EncDecMultiTaskModel, 25 languages).

Relationship to the Bhili engine
--------------------------------
`~/ASR/asr-serve/asr_engine.py` is the structural reference and its decoding math is preserved
exactly. This file is written fresh because `core` differs from that checkpoint in ways that are
not parameters -- a 10-token prompt instead of 9, `<|pnc|>` instead of `<|nopnc|>`, a romanized
slot, 25 languages instead of 1 -- and because it fixes four defects that engine inherited.

What is fixed here, and why each matters
----------------------------------------
1. **The AlignAtt token budget (the big one).**
   `aed_batched_streaming.py:500` ends `initialize_aed_model_state` with

       model_state.max_tokens_per_alignatt_step = max_tokens_per_one_second * int(chunk + right)

   It overwrites the configured value unconditionally -- so `max_tokens_per_alignatt_step` in the
   decoding config is dead -- and `int()` TRUNCATES. For every geometry with
   `chunk + right < 1.0 s` the budget is therefore **0**, and at line 335

       disable_samples_mask = steps_per_inner_loop >= max_tokens_per_alignatt_step

   is true on the first inner-loop iteration, disabling the sample before it emits a single
   token. Every configuration capable of word-by-word latency is broken by this line, which is
   why that entire region has never actually been measured. We recompute the budget with
   `round()` and a floor, then write it onto the state after initialization -- the same override
   route the reference already uses for `max_generation_length`, so no NeMo patch is needed.

2. **`max_generation_length` in both places.** `AEDStreamingState` defaults to 256 while
   `AEDStreamingDecodingConfig` defaults to 512, and `pred_tokens_ids` is allocated from the
   state. Anything past ~25 s of speech overruns the buffer and trips a CUDA device-side assert.

3. **`exclude_sink_frames` clamped every step, not just the first.** At line 245 NeMo guards the
   slice only when `i == 0`:

       if i == 0 and xatt_scores.shape[-1] <= exclude_sink_frames: exclude_sink_frames = T // 2

   On later iterations with aggressive geometry the source is short enough that
   `xatt_scores[:, :, exclude_sink_frames:]` is empty and `argmax` raises. Since the value is
   read from `self.decoding_cfg`, clamping it against this session's actual source length before
   each call gives the guard on every iteration without touching NeMo.

4. **Three inherited bugs.** `if not active: return {}` silently dropped `silent_finals`; the
   `max_batch` break sat inside the VAD branch where it could stall ready sessions; and the
   language/mode boundary was ungated.

The language boundary is real, not defensive
--------------------------------------------
core's wrapper is byte-identical to flex's and advertises 27 languages and 3 output modes. core
has **25**: `bgc` and `hne` are silently skipped at tokenizer init because their tokens are absent
from its spl vocab, and `tokenizer_config.json:prompt_langs` -- the materialised truth -- lists 25.
`<|itn|>` and `<|romanized|>` ARE in-vocab (the tokenizer asserts they exist) but are untrained on
this checkpoint, so asking for them yields fluent wrong output rather than an error. Both are
rejected at the boundary instead.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecMultiTaskModel
from nemo.collections.asr.models.aed_multitask_models import lens_to_mask
from nemo.collections.asr.parts.submodules.aed_decoding import (
    GreedyBatchedStreamingAEDComputer,
    return_decoder_input_ids,
)
from nemo.collections.asr.parts.submodules.multitask_decoding import (
    AEDStreamingDecodingConfig,
    MultiTaskDecodingConfig,
)
from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer

from nemo_patch import canary2_romanized
from vad import SileroVAD, VadConfig, VadGate

log = logging.getLogger("core.engine")

SAMPLE_RATE = 16000

#: The prompt this checkpoint was trained with: 10 tokens,
#: [<|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|L|> <|L|>
#:  <|pnc|> <|noitn|> <|noromanized|> <|notimestamp|> <|nodiarize|>]
#: Bhili's is 9 (no romanized slot, and <|nopnc|>). Asserted at load, because a silently
#: shorter prompt shifts every id_to_text offset and every _emitted_len slice.
PROMPT_LEN = 10


#: Substrings that mean the CUDA CONTEXT is gone, not just this operation. After any of these the
#: device is unusable for the life of the process -- every later kernel on it fails too -- so they
#: must be treated as fatal rather than retried. Measured the hard way: a single bad session
#: poisoned the context and the service then spent minutes accepting connections and failing every
#: one of them while still reporting healthy.
_FATAL_CUDA = (
    "illegal memory access",
    "CUDA error",
    "CUBLAS_STATUS",
    "device-side assert",
    "an illegal instruction",
    "unspecified launch failure",
)


def is_fatal_cuda(exc: BaseException) -> bool:
    """True when the CUDA context is unrecoverable and the process must be replaced."""
    text = f"{type(exc).__name__}: {exc}"
    return any(m in text for m in _FATAL_CUDA)


def load_supported_languages(hf_dir: str | Path) -> list[str]:
    """The 25 languages this checkpoint really has, from its own tokenizer_config.json.

    Deliberately NOT `indic_transcribe.LANGUAGES` (27) -- that tuple is shared across this model
    family and includes bgc/hne, which core's vocab does not carry.
    """
    cfg = json.loads((Path(hf_dir) / "tokenizer_config.json").read_text())
    return sorted(cfg["prompt_langs"])


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass
class EngineConfig:
    ckpt_path: str
    hf_dir: str = "/models/core"
    language: str = "hi"

    # Buffer geometry. Theoretical latency = chunk + right, floored to whole 80 ms encoder
    # frames (so 1.0/0.5 becomes 0.96/0.48 and reads 1.44 s).
    left_context_secs: float = 10.0
    chunk_secs: float = 1.0
    right_context_secs: float = 0.5

    # AlignAtt
    streaming_policy: str = "alignatt"
    alignatt_thr: int = 8
    # GreedyBatchedStreamingAEDComputer refuses to decode until
    #   encoder_output_len // frame_chunk_size >= waitk_lagging
    # for BOTH policies. NeMo's default of 2 holds the first partial back by an extra chunk
    # (measured TTFP 2.37 s against 1.44 s theoretical). 1 emits as soon as a full window exists.
    waitk_lagging: int = 1
    exclude_sink_frames: int = 8
    xatt_scores_layer: int = -2
    hallucinations_detector: bool = True

    #: None => max(4, round(10 * (chunk + right))). See fix (1) in the module docstring.
    #: Set an int to pin it for a sweep.
    token_budget: Optional[int] = None

    compute_dtype: str = "bfloat16"
    matmul_precision: str = "high"

    #: Ladder step 2. Time the tick with CUDA events rather than two mid-tick
    #: torch.cuda.synchronize() calls that exist only for stats and block CPU/GPU overlap.
    use_cuda_events: bool = True

    max_batch: int = 32
    #: Hard admission limit. A connection past this is refused with WebSocket 1013.
    max_sessions: int = 8
    #: How many concurrent streams this configuration can actually serve IN REAL TIME, measured.
    #: Distinct from `max_sessions` on purpose: an operator may knowingly admit more than the
    #: server can keep up with (a short burst is often better than a refusal), and in that case
    #: the fact needs to be visible in /metrics rather than inferred from latency. Measured at
    #: the T7 geometry: 57% of the chunk budget at 8 streams, 120% at 10 -- so the server stops
    #: keeping up between the two. See REPORT.md section 4.
    realtime_capacity: int = 8
    device: str = "cuda"

    max_generation_length: int = 512

    vad: VadConfig = field(default_factory=VadConfig)

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[self.compute_dtype]

    def effective_token_budget(self, chunk_eff: float, right_eff: float) -> int:
        """The corrected AlignAtt per-step token budget.

        `round()` not `int()`: truncation is what zeroes the budget for every sub-second
        geometry. The floor of 4 keeps the smallest geometries able to emit a short word rather
        than a single token.
        """
        if self.token_budget is not None:
            return int(self.token_budget)
        return max(4, round(10 * (chunk_eff + right_eff)))


@dataclass
class EngineStats:
    ticks: int = 0
    encode_ms: float = 0.0
    decode_ms: float = 0.0
    sessions_processed: int = 0
    vad_skipped: int = 0
    tick_ms: list = field(default_factory=list)
    batch_sizes: list = field(default_factory=list)
    padding_waste: list = field(default_factory=list)

    def snapshot(self) -> dict:
        t = max(1, self.ticks)

        def pct(xs, q):
            if not xs:
                return None
            s = sorted(xs)
            return round(s[min(len(s) - 1, int(q * len(s)))], 2)

        return {
            "ticks": self.ticks,
            "avg_encode_ms": round(self.encode_ms / t, 2),
            "avg_decode_ms": round(self.decode_ms / t, 2),
            "avg_sessions_per_tick": round(self.sessions_processed / t, 2),
            "vad_skipped_chunks": self.vad_skipped,
            "tick_ms_p50": pct(self.tick_ms, 0.50),
            "tick_ms_p95": pct(self.tick_ms, 0.95),
            "tick_ms_p99": pct(self.tick_ms, 0.99),
            "batch_size_hist": _hist(self.batch_sizes),
            "padding_waste_pct": pct(self.padding_waste, 0.50),
        }


def _hist(xs: list) -> dict:
    h: dict[int, int] = {}
    for x in xs:
        h[x] = h.get(x, 0) + 1
    return {str(k): h[k] for k in sorted(h)}


@dataclass
class Delta:
    """One incremental emission for a session."""

    text: str                  # newly-committed text only
    token_ids: list
    full_text: str             # everything committed so far
    latency_ms: float          # wall clock: newest audio in this chunk -> now
    is_final: bool = False


# --------------------------------------------------------------------------------------
# session
# --------------------------------------------------------------------------------------
class StreamSession:
    """One live audio stream: a ring buffer plus this session's own AED decoder state."""

    __slots__ = (
        "sid", "cfg", "engine", "lang", "_pending", "_buffer", "_state", "_emitted_len",
        "_full_text", "_closed", "_finalized", "_first_tick_done", "_audio_secs_fed",
        "_arrivals", "_samples_in", "_samples_taken", "created_at", "last_activity",
        "_lock", "_pending_finalize", "vad", "t_audio0", "n_partials", "_last_partial_at",
        "max_gap_ms", "_buffer_fed", "_split_warned", "text_origin",
    )

    def __init__(self, sid: str, engine: "StreamingEngine", lang: str):
        self.sid = sid
        self.engine = engine
        self.cfg = engine.cfg
        self.lang = lang

        self._pending = np.zeros(0, dtype=np.float32)
        self._arrivals: list[tuple[int, float]] = []
        self._samples_in = 0
        self._samples_taken = 0
        self._buffer: Optional[StreamingBatchedAudioBuffer] = None
        self._state = None
        self._emitted_len = 0
        self._full_text = ""
        self._closed = False
        self._finalized = False
        self._pending_finalize = False
        self._first_tick_done = False
        self._buffer_fed = False      # has the BUFFER received a chunk yet?
        self._split_warned = False
        # Where this turn's OWN text starts inside pred_tokens_ids. Equals
        # PROMPT_LEN normally, and prompt+carried on a seamless rotation.
        self.text_origin = PROMPT_LEN
        self._audio_secs_fed = 0.0
        self.created_at = time.monotonic()
        self.last_activity = self.created_at
        self.t_audio0: Optional[float] = None   # first audio byte accepted -> TTFP origin
        self.n_partials = 0
        self._last_partial_at: Optional[float] = None
        self.max_gap_ms = 0.0
        self._lock = threading.Lock()
        self.vad = VadGate(engine.vad, self.cfg.vad)

    # ---- audio in -------------------------------------------------------------------
    def feed(self, pcm: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError(f"session {self.sid} is closed")
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)

        # BUGFIX: classify BEFORE the audio becomes visible to the worker.
        #
        # The reference appended to _pending first and ran the VAD afterwards. The GPU worker
        # polls readiness every 2 ms, so it could tick on audio the VAD had not yet scored,
        # see has_speech still False, and discard the chunk as "no speech ever seen". With
        # small paced blocks this is a rare race; with one large block (a whole file, or a
        # client that buffers) it is deterministic -- measured here as the first 2 chunks of a
        # 3.2 s clip being dropped, truncating the transcript to its last two words.
        #
        # Silero runs on the CPU in ~40 us per 32 ms frame, so ordering it first costs nothing
        # that matters and removes the race entirely.
        self.vad.observe(pcm)

        with self._lock:
            self._pending = np.concatenate([self._pending, pcm]) if self._pending.size else pcm
            now = time.monotonic()
            if self.t_audio0 is None:
                self.t_audio0 = now
            self._samples_in += len(pcm)
            self._arrivals.append((self._samples_in, now))
            self._audio_secs_fed += len(pcm) / SAMPLE_RATE
            self.last_activity = now

    @property
    def endpointed(self) -> bool:
        return self.vad.endpointed

    def request_finalize(self, drop_pending: bool = False) -> None:
        """Ask for a final transcript.

        `drop_pending=True` is the VAD-endpoint path: the turn is over and the tail is silence
        by definition, so keep at most one chunk and discard the rest. Without it the final is
        delayed until the client stops sending, which reports the trailing-silence length as
        endpoint latency -- an artefact of the caller, not the service.
        """
        with self._lock:
            self._pending_finalize = True
            if drop_pending:
                keep = self._samples_needed
                if self._pending.size > keep:
                    self._samples_in -= self._pending.size - keep
                    self._pending = self._pending[:keep]

    def close(self) -> None:
        self._closed = True
        self._buffer = None
        self._state = None

    # ---- windowing ------------------------------------------------------------------
    @property
    def _samples_needed(self) -> int:
        """How much audio the next tick needs.

        Keyed off `_buffer_fed` -- whether the BUFFER has received a chunk -- not off whether we
        have consumed audio. The two used to be the same flag, and they diverged exactly when a
        chunk was consumed without being buffered (the VAD-skip path). The buffer then got a
        `chunk`-sized first add instead of `chunk + right` and its context split was wrong for
        the rest of the stream. `_take_chunk` still maintains `_first_tick_done` for callers that
        care about "have we ever produced a chunk"; sizing must follow the buffer.
        """
        s = self.engine.context_samples
        return s.chunk + s.right if not self._buffer_fed else s.chunk

    def ready(self) -> bool:
        if self._closed or self._finalized:
            return False
        with self._lock:
            if self._pending_finalize:
                return self._pending.size > 0 or not self._finalized
            return self._pending.size >= self._samples_needed

    def _take_chunk(self) -> tuple[np.ndarray, bool, float]:
        """Pop the next chunk -> (samples, is_last, newest_arrival_wallclock).

        The third value is when the LAST sample in this chunk was accepted -- the earliest
        instant any token derived from it could have been emitted. Measuring from the oldest
        block instead makes reported latency grow with stream length, which is simply wrong.
        """
        with self._lock:
            want = self._samples_needed
            final = self._pending_finalize and self._pending.size <= want
            take = self._pending.size if final else want
            out, self._pending = self._pending[:take], self._pending[take:]
            self._samples_taken += take
            end = self._samples_taken

            arrival = time.monotonic()
            for idx, ts in self._arrivals:
                if idx >= end:
                    arrival = ts
                    break
            self._arrivals = [(i, t) for (i, t) in self._arrivals if i > end]
            self._first_tick_done = True
            return out, final, arrival

    def note_partial(self) -> None:
        now = time.monotonic()
        self.n_partials += 1
        if self._last_partial_at is not None:
            self.max_gap_ms = max(self.max_gap_ms, (now - self._last_partial_at) * 1000)
        self._last_partial_at = now


# --------------------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------------------
class StreamingEngine:
    """Owns the model and the GPU. All GPU work happens in `tick()`, called from one worker."""

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.stats = EngineStats()
        self.sessions: dict[str, StreamSession] = {}
        self._sid_seq = 0
        self.ready = False
        #: Set once the CUDA context is unrecoverable. /health must fail on this.
        self.fatal: Optional[str] = None
        self.vad: Optional[SileroVAD] = None
        self.languages: list[str] = []
        self._prompt_cache: dict[tuple, torch.Tensor] = {}
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    # ---- startup --------------------------------------------------------------------
    def load(self) -> None:
        self._apply_nemo_patches()

        self.languages = load_supported_languages(self.cfg.hf_dir)
        log.info("checkpoint supports %d languages: %s",
                 len(self.languages), ",".join(self.languages))
        if self.cfg.language not in self.languages:
            raise ValueError(
                f"default language {self.cfg.language!r} is not one of this checkpoint's "
                f"{len(self.languages)}: {self.languages}"
            )

        if self.cfg.vad.enabled:
            try:
                self.vad = SileroVAD()
            except Exception as e:
                # A missing VAD must not take ASR down: we lose the silence gate and
                # endpointing, not transcription.
                log.warning("VAD unavailable (%s); running without silence gating", e)
                self.vad = None

        t0 = time.time()
        log.info("restoring %s", self.cfg.ckpt_path)
        self.model = EncDecMultiTaskModel.restore_from(
            restore_path=self.cfg.ckpt_path, map_location=self.cfg.device
        )
        self.model.eval()

        if self.model.cfg.preprocessor.normalize != "per_feature":
            raise RuntimeError(
                "streaming requires preprocessor.normalize == 'per_feature', got "
                f"{self.model.cfg.preprocessor.normalize!r}"
            )
        # Deterministic, unpadded features. dither only fires in train mode (we freeze below)
        # and this checkpoint already has pad_to=0, so this is belt-and-braces.
        feat = self.model.preprocessor.featurizer
        for attr, val in (("dither", 0.0), ("pad_to", 0)):
            if hasattr(feat, attr):
                setattr(feat, attr, val)

        # Order matters and must match the reference script: freeze, move, THEN cast weights.
        # No autocast -- native bf16 weights are what the parity baseline was produced with.
        self.model.freeze()
        self.model = self.model.to(self.cfg.device)
        if self.cfg.torch_dtype != torch.float32:
            self.model = self.model.to(self.cfg.torch_dtype)
        self.device = next(self.model.parameters()).device

        # The checkpoint restores with `strategy: beam` (TransformerAEDBeamInfer), but the
        # AlignAtt computer reaches for `decoding.decoding.greedy_search`, which exists only on
        # the greedy strategy.
        if hasattr(self.model, "change_decoding_strategy"):
            mt = MultiTaskDecodingConfig()
            mt.strategy = "greedy"
            self.model.change_decoding_strategy(mt)

        self._setup_contexts()
        self._setup_decoding()

        log.info(
            "loaded in %.1fs | params %.4fB | left/chunk/right = %.2f/%.2f/%.2f s "
            "| theoretical latency %.2f s | token budget %d | dtype %s",
            time.time() - t0,
            sum(p.numel() for p in self.model.parameters()) / 1e9,
            self.context_samples.left / SAMPLE_RATE,
            self.context_samples.chunk / SAMPLE_RATE,
            self.context_samples.right / SAMPLE_RATE,
            self.theoretical_latency,
            self.token_budget,
            self.cfg.compute_dtype,
        )
        self._warmup()
        self.ready = True

    @staticmethod
    def _apply_nemo_patches() -> None:
        """The two vendored NeMo fixes the checkpoint needs, in order."""
        from nemo_patch import canary2_romanized, ensure_installed

        ensure_installed()
        # For Bhili this was investigated and found NOT needed (9-token prompt, <|nopnc|>).
        # For core it inverts to REQUIRED: its prompt is 10 tokens and carries <|noromanized|>.
        canary2_romanized.apply_all()
        if not canary2_romanized.is_applied():
            raise RuntimeError("canary2 romanized slot failed to apply")

    def _setup_contexts(self) -> None:
        """Requested seconds are floored to whole subsampled encoder frames -- which is why an
        8x-subsampled 10 ms-stride model turns a 1.00 s request into a 0.96 s chunk."""
        cfg = self.cfg
        pre = self.model.cfg.preprocessor
        self.encoder_frame2audio_samples = int(
            pre.window_stride * SAMPLE_RATE * self.model.encoder.subsampling_factor
        )

        def frames(secs: float) -> int:
            return int(secs * SAMPLE_RATE) // self.encoder_frame2audio_samples

        self.context_encoder_frames = ContextSize(
            left=frames(cfg.left_context_secs),
            chunk=frames(cfg.chunk_secs),
            right=frames(cfg.right_context_secs),
        )
        if self.context_encoder_frames.chunk < 1:
            raise ValueError(f"chunk_secs={cfg.chunk_secs} is smaller than one encoder frame")
        self.context_samples = ContextSize(
            left=self.context_encoder_frames.left * self.encoder_frame2audio_samples,
            chunk=self.context_encoder_frames.chunk * self.encoder_frame2audio_samples,
            right=self.context_encoder_frames.right * self.encoder_frame2audio_samples,
        )
        self.chunk_eff = self.context_samples.chunk / SAMPLE_RATE
        self.right_eff = self.context_samples.right / SAMPLE_RATE
        self.theoretical_latency = self.chunk_eff + self.right_eff

    def _setup_decoding(self) -> None:
        cfg = self.cfg
        dec = AEDStreamingDecodingConfig()
        dec.streaming_policy = cfg.streaming_policy
        dec.alignatt_thr = cfg.alignatt_thr
        dec.waitk_lagging = cfg.waitk_lagging
        dec.exclude_sink_frames = cfg.exclude_sink_frames
        dec.xatt_scores_layer = cfg.xatt_scores_layer
        dec.hallucinations_detector = cfg.hallucinations_detector
        dec.max_generation_length = cfg.max_generation_length
        self.decoding_cfg = dec

        # Fix (1): the corrected AlignAtt budget. NeMo recomputes and overwrites this inside
        # initialize_aed_model_state, so setting it on the config alone does nothing -- it is
        # re-applied to every session's state in create_session().
        self.token_budget = cfg.effective_token_budget(self.chunk_eff, self.right_eff)
        nemo_would_use = 10 * int(self.chunk_eff + self.right_eff)
        if nemo_would_use != self.token_budget:
            log.info(
                "AlignAtt token budget: using %d (NeMo's own formula would give %d%s)",
                self.token_budget, nemo_would_use,
                " -- which disables every sample before its first token" if nemo_would_use == 0
                else "",
            )

        self.decoder_input_ids = self._build_prompt(cfg.language)
        n = self.decoder_input_ids.size(-1)
        tokens = self.model.tokenizer.ids_to_tokens(self.decoder_input_ids[0].tolist())
        log.info("prompt (%d tokens): %s", n, tokens)
        if n != PROMPT_LEN:
            raise RuntimeError(
                f"expected a {PROMPT_LEN}-token canary2 prompt for this checkpoint, got {n}: "
                f"{tokens}. A short prompt means the romanized slot patch did not apply, and "
                "every ids_to_text offset downstream would be wrong."
            )

        self.computer = GreedyBatchedStreamingAEDComputer(
            self.model,
            frame_chunk_size=self.context_encoder_frames.chunk,
            decoding_cfg=dec,
        )

    def _build_prompt(self, lang: str) -> torch.Tensor:
        """The 10-token prompt for one language, cached.

        THE PROMPT MUST BE EXPLICIT. The reference script defaults `langid` to "en"; left alone
        it transcribed Bhili audio as confident, fluent English with no error and no warning.
        For core, whose 25 languages span 11 scripts, a wrong slot means a wrong script.

        `pnc: pnc` (not nopnc) and the `romanized` slot are what distinguish core's prompt from
        Bhili's -- both read off this checkpoint's own prompt_ids_by_lang table.
        """
        key = (lang,)
        if key in self._prompt_cache:
            return self._prompt_cache[key]
        prompt = OmegaConf.create({
            "role": "user",
            "slots": {
                "source_lang": lang,
                "target_lang": lang,
                "pnc": "pnc",
                "itn": "noitn",
                # The bracketed form, not the bare word: the romanized slot's literal list is
                # copied verbatim from the fork and accepts '<|noromanized|>' but not
                # 'noromanized' (unlike pnc/itn/timestamp/diarize, which accept both).
                "romanized": canary2_romanized.ROMANIZED_FALSE,
                "timestamp": "notimestamp",
                "diarize": "nodiarize",
                "emotion": "<|emo:undefined|>",
                "decodercontext": "",
            },
        })
        # `return_decoder_input_ids` is written against the reference script's OUTER
        # TranscriptionConfig (it reads .batch_size and .prompt), so hand it a shim.
        holder = SimpleNamespace(batch_size=1, prompt=prompt)
        ids = return_decoder_input_ids(holder, self.model)
        self._prompt_cache[key] = ids
        return ids

    def _warmup(self) -> None:
        """Run a synthetic stream so CUDA kernels are hot before /health goes green."""
        n = self.context_samples.chunk + self.context_samples.right
        try:
            s = self.create_session("__warmup__")
            for _ in range(3):
                if s.sid not in self.sessions:
                    raise RuntimeError("warmup session was closed by a failing tick")
                s.feed(np.zeros(n, dtype=np.float32))
                self.tick()
        finally:
            self.close_session("__warmup__")
        self.stats = EngineStats()

    # ---- session lifecycle ----------------------------------------------------------
    def validate_request(self, lang: str, mode: str = "native") -> None:
        """Reject what this checkpoint cannot actually do, at the boundary.

        Both failures are silent otherwise: an unsupported language produces confident output
        in the wrong script, and itn/romanized are in-vocab but untrained, so they produce
        fluent wrong text rather than raising.
        """
        if lang not in self.languages:
            raise ValueError(
                f"language {lang!r} is not supported by this checkpoint. Supported "
                f"({len(self.languages)}): {', '.join(self.languages)}"
            )
        if mode != "native":
            raise ValueError(
                f"mode {mode!r} is not available on this checkpoint. <|itn|> and <|romanized|> "
                "are present in the vocabulary but untrained here, so they yield fluent wrong "
                "output rather than an error. Only 'native' is served."
            )

    def _clone_buffer(self, src: StreamingBatchedAudioBuffer,
                      keep_secs: float = 0.0) -> StreamingBatchedAudioBuffer:
        """An independent copy of an audio buffer, contents and context split included.

        A *copy*, not the same object: during a seamless rotation both sessions are briefly alive
        -- the outgoing one still owes a final tick -- and sharing one buffer would let it mutate
        the incoming session's audio underneath it.
        """
        dst = StreamingBatchedAudioBuffer(
            batch_size=1, context_samples=self.context_samples,
            dtype=torch.float32, device=self.device,
        )
        samples = src.samples
        left = int(src.context_size.left)
        chunk = int(src.context_size.chunk)
        right = int(src.context_size.right)
        if keep_secs > 0:
            # Trim the LEFT context so the carried audio roughly matches the carried text.
            # Handing over the whole ~10.4 s window gives the new decoder a long stretch it has
            # already transcribed but has no text for, and it stops producing rather than
            # continuing. chunk and right are the live edge and are never trimmed.
            keep_left = max(0, int(keep_secs * SAMPLE_RATE))
            if left > keep_left:
                drop = left - keep_left
                samples = samples[:, drop:]
                left = keep_left
        dst.samples = samples.clone()
        dst.context_size = ContextSize(left=left, chunk=chunk, right=right)
        dst.context_size_batch.left.fill_(left)
        dst.context_size_batch.chunk.copy_(src.context_size_batch.chunk)
        dst.context_size_batch.right.copy_(src.context_size_batch.right)
        return dst

    def create_session(self, sid: Optional[str] = None, lang: Optional[str] = None,
                       inherit_from: Optional[StreamSession] = None,
                       carry_tokens: int = 24,
                       carry_secs: float = 0.0) -> StreamSession:
        """Start a session, optionally taking over from one being rotated out.

        `inherit_from` makes rotation SEAMLESS. Rotation exists to reset the decoder -- whose
        `decoder_mems_list` grows unbounded against a 1024-position limit -- but the previous
        implementation also threw away the audio buffer, so the new turn restarted from an empty
        window. AlignAtt cannot commit from an empty window (it needs
        `usable - attended - 1 >= alignatt_thr`), so every rotation re-paid the full
        time-to-first-partial. Measured: cold start 2.65 s, rotation gaps 2.45 s and 2.17 s --
        the same number, because it is the same event.

        The audio buffer is a fixed-size sliding window and was never the thing growing, so it is
        carried over. The decoder state is still rebuilt, but seeded with `prompt + last
        carry_tokens emitted tokens` so the model CONTINUES the sentence instead of
        re-transcribing the window it can now see.
        """
        if len(self.sessions) >= self.cfg.max_sessions:
            raise RuntimeError(f"at capacity ({self.cfg.max_sessions} sessions)")
        lang = lang or self.cfg.language
        if sid is None:
            self._sid_seq += 1
            sid = f"s{self._sid_seq}"

        s = StreamSession(sid, self, lang)
        prompt_ids = self._build_prompt(lang)
        carried: list[int] = []

        if inherit_from is not None and inherit_from._buffer is not None:
            s._buffer = self._clone_buffer(inherit_from._buffer, keep_secs=carry_secs)
            # The window is already full, so sizing must not ask for chunk+right again.
            s._buffer_fed = True
            s._first_tick_done = True
            if carry_tokens > 0 and inherit_from._state is not None:
                eos = self.model.tokenizer.eos
                prev = inherit_from._state.pred_tokens_ids[
                    0, inherit_from.text_origin:inherit_from._emitted_len].tolist()
                carried = [int(t) for t in prev if int(t) != eos][-carry_tokens:]
        else:
            s._buffer = StreamingBatchedAudioBuffer(
                batch_size=1, context_samples=self.context_samples,
                dtype=torch.float32, device=self.device,   # audio buffer is always float32
            )

        if carried:
            prompt_ids = torch.cat(
                [prompt_ids,
                 torch.tensor([carried], dtype=prompt_ids.dtype, device=prompt_ids.device)],
                dim=-1,
            )
        s._state = GreedyBatchedStreamingAEDComputer.initialize_aed_model_state(
            asr_model=self.model,
            decoder_input_ids=prompt_ids,
            batch_size=1,
            context_encoder_frames=self.context_encoder_frames,
            chunk_secs=self.cfg.chunk_secs,
            right_context_secs=self.cfg.right_context_secs,
        )
        # --- the two state overrides that must happen AFTER initialization ---
        # (1) the AlignAtt budget NeMo just clobbered with 10*int(chunk+right)
        s._state.max_tokens_per_alignatt_step = self.token_budget
        # (2) raise the generation cap -- AND GROW THE BUFFERS TO MATCH.
        #
        # This one is a trap. `initialize_aed_model_state` allocates
        #     pred_tokens_ids = full([B, model_state.max_generation_length], eos)
        # using the state's DEFAULT of 256, and `tokens_frame_alignment` is zeros_like of it.
        # Assigning `state.max_generation_length = 512` afterwards raises the loop bound at
        # aed_batched_streaming.py:223 (`for i in range(start_from, state.max_generation_length)`)
        # without resizing anything, so the writes at lines 165/291 run off the end of a
        # 256-wide tensor. That is an out-of-bounds device write: it does not raise where it
        # happens, it corrupts the CUDA context and surfaces later as
        #     AcceleratorError: CUDA error: an illegal memory access was encountered
        # from whatever unrelated kernel touches the device next.
        #
        # Measured: baseline geometry survived it (short enough that 256 was never reached),
        # T1 at 0.8/0.4 did not -- smaller chunks mean more ticks and more accumulated tokens.
        # So "set both values" is only safe together with the reallocation below.
        self._grow_generation_buffers(s._state, self.cfg.max_generation_length)

        s._emitted_len = prompt_ids.size(-1)
        s.text_origin = prompt_ids.size(-1)
        self.sessions[sid] = s
        if carried:
            log.info("session %s: seamless handover -- %d carried tokens, %.1fs of audio context",
                     sid, len(carried),
                     float(s._buffer.samples.shape[1]) / SAMPLE_RATE)
        return s

    def _grow_generation_buffers(self, state, want: int) -> None:
        """Raise `max_generation_length` and resize the tensors sized from it.

        `pred_tokens_ids` is [B, max_generation_length] filled with eos, with the prompt written
        into its first columns; `tokens_frame_alignment` is zeros_like of it. Both must grow, and
        the existing contents must be preserved -- the prompt lives in there.
        """
        cur = state.pred_tokens_ids
        have = cur.shape[1]
        if want <= have:
            state.max_generation_length = have    # never claim more room than exists
            return

        eos = self.model.tokenizer.eos
        grown = torch.full((cur.shape[0], want), eos, dtype=cur.dtype, device=cur.device)
        grown[:, :have] = cur
        state.pred_tokens_ids = grown

        tfa = state.tokens_frame_alignment
        if tfa is not None:
            grown_tfa = torch.zeros((tfa.shape[0], want), dtype=tfa.dtype, device=tfa.device)
            grown_tfa[:, :have] = tfa
            state.tokens_frame_alignment = grown_tfa

        state.max_generation_length = want

    def close_session(self, sid: str) -> None:
        s = self.sessions.pop(sid, None)
        if s is not None:
            s.close()

    def _push_chunk(self, s: StreamSession, samples: np.ndarray, is_last: bool) -> None:
        """Put one chunk into a session's audio buffer, and verify the context split survived.

        EVERY chunk goes through here, including ones whose forward pass is skipped -- the buffer
        must stay time-aligned with the audio or `_decode_one` computes the wrong usable length
        and AlignAtt quietly stops committing.

        The divergence check exists because that failure was completely silent from this side:
        the only symptom was a third-party `WARNING root:` line from deep inside NeMo, and a demo
        that mysteriously stopped transcribing. If the split ever drifts again it says so once,
        in our own logger, naming the session.
        """
        if samples.size == 0:
            return
        t = torch.from_numpy(samples).to(self.device).unsqueeze(0)
        lens = torch.tensor([samples.size], dtype=torch.long, device=self.device)
        s._buffer.add_audio_batch_(
            t, audio_lengths=lens, is_last_chunk=bool(is_last),
            is_last_chunk_batch=torch.zeros(1, dtype=torch.bool, device=self.device),
        )
        s._buffer_fed = True

        if is_last or s._split_warned:
            return
        got = s._buffer.context_size_batch
        want = self.context_samples
        # `chunk` is legitimately 0 until the buffer has filled its first window.
        chunk_ok = int(got.chunk) in (0, want.chunk)
        if int(got.right) != want.right or not chunk_ok:
            s._split_warned = True
            log.warning(
                "session %s: audio buffer context split diverged -- got %s/%s/%s, expected "
                "%s/%s/%s (samples). AlignAtt derives its commit boundary from this, so a wrong "
                "split stalls emission. Investigate before trusting this session's transcript.",
                s.sid, int(got.left), int(got.chunk), int(got.right),
                want.left, want.chunk, want.right,
            )

    # ---- the tick -------------------------------------------------------------------
    @torch.no_grad()
    @torch.inference_mode()
    def tick(self) -> dict[str, Delta]:
        """Advance every session with a full chunk ready: one batched encoder forward, then a
        per-session decoder step. Returns only sessions that produced new text."""
        t_tick = time.perf_counter()
        candidates = [s for s in list(self.sessions.values()) if s.ready()]

        due: list[StreamSession] = []
        silent_finals: dict[str, Delta] = {}
        for s in candidates:
            # BUGFIX: the break belongs to the loop, not the VAD branch. Nested inside the else
            # it could leave ready sessions unserved while the batch was not actually full.
            if len(due) >= self.cfg.max_batch:
                break
            if s.vad.should_skip_gpu():
                # No speech has EVER been detected on this session.
                if s._pending_finalize:
                    # Do NOT decode silence: an AED decoder asked to transcribe it invents
                    # fluent text -- measured as 6 s of silence producing one token 45 times.
                    # The correct transcript of silence is "".
                    while s._pending.size:
                        s._take_chunk()
                    s._finalized = True
                    self.stats.vad_skipped += 1
                    silent_finals[s.sid] = Delta(text="", token_ids=[], full_text=s._full_text,
                                                 latency_ms=0.0, is_final=True)
                else:
                    # BUGFIX: skip the MODEL, never the BUFFER.
                    #
                    # This used to be a bare `_take_chunk()`, which consumed the audio and set
                    # `_first_tick_done` while the buffer never saw a sample. The next chunk that
                    # did reach the buffer was then `chunk`-sized instead of `chunk + right`, and
                    # NeMo's guard is a strict `elif self.right > expected.chunk:` -- which does
                    # not fire when they are exactly equal. The split settled at (0, 0, chunk) and
                    # stayed wrong for the rest of the stream: right context 0.96 s instead of
                    # 0.48 s, logged once per tick as "Expected context <any> - 15360 - 7680".
                    #
                    # `_decode_one` derives the usable length from that split, so AlignAtt's
                    # commit boundary sat 0.48 s further back than intended on top of
                    # alignatt_thr. During fluent speech attention rides the newest audio, the
                    # condition never became true, and nothing committed -- transcription simply
                    # stopped, resuming a little on each pause. Measured live: a mic sends ~2
                    # chunks of silence before the speaker starts, and every session that did so
                    # was corrupted from its first real chunk onward.
                    #
                    # add_audio_batch_ is a tensor copy, not GPU compute, so feeding it here
                    # keeps the silence gate's saving AND the hallucination suppression (the
                    # decoder still never runs on silence) while the buffer stays time-aligned.
                    self._push_chunk(s, *s._take_chunk()[:2])
                    self.stats.vad_skipped += 1
            else:
                due.append(s)

        if not due:
            # BUGFIX: the reference returned {} here, silently dropping silent_finals and
            # leaving those clients waiting forever for a final that had already been decided.
            return silent_finals

        chunks = [s._take_chunk() for s in due]
        self.stats.ticks += 1
        self.stats.sessions_processed += len(due)

        for s, (samples, is_last, _) in zip(due, chunks):
            self._push_chunk(s, samples, is_last)

        active = [(s, c) for s, c in zip(due, chunks) if s._buffer.samples.shape[1] > 0]
        if not active:
            return silent_finals   # BUGFIX: was `return {}`

        widest = max(s._buffer.samples.shape[1] for s, _ in active)
        batch = torch.zeros(len(active), widest, dtype=torch.float32, device=self.device)
        totals = torch.zeros(len(active), dtype=torch.long, device=self.device)
        real = 0
        for i, (s, _) in enumerate(active):
            n = s._buffer.samples.shape[1]
            batch[i, :n] = s._buffer.samples[0, :n]
            totals[i] = s._buffer.context_size_batch.total()[0]
            real += n
        self.stats.batch_sizes.append(len(active))
        self.stats.padding_waste.append(
            round(100.0 * (1.0 - real / max(1, len(active) * widest)), 2))

        # Audio stays float32 -- the preprocessor consumes float32 and casts internally,
        # exactly as the reference script does.
        #
        # LADDER STEP 2: time this with CUDA events instead of torch.cuda.synchronize().
        #
        # The reference called synchronize() twice per tick, purely to attribute time between
        # the encoder and the decoder. Those calls sit exactly between the two, so they force
        # the CPU to wait for the encoder to finish before it may start issuing decoder work --
        # measurement destroying the overlap it is measuring. Events record on the stream and
        # are read once at the end of the tick, so the attribution survives and the barrier
        # does not. Set CORE_CUDA_EVENTS=0 to restore the old behaviour and measure the delta.
        use_events = self.cfg.use_cuda_events and self.device.type == "cuda"
        if use_events:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev2 = torch.cuda.Event(enable_timing=True)
            ev0.record()
        else:
            t0 = time.perf_counter()

        _, encoded_len, enc_states, _ = self.model(input_signal=batch,
                                                   input_signal_length=totals)
        if use_events:
            ev1.record()
        else:
            torch.cuda.synchronize()
            self.stats.encode_ms += (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()

        out: dict[str, Delta] = dict(silent_finals)
        for i, (s, (samples, is_last, arrival)) in enumerate(active):
            try:
                d = self._decode_one(s, enc_states[i:i + 1], encoded_len[i:i + 1],
                                     is_last, arrival)
                if d is not None:
                    out[s.sid] = d
            except Exception as e:
                if is_fatal_cuda(e):
                    # NOT this session's problem any more -- the device is gone. Do not close
                    # the session and carry on as if the next one might work; it cannot.
                    self.fatal = f"{type(e).__name__}: {str(e)[:200]}"
                    log.critical("FATAL CUDA failure in session %s; engine is unusable: %s",
                                 s.sid, self.fatal)
                    raise
                log.exception("decode failed for session %s; closing it", s.sid)
                self.close_session(s.sid)

        if use_events:
            ev2.record()
            ev2.synchronize()          # one wait per tick, at the end, instead of two mid-tick
            self.stats.encode_ms += ev0.elapsed_time(ev1)
            self.stats.decode_ms += ev1.elapsed_time(ev2)
        else:
            torch.cuda.synchronize()
            self.stats.decode_ms += (time.perf_counter() - t0) * 1000

        self.stats.tick_ms.append((time.perf_counter() - t_tick) * 1000)
        return out

    def _decode_one(self, s: StreamSession, enc, enc_len, is_last: bool, arrival: float):
        # Strip right-context frames from the usable length on non-final chunks: they are
        # lookahead the model may attend to but must not yet commit tokens for. This IS the
        # latency model -- without it the emitted text runs ahead of what the policy allows.
        ctx = s._buffer.context_size_batch.subsample(factor=self.encoder_frame2audio_samples)
        no_rc = ctx.left + ctx.chunk
        is_last_t = torch.tensor([is_last], dtype=torch.bool, device=self.device)
        corrected = torch.where(is_last_t, enc_len, no_rc)

        # Fix (3): clamp exclude_sink_frames against THIS session's source length on every
        # step. NeMo only guards it when i == 0, so a later iteration with a short source makes
        # xatt_scores[:, :, exclude_sink_frames:] empty and argmax raises.
        t_src = int(corrected.min().item())
        self.decoding_cfg.exclude_sink_frames = min(self.cfg.exclude_sink_frames,
                                                    max(0, t_src // 2))

        mask = lens_to_mask(corrected, enc.shape[1]).to(enc.dtype)
        s._state.is_last_chunk_batch = is_last_t
        s._state = self.computer(
            encoder_output=enc,
            encoder_output_len=corrected,
            encoder_input_mask=mask,
            prev_batched_state=s._state,
        )

        end = int(s._state.current_context_lengths[0].item())
        if end <= s._emitted_len and not is_last:
            return None

        new_ids = s._state.pred_tokens_ids[0, s._emitted_len:end].tolist()
        eos = self.model.tokenizer.eos
        new_ids = [t for t in new_ids if t != eos]
        s._emitted_len = end

        prev = s._full_text
        # Slice from THIS turn's origin, not from PROMPT_LEN. On a seamless rotation the state is
        # seeded with prompt + carried tokens, and slicing from PROMPT_LEN would replay the
        # carried tail into full_text -- duplicating it against the cumulative transcript.
        full = self.model.tokenizer.ids_to_text(
            s._state.pred_tokens_ids[0, s.text_origin:end].tolist()
        ).strip()
        s._full_text = full
        delta_text = full[len(prev):] if full.startswith(prev) else full

        if is_last:
            s._finalized = True
        if not delta_text and not is_last:
            return None
        s.note_partial()
        return Delta(
            text=delta_text,
            token_ids=new_ids,
            full_text=full,
            latency_ms=(time.monotonic() - arrival) * 1000,
            is_final=is_last,
        )

    # ---- introspection --------------------------------------------------------------
    def metrics(self) -> dict:
        m = self.stats.snapshot()
        m.update(
            sessions_active=len(self.sessions),
            max_sessions=self.cfg.max_sessions,
            realtime_capacity=self.cfg.realtime_capacity,
            max_batch=self.cfg.max_batch,
            theoretical_latency_s=round(self.theoretical_latency, 3),
            chunk_secs_effective=round(self.chunk_eff, 3),
            right_secs_effective=round(self.right_eff, 3),
            left_secs_effective=round(self.context_samples.left / SAMPLE_RATE, 3),
            alignatt_thr=self.cfg.alignatt_thr,
            token_budget=self.token_budget,
            token_budget_nemo_default=10 * int(self.chunk_eff + self.right_eff),
            dtype=self.cfg.compute_dtype,
            n_languages=len(self.languages),
        )
        # A tick must finish inside one chunk period or the backlog grows without bound.
        budget_ms = self.chunk_eff * 1000
        p95 = m.get("tick_ms_p95")
        m["tick_budget_ms"] = round(budget_ms, 1)
        m["tick_budget_used_p95"] = round(100 * p95 / budget_ms, 1) if p95 else None

        # Over real-time capacity the service does not fail, it falls behind -- and it does so
        # silently. Latency is a poor signal for it: measured, TTFP was LOWER at 10 streams
        # (120% of budget) than at 8 (57%), so anyone watching TTFP would conclude the server
        # was fine. These two fields exist so nobody has to infer it.
        over = len(self.sessions) > self.cfg.realtime_capacity
        m["over_realtime_capacity"] = over
        if over:
            m["capacity_warning"] = (
                f"serving {len(self.sessions)} streams against a measured real-time capacity "
                f"of {self.cfg.realtime_capacity}. Transcripts are still produced but arrive "
                f"progressively later; the drift shows up as normalized latency > 1.0, not as "
                f"an error."
            )
        if torch.cuda.is_available():
            m["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            m["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
        return m
