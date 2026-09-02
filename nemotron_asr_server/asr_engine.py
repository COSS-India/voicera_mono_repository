import os
import queue
import threading
import time
import types
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass, field
import torch

# Universal Hydra & NeMo compatibility patch for legacy keyword handling
import hydra._internal.instantiate._instantiate2 as hydra_instantiate
orig_call_target = hydra_instantiate._call_target

# Config keys this patch has already reported, so a repeated load stays quiet.
_reported_drops: set = set()


def safe_call_target(_target_, _partial_, args, kwargs, full_key):
    """
    Let a checkpoint config instantiate against an older NeMo by dropping keyword
    arguments the target does not accept.

    Every drop is LOGGED. This checkpoint was trained on NeMo 3.1.0 and carries
    multisoftmax parameters that NeMo 3.0.0's RNNTDecoder/RNNTJoint/ConvASRDecoder do
    not declare; dropping them silently makes a version mismatch look like a working
    model. (Those particular parameters are inert here -- masking the joint to a
    single language's vocabulary slice was measured to change decoding not at all --
    but the next dropped parameter may not be.)
    """
    import inspect
    try:
        sig = inspect.signature(_target_)
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_var_keyword:
            valid_keys = set(sig.parameters.keys())
            dropped = sorted(k for k in kwargs if k not in valid_keys)
            if dropped:
                name = getattr(_target_, "__name__", str(_target_))
                key = (name, tuple(dropped))
                if key not in _reported_drops:
                    _reported_drops.add(key)
                    print(f"[Engine] config/runtime mismatch: {name} does not accept {dropped} "
                          f"(installed NeMo is older than the checkpoint) -- dropping them")
            kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    except Exception:
        pass
    return orig_call_target(_target_, _partial_, args, kwargs, full_key)


hydra_instantiate._call_target = safe_call_target

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from typing import Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache-aware Conformer models are float32-only: some layers force-cast their output
# to float32, so bf16/fp16 silently corrupts the streaming cache. The vendor's
# reference script raises NotImplementedError for any other compute dtype.
torch.set_float32_matmul_precision("high")

print(f"[Engine] Using compute device: {DEVICE}")

_HERE = os.path.dirname(os.path.abspath(__file__))


def _parse_att_context(raw: str) -> list[int]:
    parts = [p for p in raw.replace("[", "").replace("]", "").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"ASR_ATT_CONTEXT must be 'left,right' (e.g. '96,7'), got {raw!r}")
    return [int(p) for p in parts]


# The encoder was trained with several attention-context configs simultaneously, so the
# streaming latency is chosen at inference time. The right context determines the chunk
# size the encoder expects: chunk = (right + 1) * subsampling_factor mel frames.
#   [96,7] -> 64 frames = 640 ms (recommended by the model README)
#   [96,3] -> 32 frames = 320 ms
#   [96,1] -> 16 frames = 160 ms
ATT_CONTEXT = _parse_att_context(os.environ.get("ASR_ATT_CONTEXT", "96,7"))

# Retained for callers that drive the model directly (tests, offline tools). The
# streaming path no longer serialises on this -- it batches instead, see BatchScheduler.
model_lock = threading.Lock()

# Largest batch the scheduler will assemble, and how long it waits to fill one. The
# window is negligible against a 640 ms chunk, so it costs latency nothing worth
# measuring while letting many streams share a single GPU step.
# The model step costs ~23 ms whether the batch holds 1 stream or 32, so an unused
# batch slot is throughput thrown away. Measured under load, a cap of 32 pinned
# max_batch_seen from 64 streams upward and held capacity near 100 streams.
MAX_BATCH = int(os.environ.get("ASR_MAX_BATCH", "128"))
# 25 ms against a 320 ms chunk is negligible latency, and it is what lets a batch
# actually fill before it runs.
BATCH_WINDOW_MS = float(os.environ.get("ASR_BATCH_WINDOW_MS", "25"))
# Mel extraction is cheap (0.47 ms for a batch of 128), so it does not need a long
# window to amortise. It sits in front of the step batcher, and the two windows add up
# in the latency budget -- 25 ms here cost ~45 ms of time-to-first-word for no
# throughput gain, so it gets its own, shorter one.
MEL_WINDOW_MS = float(os.environ.get("ASR_MEL_WINDOW_MS", "5"))


def find_model_path(env_var: str, candidate_paths: list[str]) -> Optional[str]:
    env_val = os.environ.get(env_var)
    if env_val and os.path.exists(env_val):
        return env_val
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


INDIC_PATH = find_model_path(
    "INDIC_NEMO_PATH",
    [
        "/models/indic-asr-nemotron-600m/indic_nemotron_v1_1_sft_600k_lr1-averaged-40k.nemo",
        os.path.join(_HERE, "models/indic-asr-nemotron-600m/indic_nemotron_v1_1_sft_600k_lr1-averaged-40k.nemo"),
    ],
)

BHILI_PATH = find_model_path(
    "BHILI_NEMO_PATH",
    [
        "/models/bhili-asr-nemotron-600m/indic_nemotron_bhili_sft_lr1-averaged.nemo",
        os.path.join(_HERE, "models/bhili-asr-nemotron-600m/indic_nemotron_bhili_sft_lr1-averaged.nemo"),
    ],
)


def _install_batched_prompt(model):
    """
    Let the language-ID prompt vary across a batch.

    Stock NeMo does `torch.full((B,), self._inference_prompt_index)` -- one scalar
    broadcast over the whole batch, which would force every stream sharing a GPU step
    to use the same language. This accepts a (B,) tensor as well, so a single batched
    step can serve Hindi, Tamil and Bhili at once.
    """

    def _apply(self, encoded):
        if not self.concat or not hasattr(self, "_inference_prompt_index"):
            return encoded
        encoded = encoded.transpose(1, 2)                     # (B, D, T) -> (B, T, D)
        batch, steps, _ = encoded.shape
        prompt = torch.zeros(batch, steps, self.num_prompts,
                             dtype=encoded.dtype, device=encoded.device)
        idx = self._inference_prompt_index
        if torch.is_tensor(idx):
            idx = idx.to(device=encoded.device, dtype=torch.long)
        else:
            idx = torch.full((batch,), int(idx), dtype=torch.long, device=encoded.device)
        prompt.scatter_(2, idx.view(batch, 1, 1).expand(-1, steps, -1), 1.0)
        out_dtype = encoded.dtype
        encoded = self.prompt_kernel(torch.cat([encoded, prompt], dim=-1)).to(out_dtype)
        return encoded.transpose(1, 2)                        # (B, T, D) -> (B, D, T)

    model._apply_prompt_to_encoded = types.MethodType(_apply, model)


class VocabSlicer:
    """
    Restrict decoding to one language's slice of a multisoftmax output layer.

    These checkpoints have a multisoftmax head: 27 languages x 256 tokens = 6912
    classes plus one global blank. `target_lang` is meant to slice the output layer to
    a single language's head -- the model card is explicit that it "slices the output
    layers ... and must be passed on every call". Installed NeMo does not implement
    multisoftmax (see safe_call_target), so without this every language's vocabulary
    competes at once and transcripts come out as fluent-looking cross-script nonsense.

    Implemented as an additive -inf mask on each head's final layer. Two details are
    load-bearing:

    * The mask is allocated once and only ever mutated IN PLACE. RNNT greedy_batch
      decoding runs under CUDA graphs, which bake the captured tensor into the graph;
      in-place edits are seen by replays, whereas rebinding an attribute or
      monkey-patching a method is silently ignored by them.
    * Hooks are registered at load time, before any decode. A graph captured before
      registration would never contain the mask at all.
    """

    def __init__(self, model, name: str):
        self.name = name
        self.calls = 0
        self.active_lang = "<unset>"
        # Off-switch for tests and for language auto-detection, which has no slice.
        self.enabled = True

        tokenizer = model.tokenizer
        offsets = getattr(tokenizer, "token_id_offset", None)
        if not offsets:
            raise RuntimeError(f"[{name}] tokenizer exposes no token_id_offset; cannot slice vocabulary")
        self.offsets = dict(offsets)

        dec_cfg = model.cfg.decoder
        self.vocab_per_lang = int(dec_cfg.get("vocab_per_lang", 0) or 0)
        self.num_langs = int(dec_cfg.get("num_langs", 0) or 0) or len(self.offsets)
        if not self.vocab_per_lang:
            self.vocab_per_lang = len(tokenizer.vocab) // self.num_langs

        expected = self.num_langs * self.vocab_per_lang
        if len(tokenizer.vocab) != expected:
            raise RuntimeError(
                f"[{name}] vocabulary is {len(tokenizer.vocab)} tokens but "
                f"{self.num_langs} langs x {self.vocab_per_lang} = {expected}; "
                f"the slice layout does not match this checkpoint"
            )

        self._heads: list = []
        self._install(model)
        if not self._heads:
            raise RuntimeError(f"[{name}] found no output layer to slice")

    def _attach(self, module, out_features: int, channels_dim: bool):
        # One (MAX_BATCH, V) mask per head so each stream in a batch gets its own
        # language slice. Allocated once and only ever mutated in place, which is what
        # lets CUDA-graph replays observe the current values.
        mask = torch.zeros(MAX_BATCH, out_features, device=DEVICE)

        def hook(_mod, _inp, out, _v=out_features, _ch=channels_dim, _slot=len(self._heads)):
            self.calls += 1
            batch = out.size(0)
            mask_t = self._heads[_slot][0]
            if batch > mask_t.size(0):
                # A decoder may pad the batch beyond MAX_BATCH; grow rather than read
                # off the end of the tensor.
                grown = torch.zeros(batch, _v, device=mask_t.device, dtype=mask_t.dtype)
                grown[: mask_t.size(0)] = mask_t
                self._heads[_slot] = (grown, _v)
                mask_t = grown
            rows = mask_t[:batch]
            # joint: (B, T, U, V) -> (B, 1, 1, V);  ctc conv: (B, V, T) -> (B, V, 1)
            return out + (rows.view(batch, _v, 1) if _ch else rows.view(batch, *([1] * (out.dim() - 2)), _v))

        self._heads.append((mask, out_features))
        module.register_forward_hook(hook)

    def _install(self, model):
        for layer in reversed(list(model.joint.joint_net)):
            if isinstance(layer, torch.nn.Linear):
                # (B, T, U, V): mask broadcasts over the last dim.
                self._attach(layer, layer.out_features, channels_dim=False)
                break

        ctc = getattr(model, "ctc_decoder", None)
        if ctc is not None and hasattr(ctc, "decoder_layers"):
            for layer in reversed(list(ctc.decoder_layers)):
                if isinstance(layer, torch.nn.Conv1d):
                    # (B, V, T): mask must broadcast over the channel dim.
                    self._attach(layer, layer.out_channels, channels_dim=True)
                    break

    def _write_row(self, mask, out_features: int, row: int, lang: Optional[str]):
        if lang is None:
            mask[row].fill_(0.0)                   # no constraint (auto-detect)
        else:
            lo = self.offsets[lang]
            mask[row].fill_(float("-inf"))
            mask[row, lo:lo + self.vocab_per_lang] = 0.0
            mask[row, out_features - 1] = 0.0      # the global blank

    def disable(self):
        """Zero every mask and stop tracking languages -- decoding sees all 27 slices."""
        for mask, _ in self._heads:
            mask.fill_(0.0)
        self.enabled = False
        self.active_lang = "<disabled>"

    def enable(self):
        self.enabled = True
        self.active_lang = "<unset>"

    def set_language(self, lang: Optional[str]):
        """
        Constrain every row to `lang`. Used by the single-stream path and by direct
        callers; lang=None lifts the constraint (auto-detect has no single slice).
        """
        if not self.enabled or lang == self.active_lang:
            return
        if lang is not None and lang not in self.offsets:
            raise ValueError(
                f"[{self.name}] {lang!r} has no vocabulary slice; known: {sorted(self.offsets)}"
            )
        for mask, out_features in self._heads:
            for row in range(mask.size(0)):
                self._write_row(mask, out_features, row, lang)
        self.active_lang = lang

    def _ensure_rows(self, slot: int, needed: int):
        """Grow a head's mask so it can address `needed` batch rows."""
        mask, out_features = self._heads[slot]
        if needed <= mask.size(0):
            return mask
        grown = torch.zeros(needed, out_features, device=mask.device, dtype=mask.dtype)
        grown[: mask.size(0)] = mask
        self._heads[slot] = (grown, out_features)
        return grown

    def set_batch_languages(self, langs: list):
        """
        Give each element of a batch its own slice.

        Grows the mask if the batch is larger than it was allocated for. Without this
        a batch beyond MAX_BATCH raised IndexError here -- before the forward hook's
        own growth path could ever run -- which made MAX_BATCH a hard crash boundary
        rather than a tuning knob.
        """
        if not self.enabled:
            return
        for lang in langs:
            if lang is not None and lang not in self.offsets:
                raise ValueError(f"[{self.name}] {lang!r} has no vocabulary slice")
        for slot in range(len(self._heads)):
            mask = self._ensure_rows(slot, len(langs))
            out_features = self._heads[slot][1]
            for row, lang in enumerate(langs):
                self._write_row(mask, out_features, row, lang)
        self.active_lang = f"<batch:{len(langs)}>"

    def slice_bounds(self, lang: str) -> tuple:
        lo = self.offsets[lang]
        return lo, lo + self.vocab_per_lang

    def stats(self) -> dict:
        return {
            "active_lang": self.active_lang,
            "hook_calls": self.calls,
            "num_langs": self.num_langs,
            "vocab_per_lang": self.vocab_per_lang,
            "heads": len(self._heads),
        }


def configure_model(model, name: str):
    """
    Apply the inference-time setup that cache-aware streaming requires.

    Mirrors speech_to_text_cache_aware_streaming_infer.py. Skipping any of these
    leaves the model in a training-shaped configuration that produces garbled
    transcripts rather than an outright error.
    """
    # The joint is checkpointed with fuse_loss_wer=True / fused_batch_size=2, a
    # training path. -1 disables fusion for inference. change_decoding_strategy only
    # records it on the decoding object, so the joint is set directly as well.
    decoding_cfg = RNNTDecodingConfig(fused_batch_size=-1)
    # CUDA graphs pad the decoding batch to a capture size, which read past the end of
    # the per-element vocabulary mask and produced illegal memory accesses under mixed
    # batches. They also make Python-level hooks invisible to replays -- the trap that
    # hid the multisoftmax bug for two rounds. Batching supplies the throughput now.
    if hasattr(decoding_cfg, "greedy"):
        decoding_cfg.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(decoding_cfg, decoder_type="rnnt")

    # Assert rather than assume: the field is use_cuda_graph_decoder, and setting the
    # wrong name failed silently once already.
    inner = getattr(model.decoding, "decoding", None)
    if getattr(inner, "use_cuda_graph_decoder", False):
        raise RuntimeError(
            f"[{name}] CUDA graph decoder is still enabled; the per-element vocabulary "
            f"mask is not safe under graph replay"
        )
    if hasattr(model.joint, "set_fuse_loss_wer"):
        model.joint.set_fuse_loss_wer(False)
    if hasattr(model.joint, "set_fused_batch_size"):
        model.joint.set_fused_batch_size(-1)

    supported = getattr(model.encoder, "att_context_size_all", None)
    if supported is not None and list(ATT_CONTEXT) not in [list(a) for a in supported]:
        raise ValueError(f"[{name}] att_context_size {ATT_CONTEXT} not supported; available: {supported}")
    model.encoder.set_default_att_context_size(att_context_size=ATT_CONTEXT)

    # Prompted models emit "<hi-IN>"-style language tags inside the transcript.
    if hasattr(model.decoding, "set_strip_lang_tags"):
        model.decoding.set_strip_lang_tags(True)

    # Dithering adds noise to the mel spectrogram; harmless in training, but it makes
    # streaming output non-deterministic and unverifiable against the offline path.
    if hasattr(model.preprocessor, "featurizer"):
        model.preprocessor.featurizer.dither = 0.0

    model.eval()
    model.freeze()

    # Must come before any decoding happens, so CUDA-graph capture includes the mask.
    model._vocab_slicer = VocabSlicer(model, name)
    _install_batched_prompt(model)
    print(f"[Engine] {name}: vocabulary slicing active "
          f"({model._vocab_slicer.num_langs} langs x {model._vocab_slicer.vocab_per_lang} tokens, "
          f"{len(model._vocab_slicer._heads)} head(s))")

    cfg = model.encoder.streaming_cfg
    print(f"[Engine] {name}: att_context_size={ATT_CONTEXT} decoder={model.cur_decoder} "
          f"fused_batch_size={model.joint._fused_batch_size} "
          f"strip_lang_tags={getattr(model.decoding, 'strip_lang_tags', '?')}")
    print(f"[Engine] {name}: {cfg}")
    print(f"[Engine] {name}: model chunk = {cfg.chunk_size[1]} mel frames "
          f"({cfg.chunk_size[1] * 10} ms), first chunk = {cfg.chunk_size[0]} frames")
    return model


def _restore(path: str, name: str):
    """
    Load strictly, and only relax for extra tensors the checkpoint carries.

    A *missing* weight is indistinguishable from a broken model at inference time --
    it just transcribes badly -- so it must be a hard failure. These checkpoints do
    ship unexpected keys (_full_joint_out / _full_ctc_conv / _full_pred_embed: the
    full-vocabulary heads kept from training, unused by the multisoftmax decoder),
    which are harmless to ignore.
    """
    try:
        return nemo_asr.models.ASRModel.restore_from(path, map_location=DEVICE, strict=True)
    except RuntimeError as e:
        msg = str(e)
        if "Missing key(s)" in msg or "size mismatch" in msg:
            raise
        if "Unexpected key(s)" not in msg:
            raise
        extra = msg.split("Unexpected key(s) in state_dict:", 1)[1].strip()
        print(f"[Engine] {name}: ignoring unused checkpoint tensors: {extra}")
        return nemo_asr.models.ASRModel.restore_from(path, map_location=DEVICE, strict=False)


def _load(path: Optional[str], name: str):
    if not path or not os.path.exists(path):
        print(f"[Engine] Warning: {name} model file not found at {path}")
        return None
    print(f"[Engine] Loading {name} ASR model from: {path}...")
    model = _restore(path, name).to(DEVICE)
    model = configure_model(model, name)
    print(f"[Engine] {name} ASR model successfully loaded.")
    return model


indic_model = _load(INDIC_PATH, "indic-multilingual")
bhili_model = _load(BHILI_PATH, "bhili")


LANGUAGE_PROMPT_MAP = {
    "hi": "hi",       # Hindi (6)
    "mr": "mr",       # Marathi (41)
    "bn": "bn",       # Bengali (36)
    "ta": "ta",       # Tamil (39)
    "te": "te",       # Telugu (40)
    "gu": "gu",       # Gujarati (42)
    "kn": "kn",       # Kannada (43)
    "ml": "ml",       # Malayalam (44)
    "pa": "pa",       # Punjabi (86)
    "or": "or",       # Odia (59)
    "as": "as",       # Assamese (63)
    "ur": "ur",       # Urdu (37)
    "sa": "sa",       # Sanskrit (87)
    "ne": "ne",       # Nepali (46)
    "sd": "sd",       # Sindhi (89)
    "mai": "mai",     # Maithili (84)
    "doi": "doi",     # Dogri (76)
    "kok": "kok",     # Konkani (78)
    "brx": "brx",     # Bodo (75)
    "mni": "mni",     # Manipuri (85)
    "sat": "sat",     # Santali (88)
    "ks": "ks",       # Kashmiri (79)
    "bho": "bho",     # Bhojpuri (74)
    "hne": "hne",     # Chhattisgarhi (77)
    "bgc": "bgc",     # Haryanvi (72)
    "bhb": "bhb",     # Bhili (73)
    "bhili": "bhb",
    "en": "en",       # English (0)
    # NOTE: "auto" is deliberately absent. This is a multisoftmax checkpoint whose
    # output layer is sliced per language; with no slice the decoder draws tokens from
    # all 27 vocabularies and emits cross-script nonsense. Two language-ID designs were
    # measured and neither discriminates: scoring slice probability mass lands at
    # chance (0.05 vs 1/27), and scoring each language's prompt picked Urdu for Hindi
    # and Konkani for Bhili. Callers must name the language.
}


def resolve_model_and_prompt(lang_code: str):
    """
    Resolve the model and prompt string for a language code.

    An unknown code raises rather than defaulting to Hindi: silently transcribing
    Tamil audio as Hindi is indistinguishable from the model being broken.
    """
    clean_code = lang_code.strip().lower()
    is_bhili = clean_code in ["bhb", "bhili"]

    if is_bhili and bhili_model is not None:
        return bhili_model, "bhb", True

    if clean_code not in LANGUAGE_PROMPT_MAP:
        raise ValueError(
            f"Unsupported language {lang_code!r}. Supported: {sorted(LANGUAGE_PROMPT_MAP)}"
        )
    return indic_model, LANGUAGE_PROMPT_MAP[clean_code], False


def set_prompt(model, target_lang: str):
    """
    Select the decoder's language slice. The decoder is multisoftmax
    (27 languages x 256 tokens = 6912 classes); the prompt picks which 256-token
    slice decodes, so the wrong prompt yields the wrong script entirely.

    Must be re-applied before every step: the index lives on the shared model.
    Never swallow the error -- an unset prompt makes _apply_prompt_to_encoded a
    no-op, which drops language conditioning silently.

    Callers must hold model_lock. Re-setting an index that is already current is
    skipped, which avoids an INFO log line on every 640 ms step.
    """
    prompt_dict = model.cfg.model_defaults.get("prompt_dictionary", {})
    if target_lang not in prompt_dict:
        raise ValueError(
            f"Unknown target language {target_lang!r}; not a key of the model's prompt_dictionary"
        )
    # The encoder-side prompt and the output-layer slice are two halves of the same
    # switch; applying only the prompt leaves all 27 vocabularies competing.
    slicer = getattr(model, "_vocab_slicer", None)
    if slicer is not None:
        slicer.set_language(target_lang)

    if getattr(model, "_inference_prompt_index", None) == prompt_dict[target_lang]:
        return
    model.set_inference_prompt(target_lang)


def prompt_index(model, target_lang: str) -> int:
    """The encoder-side prompt id for a language, from the checkpoint's dictionary."""
    return int(model.cfg.model_defaults.get("prompt_dictionary", {})[target_lang])


def get_streaming_params(model):
    """
    Chunking geometry for one stream, read from the encoder rather than hardcoded.

    Returns a dict of [first_step, subsequent_steps] mel-frame counts plus the
    scalars conformer_stream_step needs.
    """
    cfg = model.encoder.streaming_cfg

    def as_pair(v):
        return list(v) if isinstance(v, (list, tuple)) else [v, v]

    pre_encode = getattr(model.encoder, "pre_encode", None)
    if pre_encode is not None and hasattr(pre_encode, "get_sampling_frames"):
        sampling_frames = as_pair(pre_encode.get_sampling_frames())
    else:
        sampling_frames = [1, 1]

    return {
        "chunk_size": as_pair(cfg.chunk_size),
        "shift_size": as_pair(cfg.shift_size),
        "pre_encode_cache_size": as_pair(cfg.pre_encode_cache_size),
        "drop_extra_pre_encoded": int(cfg.drop_extra_pre_encoded),
        "sampling_frames": sampling_frames,
        "hop_length": int(round(model.cfg.preprocessor.window_stride * model.cfg.preprocessor.sample_rate)),
        "feat_in": int(model.encoder._feat_in),
    }


def get_initial_cache(batch_size: int = 1, is_bhili: bool = False):
    """
    Returns initial cache tensors (cache_last_channel, cache_last_time, cache_last_channel_len)
    """
    model = bhili_model if (is_bhili and bhili_model is not None) else indic_model
    if model is None:
        raise RuntimeError("ASR Model is not loaded or initialized.")
    return model.encoder.get_initial_cache_state(batch_size=batch_size)


@dataclass
class StepRequest:
    """One stream's request for a single conformer_stream_step."""
    model_key: str
    chunk: torch.Tensor                 # (1, F, T) mel, pre-encode cache already prepended
    cache_channel: torch.Tensor         # (layers, 1, ...)
    cache_time: torch.Tensor            # (layers, 1, ...)
    cache_len: torch.Tensor             # (1,)
    prev_hyp: object
    prev_pred_out: object
    prompt_index: int
    slice_lang: Optional[str]
    drop_extra: int
    keep_all_outputs: bool
    future: Future = field(default_factory=Future)

    @property
    def group(self):
        # These are scalar arguments to conformer_stream_step, so a batch must agree on
        # them; chunk length differs between the first step and later ones.
        return (self.model_key, self.chunk.size(-1), self.drop_extra,
                self.keep_all_outputs, self.prev_hyp is None)


class BatchScheduler:
    """
    Continuous batching across streams.

    Previously every stream took a global lock for its own conformer_stream_step, so
    640 ms of audio cost 19.8 ms of exclusive GPU time and throughput capped near 32
    streams while the GPU sat ~92% idle. Here streams hand their chunk to one worker
    that assembles a batch and runs a single step for all of them.
    """

    def __init__(self, models: dict):
        self.models = models
        self.queue: queue.Queue = queue.Queue()
        self.window = BATCH_WINDOW_MS / 1000.0
        self.batches = 0
        self.items = 0
        self.max_seen = 0
        self._thread = threading.Thread(target=self._loop, daemon=True, name="asr-batcher")
        self._thread.start()

    def submit(self, request: StepRequest) -> Future:
        self.queue.put(request)
        return request.future

    def stats(self) -> dict:
        return {
            "batches": self.batches,
            "items": self.items,
            "mean_batch": round(self.items / self.batches, 2) if self.batches else 0.0,
            "max_batch_seen": self.max_seen,
            "max_batch": MAX_BATCH,
            "window_ms": BATCH_WINDOW_MS,
        }

    def _loop(self):
        while True:
            try:
                buckets = defaultdict(list)
                first = self.queue.get()
                buckets[first.group].append(first)

                # Collect whatever else shows up inside the window.
                deadline = time.perf_counter() + self.window
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        item = self.queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    buckets[item.group].append(item)

                for group, items in buckets.items():
                    for start in range(0, len(items), MAX_BATCH):
                        self._run(group, items[start:start + MAX_BATCH])
            except Exception as e:                       # never let the worker die
                print(f"[Batcher] scheduler error: {e}")

    def _run(self, group, items):
        model = self.models[group[0]]
        batch = len(items)
        try:
            with torch.inference_mode():
                signal = torch.cat([i.chunk for i in items], dim=0)
                lengths = torch.full((batch,), signal.size(-1),
                                     device=signal.device, dtype=torch.long)
                cache_channel = torch.cat([i.cache_channel for i in items], dim=1)
                cache_time = torch.cat([i.cache_time for i in items], dim=1)
                cache_len = torch.cat([i.cache_len for i in items], dim=0)

                prev_hyp = None if items[0].prev_hyp is None else [h for i in items for h in i.prev_hyp]
                prev_pred = None if items[0].prev_pred_out is None else [p for i in items for p in i.prev_pred_out]

                # Per-element language: prompt on the encoder side, slice on the output.
                model._inference_prompt_index = torch.tensor(
                    [i.prompt_index for i in items], dtype=torch.long, device=signal.device)
                slicer = getattr(model, "_vocab_slicer", None)
                if slicer is not None:
                    slicer.set_batch_languages([i.slice_lang for i in items])

                pred_out, transcriptions, cc, ct, cl, hyps = model.conformer_stream_step(
                    processed_signal=signal,
                    processed_signal_length=lengths,
                    cache_last_channel=cache_channel,
                    cache_last_time=cache_time,
                    cache_last_channel_len=cache_len,
                    keep_all_outputs=group[3],
                    previous_hypotheses=prev_hyp,
                    previous_pred_out=prev_pred,
                    drop_extra_pre_encoded=group[2],
                    return_transcription=True,
                )

            for n, item in enumerate(items):
                item.future.set_result({
                    "text": extract_transcription_text([transcriptions[n]]),
                    "cache_channel": cc[:, n:n + 1],
                    "cache_time": ct[:, n:n + 1],
                    "cache_len": cl[n:n + 1],
                    "prev_hyp": [hyps[n]] if hyps is not None else None,
                    "prev_pred_out": [pred_out[n]] if pred_out is not None else None,
                })

            self.batches += 1
            self.items += batch
            self.max_seen = max(self.max_seen, batch)
        except Exception as e:
            for item in items:
                if not item.future.done():
                    item.future.set_exception(e)


@dataclass
class MelRequest:
    """One stream's raw-audio window awaiting mel extraction."""
    model_key: str
    segment: torch.Tensor            # (L,) float32 on device
    future: Future = field(default_factory=Future)


class MelScheduler:
    """
    Batch mel extraction across streams.

    The preprocessor costs 0.66 ms for one stream and 0.47 ms for a batch of 128, yet it
    was being called once per session per wire frame on the event loop. Measured, that
    per-message cost was the throughput ceiling: halving the message count raised
    sustained steps/s from 343 to 502, and mel was 0.66 ms of the 0.92 ms per message.

    Only non-final windows are batched. A final window keeps every frame including the
    right edge, and zero-padding a batch changes the STFT's reflection padding there, so
    flushes run unbatched to stay bit-identical to offline extraction.
    """

    def __init__(self, models: dict):
        self.models = models
        self.queue: queue.Queue = queue.Queue()
        self.window = MEL_WINDOW_MS / 1000.0
        self.batches = 0
        self.items = 0
        self.max_seen = 0
        threading.Thread(target=self._loop, daemon=True, name="asr-mel-batcher").start()

    def submit(self, request: MelRequest) -> Future:
        self.queue.put(request)
        return request.future

    def stats(self) -> dict:
        return {"batches": self.batches, "items": self.items,
                "mean_batch": round(self.items / self.batches, 2) if self.batches else 0.0,
                "max_batch_seen": self.max_seen}

    def _loop(self):
        while True:
            try:
                buckets = defaultdict(list)
                first = self.queue.get()
                buckets[first.model_key].append(first)
                deadline = time.perf_counter() + self.window
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        item = self.queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    buckets[item.model_key].append(item)
                for key, items in buckets.items():
                    for start in range(0, len(items), MAX_BATCH):
                        self._run(key, items[start:start + MAX_BATCH])
            except Exception as e:
                print(f"[MelBatcher] scheduler error: {e}")

    def _run(self, model_key, items):
        if not items:
            return
        model = self.models[model_key]
        try:
            with torch.inference_mode():
                lengths = [i.segment.numel() for i in items]
                width = max(lengths)
                signal = torch.zeros(len(items), width, device=items[0].segment.device,
                                     dtype=items[0].segment.dtype)
                for n, item in enumerate(items):
                    signal[n, :lengths[n]] = item.segment
                lens = torch.tensor(lengths, device=signal.device, dtype=torch.long)
                mel, mel_len = model.preprocessor(input_signal=signal, length=lens)
            for n, item in enumerate(items):
                item.future.set_result(mel[n:n + 1, :, :int(mel_len[n])].clone())
            self.batches += 1
            self.items += len(items)
            self.max_seen = max(self.max_seen, len(items))
        except Exception as e:
            for item in items:
                if not item.future.done():
                    item.future.set_exception(e)


scheduler: Optional[BatchScheduler] = None
mel_scheduler: Optional[MelScheduler] = None


def scheduler_stats() -> Optional[dict]:
    return scheduler.stats() if scheduler is not None else None


def mel_scheduler_stats() -> Optional[dict]:
    return mel_scheduler.stats() if mel_scheduler is not None else None


def get_mel_scheduler() -> "MelScheduler":
    if mel_scheduler is None:
        start_scheduler()
    return mel_scheduler


def submit_mel(model_key: str, segment: torch.Tensor) -> Future:
    return get_mel_scheduler().submit(MelRequest(model_key=model_key, segment=segment))


def get_scheduler() -> "BatchScheduler":
    """The scheduler, started on first use so direct callers work without the server."""
    return scheduler if scheduler is not None else start_scheduler()


def start_scheduler():
    global scheduler, mel_scheduler
    models = {"indic": indic_model, "bhili": bhili_model}
    if scheduler is None:
        scheduler = BatchScheduler(models)
        mel_scheduler = MelScheduler(models)
        print(f"[Engine] batch schedulers started (max_batch={MAX_BATCH}, "
              f"step window={BATCH_WINDOW_MS} ms, mel window={MEL_WINDOW_MS} ms)")
    return scheduler


def extract_transcription_text(transcriptions) -> str:
    """
    Extracts text string safely from NeMo transcription output.
    """
    if not transcriptions:
        return ""
    t = transcriptions[0]
    if hasattr(t, 'text'):
        return str(t.text).strip()
    if isinstance(t, str):
        return t.strip()
    return str(t).strip()
