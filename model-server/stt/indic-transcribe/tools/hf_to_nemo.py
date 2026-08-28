#!/usr/bin/env python3
"""Stage 2 — convert the HF `indic-transcribe-core` port into a NeMo `.nemo` checkpoint.

Why a conversion at all
-----------------------
The streaming engine is NeMo's AlignAtt implementation over `EncDecMultiTaskModel`. The
published checkpoint is an HF port. Nothing about the weights differs; only the packaging does.

The mapping, verified empirically against both key sets before this was written
--------------------------------------------------------------------------------
Four prefix rules and **zero reshapes**. Encoder 1292 <-> 1292 tensors and decoder 630 <-> 630,
with no key on either side unmatched and no shape mismatch anywhere:

  1. ``model.encoder.X``            -> ``encoder.X``
  2. ``model.decoder.embedding.X``  -> ``transf_decoder._embedding.X``
  3. ``model.decoder.X``            -> ``transf_decoder._decoder.X``     (layers, final_layer_norm)
  4. ``lm_head.bias``               -> ``log_softmax.mlp.layer0.bias``

Plus two things that are not renames:

  * ``log_softmax.mlp.layer0.weight`` does not exist in the HF file at all. `lm_head.weight` is
    tied to ``model.decoder.embedding.token_embedding.weight`` ([7152, 1024]), so it is
    materialised here -- NeMo's TokenClassifier holds a real, untied weight.
  * ``preprocessor.featurizer.{fb,window}`` come from `feature_extractor.safetensors`, not from
    the model file.

The 32 int64 `num_batches_tracked` BatchNorm counters are preserved as int64; casting them to
float silently breaks BatchNorm's running statistics.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path

import torch

# --------------------------------------------------------------------------------------------
# The model_config.yaml for `core`.
#
# Structurally identical to the Bhili checkpoint's (same architecture: 32-layer Conformer
# encoder, 24-layer transformer decoder, d_model 1024, vocab 7152), with the differences the
# `core` checkpoint's own tokenizer_config.json dictates:
#
#   * prompt_defaults uses <|pnc|>, NOT <|nopnc|>          <- Bhili is the opposite
#   * prompt_defaults gains a `romanized` slot             <- Bhili has no such slot
#
# Both are read straight off core's prompt table, which is 10 tokens
# [7, 4, 18, L, L, 5, 9, 11, 13, 15] =
#   <|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|LANG|> <|LANG|>
#   <|pnc|> <|noitn|> <|noromanized|> <|notimestamp|> <|nodiarize|>
# against Bhili's 9 (no romanized slot, and <|nopnc|>).
#
# Training-only sections (train_ds/validation_ds/spec_augment/optim) are dropped: they point at
# paths that do not exist here and nothing on the inference path reads them.
# --------------------------------------------------------------------------------------------
CONFIG_TEMPLATE = """
sample_rate: 16000
label_smoothing: 0.0
use_loss_mask_for_prompt: false
log_prediction: true
prompt_format: canary2
prompt_defaults:
- role: user
  slots:
    decodercontext: ''
    source_lang: <|{lang}|>
    target_lang: <|{lang}|>
    emotion: <|emo:undefined|>
    pnc: <|pnc|>
    itn: <|noitn|>
    romanized: <|noromanized|>
    diarize: <|nodiarize|>
    timestamp: <|notimestamp|>
- role: user_partial
  slots:
    decodercontext: ''
model_defaults:
  asr_enc_hidden: 1024
  lm_enc_hidden: 1024
  lm_dec_hidden: 1024
tokenizer:
  dir: null
  type: agg
  langs:
    spl_tokens:
      type: bpe
      model_path: nemo:{spl_model}
      vocab_path: nemo:{spl_vocabtxt}
      spe_tokenizer_vocab: nemo:{spl_vocab}
    multilingual:
      type: bpe
      model_path: nemo:{multi_model}
      vocab_path: nemo:{multi_vocabtxt}
      spe_tokenizer_vocab: nemo:{multi_vocab}
  custom_tokenizer:
    _target_: nemo.collections.common.tokenizers.canary_multilingual_tokenizer.CanaryMultilingualTokenizer
    tokenizers: null
preprocessor:
  _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
  sample_rate: 16000
  normalize: per_feature
  window_size: 0.025
  window_stride: 0.01
  window: hann
  features: 128
  n_fft: 512
  log: true
  frame_splicing: 1
  dither: 1.0e-05
  pad_to: 0
  pad_value: 0.0
encoder:
  _target_: nemo.collections.asr.modules.ConformerEncoder
  feat_in: 128
  feat_out: -1
  n_layers: 32
  d_model: 1024
  subsampling: dw_striding
  subsampling_factor: 8
  subsampling_conv_channels: 256
  causal_downsampling: false
  reduction: null
  reduction_position: null
  reduction_factor: 1
  ff_expansion_factor: 4
  self_attention_model: rel_pos
  n_heads: 8
  att_context_size:
  - -1
  - -1
  xscaling: false
  untie_biases: true
  pos_emb_max_len: 5000
  conv_kernel_size: 9
  conv_norm_type: batch_norm
  conv_context_size: null
  dropout: 0.1
  dropout_pre_encoder: 0.1
  dropout_emb: 0.0
  dropout_att: 0.1
transf_encoder:
  _target_: nemo.collections.asr.modules.transformer.transformer_encoders.TransformerEncoder
  num_layers: 0
  hidden_size: 1024
  inner_size: 4096
  num_attention_heads: 8
  ffn_dropout: 0.1
  attn_score_dropout: 0.1
  attn_layer_dropout: 0.1
  mask_future: false
  pre_ln: true
  pre_ln_final_layer_norm: true
transf_decoder:
  _target_: nemo.collections.asr.modules.transformer.get_nemo_transformer
  model_name: null
  pretrained: false
  encoder: null
  pre_ln_final_layer_norm: true
  config_dict:
    max_sequence_length: 1024
    num_token_types: 0
    embedding_dropout: 0.1
    learn_positional_encodings: false
    hidden_size: 1024
    inner_size: 4096
    num_layers: 24
    num_attention_heads: 8
    ffn_dropout: 0.1
    attn_score_dropout: 0.1
    attn_layer_dropout: 0.1
    hidden_act: relu
    pre_ln: true
    vocab_size: None
head:
  _target_: nemo.collections.asr.parts.submodules.token_classifier.TokenClassifier
  num_layers: 1
  activation: relu
  log_softmax: true
  hidden_size: 1024
  num_classes: 7152
  dropout: 0.0
  use_transformer_init: true
decoding:
  strategy: beam
  return_best_hypothesis: true
  beam:
    beam_size: 1
    len_pen: 0.0
    max_generation_delta: 50
loss:
  _target_: nemo.collections.common.losses.smoothed_cross_entropy.SmoothedCrossEntropyLoss
  label_smoothing: 0.0
  pad_id: 2
target: nemo.collections.asr.models.aed_multitask_models.EncDecMultiTaskModel
nemo_version: 2.8.0rc0
"""


def _header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def rename(key: str) -> str | None:
    """The four prefix rules. Returns None for keys handled specially."""
    if key.startswith("model.encoder."):
        return "encoder." + key[len("model.encoder."):]
    if key.startswith("model.decoder.embedding."):
        return "transf_decoder._embedding." + key[len("model.decoder.embedding."):]
    if key.startswith("model.decoder."):
        return "transf_decoder._decoder." + key[len("model.decoder."):]
    if key == "lm_head.bias":
        return "log_softmax.mlp.layer0.bias"
    if key == "lm_head.weight":
        return None  # tied; materialised from the embedding below
    raise KeyError(f"unmapped checkpoint key: {key!r}")


def build_weights(hf_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    model_st = hf_dir / "model.safetensors"
    feat_st = hf_dir / "feature_extractor.safetensors"
    out: dict[str, torch.Tensor] = {}

    n_int64 = 0
    with safe_open(str(model_st), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open has no __iter__
            dst = rename(key)
            if dst is None:
                continue
            t = f.get_tensor(key)
            # int64 BatchNorm counters must stay int64 -- casting them breaks running stats.
            if t.dtype == torch.int64:
                n_int64 += 1
            out[dst] = t

    # lm_head.weight is absent from the file (tied to the token embedding). NeMo's
    # TokenClassifier owns a real weight, so materialise it as an independent copy.
    emb = out["transf_decoder._embedding.token_embedding.weight"]
    out["log_softmax.mlp.layer0.weight"] = emb.clone()

    # The mel front-end's two buffers live in a separate file.
    with safe_open(str(feat_st), framework="pt", device="cpu") as f:
        available = set(f.keys())
        for src, dst in (("fb", "preprocessor.featurizer.fb"),
                         ("window", "preprocessor.featurizer.window")):
            if src not in available:
                raise KeyError(f"{feat_st.name} has no {src!r}; found {sorted(available)}")
            out[dst] = f.get_tensor(src)

    print(f"[convert] {len(out)} tensors  (int64 BatchNorm counters preserved: {n_int64})",
          file=sys.stderr)
    return out


def write_tokenizer_artifacts(hf_dir: Path, stage: Path) -> dict[str, str]:
    """Copy core's OWN tokenizer models and derive the .vocab / vocab.txt companions.

    core's tokenizer must be used, not another checkpoint's: vocabularies drift across this
    model family (core carries <|af|>/<|ak|> where flex carries <|bgc|>/<|hne|>), and a
    mismatched spl vocab silently shifts every language token id.
    """
    import sentencepiece as spm

    names: dict[str, str] = {}
    for role, fname in (("spl", "tokenizer_spl_tokens.model"),
                        ("multi", "tokenizer_multilingual.model")):
        src = hf_dir / fname
        if not src.exists():
            raise FileNotFoundError(src)

        # NeMo addresses archive members as nemo:<uuid32>_<basename>.
        model_name = f"{uuid.uuid4().hex}_tokenizer.model"
        shutil.copyfile(src, stage / model_name)
        names[f"{role}_model"] = model_name

        sp = spm.SentencePieceProcessor(model_file=str(src))
        pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        scores = [sp.get_score(i) for i in range(sp.get_piece_size())]

        vocab_name = f"{uuid.uuid4().hex}_tokenizer.vocab"
        (stage / vocab_name).write_text(
            "".join(f"{p}\t{s}\n" for p, s in zip(pieces, scores)), encoding="utf-8")
        names[f"{role}_vocab"] = vocab_name

        vocabtxt_name = f"{uuid.uuid4().hex}_vocab.txt"
        (stage / vocabtxt_name).write_text(
            "".join(f"{p}\n" for p in pieces), encoding="utf-8")
        names[f"{role}_vocabtxt"] = vocabtxt_name

        print(f"[convert] {role}: {sp.get_piece_size()} pieces -> {model_name}", file=sys.stderr)
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", type=Path, default=Path("/models/core"))
    ap.add_argument("--out", type=Path, default=Path("/artifacts/indic_transcribe_core.nemo"))
    ap.add_argument("--lang", default="hi",
                    help="default prompt language baked into prompt_defaults; the engine "
                         "overrides it per request, so this is only the fallback")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    weights = build_weights(args.hf_dir)

    with tempfile.TemporaryDirectory(dir=args.out.parent) as td:
        stage = Path(td)
        names = write_tokenizer_artifacts(args.hf_dir, stage)

        cfg = CONFIG_TEMPLATE.format(
            lang=args.lang,
            spl_model=names["spl_model"], spl_vocab=names["spl_vocab"],
            spl_vocabtxt=names["spl_vocabtxt"],
            multi_model=names["multi_model"], multi_vocab=names["multi_vocab"],
            multi_vocabtxt=names["multi_vocabtxt"],
        ).lstrip()
        (stage / "model_config.yaml").write_text(cfg, encoding="utf-8")

        ckpt = stage / "model_weights.ckpt"
        torch.save(weights, ckpt)
        print(f"[convert] weights -> {ckpt.stat().st_size / 1e9:.2f} GB", file=sys.stderr)

        # .nemo is an uncompressed tar whose members are './'-prefixed.
        tmp_out = args.out.with_suffix(".nemo.tmp")
        with tarfile.open(tmp_out, "w") as tar:
            for p in sorted(stage.iterdir()):
                tar.add(p, arcname="./" + p.name)
        tmp_out.replace(args.out)

    print(f"[convert] wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
