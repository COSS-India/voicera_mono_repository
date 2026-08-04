import os
import json

import transformers
import torch
from .layers import SinusoidalPositionalEmbedding, DecoderLayer
from inference.config import device


class ParlerTTS(torch.nn.Module):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.config = json.load(
            open(os.path.join(checkpoint_dir, "indic_parler_tts_config.json"))
        )
        self.description_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/flan-t5-large"
        )
        self.prompt_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "ai4bharat/indic-parler-tts"
        )
        self.description_encoder = transformers.T5EncoderModel.from_pretrained(
            os.path.join(checkpoint_dir, "./parler_tts_text_encoder")
        )
        self.embed_prompt = torch.nn.Embedding(
            self.config["vocab_size"], self.config["decoder"]["hidden_size"]
        )
        self.embed_audio = torch.nn.ModuleList(
            [
                torch.nn.Embedding(
                    self.config["decoder"]["vocab_size"] + 1,
                    self.config["decoder"]["hidden_size"],
                )
                for _ in range(self.config["decoder"]["num_codebooks"])
            ]
        )
        self.embed_position = SinusoidalPositionalEmbedding(
            self.config["decoder"]["max_position_embeddings"],
            self.config["decoder"]["hidden_size"],
        )
        self.decoder_layers = torch.nn.ModuleList(
            [
                DecoderLayer(
                    embed_dim=self.config["decoder"]["hidden_size"],
                    num_heads=self.config["text_encoder"]["num_heads"],
                    ffn_dim=self.config["decoder"]["ffn_dim"],
                    dropout_p=self.config["decoder"]["dropout"],
                    activation_dropout_p=self.config["decoder"]["attention_dropout"],
                    activation=self.config["decoder"]["activation_function"],
                )
                for _ in range(self.config["decoder"]["num_hidden_layers"])
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(self.config["decoder"]["hidden_size"])
        self.lm_heads = torch.nn.Linear(
            self.config["decoder"]["hidden_size"],
            (
                self.config["decoder"]["vocab_size"]
                * self.config["decoder"]["num_codebooks"]
            ),
            bias=False,
        )
        self.dac = transformers.DacModel(
            transformers.DacConfig(
                sampling_rate=44100,
            )
        )
        self.eval()
        self._load_weights()
        self.half()

    def _load_weights(self):
        def _sub_state_dict(prefix, model_weights):
            prefix = prefix + "."
            return {
                k.replace(prefix, ""): v
                for k, v in model_weights.items()
                if prefix in k
            }

        model_weights = torch.load(
            os.path.join(self.checkpoint_dir, "parles_tts_state_dict.pt"),
            map_location=torch.device("cpu"),
            weights_only=True,
        )

        self.description_encoder.load_state_dict(
            _sub_state_dict("text_encoder", model_weights)
        )

        self.embed_prompt.load_state_dict(
            _sub_state_dict("embed_prompts", model_weights)
        )
        self.embed_audio.load_state_dict(
            _sub_state_dict("decoder.model.decoder.embed_tokens", model_weights)
        )
        self.embed_position.weights = torch.nn.Parameter(
            model_weights["decoder.model.decoder.embed_positions.weights"]
        )
        self.decoder_layers.load_state_dict(
            _sub_state_dict("decoder.model.decoder.layers", model_weights)
        )
        self.layer_norm.load_state_dict(
            _sub_state_dict("decoder.model.decoder.layer_norm", model_weights)
        )
        self.lm_heads.load_state_dict(
            _sub_state_dict("decoder.lm_heads", model_weights)
        )
        self.dac.load_state_dict(_sub_state_dict("audio_encoder", model_weights))

    @torch.no_grad
    def encode(
        self,
        prompts,
        descriptions,
    ):
        num_codebooks = self.config["decoder"]["num_codebooks"]
        desc_tokens = self.description_tokenizer(
            descriptions, return_tensors="pt", padding=True
        )
        desc_attention_mask = desc_tokens["attention_mask"].to(device)
        encoder_hidden_states = self.description_encoder(
            desc_tokens["input_ids"].to(device),
            attention_mask=desc_attention_mask,
        ).last_hidden_state
        prompt_tokens = self.prompt_tokenizer(
            prompts, return_tensors="pt", padding=True
        ).to(device)
        prompt_hidden_states = self.embed_prompt(prompt_tokens["input_ids"])
        return (
            encoder_hidden_states,
            prompt_hidden_states,
            desc_attention_mask,
            prompt_tokens["attention_mask"],
        )

    @torch.no_grad
    def prefill(
        self,
        decoder_input_ids,
        decoder_position_ids,
        encoder_hidden_states,
        prompt_hidden_states,
        encoder_attention_mask=None,
        prompt_attention_mask=None,
    ):
        num_decoder_layers = self.config["decoder"]["num_hidden_layers"]
        num_codebooks = self.config["decoder"]["num_codebooks"]
        vocab_size = self.config["decoder"]["vocab_size"]
        decoder_input_ids = decoder_input_ids.reshape(
            -1, num_codebooks, decoder_input_ids.shape[-1]
        )
        inputs_embeds = sum(
            [
                self.embed_audio[codebook](decoder_input_ids[:, codebook])
                for codebook in range(num_codebooks)
            ]
        )
        audio_seq_len = inputs_embeds.shape[1]
        inputs_embeds = torch.cat([prompt_hidden_states, inputs_embeds], dim=1)
        position_embeds = self.embed_position.from_position_ids(
            decoder_position_ids
        ).to(inputs_embeds.device)
        hidden_states = inputs_embeds + position_embeds

        # Only build explicit masks for real batched, mixed-length prefill
        # calls (prompt_attention_mask/encoder_attention_mask supplied). The
        # existing single-request call path passes neither, so self_attn_mask
        # and cross_attn_mask stay None and DecoderLayer.prefill falls back to
        # today's exact is_causal=True / unmasked behavior - zero change at n=1.
        self_attn_mask = None
        cross_attn_mask = None
        if prompt_attention_mask is not None:
            batch_size, full_seq_len = hidden_states.shape[0], hidden_states.shape[1]
            audio_mask = torch.ones(
                batch_size, audio_seq_len, dtype=torch.bool, device=hidden_states.device
            )
            valid = torch.cat([prompt_attention_mask.bool(), audio_mask], dim=1)
            causal = torch.tril(
                torch.ones(
                    full_seq_len, full_seq_len, dtype=torch.bool, device=hidden_states.device
                )
            )
            allowed = causal.unsqueeze(0) & valid.unsqueeze(1)
            self_attn_mask = torch.zeros(
                batch_size, 1, full_seq_len, full_seq_len,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            self_attn_mask.masked_fill_(~allowed.unsqueeze(1), float("-inf"))

        if encoder_attention_mask is not None:
            batch_size = encoder_hidden_states.shape[0]
            enc_seq_len = encoder_hidden_states.shape[1]
            cross_attn_mask = torch.zeros(
                batch_size, 1, 1, enc_seq_len,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            cross_attn_mask.masked_fill_(
                ~encoder_attention_mask.bool().unsqueeze(1).unsqueeze(1), float("-inf")
            )

        model_kv_cache = []
        model_encoder_kv_cache = []
        for layer in range(num_decoder_layers):
            hidden_states, layer_kv_cache, layer_encoder_kv_cache = self.decoder_layers[
                layer
            ].prefill(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
            )
            model_kv_cache.append(layer_kv_cache)
            model_encoder_kv_cache.append(layer_encoder_kv_cache)
        hidden_states = self.layer_norm(hidden_states)
        lm_logits = (
            self.lm_heads(hidden_states)
            .view(hidden_states.shape[0], -1, num_codebooks, vocab_size)
            .transpose(1, 2)
        )
        return lm_logits, model_kv_cache, model_encoder_kv_cache

    @torch.no_grad
    def decode(
        self,
        decoder_input_ids,
        decoder_position_ids,
        model_kv_cache_vmem,
        model_encoder_kv_cache_vmem,
    ):
        # Self-attn grows 1 token/step; cross-attn KV is static (plan cached).
        model_kv_cache_updater, model_attn_kernel = (
            model_kv_cache_vmem.get_decode_closures(grow=True)
        )
        _, model_encoder_attn_kernel = model_encoder_kv_cache_vmem.get_decode_closures(
            grow=False
        )
        return self.decode_forward(
            decoder_input_ids,
            decoder_position_ids,
            model_kv_cache_updater,
            model_attn_kernel,
            model_encoder_attn_kernel,
        )

    @torch.no_grad
    def decode_forward(
        self,
        decoder_input_ids,
        decoder_position_ids,
        model_kv_cache_updater,
        model_attn_kernel,
        model_encoder_attn_kernel,
    ):
        """GPU-only decode body (no page allocate / flashinfer.plan). CUDA-graph friendly."""
        num_decoder_layers = self.config["decoder"]["num_hidden_layers"]
        num_codebooks = self.config["decoder"]["num_codebooks"]
        vocab_size = self.config["decoder"]["vocab_size"]

        decoder_input_ids = decoder_input_ids.reshape(
            -1, num_codebooks, decoder_input_ids.shape[-1]
        )
        inputs_embeds = sum(
            [
                self.embed_audio[codebook](decoder_input_ids[:, codebook])
                for codebook in range(num_codebooks)
            ]
        )
        position_embeds = self.embed_position.from_position_ids(
            decoder_position_ids
        ).to(inputs_embeds.device)
        hidden_states = inputs_embeds + position_embeds

        for layer_id in range(num_decoder_layers):
            decoder_kv_cache_vmem_thingy = (
                lambda append_kv, lid=layer_id: model_kv_cache_updater(lid, append_kv),
                lambda q, lid=layer_id: model_attn_kernel(lid, q),
            )
            encoder_kv_cache_vmem_thingy = (
                lambda q, lid=layer_id: model_encoder_attn_kernel(lid, q)
            )

            hidden_states = self.decoder_layers[layer_id].decode(
                hidden_states=hidden_states,
                encoder_kv_cache_vmem_thingy=encoder_kv_cache_vmem_thingy,
                decoder_kv_cache_vmem_thingy=decoder_kv_cache_vmem_thingy,
            )

        hidden_states = self.layer_norm(hidden_states)
        lm_logits = (
            self.lm_heads(hidden_states)
            .view(hidden_states.shape[0], -1, num_codebooks, vocab_size)
            .transpose(1, 2)
        )
        return lm_logits
