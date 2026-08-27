import torch
import transformers
from transformers.models.musicgen.modeling_musicgen import (
    MusicgenSinusoidalPositionalEmbedding,
)


class SinusoidalPositionalEmbedding(MusicgenSinusoidalPositionalEmbedding):
    def from_position_ids(self, position_ids):
        bs, seq_len = position_ids.shape
        flat_embs = self.weights.index_select(0, position_ids.view(-1))
        return flat_embs.reshape(bs, seq_len, -1)


class AttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias).half()
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias).half()
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias).half()
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias).half()

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

    def merge_heads(self, x):
        batch_size, heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).reshape(batch_size, seq_len, heads * head_dim)


class CrossAttentionBlock(AttentionBlock):
    def prefill(self, hidden_states, key_value_states):
        """If key_value_states is not None, assumed to cross attention"""
        hidden_states = hidden_states.half()
        key_value_states = key_value_states.half()

        query = self.split_heads(self.q_proj(hidden_states))

        # cross-attention: compute KV once
        key = self.split_heads(self.k_proj(key_value_states))
        value = self.split_heads(self.v_proj(key_value_states))

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
        )

        attn_output = self.merge_heads(attn_output)
        output = self.out_proj(attn_output)
        kv_cache = (key, value)
        return output, kv_cache

    def decode(self, hidden_states, encoder_kv_cache_vmem_thingy):
        hidden_states = hidden_states.half()
        query = self.split_heads(self.q_proj(hidden_states))
        attn = encoder_kv_cache_vmem_thingy
        attn_output = attn(query)
        attn_output = self.merge_heads(attn_output)
        output = self.out_proj(attn_output)
        return output


class SelfAttentionBlock(AttentionBlock):
    def prefill(self, hidden_states, attn_mask=None):
        hidden_states = hidden_states.half()
        query = self.split_heads(self.q_proj(hidden_states))
        key = self.split_heads(self.k_proj(hidden_states))
        value = self.split_heads(self.v_proj(hidden_states))

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask, is_causal=True
        )
        attn_output = self.merge_heads(attn_output)
        output = self.out_proj(attn_output)
        kv_cache = (key, value)
        return output, kv_cache

    def decode(self, hidden_states, decoder_kv_cache_vmem_thingy):
        hidden_states = hidden_states.half()
        query = self.split_heads(self.q_proj(hidden_states))
        key = self.split_heads(self.k_proj(hidden_states))
        value = self.split_heads(self.v_proj(hidden_states))

        cache_updater, attn = decoder_kv_cache_vmem_thingy
        # update key, value into vmem
        cache_updater((key, value))
        # run attn
        attn_output = attn(query)

        attn_output = self.merge_heads(attn_output)
        output = self.out_proj(attn_output)
        return output


class DecoderLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ffn_dim,
        dropout_p,
        activation_dropout_p,
        activation="gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.dropout = torch.nn.Dropout(dropout_p)

        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.self_attn = SelfAttentionBlock(self.embed_dim, num_heads, bias=False)
        self.activation_fn = transformers.activations.ACT2FN[activation]

        self.encoder_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.encoder_attn = CrossAttentionBlock(self.embed_dim, num_heads, bias=False)

        self.activation_dropout = torch.nn.Dropout(activation_dropout_p)
        self.fc1 = torch.nn.Linear(self.embed_dim, self.ffn_dim, bias=False)
        self.fc2 = torch.nn.Linear(self.ffn_dim, self.embed_dim, bias=False)
        self.final_layer_norm = torch.nn.LayerNorm(self.embed_dim)

    def fc_layer(
        self,
        hidden_states,
    ):
        # fully connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def prefill(
        self,
        hidden_states,
        encoder_hidden_states,
    ):
        # self attn
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, kv_cache = self.self_attn.prefill(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # cross attn
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states, encoder_kv_cache = self.encoder_attn.prefill(
            hidden_states, encoder_hidden_states
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # fc
        hidden_states = self.fc_layer(hidden_states)

        return hidden_states, kv_cache, encoder_kv_cache

    def decode(
        self,
        hidden_states,
        encoder_kv_cache_vmem_thingy,
        decoder_kv_cache_vmem_thingy,
    ):
        # self attn
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn.decode(
            hidden_states, decoder_kv_cache_vmem_thingy
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # cross attn
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn.decode(
            hidden_states, encoder_kv_cache_vmem_thingy
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # fc
        hidden_states = self.fc_layer(hidden_states)

        return hidden_states
