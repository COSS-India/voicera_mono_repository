import numpy as np
from streamer import ParlerTTSStreamer
import transformers
import torch
import torch.nn as nn
import copy
from ragged_parler_utils import *

class ParlerTTSModelRunner:
    def __init__(self, model, tokenizer, description_tokenizer, play_steps=10):
        self.model = model
        self.tokenizer = tokenizer
        self.description_tokenizer = description_tokenizer
        self._stopping_criteria = None
        self.play_steps = play_steps  
        self.device = self.model.device
        self.evicted_streamers = []
        self._num_codebooks = 9     

    def _prepare_seq_idxs(self, prompt_sizes, decoder_sizes, description_sizes):
        p = sum(prompt_sizes)
        d = sum(decoder_sizes)
        n_dec = p + d
        n_enc = sum(description_sizes)
        n_seq = len(prompt_sizes)

        seq_dec_idxs = []
        p_cum_sum = np.cumsum([0, *prompt_sizes])
        d_cum_sum = np.cumsum([0, *decoder_sizes])

        seq_dec_idxs = [
            (
                list(range(p_cum_sum[seq_idx], p_cum_sum[seq_idx + 1]))
                + list(range(p + d_cum_sum[seq_idx], p + d_cum_sum[seq_idx + 1]))
            )
            for seq_idx in range(n_seq)
        ]

        e_cum_sum = np.cumsum([0, *description_sizes])
        seq_enc_idxs = [
            list(range(e_cum_sum[seq_idx], e_cum_sum[seq_idx + 1]))
            for seq_idx in range(n_seq)
        ]
                
        decoder_sequence_ids = torch.zeros(n_dec, dtype=torch.long)
        encoder_sequence_ids = torch.zeros(n_enc, dtype=torch.long)
        for seq_idx in range(n_seq): 
            decoder_sequence_ids[seq_dec_idxs[seq_idx]] = seq_idx
            encoder_sequence_ids[seq_enc_idxs[seq_idx]] = seq_idx

        return decoder_sequence_ids, encoder_sequence_ids
    
    def prepare_attention_masks(self, decoder_sequence_ids, encoder_sequence_ids):
        encoder_attn_mask = (decoder_sequence_ids[:, None] == encoder_sequence_ids[None, :])
        encoder_attn_mask = encoder_attn_mask.unsqueeze(0).unsqueeze(0).to(self.model.device)
        encoder_attn_mask_float = torch.full(
            encoder_attn_mask.shape,
            float("-inf"),
            dtype=self.model.dtype,
            device=self.model.device,
        )
        encoder_attn_mask_float[encoder_attn_mask] = 0

        decoder_attn_mask = (decoder_sequence_ids[:, None] == decoder_sequence_ids[None, :])
        decoder_attn_mask = torch.tril(decoder_attn_mask)
        decoder_attn_mask = decoder_attn_mask.unsqueeze(0).unsqueeze(0).to(self.model.device)
        decoder_attn_mask_float = torch.full(
            decoder_attn_mask.shape,
            float("-inf"),
            dtype=self.model.dtype,
            device=self.model.device,
        )
        decoder_attn_mask_float[decoder_attn_mask] = 0
        return encoder_attn_mask_float, decoder_attn_mask_float

    def _generate_decoder_position_ids(self, decoder_sequence_ids):
        num_seq = decoder_sequence_ids.max() + 1
        decoder_position_ids = torch.zeros_like(decoder_sequence_ids)

        for seq_idx in range(num_seq):
            seq_len = (decoder_sequence_ids == seq_idx).sum()
            decoder_position_ids[decoder_sequence_ids == seq_idx] = torch.arange(seq_len)

        return decoder_position_ids
    
    def create_streamer_sequence(self, num_seq):
        streamers = []
        for i in range(num_seq):
            streamers.append(ParlerTTSStreamer(self.model, play_steps=self.play_steps))
        return streamers
        
    def model_prefill(self, prompts, descriptions):
        # no batching of encoder for now
        encoder_outputs = [
            my_encoder([prompt], [description], self.model, self.tokenizer, self.description_tokenizer, self.device)
            for prompt, description in zip(prompts, descriptions)
        ]
        if self._stopping_criteria is None:
            self._stopping_criteria = encoder_outputs[0]["stopping_criteria"]

        logits_processors = [x["logits_processor"] for x in encoder_outputs]

        model_inputs = [
            self.model.prepare_inputs_for_generation(
                x["delayed_input_ids"], **x["model_kwargs"]
            )
            for x in encoder_outputs
        ]

        # encoder_outputs
        encoder_last_hidden_states = torch.cat(
            [x["encoder_outputs"].last_hidden_state for x in model_inputs], dim=1
        )
        encoder_outputs = copy.deepcopy(
            model_inputs[0]["encoder_outputs"]
        )  # copy object
        encoder_outputs.last_hidden_state = encoder_last_hidden_states

        decoder_input_ids = torch.cat(
            [x["decoder_input_ids"] for x in model_inputs], dim=1
        )
        prompt_hidden_states = torch.cat(
            [x["prompt_hidden_states"] for x in model_inputs], dim=1,
        )

        prompt_sizes = [x["prompt_hidden_states"].shape[1] for x in model_inputs]
        decoder_sizes = [x["decoder_input_ids"].shape[1] for x in model_inputs]
        description_sizes = [
            x["encoder_outputs"].last_hidden_state.shape[1] for x in model_inputs
        ]

        decoder_sequence_ids, encoder_sequence_ids = self._prepare_seq_idxs(
            prompt_sizes, decoder_sizes, description_sizes
        )
        decoder_position_ids = self._generate_decoder_position_ids(decoder_sequence_ids)
        encoder_attn_mask_float, decoder_attn_mask_float = self.prepare_attention_masks(
            decoder_sequence_ids, encoder_sequence_ids
        )

        past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        model_out = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attn_mask_float,
            prompt_hidden_states=prompt_hidden_states,
            decoder_input_ids=decoder_input_ids,
            decoder_position_ids=decoder_position_ids.unsqueeze(0).to(self.model.device),
            decoder_attention_mask=decoder_attn_mask_float,
            past_key_values=past_key_values.to_legacy_cache(),
            use_cache=True
        )

        # track a mask in decoder to classify token to audio/prompt
        # 0 -> audio, 1 -> prompt text
        decoder_token_categories = torch.zeros_like(decoder_position_ids)
        decoder_token_categories[:sum(prompt_sizes)] = 1

        # unfinished codebooks
        num_seq = len(prompts)
        unfinished_codebooks = torch.ones((self._num_codebooks, num_seq), dtype=torch.long, device=self.model.device)
        streamers = self.create_streamer_sequence(num_seq)
        model_state = {
            "encoder_outputs": encoder_outputs,
            "past_key_values": model_out.past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "logits": model_out.logits,
            "decoder_sequence_ids": decoder_sequence_ids,
            "decoder_token_categories": decoder_token_categories,
            "encoder_sequence_ids": encoder_sequence_ids,
            "logits_processors": logits_processors,
            "unfinished_codebooks": unfinished_codebooks,
            "num_seq": num_seq,
            "streamers": streamers,
            "eos_flags": self._get_eos_flags(num_seq, unfinished_codebooks)
        }
        
        for seq_id in range(num_seq):
            model_state['streamers'][seq_id].put(model_state['decoder_input_ids'][:, seq_id:seq_id+1].cpu())

        model_state = self._sample(model_state)
        return model_state

    def _sample(self, model_state, do_sample=True):
        decoder_input_ids = model_state["decoder_input_ids"]
        num_seq = model_state["num_seq"]

        next_token_logits = model_state["logits"].float().to(self.model.device)[:, -num_seq:]

        if do_sample:
            next_tokens = []
            for seq_id in range(num_seq):
                seq_idxs = self.get_decoder_input_idxs_for_seq(model_state, seq_id)
                seq_inp_ids = model_state["decoder_input_ids"][:, seq_idxs]
                seq_logits = next_token_logits[:, seq_id]
                seq_logits_processor = model_state["logits_processors"][seq_id]
                seq_scores = seq_logits_processor(seq_inp_ids, seq_logits)
                probs = nn.functional.softmax(seq_scores, dim=-1)
                next_tokens.append(torch.multinomial(probs, num_samples=1).squeeze(1))
                # next_tokens.append(seq_scores.argmax(dim=-1))
            next_tokens = torch.stack(next_tokens, dim=1)
        else:
            # TODO: wrong logic but useful for testing
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        # do padding for stopped sequences
        eos_token_id = 1024
        unfinished_codebooks = model_state["unfinished_codebooks"]
        next_tokens = next_tokens * unfinished_codebooks + eos_token_id * (1 - unfinished_codebooks)


        last_pos_ids = self._generate_decoder_position_ids(model_state["decoder_sequence_ids"])[-num_seq:]
        for seq_id in range(num_seq):
            prompt_length = torch.sum(
                (model_state["decoder_sequence_ids"] == seq_id) & 
                (model_state["decoder_token_categories"] == 1)
            )
            pos = last_pos_ids[seq_id] - prompt_length
            if pos < self._num_codebooks:
                next_tokens[pos+1:, seq_id] = 1025

        decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=1)
        
        model_state["decoder_input_ids"] = decoder_input_ids
        
        model_state["decoder_sequence_ids"] = torch.cat(
            [model_state["decoder_sequence_ids"], 
             torch.arange(0, num_seq, dtype=torch.long)]
        )
        model_state["decoder_token_categories"] = torch.cat(
            [model_state["decoder_token_categories"],
             torch.zeros(num_seq, dtype=torch.long)]
        )
        del model_state["logits"]

        for seq_id in range(num_seq):
            las_seq_idx = self.get_decoder_input_idxs_for_seq(model_state, seq_id)[-1]
            model_state['streamers'][seq_id].put(model_state['decoder_input_ids'][:, las_seq_idx].cpu())
                
        # update unfinished_codebooks
        codebooks_to_stop = self._codebooks_to_stop(model_state)
        model_state["unfinished_codebooks"] = model_state["unfinished_codebooks"] & ~codebooks_to_stop
        model_state["eos_flags"] = self._get_eos_flags(model_state["num_seq"], model_state["unfinished_codebooks"])
        return model_state
   
    def model_step(self, model_state):
        num_seq = model_state["num_seq"]
        # TODO: make this fast
        encoder_attn_mask_float, decoder_attn_mask_float = self.prepare_attention_masks(
            model_state['decoder_sequence_ids'], model_state['encoder_sequence_ids']
        )
        decoder_attn_mask_float = decoder_attn_mask_float[:, :, -num_seq:]
        encoder_attn_mask_float = encoder_attn_mask_float[:, :, -num_seq:]
        decoder_position_ids = self._generate_decoder_position_ids(model_state['decoder_sequence_ids'])
        decoder_position_ids = decoder_position_ids[-num_seq:].unsqueeze(0).to(self.model.device)

        # note that decoder_input_ids is just audio tokens
        is_audio_tok = model_state['decoder_token_categories'] == 0
        audio_idxs = torch.cumsum(is_audio_tok, 0) - 1
        decoder_input_idxs = audio_idxs[-num_seq:]
        decoder_input_ids = model_state['decoder_input_ids'][:, decoder_input_idxs]

        model_out = self.model(
            encoder_outputs=model_state['encoder_outputs'],
            attention_mask=encoder_attn_mask_float,
            decoder_input_ids=decoder_input_ids,
            decoder_position_ids=decoder_position_ids,
            decoder_attention_mask=decoder_attn_mask_float,
            past_key_values=model_state['past_key_values'],
            use_cache=True
        )

        model_state["past_key_values"] = model_out.past_key_values
        model_state["logits"] = model_out.logits
        model_state = self._sample(model_state)
        return model_state

    def _codebooks_to_stop(self, model_state):
        to_stop = []
        num_seq = model_state["num_seq"]
        for seq_id in range(num_seq):        
            seq_inp_ids = model_state["decoder_input_ids"][:, self.get_decoder_input_idxs_for_seq(model_state, seq_id)]
            to_stop_seq = self._stopping_criteria(seq_inp_ids, scores=None)
            to_stop.append(to_stop_seq)

        return torch.stack(to_stop, dim=1)

    def merge_model_states(self, model_state_0, model_state_1):
        
        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=torch.cat([
                model_state_0['encoder_outputs'].last_hidden_state,
                model_state_1['encoder_outputs'].last_hidden_state
            ], dim=1))

        kv_cache_0 = model_state_0['past_key_values']
        kv_cache_1 = model_state_1['past_key_values']

        kv_cache_merged = tuple(
            tuple(
                torch.cat([kv_cache_0[layer_idx][kv_idx], kv_cache_1[layer_idx][kv_idx]], dim=2)
                for kv_idx in range(len(kv_cache_0[0]))
            )
            for layer_idx in range(len(kv_cache_0))
        )

        num_seq_0 = model_state_0["num_seq"]
        num_seq_1 = model_state_1["num_seq"]

        dec_inp_ids_0 = model_state_0['decoder_input_ids']
        dec_inp_ids_1 = model_state_1['decoder_input_ids']

        dec_tok_cat_0 = model_state_0['decoder_token_categories']
        dec_tok_cat_1 = model_state_1['decoder_token_categories']

        dec_seq_ids_0 = model_state_0['decoder_sequence_ids']
        dec_seq_ids_1 = num_seq_0 + model_state_1['decoder_sequence_ids']
        
        decoder_input_ids = torch.cat([
            dec_inp_ids_0[:, :-num_seq_0],
            dec_inp_ids_1[:, :-num_seq_1],
            dec_inp_ids_0[:, -num_seq_0:],
            dec_inp_ids_1[:, -num_seq_1:]
        ], dim=1)
        
        decoder_token_categories = torch.cat([
            dec_tok_cat_0[:-num_seq_0],
            dec_tok_cat_1[:-num_seq_1],
            dec_tok_cat_0[-num_seq_0:],
            dec_tok_cat_1[-num_seq_1:]
        ])
                
        decoder_sequence_ids = torch.cat([
            dec_seq_ids_0[:-num_seq_0],
            dec_seq_ids_1[:-num_seq_1],
            dec_seq_ids_0[-num_seq_0:],
            dec_seq_ids_1[-num_seq_1:]
        ])
        
        encoder_sequence_ids = torch.cat([model_state_0['encoder_sequence_ids'], 
                                          num_seq_0 + model_state_1["encoder_sequence_ids"]])

        unfinished_codebooks = torch.cat(
            [model_state_0['unfinished_codebooks'],
             model_state_1['unfinished_codebooks']
            ], dim=1
            )
        logits_processors = model_state_0['logits_processors'] + model_state_1['logits_processors']
        streamers = model_state_0['streamers'] + model_state_1['streamers']
        num_seq = num_seq_0 + num_seq_1
        model_state = {
            "encoder_outputs": encoder_outputs,
            "past_key_values": kv_cache_merged,
            "decoder_input_ids": decoder_input_ids,
            "decoder_sequence_ids": decoder_sequence_ids,
            "decoder_token_categories": decoder_token_categories,
            "encoder_sequence_ids": encoder_sequence_ids,
            "logits_processors": logits_processors,
            "unfinished_codebooks": unfinished_codebooks,
            "num_seq": num_seq,
            "streamers": streamers,
            "eos_flags": self._get_eos_flags(num_seq, unfinished_codebooks)
        }
        return model_state

    def get_decoder_input_idxs_for_seq(self, model_state, seq_id):
        is_audio_tok = model_state['decoder_token_categories'] == 0
        audio_idxs = torch.cumsum(is_audio_tok, 0) - 1
        return audio_idxs[(model_state["decoder_sequence_ids"] == seq_id) & is_audio_tok]

    def evict_seq(self, model_state, seq_id):
        num_seq = model_state["num_seq"]
        assert seq_id < num_seq
        new_seq_ids_map = torch.tensor(list(range(0, seq_id)) + [-1] + list(range(seq_id, num_seq)))
        enc_seq_ids = model_state["encoder_sequence_ids"]
        dec_seq_ids = model_state["decoder_sequence_ids"]
        
        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=model_state['encoder_outputs'].last_hidden_state[:, enc_seq_ids != seq_id]
        )
        
        kv_cache = model_state['past_key_values']
        past_key_values = tuple(
            (   
                kv_cache[layer_idx][0][:, :, dec_seq_ids[:-num_seq] != seq_id],
                kv_cache[layer_idx][1][:, :, dec_seq_ids[:-num_seq] != seq_id],
                kv_cache[layer_idx][2][:, :, enc_seq_ids != seq_id],
                kv_cache[layer_idx][3][:, :, enc_seq_ids != seq_id]
            ) for layer_idx in range(len(kv_cache))
        )
        
        n_inp = model_state['decoder_input_ids'].shape[1]
        decoder_input_ids = model_state['decoder_input_ids'][:, dec_seq_ids[-n_inp:] != seq_id]
        decoder_token_categories = model_state['decoder_token_categories'][dec_seq_ids != seq_id]

        decoder_sequence_ids = new_seq_ids_map[model_state['decoder_sequence_ids'][dec_seq_ids != seq_id]]
        encoder_sequence_ids = new_seq_ids_map[model_state['encoder_sequence_ids'][enc_seq_ids != seq_id]]
        model_state["streamers"][seq_id].end()
        model_state["logits_processors"].pop(seq_id)
        stramer = model_state["streamers"].pop(seq_id)
        self.evicted_streamers.append(stramer)
        unfinished_codebooks = torch.cat((model_state['unfinished_codebooks'][:, :seq_id], model_state['unfinished_codebooks'][:, seq_id+1:]), dim=1)

        num_seq = num_seq - 1
        model_state = {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_sequence_ids": decoder_sequence_ids,
            "decoder_token_categories": decoder_token_categories,
            "encoder_sequence_ids": encoder_sequence_ids,
            "logits_processors": model_state["logits_processors"],
            "streamers": model_state["streamers"],
            "unfinished_codebooks": unfinished_codebooks,
            "num_seq": num_seq,
            "eos_flags": self._get_eos_flags(num_seq, unfinished_codebooks)
        }
        return model_state

    def _get_eos_flags(self, num_seq, unfinished_codebooks):
        get_eos_flag = {}
        for seq_id in range(num_seq):
            get_eos_flag[seq_id] = bool(torch.eq(unfinished_codebooks[:, seq_id].max(), 0))
        return(get_eos_flag)

    def indicate_stream_ended(self, model_state):
        for seq_id in range(model_state['num_seq']):
            if model_state['eos_flags'][seq_id]:
                model_state['streamers'][seq_id].end()
                return model_state
        return model_state

    def is_all_sequences_ended(self, model_state):
        return all(model_state['eos_flags'].values())

    def clear_all_model_states_at_once(self, model_state):
        while model_state['num_seq'] > 0:
            model_state = self.evict_seq(model_state, 0)
        return model_state

    def evict_sequences_ended(self, model_state):
        for seq_id in range(model_state['num_seq']):
            if model_state['eos_flags'][seq_id]:
                #TODO: NEED TO UPDATE EVICTION CODE 
                #ERROR WITH EVICTION CODE AFTER MERGING
                model_state = self.evict_seq(model_state, seq_id)
                return model_state
        return model_state

    def post_model_step_updates(self, model_state):
        model_state = self.indicate_stream_ended(model_state)
        is_all_sequences_ended = self.is_all_sequences_ended(model_state)
        if is_all_sequences_ended:
            model_state = self.clear_all_model_states_at_once(model_state)
        return is_all_sequences_ended, model_state
    

    def print_state_shapes(self, model_state):
        for k, v in model_state.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            elif k == 'encoder_outputs':
                print("encoder_outputs.last_hidden_state.shape", v.last_hidden_state.shape)
            elif k == 'past_key_values':
                print("past_key_values.self_attention_cache layer 0 shape", v[0][0].shape)
                print("past_key_values.cross_attention_cache layer 0 shape", v[0][2].shape)
            else:
                print(k, v)
        print('---')