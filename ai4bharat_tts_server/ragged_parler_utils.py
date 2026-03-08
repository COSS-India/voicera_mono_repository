import time
import copy
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.modeling_parler_tts import ParlerTTSLogitsProcessor
    
import torch
import torch.nn as nn
from transformers.cache_utils import (
    DynamicCache,
    EncoderDecoderCache,
)
from transformers import AutoTokenizer
from transformers.generation.configuration_utils import GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList



@torch.no_grad()
def _my_encoder_step(self,
    inputs,
    generation_config,
    logits_processor,
    stopping_criteria,
    streamer=None,
    synced_gpus=None,
    **kwargs):



        start = time.time()

        # 1. Handle `generation_config` and kwargs that might update it, and validate the resulting objects
        if generation_config is None:
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())


        if model_kwargs.get("encoder_outputs") is not None and type(model_kwargs["encoder_outputs"]) == tuple:
            # wrap the unconditional outputs as a BaseModelOutput for compatibility with the rest of generate
            model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=model_kwargs["encoder_outputs"][0])

        step1 = time.time() - start

        # 2. Set generation parameters if not already defined
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        step2 = time.time() - start

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=inputs_tensor.device)

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList([ParlerTTSLogitsProcessor(generation_config.eos_token_id, self.decoder.num_codebooks, batch_size, inputs_tensor.device)])
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        step3 = time.time() - start
        # print("step3: ",step3 * 1000, "ms")

        # 4. Define other model kwargs
        model_kwargs["use_cache"] = generation_config.use_cache

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )

        if "encoder_outputs" not in model_kwargs:
            # encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_text_encoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        if "prompt_hidden_states" not in model_kwargs and "prompt_input_ids" in model_kwargs:
            # `prompt_hidden_states` are created and added to `model_kwargs`
            model_kwargs = self._prepare_prompt_kwargs_for_generation(
                model_kwargs["prompt_input_ids"],
                model_kwargs,
            )

        if "decoder_input_ids" not in model_kwargs and "input_values" in model_kwargs:
            model_kwargs = self._prepare_audio_encoder_kwargs_for_generation(
                model_kwargs["input_values"],
                model_kwargs,
            )

        step4 = time.time() - start
        # print("step4: ",step4 * 1000, "ms")

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            bos_token_id=generation_config._bos_token_tensor,
            device=inputs_tensor.device,
        )

        step5 = time.time() - start
        # print("step5: ",step5 * 1000, "ms")

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        if generation_config.cache_implementation is not None and model_kwargs.get("past_key_values") is not None:
            raise ValueError(
                "Passing both `cache_implementation` (used to initialize certain caches) and `past_key_values` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                if not self.prompt_cross_attention:
                    # when we prepend prompt_hidden_state to inputs_embeds, max_cache_len needs to be actualised
                    # generation_config.max_length has already been increased by input_ids_length which is
                    # already counted in input_embeds_seq_length so we remove it
                    input_embeds_seq_length = model_kwargs["inputs_embeds"].shape[1]
                    max_cache_len = generation_config.max_length + input_embeds_seq_length - input_ids_length
                else:
                    max_cache_len = self.generation_config.max_length
                model_kwargs["past_key_values"] = self._get_cache(
                    generation_config.cache_implementation,
                    getattr(generation_config, "num_beams", 1) * batch_size,
                    max_cache_len,
                    model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                raise ValueError(
                    "This model does not support the quantized cache. If you want your model to support quantized "
                    "cache, please open an issue on the Parler-TTS repository https://github.com/huggingface/parler-tts"
                )
        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
            past = model_kwargs.get("past_key_values", None)
            requires_cross_attention_cache = (
                self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
            )
            if past is None:
                model_kwargs["past_key_values"] = (
                    DynamicCache()
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache(DynamicCache(), DynamicCache())
                )
            elif isinstance(past, tuple):
                model_kwargs["past_key_values"] = (
                    DynamicCache.from_legacy_cache(past)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(past)
                )

        # build the delay pattern mask for offsetting each codebook prediction by 1 (this behaviour is specific to Parler-TTS)
        delayed_input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids,
            bos_token_id=generation_config._bos_token_tensor,
            pad_token_id=generation_config._pad_token_tensor,
            max_length=generation_config.max_length,
        )
        # stash the delay mask so that we don't have to recompute in each forward pass
        model_kwargs["decoder_delay_pattern_mask"] = decoder_delay_pattern_mask

        # input_ids are ready to be placed on the streamer (if used)
        if streamer is not None:
            streamer.put(delayed_input_ids.cpu())

        step6 = time.time() - start
        # print("step6: ",step6 * 1000, "ms")

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode()

        step7 = time.time() - start
        # print("step7: ",step7 * 1000, "ms")

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=delayed_input_ids.device,
        )

        step8 = time.time() - start
        # print("step8: ",step8 * 1000, "ms")

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        step9 = time.time() - start
        # print("step9: ",step9 * 1000, "ms")

        if generation_mode not in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            raise ValueError(
                "Got incompatible mode for generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )

        # expand input_ids with `num_return_sequences` additional sequences per batch
        delayed_input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=delayed_input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        expanded_batch_size, cur_len = delayed_input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(expanded_batch_size, dtype=torch.long, device=delayed_input_ids.device)
        model_kwargs = self._get_initial_cache_position(delayed_input_ids, model_kwargs)

        encoder_output =  {
            "start": start,
            "delayed_input_ids": delayed_input_ids,
            "input_ids": input_ids,
            "model_kwargs": model_kwargs,
            "logits_processor": logits_processor,
            "stopping_criteria": stopping_criteria,
            "generation_config": generation_config,
            "generation_mode": generation_mode,
            "streamer": streamer,
            "pad_token_id": pad_token_id,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "output_scores":output_scores,
            "output_logits":output_logits,
            "return_dict_in_generate":return_dict_in_generate,
            "max_length":max_length,
            "has_eos_stopping_criteria":has_eos_stopping_criteria,
            "do_sample":do_sample,
            "scores": scores,
            "raw_logits":raw_logits,
            "decoder_attentions": decoder_attentions,
            "cross_attentions": cross_attentions,
            "decoder_hidden_states": decoder_hidden_states,
            "expanded_batch_size": expanded_batch_size,
            "batch_size": batch_size,
            "cur_len": cur_len,
            "this_peer_finished": this_peer_finished,
            "unfinished_sequences": unfinished_sequences,
        }

        return  encoder_output


def my_encoder(prompts, descriptions, model, tokenizer, description_tokenizer, device):
    description_input_ids = description_tokenizer(descriptions, return_tensors="pt", padding=True).to(device)
    prompt_input_ids = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    kwargs = dict(
        input_ids=description_input_ids.input_ids,
        attention_mask=description_input_ids.attention_mask,
        prompt_input_ids=prompt_input_ids.input_ids,
        prompt_attention_mask=prompt_input_ids.attention_mask,
    )
    inputs = None
    generation_config = None
    logits_processor = None
    stopping_criteria = None
    synced_gpus = None
    streamer = None
    y_encoder = _my_encoder_step(
        self=model,
        inputs=inputs,
        generation_config=generation_config,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        streamer=streamer,
        synced_gpus=synced_gpus,
        **kwargs,
    )
    return y_encoder


@torch.no_grad()
def my_decoder(self, y_output, synced_gpus):

    start = y_output['start']
    output_ids = y_output['delayed_input_ids']
    input_ids =  y_output['input_ids']
    model_kwargs = y_output['model_kwargs']
    generation_config =  y_output['generation_config']
    batch_size = y_output["batch_size"]


    output_ids = self.decoder.apply_delay_pattern_mask(output_ids, model_kwargs["decoder_delay_pattern_mask"])

    # Revert the pattern delay mask by filtering the eos and bos token ids from the delay pattern mask
    _, mask = self.decoder.build_delay_pattern_mask(
        input_ids,
        bos_token_id=generation_config.bos_token_id,
        pad_token_id=generation_config.pad_token_id,
        max_length=output_ids.shape[1],
    )


    mask = (mask != generation_config.bos_token_id) & (mask != generation_config.pad_token_id)
    output_ids = output_ids[mask].reshape(batch_size, self.decoder.num_codebooks, -1)

    # append the frame dimension back to the audio codes
    output_ids = output_ids[None, ...]

    audio_decode_kwargs = {}
    if self.use_audio_scales:
        audio_scales = model_kwargs.get("audio_scales")
        if audio_scales is None:
            audio_scales = [None] * batch_size
        audio_decode_kwargs["audio_scales"] = audio_scales


    if not self.use_4dim_audio_codes:
        # remove chunk dim
        output_ids = output_ids.squeeze(0)

    decode_sequentially = (
        generation_config.bos_token_id in output_ids
        or generation_config.pad_token_id in output_ids
        or generation_config.eos_token_id in output_ids
    )

    if not decode_sequentially:
        output_values = self.audio_encoder.decode(
            audio_codes=output_ids,
            **audio_decode_kwargs,
        ).audio_values.squeeze(1)
        output_lengths = [audio.shape[0] for audio in output_values]
    else:
        output_values = []
        for sample_id in range(batch_size):
            sample = output_ids[:, sample_id] if self.use_4dim_audio_codes else output_ids[sample_id]
            sample_mask = (sample >= self.audio_encoder.config.codebook_size)
            sample_mask = (sample_mask.sum(dim=(0, 1)) == 0) if self.use_4dim_audio_codes else (sample_mask.sum(dim=0) == 0)
            single_audio_decode_kwargs = {}
            if self.use_audio_scales:
                single_audio_decode_kwargs["audio_scales"] = [audio_decode_kwargs["audio_scales"][sample_id]]
            if sample_mask.sum() > 0:
                sample = sample[:, :, sample_mask] if self.use_4dim_audio_codes else sample[:, sample_mask]
                sample = self.audio_encoder.decode(audio_codes=sample[None, ...], **single_audio_decode_kwargs).audio_values
                sample = sample if sample.ndim == 3 else sample.unsqueeze(0)
                output_values.append(sample.transpose(0, 2))
            else:
                # output_values.append(torch.zeros((1, 1, 1)).to(self.device))
                output_values.append(torch.zeros((1, 1, 1)).to(self.device))
        output_lengths = [audio.shape[0] for audio in output_values]
        output_values = (
            torch.nn.utils.rnn.pad_sequence(output_values, batch_first=True, padding_value=0)
            .squeeze(-1)
            .squeeze(-1)
        )
    if generation_config.return_dict_in_generate:
        outputs["audios_length"] = output_lengths
        outputs.sequences = output_values
        end = time.time() - start
        print("End: ", end * 1000, "ms")

        return outputs
    else:
        end = time.time() - start
        print("End: ", end * 1000, "ms")

        return output_values


@torch.no_grad()
def model_step(self, y_encoder, synced_gpus):

        delayed_input_ids = y_encoder["delayed_input_ids"]
        cur_len = y_encoder["cur_len"]
        this_peer_finished = y_encoder["this_peer_finished"]
        unfinished_sequences = y_encoder["unfinished_sequences"]
        model_kwargs = y_encoder["model_kwargs"]

        logits_processor = y_encoder["logits_processor"]
        stopping_criteria = y_encoder["stopping_criteria"]
        streamer = y_encoder["streamer"]
        pad_token_id = y_encoder["pad_token_id"]
        do_sample = y_encoder["do_sample"]
        has_eos_stopping_criteria = y_encoder["has_eos_stopping_criteria"]
        output_attentions =y_encoder["output_attentions"]
        output_hidden_states = y_encoder["output_hidden_states"]
        scores = y_encoder["scores"]

        model_inputs = self.prepare_inputs_for_generation(delayed_input_ids, **model_kwargs) 
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        outputs = self(**model_inputs, return_dict=True)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        if synced_gpus and this_peer_finished:
            # continue
            y_encoder["cur_len"] += 1
            return y_encoder

        next_token_logits = outputs.logits.clone()[:, -1, :].float()
        next_token_logits = next_token_logits.to(delayed_input_ids.device)
        next_token_scores = logits_processor(delayed_input_ids, next_token_logits)

        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)


        delayed_input_ids = torch.cat([delayed_input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(delayed_input_ids, scores)

        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        del outputs

        y_encoder["delayed_input_ids"] = delayed_input_ids
        y_encoder["model_kwargs"] = model_kwargs
        y_encoder["unfinished_sequences"] = unfinished_sequences
        y_encoder["this_peer_finished"] = this_peer_finished
        y_encoder["cur_len"] = cur_len

        y_encoder["next_tokens"] = next_tokens

        return y_encoder


def stopping_check(self, y_output, synced_gpus):
    return self._has_unfinished_sequences(
        y_output["this_peer_finished"],
        synced_gpus,
        device=y_output["delayed_input_ids"].device,
        cur_len=y_output["cur_len"],
        max_length=y_output["max_length"],
    )

@torch.no_grad()
def my_generate(
        self:ParlerTTSForConditionalGeneration,
        prompts,
        descriptions,
        model,
        tokenizer,
        description_tokenizer,
        device,
        synced_gpus=None
):

        y_encoder = my_encoder(prompts, descriptions, model, tokenizer, description_tokenizer, device)
        y_output = y_encoder

        step_no = 0
        while stopping_check(self, y_output, synced_gpus):
            n = time.time()
            y_output = model_step(self, y_output, synced_gpus)
            print(f"step_no {step_no}", round((time.time() - n)*1000))
            step_no += 1

        step10 = time.time() - y_encoder["start"]
        print("step10: ",step10 * 1000, "ms")

        decoder_output = my_decoder(self, y_output, synced_gpus)

        return decoder_output


