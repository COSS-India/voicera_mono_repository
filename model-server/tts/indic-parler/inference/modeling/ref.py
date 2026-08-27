import torch
import transformers


class QwenModelRunner:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        ).eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

    def model_prefill(self, messages_batch):
        """Returns model state for a prefill/encoding stage"""
        # tokenize text
        texts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_batch
        ]
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        # run model
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )

        # Convert past_key_values (kv cache) into tensor for easy merging etc
        # For now we're using transformers 4.46.3, so we're getting legacy
        # cache tuples etc.
        past_key_values = model_outputs.past_key_values
        past_key_values = torch.stack([torch.stack(x) for x in past_key_values])

        model_state = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "logits": model_outputs.logits,
        }
        model_state = self._sample(model_state)
        return model_state

    def _sample(self, model_state):
        next_token_logits = model_state["logits"][:, -1, :]

        # do greedy decoding for now
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # grow attenton_mask
        attention_mask = model_state["attention_mask"]
        batch_size = attention_mask.shape[0]
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )

        return {
            "input_ids": next_tokens,
            "attention_mask": attention_mask,
            "past_key_values": model_state["past_key_values"],
            "logits": model_state["logits"],
        }

    def stopping_criteria(self, model_state):
        """Returns 0/1 for each of the batch if it has to be stopped"""
        return model_state["input_ids"][:, -1] == self.tokenizer.eos_token_id

    def model_step(self, model_state):
        model_outputs = self.model(
            input_ids=model_state["input_ids"],
            attention_mask=model_state["attention_mask"],
            past_key_values=model_state["past_key_values"],
            use_cache=True,
            return_dict=True,
        )

        # Convert past_key_values (kv cache) into tensor for easy merging etc
        # For now we're using transformers 4.46.3, so we're getting legacy
        # cache tuples etc.
        past_key_values = model_outputs.past_key_values
        past_key_values = torch.stack([torch.stack(x) for x in past_key_values])

        model_state = {
            "input_ids": model_state["input_ids"],
            "attention_mask": model_state["attention_mask"],
            "past_key_values": past_key_values,
            "logits": model_outputs.logits,
        }
        model_state = self._sample(model_state)
        return model_state

    def print_state(self, model_state):
        for k, v in model_state.items():
            print(k, v.shape)


class FlanT5ModelRunner:
    def __init__(self):
        self.model_name = "google/flan-t5-base"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, torch_dtype="auto"
        ).eval()

    def _kv_to_tensors(self, kv):
        # Convert past_key_values (kv cache) into tensor for easy merging etc
        # For now we're using transformers 4.46.3, so we're getting legacy
        # cache tuples etc.
        kv_self = [[y for y in layer[:2]] for layer in kv]
        kv_cross = [[y for y in layer[2:]] for layer in kv]
        kv_self = torch.stack([torch.stack(x) for x in kv_self])
        kv_cross = torch.stack([torch.stack(x) for x in kv_cross])
        return kv_self, kv_cross

    def _kv_to_tuples(self, kv_self, kv_cross):
        return tuple(
            (
                kv_self[layer][0],
                kv_self[layer][1],
                kv_cross[layer][0],
                kv_cross[layer][1],
            )
            for layer in range(kv_self.shape[0])
        )

    def model_prefill(self, messages_batch):
        """Returns model state for a prefill/encoding stage"""
        # tokenize text
        inputs = self.tokenizer(messages_batch, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        # encoder
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        batch_size = len(messages_batch)
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.model.device,
        )

        # decoder
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        last_hidden_state = decoder_outputs.last_hidden_state[:, -1, :]
        logits = self.model.lm_head(last_hidden_state)

        past_key_values = decoder_outputs.past_key_values
        past_key_values = self._kv_to_tensors(past_key_values)
        model_state = {
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "logits": logits,
        }
        model_state = self._sample(model_state)
        return model_state

    def _sample(self, model_state):
        next_tokens = torch.argmax(model_state["logits"], dim=-1, keepdim=True)
        model_state["input_ids"] = next_tokens
        return model_state

    def stopping_criteria(self, model_state):
        """Returns 0/1 for each of the batch if it has to be stopped"""
        return model_state["input_ids"][:, -1] == self.tokenizer.eos_token_id

    def model_step(self, model_state):
        past_key_values = self._kv_to_tuples(*model_state["past_key_values"])
        decoder_outputs = self.model.decoder(
            input_ids=model_state["input_ids"],
            attention_mask=None,
            encoder_hidden_states=model_state["encoder_hidden_states"],
            encoder_attention_mask=model_state["encoder_attention_mask"],
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        last_hidden_state = decoder_outputs.last_hidden_state[:, -1, :]
        logits = self.model.lm_head(last_hidden_state)

        past_key_values = self._kv_to_tensors(decoder_outputs.past_key_values)
        model_state = {
            "input_ids": model_state["input_ids"],
            "encoder_hidden_states": model_state["encoder_hidden_states"],
            "encoder_attention_mask": model_state["encoder_attention_mask"],
            "past_key_values": past_key_values,
            "logits": logits,
        }
        model_state = self._sample(model_state)
        return model_state

    def print_state(self, model_state):
        for k, v in model_state.items():
            if isinstance(v, tuple):
                print(k, v[0].shape, v[1].shape)
            else:
                print(k, v.shape)


with torch.inference_mode():
    runner = QwenModelRunner()
    messages_batch = [
        # Prompt 1
        [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud.",
            },
            {"role": "user", "content": "What is Grok in one short sentence?"},
        ],
        # Prompt 2
        [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud.",
            },
            {"role": "user", "content": "Give me a haiku about the moon."},
        ],
        [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud.",
            },
            {"role": "user", "content": "Give me a haiku about the moon."},
        ],
    ]
    active = torch.ones(
        len(messages_batch), dtype=torch.bool, device=runner.model.device
    )

    state = runner.model_prefill(messages_batch)
    print([runner.tokenizer.decode(x) for x in state["input_ids"]])
    active = active & ~runner.stopping_criteria(state)
    while active.any():
        state = runner.model_step(state)
        print([runner.tokenizer.decode(x) for x in state["input_ids"]])
        active = active & ~runner.stopping_criteria(state)

    runner.print_state(state)


with torch.inference_mode():
    runner = FlanT5ModelRunner()
    messages_batch = [
        "summarize: Artificial intelligence is transforming industries by automating tasks, improving decision-making, and enabling new capabilities in healthcare, finance, and transportation. However, it also raises ethical concerns about privacy, bias, and job displacement.",
        "translate English to French: The quick brown fox jumps over the lazy dog.",
        "answer the question: What is the capital of Japan?",
    ]

    active = torch.ones(
        len(messages_batch), dtype=torch.bool, device=runner.model.device
    )

    state = runner.model_prefill(messages_batch)
    print([runner.tokenizer.decode(x) for x in state["input_ids"]])
    active = active & ~runner.stopping_criteria(state)

    while active.any():
        state = runner.model_step(state)
        print([runner.tokenizer.decode(x) for x in state["input_ids"]])
        active = active & ~runner.stopping_criteria(state)

    runner.print_state(state)
