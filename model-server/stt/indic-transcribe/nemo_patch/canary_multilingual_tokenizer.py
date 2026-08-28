# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
from functools import cached_property
from pathlib import Path
from typing import Dict, List

from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
# ASR-PATCH: upstream's canary2 prompt formatter does
#   `if isinstance(prompt.tokenizer, CanaryTokenizer): eos = tokenizer.eos`
#   `else:  # SPE -> tokenizer.token_to_id(CANARY_EOS)`  <- 1-arg call
# An AggregateTokenizer needs token_to_id(token, lang_id), so the else-branch
# raises TypeError. Upstream CanaryTokenizer is a near-identical sibling of this
# class (same __init__/special_tokens/eos_id), so inheriting from it makes the
# isinstance check pass and routes to `.eos` -> `.eos_id` -> special_tokens[EOS],
# which this class already defines identically. Nothing else changes.
from nemo.collections.common.tokenizers.canary_tokenizer import CanaryTokenizer as _UpstreamCanaryTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model

from nemo.utils import logging

__all__ = ['CanaryTokenizer']

# Default tokens for compatibility with Canary.
CANARY_BOS = "<|startoftranscript|>"
CANARY_EOS = "<|endoftext|>"
CANARY_PAD = "<pad>"
CANARY_NOSPEECH = "<|nospeech|>"
CANARY_PNC = "<|pnc|>"
CANARY_NOPNC = "<|nopnc|>"
CANARY2_BOCTX = "<|startofcontext|>"
DEFAULT_TOKENS = [CANARY_NOSPEECH, CANARY_PAD, CANARY_EOS, CANARY_BOS, CANARY_PNC, CANARY_NOPNC]

CANARY_SPECIAL_TOKENIZER = "spl_tokens"


# ASR-PATCH: `super().text_to_ids(...)` -> `AggregateTokenizer.text_to_ids(self, ...)`
# Re-basing this class onto CanaryTokenizer (above) made `super()` resolve to
# CanaryTokenizer.text_to_ids, which calls _text_to_ids_maybe_with_timestamps -- overridden
# here -- i.e. infinite recursion (RecursionError). The fork's `super()` meant
# AggregateTokenizer, so we name it explicitly and preserve the original semantics.
class CanaryMultilingualTokenizer(_UpstreamCanaryTokenizer):  # ASR-PATCH: was AggregateTokenizer
    """
    Thin wrapper around AggregateTokenizer to provide quick access to special tokens
    """

    def __init__(self, tokenizers: Dict):
        super().__init__(tokenizers)

        # for easy access of special tokens
        self.special_tokens = {}
        for special in tokenizers[CANARY_SPECIAL_TOKENIZER].vocab:
            # Search for special prompting tokens
            if (special.startswith("<|") and special.endswith("|>")) or special == CANARY_PAD:
                self.special_tokens[special] = self.token_to_id(special, lang_id=CANARY_SPECIAL_TOKENIZER)

    @cached_property
    def eos_id(self) -> int:
        return self.special_tokens[CANARY_EOS]

    @cached_property
    def bos_id(self) -> int:
        return self.special_tokens[CANARY_BOS]

    @cached_property
    def nospeech_id(self) -> int:
        return self.special_tokens[CANARY_NOSPEECH]

    @cached_property
    def pad_id(self) -> int:
        return self.special_tokens[CANARY_PAD]
    
    def _text_with_special_tokens_to_ids(self, text: str, lang_id: str) -> list[int]:
        """
        Tokenize a text containing special tokens like <|music|> preserving order.

        Special tokens are tokenized by _tokenize_special_prompt,
        normal text by text_to_ids.
        """
        # Pattern to find special tokens like <|music|>
        special_token_pattern = re.compile(r"<\|.+?\|>")
        
        # Split text keeping special tokens
        parts = special_token_pattern.split(text)
        specials = special_token_pattern.findall(text)

        result_ids = []

        # Interleave tokenizing normal text and special tokens preserving order
        for i, part in enumerate(parts):
            part = part.lstrip()
            if part:
                # tokenize normal text part using base tokenizer
                result_ids.extend(AggregateTokenizer.text_to_ids(self, part, lang_id))
            if i < len(specials):
                # tokenize special token using _tokenize_special_prompt
                try:
                    ids = self._tokenize_special_prompt(specials[i])
                    result_ids.extend(ids)
                except KeyError:
                    print(f"Token {specials[i]} could not be encoded")


        return result_ids

    def _text_with_timestamps_to_ids(self, text_without_timestamps, time_text, lang_id) -> list[int]:
        trans_words = text_without_timestamps.split()

        # Get timestamp ids
        time_ids = self._tokenize_special_prompt(time_text)

        # Tokenize text word by wordd
        word_ids = []
        result_ids = []
        time_index = 0

        timestamp_every_n_words = 1  # Add timestmap for every N words
        word_index = 0
        # Both start and end time
        for word in trans_words:
            # Insert the first time_id once
            if word_index == 0 and time_index < len(time_ids):
                result_ids.append(time_ids[time_index])
                time_index += 1
            # Tokenize the word
            word_ids += AggregateTokenizer.text_to_ids(self, word, lang_id)
            result_ids += AggregateTokenizer.text_to_ids(self, word, lang_id)
            word_index += 1
            # Insert time ids every N words after the first one
            if word_index % timestamp_every_n_words == 0 and word_index != 0 and time_index < len(time_ids):
                result_ids.append(time_ids[time_index])
                time_index += 1
                if time_index < len(time_ids):
                    result_ids.append(time_ids[time_index])
                    time_index += 1
            else:
                time_index += 2
        # Ensure the last time_id is appended at the end
        if time_index < len(time_ids):
            result_ids.append(time_ids[-1])
        # Make sure the last time_id is appended only once
        if time_index < len(time_ids) and result_ids[-1] != (time_ids[-1]):
            result_ids.append(time_ids[-1])
        return result_ids

    def _text_to_ids_maybe_with_timestamps(self, text_no_eos, lang_id) -> list[int]:
        # FIX: Generalize beyod timestamps
        # time_pattern = re.compile(r"<\|\d+\|>")
        # time_text = "".join(time_pattern.findall(text_no_eos))
        # has_timestamp = bool(time_text)
        ####### Common code ######
        if lang_id in ['as', 'bn', 'brx', 'doi', 'en', 'gu', 'hi', 'kn', 'ks', 'kok', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur']:
           lang_id = 'multilingual'
        ##########################
        ### FIX: For verbatim tags
        return self._text_with_special_tokens_to_ids(text_no_eos, lang_id)
        #########################
        # if not has_timestamp:
        #     return AggregateTokenizer.text_to_ids(self, text_no_eos, lang_id)
        # else:
        #     text_without_timestamps = time_pattern.sub("", text_no_eos).strip()
        #     return self._text_with_timestamps_to_ids(text_without_timestamps, time_text, lang_id)

    def text_to_ids(self, text, lang_id) -> list[int]:
        if lang_id == CANARY_SPECIAL_TOKENIZER:
            return self._tokenize_special_prompt(text)
        lang_id = _map_canary1_to_canary2_lang(lang_id, self.langs)
        if text.endswith(CANARY_EOS):
            return self._text_to_ids_maybe_with_timestamps(text[: -len(CANARY_EOS)], lang_id) + [self.eos_id]
        return self._text_to_ids_maybe_with_timestamps(text, lang_id)

    def _tokenize_special_prompt(self, text: str) -> list[int]:
        """
        Tokenize the input special prompt of Canary family of models.

        Required because otherwise self.text_to_ids() returns a different result than what Canary had been trained with.
        """
        ans = []

        if text.startswith(CANARY2_BOCTX):
            # Canary 2 prompt format. It starts with decoder context, which should be tokenized using
            # a different tokenizer than spl_tokens. We don't really know what it is, so we'll use the
            # following HACK solution: look up 5th token which is target_lang and tokenize this part
            # using its tokenizer. We skip this when decoder context is empty.
            ans.append(self.special_tokens[CANARY2_BOCTX])
            text = text[len(CANARY2_BOCTX) :]
            ctx_end_idx = text.find(CANARY_BOS)
            if decoder_ctx := text[:ctx_end_idx]:
                target_lang = text.split("<|")[4].replace("|>", "")  # sorry
                ans.extend(self.text_to_ids(decoder_ctx, target_lang))
                text = text[ctx_end_idx:]

        num_special_tokens = text.count(">")
        for _ in range(num_special_tokens):
            token = text[: text.find(">") + 1]
            ans.append(self.special_tokens[token])
            text = text[len(token) :]
        assert len(text) == 0, text
        return ans

    def spl_token_to_id(self, token):
        if token_id := self.special_tokens.get(f"<|{token}|>", None):
            return token_id
        raise KeyError(f"Token {token} not found in tokenizer.")

    @staticmethod
    def build_special_tokenizer(
        tokens: List[str], model_dir: str | Path, force_rebuild: bool = False
    ) -> SentencePieceTokenizer:
        if force_rebuild:
            logging.info("Building special tokenizer")
            # Checks for artifacts of previous build.
            for file in ["tokenizer.model", "tokenizer.vocab", "vocab.txt", "train_text.txt"]:
                if os.path.exists(file):
                    os.remove(file)
        spl_tok_re = re.compile(r"<\|.+\|>")
        tokens = DEFAULT_TOKENS + [f"<|{t}|>" if spl_tok_re.match(t) is None else t for t in tokens]
        tokens = list(dict.fromkeys(tokens))  # remove duplicates while preserving order
        output_dir = Path(model_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        text_path = output_dir / "train_text.txt"
        train_text = "\n".join(tokens)
        text_path.write_text(train_text)
        model_path = output_dir / "tokenizer.model"
        create_spt_model(
            str(text_path),
            vocab_size=len(tokens) + 2,
            sample_size=-1,
            do_lower_case=False,
            output_dir=str(output_dir),
            user_defined_symbols=tokens,
        )
        spl_tokenizer = SentencePieceTokenizer(str(model_path))
        return spl_tokenizer


class CanaryBPETokenizer(SentencePieceTokenizer):
    """
    Thin wrapper around SPE tokenizer that overwrites SPE's BOS/EOS/PAD with Canary's special tokens
    for compatibility with CanaryTokenizer (aggregate).
    """

    @cached_property
    def eos_id(self) -> int:
        return self.token_to_id(CANARY_EOS)

    @cached_property
    def bos_id(self) -> int:
        return self.token_to_id(CANARY_BOS)

    @cached_property
    def nospeech_id(self) -> int:
        return self.token_to_id(CANARY_NOSPEECH)

    @cached_property
    def pad_id(self) -> int:
        return self.token_to_id(CANARY_PAD)


def _map_canary1_to_canary2_lang(lang: str, available_langs: list[str]) -> str:
    if len(lang) != 2 or lang in available_langs:
        return lang

    if (
        mapped := {"en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE"}.get(lang)
    ) is not None and mapped in available_langs:
        return mapped

    if lang in ['as', 'bn', 'brx', 'doi', 'en', 'gu', 'hi', 'kn', 'ks', 'kok', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur']:
        return 'multilingual'
    raise RuntimeError(f"Unsupported language: '{lang}' for CanaryTokenizer with languages: {available_langs}")
