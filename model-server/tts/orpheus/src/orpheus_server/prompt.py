"""Prompt construction, as explicit token ids.

Two reasons this emits token *ids* rather than a string:

  * The tokenizer auto-prepends BOS. Building the prompt as text and letting it
    tokenise gives you two BOS tokens and noticeably worse audio.
  * The Indic template's speaker/style markers are asymmetric special tokens
    (``<|speaker>`` opens, ``<speaker|>`` closes). Round-tripping them through
    text is fragile; addressing them by id is not.

Control-token ids are identical between upstream English Orpheus and the
AI4Bharat Indic fine-tune, so both templates live here.
"""
from __future__ import annotations

from typing import Optional

# Turn structure
TOK_SOH = 128259            # start of human turn
TOK_BOS = 128000            # begin_of_text
TOK_EOT = 128009            # end of turn
TOK_EOH = 128260            # end of human turn
TOK_SOA = 128261            # start of AI turn
TOK_SOS = 128257            # start of speech
TOK_EOS = 128258            # end of speech -- the real stop token (NOT eos 128001)

# Indic template markers (asymmetric; verified against the checkpoint tokenizer)
TOK_SPEAKER_OPEN = 156938   # <|speaker>
TOK_SPEAKER_CLOSE = 156939  # <speaker|>
TOK_STYLE_OPEN = 156940     # <|style>
TOK_STYLE_CLOSE = 156941    # <style|>

STOP_TOKEN_IDS = [TOK_EOS]

TEMPLATE_INDIC = "indic"
TEMPLATE_PLAIN = "plain"
TEMPLATES = (TEMPLATE_INDIC, TEMPLATE_PLAIN)


def build_prompt_token_ids(
    tokenizer,
    template: str,
    text: str,
    voice: str,
    style: Optional[str] = None,
) -> list[int]:
    """Build the prompt for one utterance.

    ``template`` is ``"indic"`` (speaker + style markers, the AI4Bharat
    checkpoint) or ``"plain"`` (``"{voice}: {text}"``, upstream English Orpheus).
    """

    def encode(s: str) -> list[int]:
        # add_special_tokens=False is what prevents the double-BOS above.
        return tokenizer.encode(s, add_special_tokens=False)

    if template == TEMPLATE_INDIC:
        return (
            [TOK_SOH, TOK_BOS, TOK_SPEAKER_OPEN]
            + encode(voice)
            + [TOK_SPEAKER_CLOSE]
            + encode("\n")
            + [TOK_STYLE_OPEN]
            + encode(style or "CONV")
            + [TOK_STYLE_CLOSE]
            + encode("\n")
            + encode(text)
            + [TOK_EOT, TOK_EOH, TOK_SOA, TOK_SOS]
        )
    if template == TEMPLATE_PLAIN:
        return (
            [TOK_SOH, TOK_BOS]
            + encode(f"{voice}: {text}")
            + [TOK_EOT, TOK_EOH, TOK_SOA, TOK_SOS]
        )
    raise ValueError(f"unknown prompt template {template!r}; expected one of {TEMPLATES}")
