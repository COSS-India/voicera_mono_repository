"""
ASR-PATCH (INVESTIGATED AND *NOT* NEEDED -- kept as a record; do not apply by default)
======================================================================================

VERDICT: this checkpoint does NOT want the fork's extra `romanized` prompt slot.
Evidence gathered 2026-08-23:

  1. The checkpoint's own `prompt_defaults` (inside model_config.yaml) lists exactly eight
     slots -- decodercontext, source_lang, target_lang, emotion, pnc, itn, diarize, timestamp
     -- with NO `romanized`. That matches UPSTREAM's canary2 template, not the fork's 9-slot one.
  2. Dumping the real encoded prompt gives exactly 9 tokens and they are all correct:
       <|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|bhb|> <|bhb|>
       <|nopnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
  3. Applying this patch with romanized=<|noromanized|> AND =<|romanized|> produced transcripts
     byte-identical to not applying it at all. The slot changes nothing for this model.

`<|romanized|>` / `<|noromanized|>` ARE present in the checkpoint's spl_tokens vocab (lines
11-12), which is what made the fork's 9-slot template a plausible hypothesis -- but the vocab
is shared across AI4Bharat's Canary line, so their presence proves only that the *tokenizer*
was built with them, not that this fine-tune was prompted with them.

Keep this module so the hypothesis is not re-investigated from scratch. `apply()`/`apply_all()`
still work if a future AI4Bharat checkpoint does need the slot.

Original rationale follows.

ASR-PATCH: add the `romanized` slot to upstream NeMo's canary2 prompt template.

Why
---
`canary_bhili_ft.nemo` was trained with AI4Bharat's NeMo fork, whose
`Canary2PromptFormatter` user template carries a 9th slot that upstream's does not:

  fork     : <|startofcontext|>|decodercontext|<|startoftranscript|>|emotion||source_lang|
             |target_lang||pnc||itn||romanized||timestamp||diarize|
  upstream : ... |pnc||itn|             |timestamp||diarize|      <-- no |romanized|

Evidence this matters for THIS checkpoint: `<|romanized|>` and `<|noromanized|>` are both
present in the checkpoint's own spl_tokens vocab (lines 11-12), i.e. the tokenizer was built
with them. Prompting with upstream's 8-slot template therefore feeds the decoder a prompt one
token shorter than anything it saw in training, and the transcript drifts slightly.

Rather than vendoring the fork's whole canary2.py (it has diverged ~101 lines from upstream and
lacks upstream's newer roles), we insert just this one slot into the live TEMPLATE dict.

Idempotent; safe to call more than once.
"""

from nemo.collections.common.prompts.canary2 import Canary2PromptFormatter
from nemo.collections.common.prompts.formatter import Modality

ROMANIZED_TRUE = "<|romanized|>"
ROMANIZED_FALSE = "<|noromanized|>"

# Mirrors the fork's Modality.TextLiteral argument list for this slot verbatim.
_ROMANIZED_LITERALS = (
    "yes", "no", "true", "True", "false", "False", "1", "0",
    "itn", "noitn", ROMANIZED_TRUE, ROMANIZED_FALSE,
)


def is_applied() -> bool:
    return "romanized" in Canary2PromptFormatter.TEMPLATE["user"]["slots"]


def apply() -> bool:
    """Insert |romanized| between |itn| and |timestamp|. Returns True if it changed anything."""
    if is_applied():
        return False

    user = Canary2PromptFormatter.TEMPLATE["user"]

    old = "|itn||timestamp|"
    new = "|itn||romanized||timestamp|"
    if old not in user["template"]:
        raise RuntimeError(
            "upstream canary2 user template no longer contains '|itn||timestamp|'; "
            f"refusing to guess. Template is: {user['template']!r}"
        )
    user["template"] = user["template"].replace(old, new, 1)

    # rebuild slots dict so `romanized` sits between itn and timestamp (order is cosmetic
    # for lookup but keeps the dict readable next to the template)
    slots, rebuilt = user["slots"], {}
    for k, v in slots.items():
        rebuilt[k] = v
        if k == "itn":
            rebuilt["romanized"] = Modality.TextLiteral(*_ROMANIZED_LITERALS)
    user["slots"] = rebuilt
    return True


# ---------------------------------------------------------------------------
# Supplying the slot's VALUE.
#
# Upstream's `canary2()` prompt fn fills non-required slots from a function-local
# `optional_slots` dict (decodercontext/emotion/itn/timestamp/diarize/pnc). Because that dict
# is a local, `romanized` can't be added to it from outside, and it is not in `expected_slots`
# either (only source_lang/target_lang are), so the formatter reports it missing.
#
# Duplicating upstream's ~40-line canary2() to add one key would fork logic we want to keep
# tracking upstream. Instead we wrap `encode_dialog`, which receives the fully-built turn list,
# and default the slot there. A value present in the manifest / cut.custom still wins, because
# canary2() has already copied it into slots by this point.
# ---------------------------------------------------------------------------

_ENCODE_DIALOG_PATCHED = "_asr_patch_romanized"
DEFAULT_ROMANIZED = ROMANIZED_FALSE  # Bhili is written in Devanagari, so: not romanized


def apply_default(default: str = DEFAULT_ROMANIZED) -> bool:
    """Default the `romanized` slot on user turns. Idempotent; returns True if it patched."""
    orig = Canary2PromptFormatter.encode_dialog
    if getattr(orig, _ENCODE_DIALOG_PATCHED, False):
        return False

    def encode_dialog(self, turns):
        for turn in turns:
            if turn.get("role") == "user":
                slots = turn.get("slots")
                if slots is not None and "romanized" not in slots:
                    slots["romanized"] = default
        return orig(self, turns)

    setattr(encode_dialog, _ENCODE_DIALOG_PATCHED, True)
    Canary2PromptFormatter.encode_dialog = encode_dialog
    return True


def apply_all(default: str = DEFAULT_ROMANIZED) -> None:
    """Both halves: add the slot to the template, and give it a default value."""
    apply()
    apply_default(default)
