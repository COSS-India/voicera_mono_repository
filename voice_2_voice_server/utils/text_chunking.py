"""Sentence/clause chunking shared by every translation engine.

Both the streaming LLM engine (chunking tokens as they arrive) and the NMT
engine (chunking a fully-returned translation) feed the TTS stage in the same
chunk shapes, so downstream synthesis cannot tell the engines apart. Keeping the
logic here — not in ``translation_room`` — avoids a circular import between the
room and the engines module.
"""

import re
from typing import Iterator, Optional

# Break on sentence-final punctuation (Latin + Indic danda) once whitespace
# confirms the sentence closed.
_SENTENCE_END = re.compile(r"[.!?।॥]+[\"'”’)\]]*\s")
# Fallback break for a rambling clause with no sentence-final punctuation: prefer
# a clause boundary (comma/semicolon/colon/dash) over a bare word gap.
_CLAUSE_END = re.compile(r"[,;:—–]\s")
# A rambling clause with no sentence-final punctuation would otherwise never
# flush; past this many buffered chars, break at the last clause (else word)
# boundary so audio keeps flowing. Sentence breaks are always preferred.
MAX_TTS_CHUNK_CHARS = 240
# Don't cut a chunk this short: an abbreviation ("Dr. ") or a decimal would
# otherwise become its own micro-utterance, spoken with sentence-final intonation
# and a pause. Also a floor for the on-prem Parler backend, which clips very
# short prompts.
MIN_TTS_CHUNK_CHARS = 40


def next_chunk_end(buffer: str) -> Optional[int]:
    """Index to cut ``buffer`` at for the next TTS chunk, or None to wait for more."""
    # Search past the minimum so a sentence break that lands too early ("Dr. ")
    # is skipped and folded into the following sentence instead of splitting off.
    match = _SENTENCE_END.search(buffer, MIN_TTS_CHUNK_CHARS)
    if match:
        return match.end()
    if len(buffer) >= MAX_TTS_CHUNK_CHARS:
        # Last clause boundary inside the window beats the last word gap: the
        # chunk is spoken with sentence-final intonation either way, and a pause
        # after a comma passes for natural where one mid-phrase does not.
        clause = None
        for m in _CLAUSE_END.finditer(buffer, MIN_TTS_CHUNK_CHARS, MAX_TTS_CHUNK_CHARS):
            clause = m
        if clause:
            return clause.end()
        space = buffer.rfind(" ", 0, MAX_TTS_CHUNK_CHARS)
        if space > 0:
            return space + 1
    return None


def chunk_final_text(text: str) -> Iterator[str]:
    """Split a fully-known translation into ordered TTS chunks.

    Used by non-streaming engines (NMT): the whole translation is already in
    hand, so drain it through the same boundary logic the streaming path applies
    incrementally, yielding the same chunk shapes.
    """
    buffer = text
    while True:
        cut = next_chunk_end(buffer)
        if cut is None:
            break
        chunk = buffer[:cut].strip()
        buffer = buffer[cut:].lstrip()
        if chunk:
            yield chunk
    tail = buffer.strip()
    if tail:
        yield tail
