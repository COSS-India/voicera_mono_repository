"""Pluggable translation engines for live broadcast rooms.

A translation agent picks exactly one engine and the other never enters the
picture — not imported at call time, not credential-checked, not constructed:

* ``LlmTranslator`` — streaming chat completion. Token-by-token, honours the
  agent's style/domain guidance, needs an OpenAI credential (org or platform).
  This is the original ``TranslationRoom.translate_stream`` body, moved verbatim
  so the LLM path is behaviourally unchanged.

* ``NmtTranslator`` — hosted AI4Bharat IndicTrans2 (Triton). One request returns
  the whole translation in ~0.2-0.4 s; a process-global coalescer merges every
  room's languages into shared GPU passes. No credential; language set is fixed.

Both satisfy :class:`TranslationEngine`, so ``TranslationRoom`` delegates without
knowing which is active. Neither shares code with the other beyond the TTS
chunker in ``utils.text_chunking``.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import AsyncIterator, Optional, Protocol, runtime_checkable

from loguru import logger

from config.nmt_mappings import to_nmt_code
from services.nmt import NmtError, get_nmt_client
from utils.backend_utils import fetch_integration_key
from utils.text_chunking import chunk_final_text, next_chunk_end
from .services import platform_key_fallback_enabled


TRANSLATION_LLM_TIMEOUT_SECS = float(os.getenv("TRANSLATION_LLM_TIMEOUT_SECS", "10"))
# Above this, a source segment is split at sentence boundaries into rows that
# ride the same batch — a guard against a pathological run-on, not a normal
# case (a 0.4 s VAD segment is far shorter).
NMT_MAX_SEGMENT_CHARS = int(os.getenv("NMT_MAX_SEGMENT_CHARS", "1200"))

# Sentence boundary for the long-segment split guard (Latin + Indic danda).
_SENTENCE_SPLIT = re.compile(r"[.!?।॥]+[\"'”’)\]]*\s")


def _translation_model() -> str:
    return os.getenv("TRANSLATION_MODEL") or "gpt-4o-mini"


@runtime_checkable
class TranslationEngine(Protocol):
    """One translation backend for a room. Selected once, per agent config."""

    name: str

    async def prepare(self) -> Optional[str]:
        """Pre-flight at presenter connect. Returns an error string, or None if ready."""
        ...

    def unsupported(self, target_language: str) -> Optional[str]:
        """Reason this engine cannot serve ``target_language``; None if it can."""
        ...

    def stream(
        self, text: str, target_language: str, on_token=None
    ) -> AsyncIterator[str]:
        """Yield TTS-ready chunks for ``text``, in order."""
        ...

    async def aclose(self) -> None:
        ...


# ---------------------------------------------------------------------------
# LLM engine (unchanged behaviour; moved out of TranslationRoom)
# ---------------------------------------------------------------------------

class LlmTranslator:
    name = "llm"

    def __init__(
        self,
        *,
        org_id: Optional[str],
        source_language: Optional[str],
        extra_instructions: str,
        model: Optional[str] = None,
    ):
        self._org_id = org_id
        self._source_language = source_language
        self._extra_instructions = extra_instructions
        self._model = model or _translation_model()
        self._openai = None

    def _get_client(self):
        """Resolve (and cache) the OpenAI client. Blocking integration lookup."""
        if self._openai is not None:
            return self._openai
        api_key = None
        if self._org_id:
            api_key = fetch_integration_key(self._org_id, "OpenAI")
        if not api_key and platform_key_fallback_enabled():
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        from openai import AsyncOpenAI

        self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    async def prepare(self) -> Optional[str]:
        # Blocking key lookup — keep it off the event loop so many listeners
        # joining at once cannot serialise into a stall.
        client = await asyncio.to_thread(self._get_client)
        if client is None:
            return (
                "no OpenAI key available (configure an OpenAI Integration or "
                "enable ALLOW_PLATFORM_KEY_FALLBACK)"
            )
        return None

    def unsupported(self, target_language: str) -> Optional[str]:
        return None  # the LLM handles any language pair

    async def stream(
        self, text: str, target_language: str, on_token=None
    ) -> AsyncIterator[str]:
        client = self._get_client()
        if client is None:
            logger.error("translation: no OpenAI key available (org or platform)")
            return
        system = (
            f"You are a translation engine. Translate the user's text from "
            f"{self._source_language} into {target_language}. Output only the "
            f"translation, with no commentary, labels or quotation marks."
        )
        # The agent's own prompt is extra style/domain guidance; the rules above
        # stay authoritative so the output is always just the translation.
        if self._extra_instructions:
            system = f"{system}\n\nAdditional guidance:\n{self._extra_instructions}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        # One retry, but only before any sentence has been emitted: a transient
        # LLM hiccup (429/timeout) otherwise drops the segment, yet retrying after
        # a partial emit would repeat already-spoken text.
        for attempt in range(2):
            buffer = ""
            produced = False
            stream = None
            try:
                stream = await client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.2,
                    stream=True,
                    # Bound connect + time-to-first-token; a stalled provider must
                    # not hold a language silent for the whole broadcast.
                    timeout=TRANSLATION_LLM_TIMEOUT_SECS,
                )
                # Iterated by hand so each token read carries its own inactivity
                # deadline. The deadline only covers the wait for the model:
                # while suspended at a ``yield`` (the caller back-pressuring on a
                # full synth queue) no timer is running.
                chunks = stream.__aiter__()
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            chunks.__anext__(), TRANSLATION_LLM_TIMEOUT_SECS
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        raise TimeoutError(
                            f"no token for {TRANSLATION_LLM_TIMEOUT_SECS:.0f}s"
                        )
                    delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                    if not delta:
                        continue
                    if on_token is not None:
                        on_token()
                    buffer += delta
                    while True:
                        cut = next_chunk_end(buffer)
                        if cut is None:
                            break
                        sentence = buffer[:cut].strip()
                        buffer = buffer[cut:].lstrip()
                        if sentence:
                            produced = True
                            yield sentence
                tail = buffer.strip()
                if tail:
                    yield tail
                return
            except Exception as e:
                if attempt == 0 and not produced:
                    await asyncio.sleep(0.3)
                    continue
                logger.warning(f"translation to {target_language} failed: {e}")
                return
            finally:
                if stream is not None:
                    try:
                        await stream.close()
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# NMT engine (AI4Bharat IndicTrans2)
# ---------------------------------------------------------------------------

class NmtTranslator:
    name = "nmt"

    def __init__(self, *, source_language: Optional[str]):
        self._source_language = source_language
        self._src_code = to_nmt_code(source_language) if source_language else None
        self._client = None

    async def prepare(self) -> Optional[str]:
        if not self._src_code:
            return (
                f"source language {self._source_language!r} is not supported by "
                f"the NMT engine"
            )
        try:
            self._client = await get_nmt_client()
        except ValueError as e:
            return str(e)  # NMT_SERVER_URL not set
        if not await self._client.ready():
            return "translation backend not ready"
        return None

    def unsupported(self, target_language: str) -> Optional[str]:
        if to_nmt_code(target_language) is None:
            return f"language {target_language!r} is not supported by the NMT engine"
        return None

    async def stream(
        self, text: str, target_language: str, on_token=None
    ) -> AsyncIterator[str]:
        tgt_code = to_nmt_code(target_language)
        if not self._src_code or not tgt_code:
            logger.warning(
                f"translation to {target_language}: unmapped NMT language pair"
            )
            return
        if self._client is None:
            self._client = await get_nmt_client()

        pieces = _split_source(text)
        try:
            if len(pieces) == 1:
                translated = await self._client.translate(
                    pieces[0], self._src_code, tgt_code
                )
            else:
                # Long run-on: translate parts concurrently (they coalesce into
                # the same batch) and rejoin in order before chunking.
                parts = await asyncio.gather(
                    *(
                        self._client.translate(p, self._src_code, tgt_code)
                        for p in pieces
                    )
                )
                translated = " ".join(p for p in parts if p)
        except NmtError as e:
            logger.warning(f"translation to {target_language} failed: {e}")
            return

        if on_token is not None:
            on_token()  # single "token": full translation is ready
        translated = translated.strip()
        if not translated:
            return
        for chunk in chunk_final_text(translated):
            yield chunk

    async def aclose(self) -> None:
        # The client is a process-global singleton shared across rooms; do not
        # close it when one room ends.
        self._client = None


def _split_source(text: str) -> list[str]:
    """Split a source segment into <=NMT_MAX_SEGMENT_CHARS pieces at sentence
    boundaries. Returns ``[text]`` for the normal (short) case."""
    text = text.strip()
    if len(text) <= NMT_MAX_SEGMENT_CHARS:
        return [text]
    pieces: list[str] = []
    buffer = text
    while len(buffer) > NMT_MAX_SEGMENT_CHARS:
        cut = buffer.rfind(" ", 0, NMT_MAX_SEGMENT_CHARS)
        # Prefer a sentence boundary within the window if one exists.
        for m in _SENTENCE_SPLIT.finditer(buffer, 0, NMT_MAX_SEGMENT_CHARS):
            cut = m.end()
        if cut <= 0:
            cut = NMT_MAX_SEGMENT_CHARS
        pieces.append(buffer[:cut].strip())
        buffer = buffer[cut:].lstrip()
    if buffer:
        pieces.append(buffer)
    return [p for p in pieces if p]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_translation_engine(
    engine: str,
    *,
    org_id: Optional[str],
    source_language: Optional[str],
    extra_instructions: str,
    model: Optional[str] = None,
) -> TranslationEngine:
    """Build the single engine named by ``agent_config.translation_engine``.

    Unknown / absent → "llm" (the original behaviour), so existing agents are
    unchanged.
    """
    name = (engine or "llm").strip().lower()
    if name == "nmt":
        return NmtTranslator(source_language=source_language)
    if name != "llm":
        logger.warning(f"unknown translation_engine {engine!r}; defaulting to llm")
    return LlmTranslator(
        org_id=org_id,
        source_language=source_language,
        extra_instructions=extra_instructions,
        model=model,
    )
