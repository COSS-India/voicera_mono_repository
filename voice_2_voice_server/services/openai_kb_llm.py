"""OpenAI LLM wrapper with optional Knowledge Base retrieval augmentation."""
from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService

from api.backend_utils import fetch_knowledge_chunks


class OpenAIKnowledgeLLMService(OpenAILLMService):
    """OpenAI service that prepends retrieved KB excerpts to the latest user turn."""

    def __init__(
        self,
        *,
        org_id: Optional[str] = None,
        knowledge_enabled: bool = False,
        knowledge_document_ids: Optional[list[str]] = None,
        knowledge_top_k: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._org_id = org_id
        self._knowledge_enabled = bool(knowledge_enabled)
        self._knowledge_document_ids = [d for d in (knowledge_document_ids or []) if d]
        self._knowledge_top_k = max(1, min(int(knowledge_top_k or 3), 10))

    async def _process_context(self, context: OpenAILLMContext | LLMContext):
        if not self._knowledge_enabled or not self._org_id:
            return await super()._process_context(context)

        messages = context.get_messages()
        if not messages:
            return await super()._process_context(context)

        user_index = None
        user_text = ""
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "user":
                user_index = i
                user_text = (msg.get("content") or "").strip()
                break

        if user_index is None or not user_text:
            return await super()._process_context(context)

        # Guardrail: only retrieve when at least one document is selected.
        # Without this, passing an empty document_ids list would make retrieval
        # search across all org knowledge documents.
        if not self._knowledge_document_ids:
            return await super()._process_context(context)

        chunks = await asyncio.to_thread(
            fetch_knowledge_chunks,
            org_id=self._org_id,
            question=user_text,
            document_ids=self._knowledge_document_ids,
            top_k=self._knowledge_top_k,
            timeout=0.8,
        )
        if not chunks:
            # Graceful fallback: no relevant KB excerpts found.
            # Let the LLM respond naturally using its system prompt / persona
            # instead of forcing a "no info available" dead-end.
            logger.debug(
                "OpenAI KB retrieval returned 0 excerpts; falling back to natural LLM response"
            )
            return await super()._process_context(context)

        excerpt_lines: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue
            source = chunk.get("source_filename") or chunk.get("document_id") or "knowledge_base"
            excerpt_lines.append(f"[Excerpt {idx} | {source}]\n{text}")
        if not excerpt_lines:
            return await super()._process_context(context)

        augmented = (
            f"User question:\n{user_text}\n\n"
            "Knowledge Base excerpts:\n"
            f"{chr(10).join(excerpt_lines)}\n\n"
            "Answer using the excerpts when relevant. If excerpts are insufficient, answer naturally and be transparent."
        )
        original = messages[user_index].get("content")
        messages[user_index]["content"] = augmented
        try:
            return await super()._process_context(context)
        finally:
            # Avoid permanently polluting memory context with expanded payload.
            messages[user_index]["content"] = original
            logger.debug("OpenAI KB retrieval used: %s excerpts", len(excerpt_lines))
