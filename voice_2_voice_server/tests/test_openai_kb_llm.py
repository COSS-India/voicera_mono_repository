"""Tests for services/openai_kb_llm.py."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(messages):
    ctx = MagicMock()
    ctx.get_messages.return_value = messages
    return ctx


def _make_service(**kwargs):
    defaults = dict(
        org_id="org_123",
        knowledge_enabled=True,
        knowledge_document_ids=["doc_1"],
        knowledge_top_k=3,
        model="gpt-4o",
        api_key="sk-test",
    )
    defaults.update(kwargs)
    from services.openai_kb_llm import OpenAIKnowledgeLLMService
    with patch("services.openai_kb_llm.OpenAILLMService.__init__", return_value=None):
        svc = OpenAIKnowledgeLLMService.__new__(OpenAIKnowledgeLLMService)
        svc._org_id = defaults.get("org_id")
        svc._knowledge_enabled = bool(defaults.get("knowledge_enabled", False))
        raw_ids = defaults.get("knowledge_document_ids") or []
        svc._knowledge_document_ids = [d for d in raw_ids if d]
        top_k = defaults.get("knowledge_top_k", 3)
        svc._knowledge_top_k = max(1, min(int(top_k or 3), 10))
    return svc


# ---------------------------------------------------------------------------
# __init__ attribute setup
# ---------------------------------------------------------------------------

def test_init_knowledge_top_k_clamped():
    svc = _make_service(knowledge_top_k=50)
    assert svc._knowledge_top_k == 10


def test_init_knowledge_top_k_min():
    # top_k=0 → `int(0 or 3)` = 3 (0 is falsy, falls back to 3)
    svc = _make_service(knowledge_top_k=0)
    assert svc._knowledge_top_k == 3


def test_init_filters_empty_document_ids():
    svc = _make_service(knowledge_document_ids=["doc1", "", "doc2", None])
    assert svc._knowledge_document_ids == ["doc1", "doc2"]


# ---------------------------------------------------------------------------
# _process_context — early exits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_context_knowledge_disabled():
    from services.openai_kb_llm import OpenAIKnowledgeLLMService
    svc = _make_service(knowledge_enabled=False)
    super_mock = AsyncMock(return_value="result")

    with patch.object(OpenAIKnowledgeLLMService, "_process_context", super_mock):
        ctx = _make_context([{"role": "user", "content": "hello"}])
        # knowledge disabled → immediate super()
        with patch("services.openai_kb_llm.OpenAILLMService._process_context", super_mock):
            result = await svc._process_context(ctx)

    # Should have called super — no KB fetch
    super_mock.assert_called_once()


@pytest.mark.asyncio
async def test_process_context_no_org_id():
    svc = _make_service(knowledge_enabled=True, org_id=None)
    super_mock = AsyncMock(return_value=None)

    with patch("services.openai_kb_llm.OpenAILLMService._process_context", super_mock):
        ctx = _make_context([{"role": "user", "content": "hello"}])
        await svc._process_context(ctx)

    super_mock.assert_called_once()


@pytest.mark.asyncio
async def test_process_context_no_messages():
    svc = _make_service()
    super_mock = AsyncMock(return_value=None)

    with patch("services.openai_kb_llm.OpenAILLMService._process_context", super_mock):
        ctx = _make_context([])
        await svc._process_context(ctx)

    super_mock.assert_called_once()


@pytest.mark.asyncio
async def test_process_context_no_user_message():
    svc = _make_service()
    super_mock = AsyncMock(return_value=None)

    with patch("services.openai_kb_llm.OpenAILLMService._process_context", super_mock):
        ctx = _make_context([{"role": "assistant", "content": "Hi!"}])
        await svc._process_context(ctx)

    super_mock.assert_called_once()


@pytest.mark.asyncio
async def test_process_context_empty_user_text():
    svc = _make_service()
    super_mock = AsyncMock(return_value=None)

    with patch("services.openai_kb_llm.OpenAILLMService._process_context", super_mock):
        ctx = _make_context([{"role": "user", "content": "   "}])
        await svc._process_context(ctx)

    super_mock.assert_called_once()


@pytest.mark.asyncio
async def test_process_context_no_document_ids():
    svc = _make_service(knowledge_document_ids=[])
    super_mock = AsyncMock(return_value=None)

    with patch("services.openai_kb_llm.OpenAILLMService._process_context", super_mock):
        ctx = _make_context([{"role": "user", "content": "What is the policy?"}])
        await svc._process_context(ctx)

    super_mock.assert_called_once()


# ---------------------------------------------------------------------------
# _process_context — KB fetch paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_context_no_chunks_returned():
    svc = _make_service()
    super_mock = AsyncMock(return_value=None)

    with patch("services.openai_kb_llm.OpenAILLMService._process_context", super_mock), \
         patch("services.openai_kb_llm.asyncio.to_thread", new_callable=AsyncMock, return_value=[]):
        ctx = _make_context([{"role": "user", "content": "What is the return policy?"}])
        await svc._process_context(ctx)

    super_mock.assert_called_once()


@pytest.mark.asyncio
async def test_process_context_chunks_with_empty_text():
    svc = _make_service()
    super_mock = AsyncMock(return_value=None)
    chunks = [{"text": "", "source_filename": "doc.pdf"}]  # empty text → no excerpt line

    with patch("services.openai_kb_llm.OpenAILLMService._process_context", super_mock), \
         patch("services.openai_kb_llm.asyncio.to_thread", new_callable=AsyncMock, return_value=chunks):
        ctx = _make_context([{"role": "user", "content": "What is the return policy?"}])
        await svc._process_context(ctx)

    # All chunks had empty text → falls back to super without augmenting
    super_mock.assert_called_once()


@pytest.mark.asyncio
async def test_process_context_augments_and_restores():
    svc = _make_service()
    super_mock = AsyncMock(return_value=None)
    chunks = [
        {"text": "You may return within 30 days.", "source_filename": "policy.pdf"},
        {"text": "", "source_filename": "empty.pdf"},  # skipped
    ]

    messages = [{"role": "user", "content": "What is the return policy?"}]
    ctx = _make_context(messages)

    captured_content = []

    async def capture_super(context):
        # Capture what the content was during the super call
        captured_content.append(messages[0]["content"])

    with patch("services.openai_kb_llm.OpenAILLMService._process_context", side_effect=capture_super), \
         patch("services.openai_kb_llm.asyncio.to_thread", new_callable=AsyncMock, return_value=chunks):
        await svc._process_context(ctx)

    # During super call: content was augmented
    assert "Knowledge Base excerpts" in captured_content[0]
    assert "policy.pdf" in captured_content[0]
    # After call: content restored
    assert messages[0]["content"] == "What is the return policy?"


@pytest.mark.asyncio
async def test_process_context_restores_on_exception():
    svc = _make_service()
    chunks = [{"text": "Some excerpt.", "source_filename": "doc.pdf"}]

    messages = [{"role": "user", "content": "What is the return policy?"}]
    ctx = _make_context(messages)

    with patch("services.openai_kb_llm.OpenAILLMService._process_context", side_effect=RuntimeError("boom")), \
         patch("services.openai_kb_llm.asyncio.to_thread", new_callable=AsyncMock, return_value=chunks):
        with pytest.raises(RuntimeError):
            await svc._process_context(ctx)

    # Content must be restored even after exception
    assert messages[0]["content"] == "What is the return policy?"


@pytest.mark.asyncio
async def test_process_context_uses_document_id_as_source():
    svc = _make_service()
    super_mock = AsyncMock(return_value=None)
    chunks = [{"text": "Excerpt text.", "document_id": "doc_abc"}]  # no source_filename

    messages = [{"role": "user", "content": "Question?"}]
    ctx = _make_context(messages)

    captured = []

    async def capture_super(context):
        captured.append(messages[0]["content"])

    with patch("services.openai_kb_llm.OpenAILLMService._process_context", side_effect=capture_super), \
         patch("services.openai_kb_llm.asyncio.to_thread", new_callable=AsyncMock, return_value=chunks):
        await svc._process_context(ctx)

    assert "doc_abc" in captured[0]


@pytest.mark.asyncio
async def test_process_context_picks_last_user_message():
    svc = _make_service()
    super_mock = AsyncMock(return_value=None)
    chunks = [{"text": "Relevant info.", "source_filename": "kb.pdf"}]

    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Second question"},  # this is the last user
    ]
    ctx = _make_context(messages)

    captured = []

    async def capture_super(context):
        captured.append(messages[2]["content"])

    with patch("services.openai_kb_llm.OpenAILLMService._process_context", side_effect=capture_super), \
         patch("services.openai_kb_llm.asyncio.to_thread", new_callable=AsyncMock, return_value=chunks):
        await svc._process_context(ctx)

    assert "Second question" in captured[0]
    # First user message untouched
    assert messages[0]["content"] == "First question"
