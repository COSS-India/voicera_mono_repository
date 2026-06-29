"""Tests for app/routers/rag.py"""

import pytest
from unittest.mock import patch

from app.services.knowledge_service import KnowledgeRetrievalError

BASE = "/api/v1/rag"

CHUNKS = [{"text": "chunk text", "chunk_id": "c1", "document_id": "d1", "source_filename": "f.pdf", "distance": 0.1}]
PAYLOAD = {"org_id": "testorg1", "question": "What is X?", "top_k": 3}


class TestRetrieveKnowledgeChunks:
    def test_success_returns_chunks(self, client):
        with patch("app.routers.rag.knowledge_service.retrieve_chunks_for_query", return_value=CHUNKS):
            resp = client.post(BASE + "/retrieve", json=PAYLOAD)
        assert resp.status_code == 200
        assert resp.json()["chunks"][0]["text"] == "chunk text"

    def test_retrieval_error_returns_500(self, client):
        err = KnowledgeRetrievalError("Chroma unavailable")
        err.message = "Chroma unavailable"
        with patch("app.routers.rag.knowledge_service.retrieve_chunks_for_query", side_effect=err):
            resp = client.post(BASE + "/retrieve", json=PAYLOAD)
        assert resp.status_code == 500
        assert "Chroma unavailable" in resp.json()["detail"]
