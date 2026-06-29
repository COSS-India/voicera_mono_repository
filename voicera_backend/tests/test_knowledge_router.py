"""
Integration tests for /api/v1/knowledge endpoints.
"""
import io
import pytest
from unittest.mock import patch

from app.services.knowledge_service import (
    KnowledgeDocumentNotFoundError,
    KnowledgeChromaDeleteError,
)

BASE = "/api/v1/knowledge"

DOC_RESPONSE = {
    "document_id": "doc-001",
    "org_id": "testorg1",
    "original_filename": "report.pdf",
    "status": "ready",
    "chunk_count": 5,
    "embedding_model": "text-embedding-3-small",
    "error_message": None,
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z",
}

VALID_PDF = b"%PDF-1.4 fake pdf content for testing"


# ── GET /knowledge ────────────────────────────────────────────────────────

class TestListKnowledgeDocuments:
    def test_success_returns_list(self, client):
        with patch("app.services.knowledge_service.list_documents", return_value=[DOC_RESPONSE]):
            resp = client.get(BASE)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["document_id"] == "doc-001"

    def test_empty_list_returns_200(self, client):
        with patch("app.services.knowledge_service.list_documents", return_value=[]):
            resp = client.get(BASE)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_unauthenticated_returns_401(self, unauth_client):
        resp = unauth_client.get(BASE)
        assert resp.status_code == 401


# ── POST /knowledge/upload ────────────────────────────────────────────────

class TestUploadKnowledgePdf:
    def _upload(self, client, *, org_id="testorg1", filename="doc.pdf", content=VALID_PDF):
        return client.post(
            f"{BASE}/upload",
            files={"file": (filename, io.BytesIO(content), "application/pdf")},
            data={"org_id": org_id},
        )

    def test_success_returns_201_with_processing_status(self, client):
        with patch("app.services.knowledge_service.create_document_pending",
                   return_value="doc-001"), \
             patch("app.services.knowledge_service.run_ingest_job"):
            resp = self._upload(client)
        assert resp.status_code == 201
        assert resp.json()["status"] == "processing"
        assert resp.json()["document_id"] == "doc-001"

    def test_wrong_org_returns_403(self, client):
        resp = self._upload(client, org_id="otherorg9")
        assert resp.status_code == 403

    def test_non_pdf_file_returns_400(self, client):
        resp = self._upload(client, filename="doc.txt")
        assert resp.status_code == 400
        assert "PDF" in resp.json()["detail"]

    def test_empty_file_returns_400(self, client):
        resp = self._upload(client, content=b"")
        assert resp.status_code == 400

    def test_oversized_file_returns_413(self, client):
        big_content = b"%PDF-1.4 " + b"x" * (26 * 1024 * 1024)
        resp = self._upload(client, content=big_content)
        assert resp.status_code == 413


# ── DELETE /knowledge/{document_id} ──────────────────────────────────────

class TestDeleteKnowledgeDocument:
    def test_success_returns_deleted_true(self, client):
        with patch("app.services.knowledge_service.delete_knowledge_document"):
            resp = client.delete(f"{BASE}/doc-001")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_not_found_returns_404(self, client):
        with patch("app.services.knowledge_service.delete_knowledge_document",
                   side_effect=KnowledgeDocumentNotFoundError()):
            resp = client.delete(f"{BASE}/missing-doc")
        assert resp.status_code == 404

    def test_chroma_delete_error_returns_500(self, client):
        err = KnowledgeChromaDeleteError("Chroma cleanup failed")
        err.message = "Chroma cleanup failed"
        with patch("app.services.knowledge_service.delete_knowledge_document",
                   side_effect=err):
            resp = client.delete(f"{BASE}/doc-001")
        assert resp.status_code == 500

    def test_unauthenticated_returns_401(self, unauth_client):
        resp = unauth_client.delete(f"{BASE}/doc-001")
        assert resp.status_code == 401
