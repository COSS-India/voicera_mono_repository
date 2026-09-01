"""
Unit tests for app.services.knowledge_service.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch, call

from app.services.knowledge_service import (
    _org_chroma_subdir,
    create_document_pending,
    update_document,
    list_documents,
    delete_knowledge_document,
    run_ingest_job,
    KnowledgeDocumentNotFoundError,
    KnowledgeChromaDeleteError,
    KnowledgeRetrievalError,
    retrieve_chunks_for_query,
)
from tests.helpers import make_mock_db

# ── Sample data ───────────────────────────────────────────────────────────────

ORG_ID = "testorg1"
DOC_ID = "doc-test-001"

DOCUMENT_DOC = {
    "document_id": DOC_ID,
    "org_id": ORG_ID,
    "original_filename": "report.pdf",
    "status": "ready",
    "chunk_count": 5,
    "embedding_model": "text-embedding-3-small",
    "error_message": None,
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z",
}

PDF_BYTES = b"%PDF-1.4 test content"


def _make_doc_db(doc=None):
    docs_coll = MagicMock()
    docs_coll.find_one.return_value = doc
    db = make_mock_db(KnowledgeDocuments=docs_coll)
    return db, docs_coll


# ── TestOrgChromaSubdir ───────────────────────────────────────────────────

class TestOrgChromaSubdir:
    def test_returns_48_char_sha256_prefix(self):
        result = _org_chroma_subdir(ORG_ID)
        assert len(result) == 48
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic_for_same_org(self):
        assert _org_chroma_subdir(ORG_ID) == _org_chroma_subdir(ORG_ID)

    def test_different_orgs_produce_different_subdirs(self):
        assert _org_chroma_subdir("orgA") != _org_chroma_subdir("orgB")


# ── TestCreateDocumentPending ─────────────────────────────────────────────

class TestCreateDocumentPending:
    def test_inserts_with_processing_status_and_returns_uuid(self):
        db, docs_coll = _make_doc_db()
        with patch("app.services.knowledge_service.get_database", return_value=db):
            doc_id = create_document_pending(ORG_ID, "report.pdf")
        assert isinstance(doc_id, str)
        assert len(doc_id) == 36  # UUID format
        docs_coll.insert_one.assert_called_once()
        inserted = docs_coll.insert_one.call_args[0][0]
        assert inserted["status"] == "processing"
        assert inserted["org_id"] == ORG_ID
        assert inserted["original_filename"] == "report.pdf"

    def test_returns_unique_ids_each_call(self):
        db, _ = _make_doc_db()
        with patch("app.services.knowledge_service.get_database", return_value=db):
            id1 = create_document_pending(ORG_ID, "a.pdf")
            id2 = create_document_pending(ORG_ID, "b.pdf")
        assert id1 != id2


# ── TestUpdateDocument ────────────────────────────────────────────────────

class TestUpdateDocument:
    def test_updates_status_and_timestamp(self):
        db, docs_coll = _make_doc_db()
        with patch("app.services.knowledge_service.get_database", return_value=db):
            update_document(DOC_ID, ORG_ID, status="failed", error_message="oops")
        docs_coll.update_one.assert_called_once()
        update_set = docs_coll.update_one.call_args[0][1]["$set"]
        assert update_set["status"] == "failed"
        assert update_set["error_message"] == "oops"

    def test_sets_chunk_count_and_model_when_provided(self):
        db, docs_coll = _make_doc_db()
        with patch("app.services.knowledge_service.get_database", return_value=db):
            update_document(
                DOC_ID, ORG_ID, status="ready",
                chunk_count=10, embedding_model="text-embedding-3-small"
            )
        update_set = docs_coll.update_one.call_args[0][1]["$set"]
        assert update_set["chunk_count"] == 10
        assert update_set["embedding_model"] == "text-embedding-3-small"

    def test_clears_error_message_on_ready_status(self):
        db, docs_coll = _make_doc_db()
        with patch("app.services.knowledge_service.get_database", return_value=db):
            update_document(DOC_ID, ORG_ID, status="ready")
        update_set = docs_coll.update_one.call_args[0][1]["$set"]
        assert update_set.get("error_message") is None


# ── TestListDocuments ─────────────────────────────────────────────────────

class TestListDocuments:
    def test_returns_docs_without_id_field(self):
        from bson import ObjectId
        doc_with_id = {**DOCUMENT_DOC, "_id": ObjectId()}
        docs_coll = MagicMock()
        docs_coll.find.return_value.sort.return_value = [doc_with_id]
        db = make_mock_db(KnowledgeDocuments=docs_coll)
        with patch("app.services.knowledge_service.get_database", return_value=db):
            result = list_documents(ORG_ID)
        assert len(result) == 1
        assert "_id" not in result[0]

    def test_returns_empty_list_when_no_docs(self):
        docs_coll = MagicMock()
        docs_coll.find.return_value.sort.return_value = []
        db = make_mock_db(KnowledgeDocuments=docs_coll)
        with patch("app.services.knowledge_service.get_database", return_value=db):
            result = list_documents(ORG_ID)
        assert result == []


# ── TestRetrieveChunksForQuery ────────────────────────────────────────────

class TestRetrieveChunksForQuery:
    def test_empty_question_returns_empty_list(self):
        with patch("app.services.knowledge_service.resolve_openai_key_for_org", return_value="key"):
            result = retrieve_chunks_for_query(org_id=ORG_ID, question="")
        assert result == []

    def test_empty_document_ids_list_returns_empty(self):
        with patch("app.services.knowledge_service.resolve_openai_key_for_org", return_value="key"):
            result = retrieve_chunks_for_query(org_id=ORG_ID, question="hello", document_ids=[])
        assert result == []

    def test_missing_api_key_raises_retrieval_error(self):
        with patch("app.services.knowledge_service.resolve_openai_key_for_org", return_value=None):
            with pytest.raises(KnowledgeRetrievalError):
                retrieve_chunks_for_query(org_id=ORG_ID, question="test query")

    def test_missing_chroma_dir_returns_empty_list(self):
        mock_path = MagicMock()
        mock_path.is_dir.return_value = False
        with patch("app.services.knowledge_service.resolve_openai_key_for_org", return_value="sk-test"), \
             patch("app.services.knowledge_service.chroma_dir_for_org", return_value=mock_path):
            result = retrieve_chunks_for_query(org_id=ORG_ID, question="test query")
        assert result == []

    def test_chromadb_import_error_raises_retrieval_error(self):
        mock_path = MagicMock()
        mock_path.is_dir.return_value = True
        with patch("app.services.knowledge_service.resolve_openai_key_for_org", return_value="sk-test"), \
             patch("app.services.knowledge_service.chroma_dir_for_org", return_value=mock_path), \
             patch.dict(sys.modules, {"chromadb": None}):
            with pytest.raises(KnowledgeRetrievalError):
                retrieve_chunks_for_query(org_id=ORG_ID, question="test query")


# ── TestDeleteKnowledgeDocument ───────────────────────────────────────────

class TestDeleteKnowledgeDocument:
    def test_success_deletes_from_mongo(self):
        db, docs_coll = _make_doc_db(doc=DOCUMENT_DOC)
        docs_coll.delete_one.return_value.deleted_count = 1
        with patch("app.services.knowledge_service.get_database", return_value=db), \
             patch("app.services.knowledge_service._delete_chunks_local_disk"):
            delete_knowledge_document(ORG_ID, DOC_ID)
        docs_coll.delete_one.assert_called_once()

    def test_not_found_raises_document_not_found_error(self):
        db, _ = _make_doc_db(doc=None)
        with patch("app.services.knowledge_service.get_database", return_value=db):
            with pytest.raises(KnowledgeDocumentNotFoundError):
                delete_knowledge_document(ORG_ID, "missing-doc")

    def test_wrong_org_raises_document_not_found_error(self):
        doc_other_org = {**DOCUMENT_DOC, "org_id": "other-org"}
        db, _ = _make_doc_db(doc=doc_other_org)
        with patch("app.services.knowledge_service.get_database", return_value=db):
            with pytest.raises(KnowledgeDocumentNotFoundError):
                delete_knowledge_document(ORG_ID, DOC_ID)

    def test_chroma_delete_failure_raises_and_skips_mongo_delete(self):
        db, docs_coll = _make_doc_db(doc=DOCUMENT_DOC)
        with patch("app.services.knowledge_service.get_database", return_value=db), \
             patch("app.services.knowledge_service._delete_chunks_local_disk",
                   side_effect=KnowledgeChromaDeleteError("Chroma failed")):
            with pytest.raises(KnowledgeChromaDeleteError):
                delete_knowledge_document(ORG_ID, DOC_ID)
        docs_coll.delete_one.assert_not_called()


# ── TestRunIngestJob ──────────────────────────────────────────────────────

class TestRunIngestJob:
    def test_no_api_key_marks_document_failed(self):
        db, docs_coll = _make_doc_db()
        with patch("app.services.knowledge_service.get_database", return_value=db), \
             patch("app.services.knowledge_service.resolve_openai_key_for_org", return_value=None):
            run_ingest_job(DOC_ID, ORG_ID, "report.pdf", PDF_BYTES)
        docs_coll.update_one.assert_called()
        update_set = docs_coll.update_one.call_args[0][1]["$set"]
        assert update_set["status"] == "failed"

    def test_rag_import_error_marks_document_failed(self):
        db, docs_coll = _make_doc_db()
        with patch("app.services.knowledge_service.get_database", return_value=db), \
             patch("app.services.knowledge_service.resolve_openai_key_for_org", return_value="sk-test"), \
             patch("app.services.knowledge_service.chroma_dir_for_org", return_value=MagicMock()), \
             patch.dict(sys.modules, {"rag_system": None, "rag_system.ingest_pipeline": None}):
            run_ingest_job(DOC_ID, ORG_ID, "report.pdf", PDF_BYTES)
        update_set = docs_coll.update_one.call_args[0][1]["$set"]
        assert update_set["status"] == "failed"

    def test_ingest_pipeline_error_marks_document_failed(self):
        db, docs_coll = _make_doc_db()
        mock_pipeline = MagicMock()
        mock_pipeline.IngestPipelineError = Exception
        mock_pipeline.ingest_pdf_bytes.side_effect = Exception("Parse failed")

        with patch("app.services.knowledge_service.get_database", return_value=db), \
             patch("app.services.knowledge_service.resolve_openai_key_for_org", return_value="sk-test"), \
             patch("app.services.knowledge_service.chroma_dir_for_org", return_value=MagicMock()), \
             patch.dict(sys.modules, {
                 "rag_system": MagicMock(ingest_pipeline=mock_pipeline),
                 "rag_system.ingest_pipeline": mock_pipeline,
             }):
            run_ingest_job(DOC_ID, ORG_ID, "report.pdf", PDF_BYTES)
        update_set = docs_coll.update_one.call_args[0][1]["$set"]
        assert update_set["status"] == "failed"

    def test_success_marks_document_ready_with_chunk_count(self):
        db, docs_coll = _make_doc_db()
        mock_pipeline = MagicMock()
        mock_pipeline.IngestPipelineError = type("IngestPipelineError", (Exception,), {})
        mock_result = MagicMock()
        mock_result.num_chunks = 7
        mock_result.embedding_model = "text-embedding-3-small"
        mock_pipeline.ingest_pdf_bytes.return_value = mock_result

        with patch("app.services.knowledge_service.get_database", return_value=db), \
             patch("app.services.knowledge_service.resolve_openai_key_for_org", return_value="sk-test"), \
             patch("app.services.knowledge_service.chroma_dir_for_org", return_value=MagicMock()), \
             patch.dict(sys.modules, {
                 "rag_system": MagicMock(ingest_pipeline=mock_pipeline),
                 "rag_system.ingest_pipeline": mock_pipeline,
             }):
            run_ingest_job(DOC_ID, ORG_ID, "report.pdf", PDF_BYTES)
        update_set = docs_coll.update_one.call_args[0][1]["$set"]
        assert update_set["status"] == "ready"
        assert update_set["chunk_count"] == 7
