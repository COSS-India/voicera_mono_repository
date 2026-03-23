"""
Knowledge base: org-scoped PDF metadata in Mongo + ingest to Chroma.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from app.config import settings
from app.database import get_database
from app.services import integration_service

logger = logging.getLogger(__name__)

# Do not import rag_system at module load: chromadb/openai stack can fail in some
# environments and would prevent the entire API (including login) from starting.

COLLECTION_NAME = "KnowledgeDocuments"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


class KnowledgeDocumentNotFoundError(Exception):
    """No KnowledgeDocuments row for this org and document_id."""


class KnowledgeChromaDeleteError(Exception):
    """Chroma / RAG ingest delete failed; Mongo row was not removed."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class KnowledgeRetrievalError(Exception):
    """Knowledge retrieval failed for the current query."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _org_chroma_subdir(org_id: str) -> str:
    """Stable filesystem-safe directory name per org."""
    return hashlib.sha256(org_id.encode("utf-8")).hexdigest()[:48]


def chroma_dir_for_org(org_id: str) -> Path:
    base = Path(settings.CHROMA_BASE_DIR)
    return base / "orgs" / _org_chroma_subdir(org_id)


def resolve_openai_key_for_org(org_id: str) -> Optional[str]:
    """OpenAI API key from org Integrations only (Integrations tab, model name 'OpenAI')."""
    return integration_service.get_openai_api_key_for_org(org_id)


def create_document_pending(org_id: str, original_filename: str) -> str:
    """Insert Mongo row with status processing; return document_id."""
    db = get_database()
    table = db[COLLECTION_NAME]
    document_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    table.insert_one(
        {
            "document_id": document_id,
            "org_id": org_id,
            "original_filename": original_filename,
            "status": "processing",
            "chunk_count": None,
            "embedding_model": None,
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    return document_id


def update_document(
    document_id: str,
    org_id: str,
    *,
    status: str,
    chunk_count: Optional[int] = None,
    embedding_model: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    db = get_database()
    table = db[COLLECTION_NAME]
    now = datetime.utcnow().isoformat() + "Z"
    update: dict[str, Any] = {
        "status": status,
        "updated_at": now,
    }
    if chunk_count is not None:
        update["chunk_count"] = chunk_count
    if embedding_model is not None:
        update["embedding_model"] = embedding_model
    if error_message is not None:
        update["error_message"] = error_message
    elif status == "ready":
        update["error_message"] = None

    table.update_one(
        {"document_id": document_id, "org_id": org_id},
        {"$set": update},
    )


def list_documents(org_id: str) -> List[dict]:
    db = get_database()
    table = db[COLLECTION_NAME]
    cursor = table.find({"org_id": org_id}).sort("created_at", -1)
    out: List[dict] = []
    for doc in cursor:
        doc.pop("_id", None)
        out.append(doc)
    return out


def retrieve_chunks_for_query(
    *,
    org_id: str,
    question: str,
    document_ids: Optional[List[str]] = None,
    top_k: int = 3,
) -> List[dict]:
    """
    Retrieve top-k relevant chunk texts from org-scoped Chroma.

    Uses OpenAI query embeddings and Chroma ANN query on ``rag_docs``.
    If ``document_ids`` are provided, retrieval is constrained to those documents.
    """
    query = (question or "").strip()
    if not query:
        return []

    # Safety belt: if the caller explicitly passed an empty list, do not
    # fall back to "search across all docs".
    if document_ids is not None and len(document_ids) == 0:
        return []

    k = max(1, min(int(top_k or 3), 10))
    selected_ids = [d.strip() for d in (document_ids or []) if d and d.strip()]

    api_key = resolve_openai_key_for_org(org_id)
    if not api_key:
        raise KnowledgeRetrievalError(
            "No OpenAI API key: add your OpenAI key in Integrations for this organization."
        )

    chroma_dir = chroma_dir_for_org(org_id)
    if not chroma_dir.is_dir():
        return []

    try:
        import chromadb
        from openai import OpenAI
    except ImportError as e:
        raise KnowledgeRetrievalError(
            "RAG dependencies not installed in the backend process."
        ) from e

    client = OpenAI(api_key=api_key)
    try:
        emb_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        q_emb = list(emb_resp.data[0].embedding)
    except Exception as e:
        raise KnowledgeRetrievalError(f"Embedding failed: {e}") from e

    try:
        chroma = chromadb.PersistentClient(path=str(chroma_dir.resolve()))
        collection = chroma.get_collection(name="rag_docs")
    except Exception:
        return []

    include = ["documents", "distances", "metadatas"]
    n_results = max(k, min(25, k * 4 if selected_ids else k))
    where = {"document_id": {"$in": selected_ids}} if selected_ids else None
    results = None
    try:
        if where:
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=n_results,
                include=include,
                where=where,
            )
        else:
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=n_results,
                include=include,
            )
    except Exception:
        # Fallback for older Chroma filtering behavior.
        try:
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=n_results,
                include=include,
            )
        except Exception as e:
            raise KnowledgeRetrievalError(f"Chroma query failed: {e}") from e

    ids_batch = results.get("ids") or []
    docs_batch = results.get("documents") or []
    dists_batch = results.get("distances") or []
    metas_batch = results.get("metadatas") or []
    if not ids_batch or not ids_batch[0]:
        return []

    ids = ids_batch[0]
    docs = docs_batch[0] if docs_batch else []
    dists = dists_batch[0] if dists_batch else [None] * len(ids)
    metas = metas_batch[0] if metas_batch else [None] * len(ids)
    out: List[dict] = []
    for cid, doc, dist, meta in zip(ids, docs, dists, metas, strict=False):
        metadata = meta or {}
        doc_id = metadata.get("document_id") or metadata.get("chunk_id_prefix")
        if selected_ids and doc_id not in selected_ids:
            continue
        text = (doc or "").strip()
        if not text:
            continue
        out.append(
            {
                "chunk_id": cid,
                "document_id": doc_id,
                "source_filename": metadata.get("source_filename"),
                "text": text,
                "distance": dist,
            }
        )
        if len(out) >= k:
            break
    return out


def _delete_chunks_local_disk(chroma_dir: Path, document_id: str) -> None:
    """Delete vectors for ``document_id`` from Chroma on disk (this API process)."""
    try:
        from rag_system.ingest_pipeline import (
            IngestPipelineError,
            delete_chunks_for_document,
        )
    except ImportError as e:
        logger.error("Knowledge delete RAG dependencies missing: %s", e)
        raise KnowledgeChromaDeleteError(
            "RAG dependencies not installed in the API process. "
            "Install voicera_backend requirements (e.g. pip install -r requirements.txt)."
        ) from e
    try:
        delete_chunks_for_document(
            chroma_dir,
            document_id,
            collection_name="rag_docs",
            raise_on_delete_error=True,
        )
    except IngestPipelineError as e:
        raise KnowledgeChromaDeleteError(str(e)) from e


def delete_knowledge_document(org_id: str, document_id: str) -> None:
    """
    Remove Chroma vectors for ``document_id`` then delete the Mongo row.
    Raises :class:`KnowledgeDocumentNotFoundError` or :class:`KnowledgeChromaDeleteError`.
    """
    db = get_database()
    table = db[COLLECTION_NAME]
    existing = table.find_one({"document_id": document_id})
    if not existing or existing.get("org_id") != org_id:
        raise KnowledgeDocumentNotFoundError()

    chroma_dir = chroma_dir_for_org(org_id)
    _delete_chunks_local_disk(chroma_dir, document_id)

    deleted = table.delete_one({"document_id": document_id, "org_id": org_id})
    if deleted.deleted_count == 0:
        raise KnowledgeDocumentNotFoundError()


def run_ingest_job(document_id: str, org_id: str, original_filename: str, pdf_bytes: bytes) -> None:
    """Background task: embed PDF into org Chroma and update Mongo."""
    api_key = resolve_openai_key_for_org(org_id)
    if not api_key:
        update_document(
            document_id,
            org_id,
            status="failed",
            error_message="No OpenAI API key: add your OpenAI key in Integrations for this organization.",
        )
        return

    chroma_dir = chroma_dir_for_org(org_id)

    try:
        from rag_system.ingest_pipeline import IngestPipelineError, ingest_pdf_bytes
    except ImportError as e:
        logger.exception("Knowledge ingest RAG import failed")
        detail = str(e).strip() or type(e).__name__
        update_document(
            document_id,
            org_id,
            status="failed",
            error_message=(
                "RAG import failed in the API process. "
                "If you use Docker: rebuild the backend image so chromadb/onnxruntime install correctly "
                "(e.g. `docker compose build --no-cache backend`). "
                f"Detail: {detail}"[:2000],
            ),
        )
        return

    try:
        result = ingest_pdf_bytes(
            pdf_bytes=pdf_bytes,
            filename=original_filename,
            chunk_id_prefix=document_id,
            chroma_dir=chroma_dir,
            collection="rag_docs",
            openai_api_key=api_key,
            org_id=org_id,
            document_id=document_id,
            reset_collection=False,
        )
        update_document(
            document_id,
            org_id,
            status="ready",
            chunk_count=result.num_chunks,
            embedding_model=result.embedding_model,
        )
        logger.info(
            "Knowledge ingest ready document_id=%s org_id=%s chunks=%s",
            document_id,
            org_id,
            result.num_chunks,
        )
    except IngestPipelineError as e:
        logger.warning("Knowledge ingest failed: %s", e)
        update_document(
            document_id,
            org_id,
            status="failed",
            error_message=str(e)[:2000],
        )
    except Exception as e:
        logger.exception("Knowledge ingest unexpected error")
        update_document(
            document_id,
            org_id,
            status="failed",
            error_message=str(e)[:2000],
        )
