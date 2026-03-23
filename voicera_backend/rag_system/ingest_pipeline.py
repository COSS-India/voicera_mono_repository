"""
Shared PDF -> chunk -> embed -> Chroma pipeline.

Used by the main Voicera API (Knowledge Base upload / ingest).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from openai import OpenAI

from .chunk_text import chunk_text
from .embed_chunks import embed_openai
from .pdf_to_text import extract_text_from_pdf

# Default Chroma root (per-org subdirs used by knowledge API)
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent / "chroma_data"
DEFAULT_COLLECTION = "rag_docs"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
DEFAULT_BATCH_SIZE = 100


class IngestPipelineError(Exception):
    """Raised when ingest cannot complete (caller maps to HTTP 400/503)."""


@dataclass
class IngestResult:
    chunk_id_prefix: str
    filename: str
    characters_extracted: int
    num_chunks: int
    embedding_dim: int
    chroma_dir: str
    collection: str
    embedding_model: str


def resolve_openai_api_key(explicit: str | None = None) -> str:
    """Use only the explicit key from the caller (org Integrations via knowledge_service)."""
    key = (explicit or "").strip()
    if not key:
        raise IngestPipelineError(
            "No OpenAI API key: add your OpenAI key in Integrations for this organization."
        )
    return key


def upsert_chroma(
    *,
    chroma_dir: Path,
    collection_name: str,
    embeddings: np.ndarray,
    texts: list[str],
    ids: list[str],
    metadatas: list[dict],
    model_name: str,
    reset_collection: bool,
) -> None:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    _, dim = embeddings.shape
    client = chromadb.PersistentClient(path=str(chroma_dir.resolve()))

    if reset_collection:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_dim": str(dim),
            "embedding_model": model_name,
        },
    )

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )


def delete_chunks_for_document(
    chroma_dir: Path,
    document_id: str,
    collection_name: str = DEFAULT_COLLECTION,
    *,
    raise_on_delete_error: bool = False,
) -> None:
    """
    Remove all vectors for a Knowledge Base document from Chroma.

    Matches metadata ``document_id`` or ``chunk_id_prefix`` (KB uses the same UUID for both).
    No-op if the persist dir or collection does not exist.
    If ``raise_on_delete_error`` is True, failures after the collection is opened propagate as
    :class:`IngestPipelineError` (used when the API must not delete Mongo before Chroma succeeds).
    """
    doc_id = (document_id or "").strip()
    if not doc_id:
        return
    path = Path(chroma_dir).resolve()
    if not path.is_dir():
        return
    try:
        client = chromadb.PersistentClient(path=str(path))
        collection = client.get_collection(name=collection_name)
    except Exception:
        return
    wheres: list[dict] = [
        {"$or": [{"document_id": doc_id}, {"chunk_id_prefix": doc_id}]},
        {"document_id": doc_id},
        {"chunk_id_prefix": doc_id},
    ]
    last_err: BaseException | None = None
    for where in wheres:
        try:
            collection.delete(where=where)
            return
        except Exception as e:
            last_err = e
    if raise_on_delete_error and last_err is not None:
        raise IngestPipelineError(f"Chroma delete failed: {last_err}") from last_err


def ingest_pdf_bytes(
    *,
    pdf_bytes: bytes,
    filename: str,
    chunk_id_prefix: str,
    chroma_dir: Path,
    collection: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_OVERLAP,
    dimensions: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reset_collection: bool = False,
    openai_api_key: str | None = None,
    org_id: str | None = None,
    document_id: str | None = None,
) -> IngestResult:
    """
    Full pipeline: PDF bytes -> text -> chunks -> embeddings -> Chroma upsert.

    Chunk vector ids are ``{chunk_id_prefix}_{i}``.
    """
    if not filename.lower().endswith(".pdf"):
        raise IngestPipelineError("Expected a .pdf file")

    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise IngestPipelineError(
            "Invalid chunk_size / chunk_overlap (overlap must be < chunk_size)"
        )

    if not pdf_bytes:
        raise IngestPipelineError("Empty file")

    api_key = resolve_openai_api_key(openai_api_key)
    suffix = Path(filename).suffix or ".pdf"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(pdf_bytes)

        text = extract_text_from_pdf(tmp_path)
        if not text.strip():
            raise IngestPipelineError(
                "No extractable text (empty or image-only PDF)."
            )

        chunks = chunk_text(text, chunk_size, chunk_overlap)
        if not chunks:
            raise IngestPipelineError("Chunking produced no segments")

        client = OpenAI(api_key=api_key)
        embeddings = embed_openai(
            client,
            chunks,
            model=embedding_model,
            batch_size=batch_size,
            dimensions=dimensions,
        )

        n = embeddings.shape[0]
        ids = [f"{chunk_id_prefix}_{i}" for i in range(n)]
        metadatas = []
        for i in range(n):
            meta: dict = {
                "chunk_index": i,
                "chunk_id_prefix": chunk_id_prefix,
                "source_filename": filename,
                "embedding_model": embedding_model,
            }
            if org_id is not None:
                meta["org_id"] = org_id
            if document_id is not None:
                meta["document_id"] = document_id
            metadatas.append(meta)

        upsert_chroma(
            chroma_dir=chroma_dir,
            collection_name=collection,
            embeddings=embeddings,
            texts=chunks,
            ids=ids,
            metadatas=metadatas,
            model_name=embedding_model,
            reset_collection=reset_collection,
        )

        return IngestResult(
            chunk_id_prefix=chunk_id_prefix,
            filename=filename,
            characters_extracted=len(text),
            num_chunks=n,
            embedding_dim=int(embeddings.shape[1]),
            chroma_dir=str(chroma_dir.resolve()),
            collection=collection,
            embedding_model=embedding_model,
        )
    finally:
        if tmp_path is not None and tmp_path.is_file():
            try:
                tmp_path.unlink()
            except OSError:
                pass
