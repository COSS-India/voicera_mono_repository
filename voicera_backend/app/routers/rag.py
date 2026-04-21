"""
RAG retrieval API for runtime grounding.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import verify_api_key
from app.models.schemas import KnowledgeRetrieveRequest, KnowledgeRetrieveResponse
from app.services import knowledge_service

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/retrieve", response_model=KnowledgeRetrieveResponse)
async def retrieve_knowledge_chunks(
    payload: KnowledgeRetrieveRequest,
    _: bool = Depends(verify_api_key),
) -> KnowledgeRetrieveResponse:
    """Retrieve top-k relevant chunks from org-scoped Knowledge Base."""
    try:
        chunks = knowledge_service.retrieve_chunks_for_query(
            org_id=payload.org_id,
            question=payload.question,
            document_ids=payload.document_ids,
            top_k=payload.top_k,
        )
        return KnowledgeRetrieveResponse(chunks=chunks)
    except knowledge_service.KnowledgeRetrievalError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        ) from e
