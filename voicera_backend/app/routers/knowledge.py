"""
Knowledge base API: org-scoped PDF upload and ingest status.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status

from app.auth import get_current_user
from app.models.schemas import (
    KnowledgeDeleteResponse,
    KnowledgeDocumentResponse,
    KnowledgeUploadResponse,
)
from app.services import knowledge_service

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("", response_model=List[KnowledgeDocumentResponse])
async def list_knowledge_documents(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List all knowledge PDFs for the authenticated user's organization."""
    org_id = current_user["org_id"]
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has no organization",
        )
    rows = knowledge_service.list_documents(org_id)
    return rows


@router.post("/upload", response_model=KnowledgeUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_knowledge_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    org_id: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> KnowledgeUploadResponse:
    """
    Upload a PDF and schedule background ingest into org-scoped Chroma.
    Returns immediately with status processing.
    """
    if org_id != current_user.get("org_id"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized for this organization",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )
    if len(content) > knowledge_service.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (max {knowledge_service.MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
        )

    document_id = knowledge_service.create_document_pending(org_id, file.filename)
    background_tasks.add_task(
        knowledge_service.run_ingest_job,
        document_id,
        org_id,
        file.filename,
        content,
    )

    return KnowledgeUploadResponse(
        document_id=document_id,
        org_id=org_id,
        original_filename=file.filename,
        status="processing",
    )


@router.delete(
    "/{document_id}",
    response_model=KnowledgeDeleteResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_knowledge_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> KnowledgeDeleteResponse:
    """Delete a knowledge document and its Chroma vectors for the user's organization."""
    org_id = current_user.get("org_id")
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has no organization",
        )
    try:
        knowledge_service.delete_knowledge_document(org_id, document_id)
    except knowledge_service.KnowledgeDocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        ) from None
    except knowledge_service.KnowledgeChromaDeleteError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        ) from e
    return KnowledgeDeleteResponse(deleted=True)
