"""
Batch CSV API routes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile, status
import requests

from app.auth import get_current_user, verify_api_key
from app.config import settings
from app.models.schemas import (
    BatchDeleteResponse,
    BatchResponse,
    BatchRunRequest,
    BatchScheduleRequest,
    BatchUploadResponse,
)
from app.services import batch_service

router = APIRouter(prefix="/batches", tags=["batches"])


def _voice_server_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.INTERNAL_API_KEY:
        headers["X-API-Key"] = settings.INTERNAL_API_KEY
    return headers


@router.get("", response_model=List[BatchResponse])
async def get_batches(
    agent_type: Optional[str] = Query(default=None),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List immutable batch uploads for the authenticated org."""
    org_id = current_user.get("org_id")
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has no organization",
        )
    return batch_service.list_batches(org_id=org_id, agent_type=agent_type)


@router.post("/upload", response_model=BatchUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_batch_csv(
    file: UploadFile = File(...),
    org_id: str = Form(...),
    batch_name: str = Form(...),
    agent_type: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> BatchUploadResponse:
    """
    Upload immutable CSV batch for an org+agent_type and persist parsed contacts.
    """
    user_org_id = current_user.get("org_id")
    if org_id != user_org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized for this organization",
        )

    normalized_agent_type = agent_type.strip()
    if not normalized_agent_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="agent_type is required",
        )

    if not batch_service.validate_agent_for_org(org_id, normalized_agent_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid agent_type for this organization",
        )

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are allowed",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )

    try:
        batch_doc = batch_service.create_batch_from_csv(
            org_id=org_id,
            batch_name=batch_name,
            agent_type=normalized_agent_type,
            original_filename=file.filename,
            csv_bytes=content,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process batch CSV: {str(e)}",
        ) from e

    return BatchUploadResponse(**batch_doc)


@router.delete("/{batch_id}", response_model=BatchDeleteResponse, status_code=status.HTTP_200_OK)
async def delete_batch(
    batch_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> BatchDeleteResponse:
    """Delete one batch for the authenticated organization."""
    org_id = current_user.get("org_id")
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has no organization",
        )

    try:
        batch_service.delete_batch(org_id=org_id, batch_id=batch_id)
    except batch_service.BatchNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found",
        ) from None

    return BatchDeleteResponse(deleted=True)


@router.post("/{batch_id}/run", status_code=status.HTTP_200_OK)
async def run_batch(
    batch_id: str,
    run_request: BatchRunRequest = Body(default=BatchRunRequest()),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Start batch execution worker with bounded concurrency."""
    org_id = current_user.get("org_id")
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has no organization",
        )

    try:
        result = batch_service.run_batch(
            org_id=org_id,
            batch_id=batch_id,
            agent_type=run_request.agent_type,
            concurrency=run_request.concurrency,
        )
        response = requests.post(
            f"{settings.VOICE_SERVER_URL}/outbound/batch/run/",
            json={
                "org_id": org_id,
                "batch_id": batch_id,
                "agent_type": result.get("agent_type"),
                "concurrency": result.get("concurrency"),
            },
            headers=_voice_server_headers(),
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json() if response.text else {}
        return payload if isinstance(payload, dict) else result
    except batch_service.BatchNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found",
        ) from None
    except batch_service.BatchRunStateError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except requests.RequestException as e:
        batch_service.mark_batch_start_failure(
            org_id=org_id,
            batch_id=batch_id,
            error_message=f"Failed to start batch on voice server: {str(e)}",
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to start batch on voice server: {str(e)}",
        ) from e


@router.post("/{batch_id}/schedule", status_code=status.HTTP_200_OK)
async def schedule_batch(
    batch_id: str,
    schedule_request: BatchScheduleRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    org_id = current_user.get("org_id")
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has no organization",
        )

    try:
        return batch_service.schedule_batch(
            org_id=org_id,
            batch_id=batch_id,
            scheduled_at_local=schedule_request.scheduled_at_local,
            timezone_name=schedule_request.timezone,
            scheduled_by=current_user.get("email"),
            agent_type=schedule_request.agent_type,
            concurrency=schedule_request.concurrency,
        )
    except batch_service.BatchNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found") from None
    except (batch_service.BatchRunStateError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/{batch_id}/schedule/cancel", status_code=status.HTTP_200_OK)
async def cancel_batch_schedule(
    batch_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    org_id = current_user.get("org_id")
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has no organization",
        )

    try:
        return batch_service.cancel_scheduled_batch(org_id=org_id, batch_id=batch_id)
    except batch_service.BatchNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found") from None
    except batch_service.BatchRunStateError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/{batch_id}/schedule/reschedule", status_code=status.HTTP_200_OK)
async def reschedule_batch(
    batch_id: str,
    schedule_request: BatchScheduleRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    org_id = current_user.get("org_id")
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has no organization",
        )

    try:
        return batch_service.reschedule_batch(
            org_id=org_id,
            batch_id=batch_id,
            scheduled_at_local=schedule_request.scheduled_at_local,
            timezone_name=schedule_request.timezone,
            scheduled_by=current_user.get("email"),
            agent_type=schedule_request.agent_type,
            concurrency=schedule_request.concurrency,
        )
    except batch_service.BatchNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found") from None
    except (batch_service.BatchRunStateError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/{batch_id}/stop", status_code=status.HTTP_200_OK)
async def stop_batch(
    batch_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Request batch execution worker to stop scheduling new calls."""
    org_id = current_user.get("org_id")
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has no organization",
        )

    try:
        batch_service.stop_batch(org_id=org_id, batch_id=batch_id)
        response = requests.post(
            f"{settings.VOICE_SERVER_URL}/outbound/batch/stop/",
            json={"org_id": org_id, "batch_id": batch_id},
            headers=_voice_server_headers(),
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json() if response.text else {}
        return payload if isinstance(payload, dict) else {"status": "success", "message": "Batch stop requested"}
    except batch_service.BatchNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found",
        ) from None
    except batch_service.BatchRunStateError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except requests.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to forward stop to voice server: {str(e)}",
        ) from e


@router.post("/worker/claim-next", status_code=status.HTTP_200_OK)
async def worker_claim_next_contact(
    payload: Dict[str, Any] = Body(default={}),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    org_id = str(payload.get("org_id") or "").strip()
    batch_id = str(payload.get("batch_id") or "").strip()
    if not org_id or not batch_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="org_id and batch_id are required")

    contact = batch_service.claim_next_contact_for_execution(org_id=org_id, batch_id=batch_id)
    if not contact:
        return {"contact": None}
    return {"contact": contact}


@router.post("/worker/agent-config", status_code=status.HTTP_200_OK)
async def worker_get_agent_call_config(
    payload: Dict[str, Any] = Body(default={}),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    org_id = str(payload.get("org_id") or "").strip()
    agent_type = str(payload.get("agent_type") or "").strip()
    if not org_id or not agent_type:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="org_id and agent_type are required")

    try:
        config = batch_service.get_agent_call_config_for_batch(org_id=org_id, agent_type=agent_type)
        return config
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/worker/report", status_code=status.HTTP_200_OK)
async def worker_report_contact_result(
    payload: Dict[str, Any] = Body(default={}),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    org_id = str(payload.get("org_id") or "").strip()
    batch_id = str(payload.get("batch_id") or "").strip()
    row_number = payload.get("row_number")
    ok = bool(payload.get("ok"))
    error = payload.get("error")

    if not org_id or not batch_id or row_number is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="org_id, batch_id and row_number are required",
        )

    batch_service.report_contact_execution_result(
        org_id=org_id,
        batch_id=batch_id,
        row_number=int(row_number),
        ok=ok,
        error=str(error) if error else None,
    )
    return {"updated": True}


@router.post("/worker/finalize", status_code=status.HTTP_200_OK)
async def worker_finalize_batch(
    payload: Dict[str, Any] = Body(default={}),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    org_id = str(payload.get("org_id") or "").strip()
    batch_id = str(payload.get("batch_id") or "").strip()
    stopped = bool(payload.get("stopped", False))
    if not org_id or not batch_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="org_id and batch_id are required",
        )
    return batch_service.finalize_batch_execution(
        org_id=org_id,
        batch_id=batch_id,
        stopped=stopped,
    )
