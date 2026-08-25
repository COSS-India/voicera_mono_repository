"""
Plivo API routes.
"""
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status, Depends

from app.auth import get_current_user
from app.models.schemas import (
    PlivoApplicationCreate,
    PlivoApplicationResponse,
    PlivoNumberLink,
    PlivoNumberUnlink,
)
from app.services import plivo

router = APIRouter(prefix="/plivo", tags=["plivo"])


@router.post("/application", response_model=PlivoApplicationResponse, status_code=status.HTTP_201_CREATED)
async def create_plivo_application_endpoint(
    request: PlivoApplicationCreate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    try:
        result = await plivo.create_plivo_application(
            current_user["org_id"],
            request.agent_type,
            request.answer_url,
        )
        if result["status"] == "fail":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating Plivo application: {str(e)}"
        )
    return result


@router.get("/numbers", response_model=Dict[str, Any])
async def get_plivo_numbers(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    result = await plivo.get_plivo_numbers(current_user["org_id"])
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Failed to fetch Plivo numbers"),
        )
    return {"status": "success", "numbers": result.get("numbers", [])}


@router.delete("/application/{application_id}", response_model=Dict[str, Any])
async def delete_plivo_application_endpoint(
    application_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    try:
        result = await plivo.delete_plivo_application(current_user["org_id"], application_id)
        if result["status"] == "fail":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting Plivo application: {str(e)}"
        )
    return result


@router.post("/numbers/link", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def link_number_to_application_endpoint(
    request: PlivoNumberLink,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    try:
        result = await plivo.link_number_to_application(
            current_user["org_id"],
            request.phone_number,
            request.application_id,
        )
        if result["status"] == "fail":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error linking phone number to application: {str(e)}"
        )
    return result


@router.delete("/numbers/unlink", response_model=Dict[str, Any])
async def unlink_number_from_application_endpoint(
    request: PlivoNumberUnlink,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    try:
        result = await plivo.unlink_number_from_application(
            current_user["org_id"],
            request.phone_number,
        )
        if result["status"] == "fail":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error unlinking phone number from application: {str(e)}"
        )
    return result
