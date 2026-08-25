"""
Audience API routes.
"""
from fastapi import APIRouter, HTTPException, status, Query, Depends
from app.models.schemas import AudienceCreate, AudienceResponse
from app.services import audience_service
from app.auth import get_current_user
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/audience", tags=["audience"])

@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_audience(
    audience_data: AudienceCreate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new audience entry (protected endpoint).
    """
    result = audience_service.create_audience(audience_data)
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result

@router.get("/{audience_name}", response_model=AudienceResponse)
async def get_audience(
    audience_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get audience by name (protected endpoint).
    """
    audience = audience_service.get_audience_by_name(audience_name)
    if not audience:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audience not found"
        )
    return audience

@router.get("", response_model=List[AudienceResponse])
async def get_all_audiences(
    phone_number: Optional[str] = Query(None, description="Filter by phone number"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all audiences, optionally filtered by phone number (protected endpoint).
    """
    audiences = audience_service.get_all_audiences(phone_number)
    return audiences

