"""
Call recording API routes.
"""
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import CallRecordingCreate
from app.services import call_recording_service
from typing import Dict, Any

router = APIRouter(prefix="/call-recordings", tags=["call-recordings"])

@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def save_call_recording(recording_data: CallRecordingCreate):
    """
    Save or update call recording data.
    
    This endpoint is called by the voice server after a call completes.
    It updates the meeting record with recording URLs, transcript, and call metadata.
    
    Note: This endpoint is currently unauthenticated for service-to-service calls.
    Consider adding API key authentication or service token in production.
    
    Args:
        recording_data: Call recording data including URLs and transcript
        
    Returns:
        Updated meeting document
    """
    result = call_recording_service.save_call_recording(recording_data)
    
    if isinstance(result, dict) and result.get("status") == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Failed to save call recording")
        )
    
    return result
