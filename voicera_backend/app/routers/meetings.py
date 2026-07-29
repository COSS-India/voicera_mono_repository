"""
Meeting API routes.
"""
from fastapi import APIRouter, HTTPException, status, Depends, Query, Response
from fastapi.responses import StreamingResponse
from app.models.schemas import (
    MeetingCreate,
    MeetingResponse,
    MeetingUpdate,
    PaginatedMeetingsResponse,
    MeetingFilterOptionsResponse,
)
from app.services import meeting_service
from app.auth import get_current_user, verify_api_key
from app.storage.minio_client import MinIOStorage
from app.config import settings
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/meetings", tags=["meetings"])


# ============================================================================
# Bot Endpoints (API Key Authentication)
# ============================================================================

@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_meeting(
    meeting_data: MeetingCreate,
    _: bool = Depends(verify_api_key)
):
    """
    Create a new meeting when call starts.
    
    This endpoint is called by the voice bot when a call begins.
    Requires X-API-Key header for authentication.
    """
    result = meeting_service.setup_meeting_id(meeting_data)
    if isinstance(result, dict) and result.get("status") == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    
    return result


@router.patch("/{meeting_id}", response_model=Dict[str, Any])
async def update_meeting(
    meeting_id: str,
    update_data: MeetingUpdate,
    _: bool = Depends(verify_api_key)
):
    """
    Update a meeting when call ends.
    
    This endpoint is called by the voice bot when a call ends.
    Requires X-API-Key header for authentication.
    """
    result = meeting_service.update_meeting_end_time(meeting_id, update_data.end_time_utc)
    if isinstance(result, dict) and result.get("status") == "fail":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["message"]
        )
    
    return result


# ============================================================================
# Frontend Endpoints (User JWT Authentication)
# ============================================================================

@router.get("/filter-options", response_model=MeetingFilterOptionsResponse)
async def get_meeting_filter_options(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Distinct agent types and phone numbers for History filter dropdowns."""
    org_id = current_user["org_id"]
    options = meeting_service.fetch_meeting_filter_options(org_id)
    return options


@router.get("", response_model=PaginatedMeetingsResponse)
async def get_meetings(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    limit: int = Query(50, ge=1, le=10000, description="Page size (max 50; higher when for_export)"),
    for_export: bool = Query(False, description="Allow large limit for export"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    from_number: Optional[str] = Query(None),
    to_number: Optional[str] = Query(None),
    inbound: Optional[bool] = Query(None, description="True=inbound, False=outbound"),
    call_status: Optional[str] = Query(None, description="Busy, Completed, or In Progress"),
    date_from: Optional[str] = Query(None, description="ISO date start (inclusive)"),
    date_to: Optional[str] = Query(None, description="ISO date end (inclusive)"),
    date_sort_order: Optional[str] = Query("latest", description="latest or oldest"),
    duration_sort_order: Optional[str] = Query(
        None, description="longest or shortest; overrides date sort when set"
    ),
    has_latency_metrics: Optional[bool] = Query(
        None, description="When true, only return meetings with latency turn data"
    ),
    search: Optional[str] = Query(
        None, description="Search meeting id, agent, phone numbers, or call direction"
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get paginated meetings for the current user's organization.
    """
    org_id = current_user["org_id"]
    effective_limit = limit if for_export else min(limit, 50)

    result = meeting_service.fetch_meetings_paginated(
        org_id=org_id,
        page=page,
        limit=effective_limit,
        agent_type=agent_type,
        from_number=from_number,
        to_number=to_number,
        inbound=inbound,
        call_status=call_status,
        date_from=date_from,
        date_to=date_to,
        date_sort_order=date_sort_order or "latest",
        duration_sort_order=duration_sort_order,
        has_latency_metrics=has_latency_metrics,
        search=search,
    )
    return result


@router.get("/{meeting_id}", response_model=MeetingResponse)
async def get_meeting(
    meeting_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get meeting details by meeting_id.
    
    Only returns the meeting if it belongs to the user's organization.
    """
    meeting = meeting_service.fetch_meeting_details(meeting_id)
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meeting not found"
        )
    
    # Verify meeting belongs to user's org
    if meeting.get("org_id") != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this meeting"
        )
    
    # Transform meeting data (convert minio:// URLs, parse transcript)
    return meeting_service.transform_meeting_for_frontend(meeting)


@router.get(
    "/{meeting_id}/recording",
    responses={
        200: {
            "content": {
                "audio/wav": {},
                "audio/mpeg": {},
                "audio/mp4": {},
            },
            "description": "Audio file stream",
        },
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Meeting or recording not found"},
    },
)
async def get_meeting_recording(
    meeting_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Stream audio recording for a meeting.
    
    Only returns the recording if the meeting belongs to the user's organization.
    Supports minio:// URLs by proxying from MinIO storage.
    
    Returns the audio file as a stream (WAV, MP3, or M4A format).
    In Swagger UI, click "Try it out" → "Execute", then click the response URL to play/download the audio.
    """
    # Verify meeting exists and belongs to user's org
    meeting = meeting_service.fetch_meeting_details(meeting_id)
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meeting not found"
        )
    
    if meeting.get("org_id") != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this meeting"
        )
    
    # Get recording URL
    recording_url = meeting.get("recording_url")
    if not recording_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found for this meeting"
        )
    
    # If it's already an HTTP URL, redirect or proxy it
    if recording_url.startswith("http://") or recording_url.startswith("https://"):
        # For now, return the URL - could proxy it here if needed
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Direct HTTP URLs not supported. Use minio:// URLs."
        )
    
    # Parse minio:// URL
    storage = MinIOStorage()
    parsed = storage.parse_minio_url(recording_url)
    if not parsed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid recording URL format: {recording_url}"
        )
    
    bucket_name, object_name = parsed
    
    # Check if object exists
    if not storage.object_exists(bucket_name, object_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording file not found: {object_name}"
        )
    
    try:
        # Get object from MinIO and stream it
        response = await storage.get_object(bucket_name, object_name)
        
        # Get content length from response headers if available
        content_length = response.headers.get("content-length")
        
        def audio_stream():
            try:
                # Stream chunks from MinIO response
                for chunk in response.stream(32 * 1024):  # 32KB chunks
                    yield chunk
            finally:
                response.close()
                response.release_conn()
        
        # Determine content type from file extension
        content_type = "audio/wav"
        if object_name.endswith(".mp3"):
            content_type = "audio/mpeg"
        elif object_name.endswith(".m4a"):
            content_type = "audio/mp4"
        
        # Build headers
        headers = {
            "Content-Disposition": f'inline; filename="{object_name}"',
            "Accept-Ranges": "bytes",
        }
        if content_length:
            headers["Content-Length"] = content_length
        
        return StreamingResponse(
            audio_stream(),
            media_type=content_type,
            headers=headers
        )
    except Exception as e:
        logger.error(f"Error streaming recording {meeting_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error streaming recording: {str(e)}"
        )