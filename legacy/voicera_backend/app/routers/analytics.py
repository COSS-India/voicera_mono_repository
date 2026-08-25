"""
Analytics API routes.
"""
from fastapi import APIRouter, HTTPException, status, Depends, Query
from app.models.schemas import AnalyticsResponse
from app.services import analytics_service
from app.auth import get_current_user
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("", response_model=AnalyticsResponse)
async def get_analytics(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    phone_number: Optional[str] = Query(None, description="Filter by phone number"),
    start_date: Optional[str] = Query(None, description="Start date in ISO format (YYYY-MM-DD or ISO datetime)"),
    end_date: Optional[str] = Query(None, description="End date in ISO format (YYYY-MM-DD or ISO datetime)"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get analytics metrics for the current user's organization.
    
    Calculates metrics on-demand from CallLogs collection:
    - Calls Attempted: Total number of call records
    - Calls Connected: Number of calls that successfully connected
    - Average Call Duration: Average duration of connected calls (in minutes)
    - Total Minutes Connected: Sum of all connected call durations (in minutes)
    - Most Used Agent: Agent type with the highest call count
    
    The org_id is automatically extracted from the JWT token.
    
    Optional query parameters:
    - agent_type: Filter by specific agent type
    - phone_number: Filter by specific phone number
    - start_date: Filter by start date (ISO format)
    - end_date: Filter by end date (ISO format)
    
    Examples:
        - GET /analytics - Get all analytics for your org
        - GET /analytics?agent_type=sales - Get analytics for sales agent only
        - GET /analytics?start_date=2024-01-01&end_date=2024-01-31 - Get analytics for January 2024
    """
    try:
        org_id = current_user["org_id"]
        
        # If date range is provided, use date range function
        if start_date or end_date:
            analytics = analytics_service.get_analytics_by_date_range(
                org_id=org_id,
                start_date=start_date,
                end_date=end_date,
                agent_type=agent_type
            )
        else:
            # Otherwise use standard analytics function
            analytics = analytics_service.get_analytics(
                org_id=org_id,
                agent_type=agent_type,
                phone_number=phone_number
            )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error fetching analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating analytics: {str(e)}"
        )
