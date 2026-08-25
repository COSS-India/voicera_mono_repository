"""
Campaign API routes.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from app.models.schemas import CampaignCreate, CampaignResponse
from app.services import campaign_service
from app.auth import get_current_user
from typing import List, Dict, Any

router = APIRouter(prefix="/campaigns", tags=["campaigns"])

@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_campaign(
    campaign_data: CampaignCreate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new campaign (protected endpoint).
    """
    # Set org_id from current user if not provided
    if not campaign_data.org_id:
        campaign_data.org_id = current_user["org_id"]
    
    # Ensure user can only create campaigns for their own org
    if campaign_data.org_id != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to create campaigns for this organization"
        )
    
    result = campaign_service.create_campaign(campaign_data)
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result

@router.get("/org/{org_id}", response_model=List[CampaignResponse])
async def get_campaigns_by_org(
    org_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all campaigns for a given organization (protected endpoint).
    """
    # Ensure user can only access their own org's campaigns
    if org_id != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this organization's campaigns"
        )
    
    campaigns = campaign_service.get_all_campaigns(org_id)
    return campaigns

@router.get("/{campaign_name}", response_model=CampaignResponse)
async def get_campaign(
    campaign_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get campaign by name (protected endpoint).
    """
    campaign = campaign_service.get_campaign_by_name(campaign_name)
    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Campaign not found"
        )
    
    # Ensure user can only access campaigns from their own org
    if campaign.get("org_id") != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this campaign"
        )
    
    return campaign

