"""
Campaign service for handling campaign-related database operations.
"""
from typing import Optional, Dict, Any, List
from app.database import get_database
from app.models.schemas import CampaignCreate
import logging

logger = logging.getLogger(__name__)

def create_campaign(campaign_data: CampaignCreate) -> Dict[str, Any]:
    """
    Create a new campaign.
    
    Args:
        campaign_data: Campaign creation data
        
    Returns:
        Dict with status and message
    """
    try:
        db = get_database()
        campaign_table = db["Campaigns"]
        
        # Check if campaign already exists
        existing_campaign = campaign_table.find_one({"campaign_name": campaign_data.campaign_name})
        if existing_campaign:
            return {"status": "fail", "message": "Campaign with this name already exists"}
        
        campaign_doc = {
            "campaign_name": campaign_data.campaign_name
        }
        
        if campaign_data.org_id:
            campaign_doc["org_id"] = campaign_data.org_id
        if campaign_data.agent_type:
            campaign_doc["agent_type"] = campaign_data.agent_type
        if campaign_data.status:
            campaign_doc["status"] = campaign_data.status
        if campaign_data.campaign_information:
            campaign_doc["campaign_information"] = campaign_data.campaign_information
        
        campaign_table.insert_one(campaign_doc)
        logger.info(f"Campaign created successfully: {campaign_data.campaign_name}")
        return {"status": "success", "message": "Campaign created successfully"}
        
    except Exception as e:
        logger.error(f"Error creating campaign: {str(e)}")
        return {"status": "fail", "message": f"Error creating campaign: {str(e)}"}

def get_all_campaigns(org_id: str) -> List[Dict[str, Any]]:
    """
    Get all campaigns for a given org.
    
    Args:
        org_id: Organization ID
        
    Returns:
        List of campaign documents
    """
    try:
        db = get_database()
        campaign_table = db["Campaigns"]
        campaigns = list(campaign_table.find({"org_id": org_id}))
        return campaigns
    except Exception as e:
        logger.error(f"Error fetching campaigns: {str(e)}")
        return []

def get_campaign_by_name(campaign_name: str) -> Optional[Dict[str, Any]]:
    """
    Get campaign by name.
    
    Args:
        campaign_name: Campaign name
        
    Returns:
        Campaign document or None
    """
    try:
        db = get_database()
        campaign_table = db["Campaigns"]
        campaign = campaign_table.find_one({"campaign_name": campaign_name})
        return campaign
    except Exception as e:
        logger.error(f"Error fetching campaign: {str(e)}")
        return None

