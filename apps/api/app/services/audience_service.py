"""
Audience service for handling audience-related database operations.
"""
from typing import Optional, Dict, Any, List
from app.database import get_database
from app.models.schemas import AudienceCreate
import logging

logger = logging.getLogger(__name__)

def create_audience(audience_data: AudienceCreate) -> Dict[str, Any]:
    """
    Create a new audience entry.
    
    Args:
        audience_data: Audience creation data
        
    Returns:
        Dict with status and message
    """
    try:
        db = get_database()
        audience_table = db["Audience"]
        
        # Check if audience already exists
        existing_audience = audience_table.find_one({"audience_name": audience_data.audience_name})
        if existing_audience:
            return {"status": "fail", "message": "Audience with this name already exists"}
        
        audience_doc = {
            "audience_name": audience_data.audience_name,
            "phone_number": audience_data.phone_number
        }
        
        if audience_data.parameters:
            audience_doc["parameters"] = audience_data.parameters
        
        audience_table.insert_one(audience_doc)
        logger.info(f"Audience created successfully: {audience_data.audience_name}")
        return {"status": "success", "message": "Audience created successfully"}
        
    except Exception as e:
        logger.error(f"Error creating audience: {str(e)}")
        return {"status": "fail", "message": f"Error creating audience: {str(e)}"}

def get_audience_by_name(audience_name: str) -> Optional[Dict[str, Any]]:
    """
    Get audience by name.
    
    Args:
        audience_name: Audience name
        
    Returns:
        Audience document or None
    """
    try:
        db = get_database()
        audience_table = db["Audience"]
        audience = audience_table.find_one({"audience_name": audience_name})
        return audience
    except Exception as e:
        logger.error(f"Error fetching audience: {str(e)}")
        return None

def get_all_audiences(phone_number: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get all audiences, optionally filtered by phone number.
    
    Args:
        phone_number: Optional phone number filter
        
    Returns:
        List of audience documents
    """
    try:
        db = get_database()
        audience_table = db["Audience"]
        
        if phone_number:
            audiences = list(audience_table.find({"phone_number": phone_number}))
        else:
            audiences = list(audience_table.find({}))
        
        return audiences
    except Exception as e:
        logger.error(f"Error fetching audiences: {str(e)}")
        return []

