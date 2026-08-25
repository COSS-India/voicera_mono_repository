"""
Phone number service for handling phone number-related database operations.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.database import get_database
from app.services import agent_service
import logging

logger = logging.getLogger(__name__)


def _last_link_fields(
    action: str, agent_type: Optional[str], member_email: Optional[str], at: str
) -> Dict[str, str]:
    """Optional audit fields for Numbers UI; omitted when member_email is missing."""
    if not member_email:
        return {}
    return {
        "last_link_action": action,
        "last_link_agent_type": agent_type or "",
        "last_link_by_email": member_email,
        "last_link_at": at,
    }


def get_all_phone_numbers_by_org(org_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all phone numbers for a given organization.
    
    Args:
        org_id: Organization ID
        
    Returns:
        List of phone number documents
    """
    try:
        db = get_database()
        phone_number_table = db["PhoneNumber"]
        phone_numbers = list(phone_number_table.find({"org_id": org_id}))
        return phone_numbers
    except Exception as e:
        logger.error(f"Error fetching phone numbers for org {org_id}: {str(e)}")
        return []

def attach_phone_number_to_agent(
    phone_number: str,
    provider: str,
    agent_type: Optional[str] = None,
    org_id: Optional[str] = None,
    member_email: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Attach a phone number to an agent by agent_type.
    
    Args:
        phone_number: Phone number to attach
        provider: Phone number provider (required)
        agent_type: Agent type identifier (optional)
        org_id: Organization ID (optional, will be fetched from agent if agent_type provided)
        
    Returns:
        Dict with status and message
    """
    try:
        db = get_database()
        phone_number_table = db["PhoneNumber"]
        
        # If agent_type is provided, fetch agent config to get org_id
        if agent_type:
            agent = agent_service.fetch_agent_config(agent_type)
            if not agent:
                return {"status": "fail", "message": "Agent type not found"}
            
            fetched_org_id = agent.get("org_id")
            if not fetched_org_id:
                return {"status": "fail", "message": "Agent does not have an org_id"}
            
            # Use org_id from agent if not provided
            if not org_id:
                org_id = fetched_org_id
            
            # Update agent config with phone number
            agent_table = db["AgentConfig"]
            agent_table.update_one(
                {"agent_type": agent_type},
                {"$set": {"phone_number": phone_number, "updated_at": datetime.now().isoformat()}}
            )
        elif not org_id:
            return {"status": "fail", "message": "Either agent_type or org_id must be provided"}
        
        # Check if phone number already exists
        existing_phone = phone_number_table.find_one({"phone_number": phone_number})
        current_time = datetime.now().isoformat()
        
        if existing_phone:
            # Update existing phone number
            update_doc = {
                "provider": provider,
                "updated_at": current_time,
                **_last_link_fields("attached", agent_type, member_email, current_time),
            }
            
            if agent_type:
                update_doc["agent_type"] = agent_type
            if org_id:
                update_doc["org_id"] = org_id
            
            result = phone_number_table.update_one(
                {"phone_number": phone_number},
                {"$set": update_doc}
            )
            
            if result.modified_count > 0:
                logger.info(f"Phone number {phone_number} updated")
                return {"status": "success", "message": "Phone number updated successfully"}
            else:
                logger.info(f"Phone number {phone_number} already has the same configuration")
                return {"status": "success", "message": "Phone number already configured"}
        else:
            # Create new phone number record
            phone_doc = {
                "phone_number": phone_number,
                "provider": provider,
                "created_at": current_time,
                "updated_at": current_time,
                **_last_link_fields("attached", agent_type, member_email, current_time),
            }
            
            if agent_type:
                phone_doc["agent_type"] = agent_type
            if org_id:
                phone_doc["org_id"] = org_id
            
            phone_number_table.insert_one(phone_doc)
            logger.info(f"Phone number {phone_number} created with provider {provider}")
            return {"status": "success", "message": "Phone number created successfully"}
        
    except Exception as e:
        logger.error(f"Error attaching phone number to agent: {str(e)}")
        return {"status": "fail", "message": f"Error attaching phone number: {str(e)}"}

def get_phone_number_by_agent_type(agent_type: str, org_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the phone number attached to an agent by agent_type.
    
    Args:
        agent_type: Agent type identifier
        org_id: Organization ID for validation
        
    Returns:
        Phone number document if found, None otherwise
    """
    try:
        db = get_database()
        phone_number_table = db["PhoneNumber"]
        
        phone_number_doc = phone_number_table.find_one({
            "agent_type": agent_type,
            "org_id": org_id
        })
        
        return phone_number_doc
        
    except Exception as e:
        logger.error(f"Error fetching phone number for agent {agent_type}: {str(e)}")
        return None

def detach_phone_number(
    phone_number: str, org_id: str, member_email: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detach a phone number from its agent by removing the agent_type.
    
    Args:
        phone_number: Phone number to detach
        org_id: Organization ID for validation
        
    Returns:
        Dict with status and message
    """
    try:
        db = get_database()
        phone_number_table = db["PhoneNumber"]
        
        # Check if phone number exists and belongs to the org
        existing_phone = phone_number_table.find_one({"phone_number": phone_number})
        if not existing_phone:
            return {"status": "fail", "message": "Phone number not found"}
        
        if existing_phone.get("org_id") != org_id:
            return {"status": "fail", "message": "Not authorized to detach this phone number"}
        
        if not existing_phone.get("agent_type"):
            return {"status": "fail", "message": "Phone number is not attached to any agent"}
        
        agent_type = existing_phone.get("agent_type")
        
        # Remove agent_type association
        current_time = datetime.now().isoformat()
        set_doc = {
            "updated_at": current_time,
            **_last_link_fields("detached", agent_type, member_email, current_time),
        }
        result = phone_number_table.update_one(
            {"phone_number": phone_number},
            {"$unset": {"agent_type": ""}, "$set": set_doc},
        )
        
        # Remove phone number from agent config
        if agent_type:
            agent_table = db["AgentConfig"]
            agent_table.update_one(
                {"agent_type": agent_type},
                {"$unset": {"phone_number": ""}, "$set": {"updated_at": current_time}}
            )
        
        if result.modified_count > 0:
            logger.info(f"Phone number {phone_number} detached from agent")
            return {"status": "success", "message": "Phone number detached successfully"}
        else:
            return {"status": "fail", "message": "Failed to detach phone number"}
        
    except Exception as e:
        logger.error(f"Error detaching phone number: {str(e)}")
        return {"status": "fail", "message": f"Error detaching phone number: {str(e)}"}
