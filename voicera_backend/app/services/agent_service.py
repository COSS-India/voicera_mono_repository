"""
Agent service for handling agent-related database operations.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.database import get_database
from app.models.schemas import AgentConfigCreate, AgentConfigUpdate
import logging
import string

logger = logging.getLogger(__name__)

def create_agent(agent_data: AgentConfigCreate) -> Dict[str, Any]:
    """
    Create a new agent type for a given org.
    
    Args:
        agent_data: Agent creation data
        
    Returns:
        Dict with status and message
    """
    try:
        db = get_database()
        agent_table = db["AgentConfig"]
        
        # Check if agent_type already exists for this organization
        existing_agent = agent_table.find_one({
            "agent_type": agent_data.agent_type,
            "org_id": agent_data.org_id
        })
        if existing_agent:
            return {"status": "fail", "message": "Agent type already exists for this organization"}
        
        # Check if agent_id already exists for this organization
        existing_agent_by_id = agent_table.find_one({
            "agent_id": agent_data.agent_id,
            "org_id": agent_data.org_id
        })
        if existing_agent_by_id:
            return {"status": "fail", "message": "Agent ID already exists for this organization"}
        
        agent_doc = {
            "agent_type": agent_data.agent_type,
            "agent_id": agent_data.agent_id,
            "agent_config": agent_data.agent_config,
            "org_id": agent_data.org_id,
            "updated_at": datetime.now().isoformat()
        }
        
        if agent_data.agent_category:
            agent_doc["agent_category"] = agent_data.agent_category
        if agent_data.phone_number:
            agent_doc["phone_number"] = agent_data.phone_number
        if agent_data.app_id:
            agent_doc["app_id"] = agent_data.app_id
        if agent_data.telephony_provider:
            agent_doc["telephony_provider"] = agent_data.telephony_provider
        if agent_data.greeting_message:
            # Remove punctuation from greeting message
            greeting_message = agent_data.greeting_message.translate(
                str.maketrans('', '', string.punctuation)
            )
            agent_doc["agent_config"]["greeting_message"] = greeting_message
        if agent_data.vobiz_app_id:
            agent_doc["vobiz_app_id"] = agent_data.vobiz_app_id
        if agent_data.vobiz_answer_url:
            agent_doc["vobiz_answer_url"] = agent_data.vobiz_answer_url
        if agent_data.plivo_app_id:
            agent_doc["plivo_app_id"] = agent_data.plivo_app_id
        if agent_data.plivo_answer_url:
            agent_doc["plivo_answer_url"] = agent_data.plivo_answer_url
        
        agent_table.insert_one(agent_doc)
        logger.info(f"Agent created successfully: {agent_data.agent_type}")
        return {"status": "success", "message": "Agent type created successfully"}
        
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        return {"status": "fail", "message": f"Error creating agent type: {str(e)}"}

def fetch_agent_config(agent_type: str) -> Optional[Dict[str, Any]]:
    """
    Fetch agent config for a given agent type.
    
    Args:
        agent_type: Agent type identifier
        
    Returns:
        Agent config document or None
    """
    try:
        db = get_database()
        agent_table = db["AgentConfig"]
        agent = agent_table.find_one({"agent_type": agent_type})
        return agent
    except Exception as e:
        logger.error(f"Error fetching agent config: {str(e)}")
        return None

def fetch_agent_config_by_id(agent_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch agent config for a given agent ID.
    
    Args:
        agent_id: Agent ID identifier
        
    Returns:
        Agent config document or None
    """
    try:
        db = get_database()
        agent_table = db["AgentConfig"]
        agent = agent_table.find_one({"agent_id": agent_id})
        return agent
    except Exception as e:
        logger.error(f"Error fetching agent config by ID: {str(e)}")
        return None

def fetch_agents_of_org(org_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all agents for a given org.
    
    Args:
        org_id: Organization ID
        
    Returns:
        List of agent documents
    """
    try:
        db = get_database()
        agent_table = db["AgentConfig"]
        agents = list(agent_table.find({"org_id": org_id}))
        return agents
    except Exception as e:
        logger.error(f"Error fetching agents: {str(e)}")
        return []

def update_agent_config(agent_type: str, agent_data: AgentConfigUpdate) -> Dict[str, Any]:
    """
    Update agent config.
    
    Args:
        agent_type: Agent type identifier
        agent_data: Updated agent data
        
    Returns:
        Dict with status and message
    """
    try:
        db = get_database()
        agent_table = db["AgentConfig"]
        
        update_doc = {
            "agent_config": agent_data.agent_config,
            "updated_at": datetime.now().isoformat()
        }
        
        if agent_data.agent_category:
            update_doc["agent_category"] = agent_data.agent_category
        if agent_data.phone_number:
            update_doc["phone_number"] = agent_data.phone_number
        if agent_data.app_id:
            update_doc["app_id"] = agent_data.app_id
        if agent_data.telephony_provider:
            update_doc["telephony_provider"] = agent_data.telephony_provider
        if agent_data.greeting_message:
            greeting_message = agent_data.greeting_message.translate(
                str.maketrans('', '', string.punctuation)
            )
            update_doc["agent_config"]["greeting_message"] = greeting_message
        if agent_data.vobiz_app_id:
            update_doc["vobiz_app_id"] = agent_data.vobiz_app_id
        if agent_data.vobiz_answer_url:
            update_doc["vobiz_answer_url"] = agent_data.vobiz_answer_url
        if agent_data.plivo_app_id:
            update_doc["plivo_app_id"] = agent_data.plivo_app_id
        if agent_data.plivo_answer_url:
            update_doc["plivo_answer_url"] = agent_data.plivo_answer_url
        
        result = agent_table.update_one(
            {"agent_type": agent_type},
            {"$set": update_doc}
        )
        
        if result.matched_count == 0:
            return {"status": "fail", "message": "Agent type not found"}
        
        logger.info(f"Agent updated successfully: {agent_type}")
        return {"status": "success", "message": "Agent config updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating agent: {str(e)}")
        return {"status": "fail", "message": f"Error updating agent: {str(e)}"}

def delete_agent(agent_type: str) -> Dict[str, Any]:
    """
    Delete an agent by agent_type.
    
    Args:
        agent_type: Agent type identifier
        
    Returns:
        Dict with status and message
    """
    try:
        db = get_database()
        agent_table = db["AgentConfig"]
        
        result = agent_table.delete_one({"agent_type": agent_type})
        
        if result.deleted_count == 0:
            return {"status": "fail", "message": "Agent type not found"}
        
        logger.info(f"Agent deleted successfully: {agent_type}")
        return {"status": "success", "message": "Agent deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting agent: {str(e)}")
        return {"status": "fail", "message": f"Error deleting agent: {str(e)}"}

def fetch_agent_by_phone_number(phone_number: str) -> Optional[Dict[str, Any]]:
    """
    Fetch agent config by phone number.
    
    Args:
        phone_number: Phone number to search for
        
    Returns:
        Agent config document or None
    """
    try:
        db = get_database()
        agent_table = db["AgentConfig"]
        agent = agent_table.find_one({"phone_number": phone_number})
        return agent
    except Exception as e:
        logger.error(f"Error fetching agent by phone number: {str(e)}")
        return None

