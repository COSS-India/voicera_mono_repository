"""
Agent service for handling agent-related database operations.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.database import get_database
from app.models.schemas import AgentConfigCreate, AgentConfigUpdate
import logging
import re
import string

logger = logging.getLogger(__name__)

VALID_INTERACTION_MODES = {"conversational", "non_conversational"}
_AGENT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_AGENT_NAME_ERROR = (
    "Agent name may only contain letters, numbers, underscores, and hyphens (no spaces)"
)


def _validate_agent_type_name(agent_type: str) -> Optional[str]:
    normalized = (agent_type or "").strip()
    if not normalized:
        return "Agent name cannot be empty"
    if not _AGENT_NAME_PATTERN.match(normalized):
        return _AGENT_NAME_ERROR
    return None


def _get_interaction_mode(agent_config: Dict[str, Any]) -> str:
    mode = (agent_config or {}).get("interaction_mode") or "conversational"
    return mode if mode in VALID_INTERACTION_MODES else "conversational"


def _validate_agent_config_for_mode(agent_config: Dict[str, Any]) -> Optional[str]:
    mode = _get_interaction_mode(agent_config)
    if mode == "non_conversational":
        greeting = str((agent_config or {}).get("greeting_message") or "").strip()
        if not greeting:
            return "Alert message is required for non-conversational agents"
        tts_model = (agent_config or {}).get("tts_model")
        if not isinstance(tts_model, dict) or not tts_model.get("name"):
            return "TTS configuration is required for non-conversational agents"
    return None


def _normalize_agent_language_fields(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """Keep one primary ``language`` and optional ``secondary_languages`` in sync."""
    config = dict(agent_config or {})

    primary = ""
    secondaries: List[str] = []

    languages_list = config.get("languages")
    secondary_languages_list = config.get("secondary_languages")

    if isinstance(languages_list, list) and languages_list:
        deduped: List[str] = []
        seen: set[str] = set()
        for item in languages_list:
            value = str(item).strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(value)
        if deduped:
            primary = deduped[0]
            secondaries = deduped[1:]
    elif isinstance(secondary_languages_list, list) and secondary_languages_list:
        primary = str(config.get("language") or "").strip()
        seen: set[str] = {primary.lower()} if primary else set()
        for item in secondary_languages_list:
            value = str(item).strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            secondaries.append(value)
    else:
        primary = str(config.get("language") or "").strip()
        secondary = str(config.get("secondary_language") or "").strip()
        if secondary and (not primary or secondary.lower() != primary.lower()):
            secondaries = [secondary]

    if not primary:
        config.pop("language", None)
        config.pop("languages", None)
        config.pop("secondary_languages", None)
        config.pop("secondary_language", None)
        return config

    config["language"] = primary
    if secondaries:
        config["secondary_languages"] = secondaries
        config["languages"] = [primary, *secondaries]
        config["secondary_language"] = secondaries[0]
    else:
        config.pop("languages", None)
        config.pop("secondary_languages", None)
        config.pop("secondary_language", None)

    return config


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

        agent_type = (agent_data.agent_type or "").strip()
        name_error = _validate_agent_type_name(agent_type)
        if name_error:
            return {"status": "fail", "message": name_error}
        
        # Check if agent_type already exists for this organization
        existing_agent = agent_table.find_one({
            "agent_type": agent_type,
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

        agent_config = _normalize_agent_language_fields(dict(agent_data.agent_config or {}))
        if not agent_config.get("interaction_mode"):
            agent_config["interaction_mode"] = "conversational"
        validation_error = _validate_agent_config_for_mode(agent_config)
        if validation_error:
            return {"status": "fail", "message": validation_error}
        
        now_iso = datetime.now().isoformat()
        agent_doc = {
            "agent_type": agent_type,
            "agent_id": agent_data.agent_id,
            "agent_config": agent_config,
            "org_id": agent_data.org_id,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        
        if agent_data.agent_category:
            agent_doc["agent_category"] = agent_data.agent_category
        provider = (agent_data.telephony_provider or "").strip()
        category = (agent_data.agent_category or "").strip()
        is_websocket = provider == "WebSocket" or category == "voicera_websocket"
        if not is_websocket and category == "voicera_telephony" and provider not in ("Vobiz", "Plivo"):
            return {
                "status": "fail",
                "message": "Telephony provider must be Vobiz or Plivo for telephony agents",
            }
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
        logger.info(f"Agent created successfully: {agent_type}")
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

def fetch_agent_config_for_org(agent_type: str, org_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch agent config for a given agent type scoped to an organization.
    """
    try:
        db = get_database()
        agent_table = db["AgentConfig"]
        agent = agent_table.find_one({"agent_type": agent_type, "org_id": org_id})
        return agent
    except Exception as e:
        logger.error(f"Error fetching org-scoped agent config: {str(e)}")
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
        agents = list(
            agent_table.find({"org_id": org_id}).sort(
                [("created_at", -1), ("updated_at", -1)]
            )
        )
        return agents
    except Exception as e:
        logger.error(f"Error fetching agents: {str(e)}")
        return []

def update_agent_config(agent_type: str, agent_data: AgentConfigUpdate, org_id: str) -> Dict[str, Any]:
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
        
        existing_agent = agent_table.find_one({"agent_type": agent_type, "org_id": org_id})
        if not existing_agent:
            return {"status": "fail", "message": "Agent type not found"}

        target_agent_type = (agent_data.agent_type or agent_type).strip()
        name_error = _validate_agent_type_name(target_agent_type)
        if name_error:
            return {"status": "fail", "message": name_error}

        if target_agent_type != agent_type:
            duplicate = agent_table.find_one({"agent_type": target_agent_type, "org_id": org_id})
            if duplicate:
                return {"status": "fail", "message": "Agent type already exists for this organization"}

        existing_mode = _get_interaction_mode(existing_agent.get("agent_config") or {})
        incoming_config = _normalize_agent_language_fields(dict(agent_data.agent_config or {}))
        incoming_mode = _get_interaction_mode(incoming_config)

        if existing_mode == "non_conversational":
            if incoming_mode != "non_conversational":
                return {
                    "status": "fail",
                    "message": "Cannot change a non-conversational agent to conversational",
                }
            incoming_config["interaction_mode"] = "non_conversational"
        elif not incoming_config.get("interaction_mode"):
            incoming_config["interaction_mode"] = existing_mode

        validation_error = _validate_agent_config_for_mode(incoming_config)
        if validation_error:
            return {"status": "fail", "message": validation_error}

        update_doc = {
            "agent_config": incoming_config,
            "updated_at": datetime.now().isoformat(),
            "agent_type": target_agent_type,
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

        if existing_agent.get("created_at") is None:
            update_doc["created_at"] = (
                existing_agent.get("updated_at") or datetime.now().isoformat()
            )

        result = agent_table.update_one(
            {"agent_type": agent_type, "org_id": org_id},
            {"$set": update_doc}
        )
        
        if result.matched_count == 0:
            return {"status": "fail", "message": "Agent type not found"}
        
        if target_agent_type != agent_type:
            collection_names = [
                "PhoneNumber",
                "Meetings",
                "CallLogs",
                "CallRecordings",
                "Campaigns",
                "Batches",
                "BatchContacts",
            ]
            for collection_name in collection_names:
                db[collection_name].update_many(
                    {"org_id": org_id, "agent_type": agent_type},
                    {"$set": {"agent_type": target_agent_type}},
                )

        logger.info(f"Agent updated successfully: {agent_type} -> {target_agent_type}")
        return {
            "status": "success",
            "message": "Agent config updated successfully",
            "agent_type": target_agent_type,
        }
        
    except Exception as e:
        logger.error(f"Error updating agent: {str(e)}")
        return {"status": "fail", "message": f"Error updating agent: {str(e)}"}

def delete_agent(agent_type: str, org_id: Optional[str] = None) -> Dict[str, Any]:
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
        
        query: Dict[str, Any] = {"agent_type": agent_type}
        if org_id:
            query["org_id"] = org_id

        result = agent_table.delete_one(query)
        
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
