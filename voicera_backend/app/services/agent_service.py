"""
Agent service for handling agent-related database operations.
"""
import json
import logging
import secrets
import string
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from pydantic import BaseModel

from app.config import settings
from app.database import get_database
from app.models.schemas import AgentConfigCreate, AgentConfigUpdate

logger = logging.getLogger(__name__)

VALID_INTERACTION_MODES = {"conversational", "non_conversational", "translation"}

# Interaction modes whose behaviour is locked once an agent is created: switching
# in or out of them would change the whole runtime pipeline, so we reject it.
IMMUTABLE_INTERACTION_MODES = {"non_conversational", "translation"}


def generate_share_token() -> str:
    """Random URL-safe token used as the public key for a shareable agent."""
    return secrets.token_urlsafe(16)

# Pre-configured agents, seeded once per org from app/config/default_agents.json.
# Platform .env credential fallback on the voice server is now available to all
# agents (gated only by ALLOW_PLATFORM_KEY_FALLBACK), not just these.
DEFAULT_AGENTS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default_agents.json"


class _ModelConfig(BaseModel):
    name: str
    model: str
    speaker: Optional[str] = None
    # Natural-language voice-style prompt for indic-parler-tts (drives pace/tone
    # on the voice server; surfaced as the "voice description" in the frontend).
    description: Optional[str] = None


class DefaultAgentTemplate(BaseModel):
    agent_type: str
    # Stable prefix for this template's agent_id (agent_id = f"{id_prefix}_{org_id}"),
    # kept separate from agent_type so renaming the display name never changes the id.
    id_prefix: str
    language: str
    telephony_provider: str = "Vobiz"
    system_prompt: str
    greeting_message: str
    llm_model: _ModelConfig
    stt_model: _ModelConfig
    tts_model: _ModelConfig


def _load_default_agent_templates() -> List[DefaultAgentTemplate]:
    raw = json.loads(DEFAULT_AGENTS_CONFIG_PATH.read_text(encoding="utf-8"))
    return [DefaultAgentTemplate(**entry) for entry in raw]


# Loaded once at process start; add/edit entries in default_agents.json, no code change needed.
DEFAULT_AGENT_TEMPLATES = _load_default_agent_templates()


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
    elif mode == "translation":
        source = str((agent_config or {}).get("source_language") or "").strip()
        if not source:
            return "Source language is required for translation agents"
        targets = (agent_config or {}).get("target_languages")
        if not isinstance(targets, list) or not [t for t in targets if str(t).strip()]:
            return "At least one target language is required for translation agents"
        stt_model = (agent_config or {}).get("stt_model")
        if not isinstance(stt_model, dict) or not stt_model.get("name"):
            return "STT configuration is required for translation agents"
        tts_model = (agent_config or {}).get("tts_model")
        if not isinstance(tts_model, dict) or not tts_model.get("name"):
            return "TTS configuration is required for translation agents"
        # Which translation engine drives the broadcast. Absent → "llm" (the
        # original behaviour). Per-language support is enforced by the voice
        # server, which owns the engine-specific language maps; duplicating them
        # here would drift.
        engine = str((agent_config or {}).get("translation_engine") or "llm").strip().lower()
        if engine not in ("llm", "nmt"):
            return "Translation engine must be 'llm' or 'nmt'"
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


def _create_default_agent(org_id: str, template: DefaultAgentTemplate) -> Dict[str, Any]:
    """
    Create one pre-configured default agent for an org from its template.

    Uses env-less AI4Bharat STT/TTS; the OpenAI key is resolved on the voice
    server (org integration first, platform .env fallback for default agents).
    """
    try:
        db = get_database()
        agent_table = db["AgentConfig"]

        if agent_table.find_one({"agent_type": template.agent_type, "org_id": org_id}):
            return {"status": "success", "message": "Default agent already exists"}

        now_iso = datetime.now().isoformat()
        agent_doc = {
            "agent_type": template.agent_type,
            # agent_id must be globally unique: the voice server looks agents
            # up by agent_id without an org filter.
            "agent_id": f"{template.id_prefix}_{org_id}",
            "org_id": org_id,
            "agent_category": "voicera_telephony",
            "telephony_provider": template.telephony_provider,
            "agent_config": {
                "interaction_mode": "conversational",
                "system_prompt": template.system_prompt,
                "greeting_message": template.greeting_message,
                "ignore_user_speech_before_greeting": True,
                "interruption_min_words": 1,
                "user_silence_hangup_seconds": 30,
                "call_timeout_seconds": 300,
                "hold_messages": [],
                "hold_message_timeout_seconds": 0.3,
                "language": template.language,
                "knowledge_base_enabled": False,
                "knowledge_document_ids": [],
                "knowledge_top_k": 3,
                "llm_model": template.llm_model.model_dump(exclude_none=True),
                "stt_model": template.stt_model.model_dump(exclude_none=True),
                "tts_model": template.tts_model.model_dump(exclude_none=True),
            },
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        agent_table.insert_one(agent_doc)
        logger.info(f"Default agent '{template.agent_type}' created for org: {org_id}")
        return {"status": "success", "message": "Default agent created"}
    except Exception as e:
        logger.error(f"Error creating default agent '{template.agent_type}' for org {org_id}: {str(e)}")
        return {"status": "fail", "message": f"Error creating default agent: {str(e)}"}


def ensure_default_agent_seeded(org_id: str) -> None:
    """
    Seed each platform default agent template once per org, tracked by a
    marker collection so a deleted or renamed default agent stays gone (and
    so a template added later still backfills existing orgs on their next
    agent listing, per-template rather than all-or-nothing).

    Gated by DEMO_AGENTS_ENABLED: when disabled, no seeding/backfill runs and
    already-seeded agents are left as-is.
    """
    if not settings.DEMO_AGENTS_ENABLED:
        return
    db = get_database()
    marker_table = db["DefaultAgentSeeded"]
    for template in DEFAULT_AGENT_TEMPLATES:
        try:
            if marker_table.find_one({"org_id": org_id, "agent_type": template.agent_type}):
                continue
            result = _create_default_agent(org_id, template)
            if result["status"] == "success":
                marker_table.insert_one({
                    "org_id": org_id,
                    "agent_type": template.agent_type,
                    "created_at": datetime.now().isoformat(),
                })
        except Exception as e:
            logger.error(f"Error seeding default agent '{template.agent_type}' for org {org_id}: {str(e)}")


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

        agent_config = _normalize_agent_language_fields(dict(agent_data.agent_config or {}))
        if not agent_config.get("interaction_mode"):
            agent_config["interaction_mode"] = "conversational"
        validation_error = _validate_agent_config_for_mode(agent_config)
        if validation_error:
            return {"status": "fail", "message": validation_error}

        now_iso = datetime.now().isoformat()
        agent_doc = {
            "agent_type": agent_data.agent_type,
            "agent_id": agent_data.agent_id,
            "agent_config": agent_config,
            "org_id": agent_data.org_id,
            "created_at": now_iso,
            "updated_at": now_iso,
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
        if agent_data.public_share_enabled:
            agent_doc["public_share_enabled"] = True
            agent_doc["share_token"] = generate_share_token()

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

def fetch_agent_by_share_token(share_token: str) -> Optional[Dict[str, Any]]:
    """Fetch a publicly-shared agent by its share_token.

    Returns None unless the token matches an agent with public sharing enabled,
    so a disabled or unknown token never resolves.
    """
    if not share_token:
        return None
    try:
        db = get_database()
        agent_table = db["AgentConfig"]
        agent = agent_table.find_one(
            {"share_token": share_token, "public_share_enabled": True}
        )
        return agent
    except Exception as e:
        logger.error(f"Error fetching agent by share token: {str(e)}")
        return None


def rotate_share_token(agent_type: str, org_id: str) -> Dict[str, Any]:
    """Regenerate the share_token for an agent, invalidating old public links."""
    try:
        db = get_database()
        agent_table = db["AgentConfig"]
        existing = agent_table.find_one({"agent_type": agent_type, "org_id": org_id})
        if not existing:
            return {"status": "fail", "message": "Agent type not found"}
        new_token = generate_share_token()
        agent_table.update_one(
            {"agent_type": agent_type, "org_id": org_id},
            {"$set": {
                "share_token": new_token,
                "public_share_enabled": True,
                "updated_at": datetime.now().isoformat(),
            }},
        )
        return {"status": "success", "share_token": new_token}
    except Exception as e:
        logger.error(f"Error rotating share token: {str(e)}")
        return {"status": "fail", "message": f"Error rotating share token: {str(e)}"}


def build_public_agent_projection(
    agent: Dict[str, Any], include_agent_id: bool = False
) -> Dict[str, Any]:
    """Secret-stripped view of an agent for unauthenticated consumers.

    ``agent_id`` is withheld by default: the voice server's WebSocket routes are
    unauthenticated, so anyone holding an agent_id can open a session on the
    org's STT/LLM/TTS credentials. Only the internal (X-API-Key) caller, which
    needs it to resolve the room, may ask for it.
    """
    config = agent.get("agent_config") or {}
    targets = config.get("target_languages")
    if not isinstance(targets, list):
        targets = []
    projection: Dict[str, Any] = {
        "display_name": agent.get("agent_type"),
        "interaction_mode": _get_interaction_mode(config),
        "source_language": config.get("source_language"),
        "target_languages": [str(t) for t in targets if str(t).strip()],
        "greeting_message": config.get("greeting_message"),
    }
    if include_agent_id:
        projection["agent_id"] = agent.get("agent_id")
    return projection


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
        if not target_agent_type:
            return {"status": "fail", "message": "Agent type cannot be empty"}

        if target_agent_type != agent_type:
            duplicate = agent_table.find_one({"agent_type": target_agent_type, "org_id": org_id})
            if duplicate:
                return {"status": "fail", "message": "Agent type already exists for this organization"}

        existing_mode = _get_interaction_mode(existing_agent.get("agent_config") or {})
        incoming_config = _normalize_agent_language_fields(dict(agent_data.agent_config or {}))
        incoming_mode = _get_interaction_mode(incoming_config)

        if existing_mode in IMMUTABLE_INTERACTION_MODES:
            if incoming_mode != existing_mode:
                return {
                    "status": "fail",
                    "message": f"Cannot change the interaction mode of a {existing_mode} agent",
                }
            incoming_config["interaction_mode"] = existing_mode
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
        if agent_data.public_share_enabled is not None:
            update_doc["public_share_enabled"] = bool(agent_data.public_share_enabled)
            # Mint a token on first enable; preserve it across toggles so an
            # existing share link keeps working when re-enabled.
            if agent_data.public_share_enabled and not existing_agent.get("share_token"):
                update_doc["share_token"] = generate_share_token()

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
