"""Backend API utility functions for voice bot integration."""

import os
import time
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

from loguru import logger
import requests
from storage.minio_client import MinIOStorage


# ============================================================================
# Backend API Helper Functions
# ============================================================================

def _get_backend_url() -> str:
    """Get backend API URL from environment."""
    return os.getenv("VOICERA_BACKEND_URL", "http://localhost:8000")


def _get_api_key() -> Optional[str]:
    """Get internal API key from environment."""
    return os.getenv("INTERNAL_API_KEY")


def _get_api_headers() -> Dict[str, str]:
    """Get headers for backend API calls."""
    headers = {"Content-Type": "application/json"}
    api_key = _get_api_key()
    if api_key:
        headers["X-API-Key"] = api_key
    else:
        logger.warning("⚠️ INTERNAL_API_KEY not set - API calls may fail")
    return headers


def fetch_integration_key(org_id: str, model: str) -> Optional[str]:
    """
    Fetch integration API key for a given org and provider model from the backend.

    Uses the bot endpoint which requires X-API-Key (INTERNAL_API_KEY).
    Returns None on 404 or any error so callers can fall back to .env.

    Args:
        org_id: Organization ID
        model: Provider/model name (e.g. "OpenAI", "Deepgram", "ElevenLabs")

    Returns:
        The API key string, or None if not found or on error
    """
    backend_url = _get_backend_url()
    api_endpoint = f"{backend_url}/api/v1/integrations/bot/get-api-key"
    headers = _get_api_headers()

    try:
        response = requests.post(
            api_endpoint,
            json={"org_id": org_id, "model": model},
            headers=headers,
            timeout=10,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        return data.get("api_key")
    except requests.exceptions.RequestException as e:
        logger.debug(f"Integration key fetch failed for org={org_id} model={model}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Integration key fetch error for org={org_id} model={model}: {e}")
        return None


def fetch_knowledge_chunks(
    *,
    org_id: str,
    question: str,
    document_ids: Optional[list[str]] = None,
    top_k: int = 3,
    timeout: float = 0.8,
) -> list[dict]:
    """
    Retrieve top-k knowledge chunks for runtime RAG augmentation.

    Uses internal API-key auth and returns an empty list on failures (fail-open).
    """
    backend_url = _get_backend_url()
    api_endpoint = f"{backend_url}/api/v1/rag/retrieve"
    headers = _get_api_headers()
    payload = {
        "org_id": org_id,
        "question": question,
        "top_k": top_k,
        "document_ids": document_ids or [],
    }
    try:
        response = requests.post(
            api_endpoint,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        chunks = data.get("chunks") if isinstance(data, dict) else []
        return chunks if isinstance(chunks, list) else []
    except Exception as e:
        logger.debug("Knowledge retrieval failed: %s", e)
        return []


async def fetch_agent_config_from_backend(agent_id: str) -> dict:
    """
    Fetch agent configuration from backend API.
    
    Args:
        agent_id: Agent ID to fetch config for
        
    Returns:
        Agent configuration dictionary
    """
    backend_url = _get_backend_url()
    api_endpoint = f"{backend_url}/api/v1/agents/config/id/{agent_id}"
    headers = _get_api_headers()
    
    try:
        logger.info(f"📥 Fetching agent config from backend: {agent_id}")
        response = requests.get(api_endpoint, headers=headers, timeout=10)
        response.raise_for_status()
        
        agent_data = response.json()
        # Extract agent_config from response
        agent_config = agent_data.get("agent_config", {})
        logger.info(f"📥 Agent config: {agent_config}")
        
        # Add other fields that might be needed
        if "org_id" in agent_data:
            agent_config["org_id"] = agent_data["org_id"]
        if "agent_type" in agent_data:
            agent_config["agent_type"] = agent_data["agent_type"]
        if "greeting_message" in agent_data:
            agent_config["greeting_message"] = agent_data["greeting_message"]
            
        logger.info(f"Agent config fetched successfully: {agent_id}")
        return agent_config
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch agent config from backend: {e}")
        logger.debug(f"API endpoint: {api_endpoint}")
        return None
    except Exception as e:
        logger.error(f"Error parsing agent config response: {e}")
        logger.debug(traceback.format_exc())
        return None


async def create_meeting_in_backend(payload) :
    """Create a meeting record in the backend when call starts."""
    backend_url = _get_backend_url()
    api_endpoint = f"{backend_url}/api/v1/meetings"
    headers = _get_api_headers()

    try:
        logger.info(f"Creating meeting in backend: {payload}")
        response = requests.post(api_endpoint, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to create meeting in backend: {e}")
        return None
    except Exception as e:
        logger.error(f"Error creating meeting in backend: {e}")
        return None


async def create_rejected_call_meeting(
    call_uuid: str,
    agent_type: str,
    form_data: Dict[str, Any],
    from_number: Optional[str] = None,
    to_number: Optional[str] = None,
) -> bool:
    """
    Create a meeting record for a rejected/busy call.
    
    This function is called when a call is rejected (CALL_REJECTED) before
    the websocket connection is established. It creates a minimal meeting
    record with available metadata from the webhook.
    
    Args:
        call_uuid: Call UUID from Vobiz (used as meeting_id)
        agent_type: Type of agent used
        form_data: Form data from Vobiz webhook containing call metadata (dict-like)
        from_number: From number from query params (fallback if not in form_data)
        to_number: To number from query params (fallback if not in form_data)
        
    Returns:
        True if meeting was created successfully, False otherwise
    """
    backend_url = _get_backend_url()
    api_endpoint = f"{backend_url}/api/v1/meetings"
    headers = _get_api_headers()
    
    try:
        # Fetch agent config to get org_id
        agent_config = await fetch_agent_config_from_backend(agent_type)
        if not agent_config:
            logger.warning(f"⚠️ Could not fetch agent config for {agent_type}, skipping meeting creation")
            return False
        
        # Helper function to safely get value from form_data (handle both dict and FormData)
        def get_form_value(key: str, default: str = '') -> str:
            value = form_data.get(key, default)
            # Handle case where value might be a list (multi-value form fields)
            if isinstance(value, list):
                return value[0] if value else default
            return str(value) if value else default
        
        # Extract phone numbers - prefer form_data, fallback to query params
        from_num = get_form_value('From') or get_form_value('from_number') or from_number or ''
        to_num = get_form_value('To') or get_form_value('to_number') or to_number or ''
        
        # Normalize phone numbers (remove leading/trailing spaces)
        from_num = from_num.strip() if from_num else None
        to_num = to_num.strip() if to_num else None
        
        # Parse timestamps from form_data
        start_time_str = get_form_value('StartTime')
        end_time_str = get_form_value('EndTime')
        
        # Convert to ISO format UTC timestamps
        start_time_utc = None
        end_time_utc = None
        
        if start_time_str:
            try:
                # Parse the datetime string (format: '2026-01-14 17:04:30')
                start_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                start_time_utc = start_dt.isoformat() + 'Z'
            except (ValueError, TypeError) as e:
                logger.warning(f"⚠️ Could not parse StartTime '{start_time_str}': {e}")
                start_time_utc = datetime.utcnow().isoformat() + 'Z'
        else:
            start_time_utc = datetime.utcnow().isoformat() + 'Z'
        
        if end_time_str:
            try:
                # Parse the datetime string (format: '2026-01-14 17:04:30')
                end_dt = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
                end_time_utc = end_dt.isoformat() + 'Z'
            except (ValueError, TypeError) as e:
                logger.warning(f"⚠️ Could not parse EndTime '{end_time_str}': {e}")
                end_time_utc = start_time_utc  # Use start time as fallback
        else:
            end_time_utc = start_time_utc  # If no end time, use start time
        
        # Determine if inbound or outbound from Direction field
        direction = get_form_value('Direction', '').lower()
        inbound = direction == 'inbound'
        
        # Build payload
        payload = {
            "meeting_id": call_uuid,
            "agent_type": agent_type,
            "start_time_utc": start_time_utc,
            "end_time_utc": end_time_utc,
            "created_at": start_time_utc,
            "from_number": from_num,
            "to_number": to_num,
            "inbound": inbound,
            "call_busy": True,
        }
        
        # Add org_id if available in agent config
        if "org_id" in agent_config:
            payload["org_id"] = agent_config["org_id"]
        
        logger.info(f"📤 Creating rejected call meeting in backend: {call_uuid}")
        logger.info(f"📤 Payload: {payload}")
        
        response = requests.post(api_endpoint, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ Rejected call meeting created successfully: {call_uuid}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to create rejected call meeting in backend: {e}")
        logger.debug(f"API endpoint: {api_endpoint}")
        return False
    except Exception as e:
        logger.error(f"❌ Error creating rejected call meeting: {e}")
        logger.debug(traceback.format_exc())
        return False


async def update_meeting_end_time(
    call_sid: str,
    end_time_utc: str
) -> bool:
    """
    Update meeting end_time_utc in the backend.
    
    Args:
        call_sid: Call identifier (meeting_id)
        end_time_utc: ISO format UTC timestamp when call ended
        
    Returns:
        True if meeting was updated successfully, False otherwise
    """
    backend_url = _get_backend_url()
    api_endpoint = f"{backend_url}/api/v1/meetings/{call_sid}"
    headers = _get_api_headers()
    
    payload = {
        "end_time_utc": end_time_utc
    }
    
    try:
        logger.info(f"📤 Updating meeting end time in backend: {call_sid}")
        response = requests.patch(api_endpoint, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ Meeting end time updated successfully: {call_sid}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to update meeting end time: {e}")
        logger.debug(f"API endpoint: {api_endpoint}, payload: {payload}")
        return False


def fetch_batch_agent_call_config(org_id: str, agent_type: str) -> Optional[Dict[str, Any]]:
    backend_url = _get_backend_url()
    api_endpoint = f"{backend_url}/api/v1/batches/worker/agent-config"
    headers = _get_api_headers()
    try:
        response = requests.post(
            api_endpoint,
            json={"org_id": org_id, "agent_type": agent_type},
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.error(f"Failed to fetch batch agent call config: {e}")
        return None


def claim_next_batch_contact(org_id: str, batch_id: str) -> Optional[Dict[str, Any]]:
    backend_url = _get_backend_url()
    api_endpoint = f"{backend_url}/api/v1/batches/worker/claim-next"
    headers = _get_api_headers()
    try:
        response = requests.post(
            api_endpoint,
            json={"org_id": org_id, "batch_id": batch_id},
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json() if response.text else {}
        if not isinstance(data, dict):
            return None
        contact = data.get("contact")
        return contact if isinstance(contact, dict) else None
    except Exception as e:
        logger.error(f"Failed to claim next batch contact: {e}")
        return None


def report_batch_contact_result(
    *,
    org_id: str,
    batch_id: str,
    row_number: int,
    ok: bool,
    error: Optional[str] = None,
) -> None:
    backend_url = _get_backend_url()
    api_endpoint = f"{backend_url}/api/v1/batches/worker/report"
    headers = _get_api_headers()
    try:
        requests.post(
            api_endpoint,
            json={
                "org_id": org_id,
                "batch_id": batch_id,
                "row_number": row_number,
                "ok": ok,
                "error": error,
            },
            headers=headers,
            timeout=10,
        ).raise_for_status()
    except Exception as e:
        logger.error(f"Failed to report batch contact result: {e}")


def finalize_batch_execution(org_id: str, batch_id: str, stopped: bool) -> None:
    backend_url = _get_backend_url()
    api_endpoint = f"{backend_url}/api/v1/batches/worker/finalize"
    headers = _get_api_headers()
    try:
        requests.post(
            api_endpoint,
            json={"org_id": org_id, "batch_id": batch_id, "stopped": stopped},
            headers=headers,
            timeout=10,
        ).raise_for_status()
    except Exception as e:
        logger.error(f"Failed to finalize batch execution: {e}")
    except Exception as e:
        logger.error(f"❌ Error updating meeting end time: {e}")
        logger.debug(traceback.format_exc())
        return False


async def fetch_agent_by_phone_number(phone_number: str) -> Optional[Dict[str, Any]]:
    """
    Fetch agent configuration by phone number.

    Args:
        phone_number: Phone number to look up (format: +918071387434)

    Returns:
        Agent configuration dictionary or None
    """
    # Normalize phone number - convert local format to E.164
    # Vobiz may send "08071387434", DB has "+918071387434"
    normalized_number = phone_number
    if phone_number.startswith('0'):
        normalized_number = '+91' + phone_number[1:]  # Remove leading 0, add +91
    elif not phone_number.startswith('+'):
        normalized_number = '+' + phone_number  # Just add + if missing

    logger.info(f"📞 Normalizing phone: {phone_number} → {normalized_number}")

    backend_url = _get_backend_url()
    from urllib.parse import quote
    encoded_phone = quote(normalized_number, safe='')
    api_endpoint = f"{backend_url}/api/v1/agents/by-phone/{encoded_phone}"
    headers = _get_api_headers()

    try:
        logger.info(f"📥 Fetching agent by phone number: {normalized_number}")
        response = requests.get(api_endpoint, headers=headers, timeout=10)
        response.raise_for_status()

        agent_data = response.json()
        logger.info(f"✅ Agent found: {agent_data.get('agent_type')}")
        return agent_data

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to fetch agent by phone number: {e}")
        return None


async def submit_call_recording(
    call_sid: str,
    agent_type: str,
    agent_config: dict,
    storage: MinIOStorage,
    call_start_time: float,
    start_time_utc: str
) -> None:
    """
    Submit call recording data to the backend API after a call ends.
    
    This function:
    1. Reads the transcript from MinIO
    2. Sends call recording data to backend API
    3. Updates meeting end_time_utc in backend
    
    Args:
        call_sid: Call identifier (same as meeting_id)
        agent_type: Type of agent used for the call
        agent_config: Agent configuration dictionary
        storage: MinIOStorage instance for accessing stored files
        call_start_time: Monotonic time when call started
        start_time_utc: ISO format UTC timestamp when call started
    """
    backend_url = _get_backend_url()
    headers = _get_api_headers()
    
    try:
        call_end_time = time.monotonic()
        call_duration = call_end_time - call_start_time
        end_time_utc = datetime.utcnow().isoformat()
        
        # Build MinIO object URLs
        recording_url = f"minio://recordings/{call_sid}.wav"
        transcript_url = f"minio://transcripts/{call_sid}.txt"
        
        # Read transcript content from MinIO
        transcript_content = None
        try:
            response = await storage.get_object("transcripts", f"{call_sid}.txt")
            transcript_content = response.read().decode("utf-8")
            response.close()
            response.release_conn()
        except Exception as e:
            logger.warning(f"⚠️ Could not read transcript: {e}")
        
        # 1. Send call recording data to backend API
        api_endpoint = f"{backend_url}/api/v1/call-recordings"
        payload = {
            "call_sid": call_sid,
            "recording_url": recording_url,
            "transcript_url": transcript_url,
            "transcript_content": transcript_content,
            "agent_type": agent_type,
            "call_duration": call_duration,
            "start_time_utc": start_time_utc,
            "end_time_utc": end_time_utc,
        }
        
        # Add org_id if available in agent config
        if "org_id" in agent_config:
            payload["org_id"] = agent_config["org_id"]
        
        try:
            logger.info(f"📤 Sending call recording data to backend: {call_sid}")
            response = requests.post(api_endpoint, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            logger.info(f"✅ Call recording data saved successfully: {call_sid}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to send call recording data: {e}")
        except Exception as e:
            logger.error(f"❌ Error processing call recording data: {e}")
            logger.debug(traceback.format_exc())
        
        # 2. Update meeting end_time_utc
        await update_meeting_end_time(call_sid, end_time_utc)
        
    except Exception as e:
        logger.error(f"❌ Error in submit_call_recording: {e}")
        logger.debug(traceback.format_exc())
