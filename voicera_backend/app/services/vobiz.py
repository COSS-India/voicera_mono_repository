"""
Vobiz service for handling Vobiz API operations.

Credentials (X-Auth-ID / X-Auth-Token) are loaded per organization from the Integrations
collection (models VobizAuthId and VobizAuthToken). API base URL remains from settings.
"""
from typing import Dict, Any, Optional, Tuple
from app.config import settings
from app.services import integration_service as _integration_service
import httpx
import logging

logger = logging.getLogger(__name__)


def _get_vobiz_auth_for_org(org_id: str) -> Optional[Tuple[str, str]]:
    """Return (auth_id, auth_token) from integrations, or None if either is missing."""
    id_doc = _integration_service.get_integration(org_id, "VobizAuthId")
    tok_doc = _integration_service.get_integration(org_id, "VobizAuthToken")
    if not id_doc or not tok_doc:
        return None
    auth_id = (id_doc.get("api_key") or "").strip()
    auth_token = (tok_doc.get("api_key") or "").strip()
    if not auth_id or not auth_token:
        return None
    return auth_id, auth_token


async def create_vobiz_application(org_id: str, agent_type: str, answer_url: str) -> Dict[str, Any]:
    """
    Create a Vobiz application via API.
    
    Args:
        agent_type: Agent type identifier (used as app_name)
        answer_url: Answer URL for the application
        
    Returns:
        Dict with status, message, and app_id if successful
    """
    try:
        creds = _get_vobiz_auth_for_org(org_id)
        if not creds:
            return {
                "status": "fail",
                "message": "Vobiz Auth ID and Auth Token must be configured in Integrations (Telephony) for this organization.",
            }
        auth_id, auth_token = creds

        # Construct the Vobiz API URL
        url = f"{settings.VOBIZ_API_BASE_URL}/Account/{auth_id}/Application/"

        # Prepare headers
        headers = {
            "X-Auth-ID": auth_id,
            "X-Auth-Token": auth_token,
            "Content-Type": "application/json"
        }
        
        # Prepare request body
        payload = {
            "app_name": agent_type,
            "answer_url": answer_url,
            "answer_method": "POST"
        }
        
        # Make the API request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Extract app_id from response if available
            app_id = data.get("app_id") or data.get("id") or data.get("application_id")
            
            logger.info(f"Vobiz application created successfully for agent_type: {agent_type}")
            return {
                "status": "success",
                "message": "Vobiz application created successfully",
                "app_id": app_id
            }
            
    except httpx.HTTPStatusError as e:
        error_message = f"Vobiz API error: {e.response.text}"
        logger.error(error_message)
        return {
            "status": "fail",
            "message": error_message
        }
    except httpx.RequestError as e:
        error_message = f"Failed to connect to Vobiz API: {str(e)}"
        logger.error(error_message)
        return {
            "status": "fail",
            "message": error_message
        }
    except Exception as e:
        error_message = f"Error creating Vobiz application: {str(e)}"
        logger.error(error_message)
        return {
            "status": "fail",
            "message": error_message
        }

async def delete_vobiz_application(org_id: str, application_id: str) -> Dict[str, Any]:
    """
    Delete a Vobiz application via API.
    
    Args:
        application_id: Vobiz application ID to delete
        
    Returns:
        Dict with status and message
    """
    try:
        creds = _get_vobiz_auth_for_org(org_id)
        if not creds:
            return {
                "status": "fail",
                "message": "Vobiz Auth ID and Auth Token must be configured in Integrations (Telephony) for this organization.",
            }
        auth_id, auth_token = creds

        # Construct the Vobiz API URL
        url = f"{settings.VOBIZ_API_BASE_URL}/Account/{auth_id}/Application/{application_id}/"

        # Prepare headers
        headers = {
            "X-Auth-ID": auth_id,
            "X-Auth-Token": auth_token
        }
        
        # Make the API request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(url, headers=headers)
            response.raise_for_status()
            
            logger.info(f"Vobiz application deleted successfully: {application_id}")
            return {
                "status": "success",
                "message": "Vobiz application deleted successfully"
            }
            
    except httpx.HTTPStatusError as e:
        error_message = f"Vobiz API error: {e.response.text}"
        logger.error(error_message)
        return {
            "status": "fail",
            "message": error_message
        }
    except httpx.RequestError as e:
        error_message = f"Failed to connect to Vobiz API: {str(e)}"
        logger.error(error_message)
        return {
            "status": "fail",
            "message": error_message
        }
    except Exception as e:
        error_message = f"Error deleting Vobiz application: {str(e)}"
        logger.error(error_message)
        return {
            "status": "fail",
            "message": error_message
        }

async def link_number_to_application(org_id: str, phone_number: str, application_id: str) -> Dict[str, Any]:
    """
    Link a phone number to a Vobiz application via API.
    
    Args:
        phone_number: Phone number to link (e164 format)
        application_id: Vobiz application ID
        
    Returns:
        Dict with status and message
    """
    try:
        creds = _get_vobiz_auth_for_org(org_id)
        if not creds:
            return {
                "status": "fail",
                "message": "Vobiz Auth ID and Auth Token must be configured in Integrations (Telephony) for this organization.",
            }
        auth_id, auth_token = creds

        # Construct the Vobiz API URL
        url = f"{settings.VOBIZ_API_BASE_URL}/account/{auth_id}/numbers/{phone_number}/application"

        # Prepare headers
        headers = {
            "X-Auth-ID": auth_id,
            "X-Auth-Token": auth_token,
            "Content-Type": "application/json"
        }
        
        # Prepare request body
        payload = {
            "application_id": application_id
        }
        
        # Make the API request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            logger.info(f"Phone number {phone_number} linked to application {application_id} successfully")
            return {
                "status": "success",
                "message": f"Phone number linked to application successfully"
            }
            
    except httpx.HTTPStatusError as e:
        error_message = f"Vobiz API error: {e.response.text}"
        logger.error(error_message)
        return {
            "status": "fail",
            "message": error_message
        }
    except httpx.RequestError as e:
        error_message = f"Failed to connect to Vobiz API: {str(e)}"
        logger.error(error_message)
        return {
            "status": "fail",
            "message": error_message
        }
    except Exception as e:
        error_message = f"Error linking phone number to application: {str(e)}"
        logger.error(error_message)
        return {
            "status": "fail",
            "message": error_message
        }

async def unlink_number_from_application(org_id: str, phone_number: str) -> Dict[str, Any]:
    """
    Unlink a phone number from a Vobiz application via API.
    
    Args:
        phone_number: Phone number to unlink (e164 format)
        
    Returns:
        Dict with status and message
    """
    try:
        creds = _get_vobiz_auth_for_org(org_id)
        if not creds:
            return {
                "status": "fail",
                "message": "Vobiz Auth ID and Auth Token must be configured in Integrations (Telephony) for this organization.",
            }
        auth_id, auth_token = creds

        # Construct the Vobiz API URL
        url = f"{settings.VOBIZ_API_BASE_URL}/account/{auth_id}/numbers/{phone_number}/application"

        # Prepare headers
        headers = {
            "X-Auth-ID": auth_id,
            "X-Auth-Token": auth_token
        }
        
        # Make the API request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(url, headers=headers)
            response.raise_for_status()
            
            logger.info(f"Phone number {phone_number} unlinked from application successfully")
            return {
                "status": "success",
                "message": "Phone number unlinked from application successfully"
            }
            
    except httpx.HTTPStatusError as e:
        error_message = f"Vobiz API error: {e.response.text}"
        logger.error(error_message)
        return {
            "status": "fail",
            "message": error_message
        }
    except httpx.RequestError as e:
        error_message = f"Failed to connect to Vobiz API: {str(e)}"
        logger.error(error_message)
        return {
            "status": "fail",
            "message": error_message
        }
    except Exception as e:
        error_message = f"Error unlinking phone number from application: {str(e)}"
        logger.error(error_message)
        return {
            "status": "fail",
            "message": error_message
        }


async def get_vobiz_numbers(org_id: str) -> Dict[str, Any]:
    """
    List phone numbers from the Vobiz API for this organization.
    """
    creds = _get_vobiz_auth_for_org(org_id)
    if not creds:
        return {
            "status": "fail",
            "message": "Vobiz Auth ID and Auth Token must be configured in Integrations (Telephony) for this organization.",
            "numbers": [],
        }
    auth_id, auth_token = creds
    url = f"{settings.VOBIZ_API_BASE_URL}/account/{auth_id}/numbers"
    headers = {
        "X-Auth-ID": auth_id,
        "X-Auth-Token": auth_token,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            e164_numbers = [
                item.get("e164")
                for item in data.get("items", [])
                if item.get("e164")
            ]
            return {"status": "success", "numbers": e164_numbers}
    except httpx.HTTPStatusError as e:
        logger.error(f"Vobiz numbers API error: {e.response.text}")
        return {
            "status": "fail",
            "message": f"Vobiz API error: {e.response.text}",
            "numbers": [],
        }
    except httpx.RequestError as e:
        logger.error(f"Vobiz numbers request failed: {e}")
        return {
            "status": "fail",
            "message": f"Failed to connect to Vobiz API: {str(e)}",
            "numbers": [],
        }
