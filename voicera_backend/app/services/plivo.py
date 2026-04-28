"""
Plivo service for handling Plivo API operations.

Credentials (auth_id / auth_token) are loaded per organization from the Integrations
collection (models PlivoAuthId and PlivoAuthToken). API base URL remains from settings.
"""
from typing import Dict, Any, Optional, Tuple
import logging

import httpx

from app.config import settings
from app.services import integration_service as _integration_service

logger = logging.getLogger(__name__)


def _get_plivo_auth_for_org(org_id: str) -> Optional[Tuple[str, str]]:
    """Return (auth_id, auth_token) from integrations, or None if either is missing."""
    id_doc = _integration_service.get_integration(org_id, "PlivoAuthId")
    tok_doc = _integration_service.get_integration(org_id, "PlivoAuthToken")
    if not id_doc or not tok_doc:
        return None
    auth_id = (id_doc.get("api_key") or "").strip()
    auth_token = (tok_doc.get("api_key") or "").strip()
    if not auth_id or not auth_token:
        return None
    return auth_id, auth_token


def _auth_headers(include_content_type: bool = False) -> Dict[str, str]:
    headers: Dict[str, str] = {"Accept": "application/json"}
    if include_content_type:
        headers["Content-Type"] = "application/json"
    return headers


async def create_plivo_application(org_id: str, agent_type: str, answer_url: str) -> Dict[str, Any]:
    """Create a Plivo application via API."""
    try:
        creds = _get_plivo_auth_for_org(org_id)
        if not creds:
            return {
                "status": "fail",
                "message": "Plivo Auth ID and Auth Token must be configured in Integrations (Telephony) for this organization.",
            }
        auth_id, auth_token = creds

        url = f"{settings.PLIVO_API_BASE_URL}/Account/{auth_id}/Application/"
        payload = {
            "app_name": agent_type,
            "answer_url": answer_url,
            "answer_method": "POST",
            "hangup_url": answer_url.replace("/plivo/answer", "/plivo/hangup"),
            "hangup_method": "POST",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=_auth_headers(include_content_type=True),
                auth=(auth_id, auth_token),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            app_id = data.get("app_id") or data.get("id") or data.get("application_id")
            return {
                "status": "success",
                "message": "Plivo application created successfully",
                "app_id": app_id,
            }
    except httpx.HTTPStatusError as e:
        message = f"Plivo API error: {e.response.text}"
        logger.error(message)
        return {"status": "fail", "message": message}
    except httpx.RequestError as e:
        message = f"Failed to connect to Plivo API: {str(e)}"
        logger.error(message)
        return {"status": "fail", "message": message}
    except Exception as e:
        message = f"Error creating Plivo application: {str(e)}"
        logger.error(message)
        return {"status": "fail", "message": message}


async def delete_plivo_application(org_id: str, application_id: str) -> Dict[str, Any]:
    """Delete a Plivo application via API."""
    try:
        creds = _get_plivo_auth_for_org(org_id)
        if not creds:
            return {
                "status": "fail",
                "message": "Plivo Auth ID and Auth Token must be configured in Integrations (Telephony) for this organization.",
            }
        auth_id, auth_token = creds

        url = f"{settings.PLIVO_API_BASE_URL}/Account/{auth_id}/Application/{application_id}/"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                url,
                headers=_auth_headers(),
                auth=(auth_id, auth_token),
            )
            response.raise_for_status()
            return {
                "status": "success",
                "message": "Plivo application deleted successfully",
            }
    except httpx.HTTPStatusError as e:
        message = f"Plivo API error: {e.response.text}"
        logger.error(message)
        return {"status": "fail", "message": message}
    except httpx.RequestError as e:
        message = f"Failed to connect to Plivo API: {str(e)}"
        logger.error(message)
        return {"status": "fail", "message": message}
    except Exception as e:
        message = f"Error deleting Plivo application: {str(e)}"
        logger.error(message)
        return {"status": "fail", "message": message}


async def link_number_to_application(org_id: str, phone_number: str, application_id: str) -> Dict[str, Any]:
    """Link a phone number to a Plivo application via API."""
    try:
        creds = _get_plivo_auth_for_org(org_id)
        if not creds:
            return {
                "status": "fail",
                "message": "Plivo Auth ID and Auth Token must be configured in Integrations (Telephony) for this organization.",
            }
        auth_id, auth_token = creds

        url = f"{settings.PLIVO_API_BASE_URL}/Account/{auth_id}/Number/{phone_number}/"
        payload = {"app_id": application_id}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=_auth_headers(include_content_type=True),
                auth=(auth_id, auth_token),
                json=payload,
            )
            response.raise_for_status()
            return {
                "status": "success",
                "message": "Phone number linked to application successfully",
            }
    except httpx.HTTPStatusError as e:
        message = f"Plivo API error: {e.response.text}"
        logger.error(message)
        return {"status": "fail", "message": message}
    except httpx.RequestError as e:
        message = f"Failed to connect to Plivo API: {str(e)}"
        logger.error(message)
        return {"status": "fail", "message": message}
    except Exception as e:
        message = f"Error linking phone number to application: {str(e)}"
        logger.error(message)
        return {"status": "fail", "message": message}


async def unlink_number_from_application(org_id: str, phone_number: str) -> Dict[str, Any]:
    """Unlink a phone number from a Plivo application via API."""
    try:
        creds = _get_plivo_auth_for_org(org_id)
        if not creds:
            return {
                "status": "fail",
                "message": "Plivo Auth ID and Auth Token must be configured in Integrations (Telephony) for this organization.",
            }
        auth_id, auth_token = creds

        url = f"{settings.PLIVO_API_BASE_URL}/Account/{auth_id}/Number/{phone_number}/"
        payload = {"app_id": None}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=_auth_headers(include_content_type=True),
                auth=(auth_id, auth_token),
                json=payload,
            )
            response.raise_for_status()
            return {
                "status": "success",
                "message": "Phone number unlinked from application successfully",
            }
    except httpx.HTTPStatusError as e:
        message = f"Plivo API error: {e.response.text}"
        logger.error(message)
        return {"status": "fail", "message": message}
    except httpx.RequestError as e:
        message = f"Failed to connect to Plivo API: {str(e)}"
        logger.error(message)
        return {"status": "fail", "message": message}
    except Exception as e:
        message = f"Error unlinking phone number from application: {str(e)}"
        logger.error(message)
        return {"status": "fail", "message": message}


async def get_plivo_numbers(org_id: str) -> Dict[str, Any]:
    """List phone numbers from the Plivo API for this organization."""
    creds = _get_plivo_auth_for_org(org_id)
    if not creds:
        return {
            "status": "fail",
            "message": "Plivo Auth ID and Auth Token must be configured in Integrations (Telephony) for this organization.",
            "numbers": [],
        }
    auth_id, auth_token = creds
    url = f"{settings.PLIVO_API_BASE_URL}/Account/{auth_id}/Number/"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                headers=_auth_headers(),
                auth=(auth_id, auth_token),
            )
            response.raise_for_status()
            data = response.json()
            objects = data.get("objects", [])
            e164_numbers = [item.get("number") for item in objects if item.get("number")]
            return {"status": "success", "numbers": e164_numbers}
    except httpx.HTTPStatusError as e:
        logger.error(f"Plivo numbers API error: {e.response.text}")
        return {
            "status": "fail",
            "message": f"Plivo API error: {e.response.text}",
            "numbers": [],
        }
    except httpx.RequestError as e:
        logger.error(f"Plivo numbers request failed: {e}")
        return {
            "status": "fail",
            "message": f"Failed to connect to Plivo API: {str(e)}",
            "numbers": [],
        }
