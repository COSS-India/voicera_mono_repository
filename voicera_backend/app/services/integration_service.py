"""
Integration service for handling integration-related database operations.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.database import get_database
from app.models.schemas import IntegrationCreate
import logging

logger = logging.getLogger(__name__)

# These models store a non-secret identifier (not a secret key/token) in the
# api_key field, and the frontend needs it back to prefill the form.
_NON_SECRET_MODELS = {"VobizAuthId", "PlivoAuthId"}


def _to_public_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the api_key field unless this integration's model is non-secret."""
    public_doc = dict(doc)
    if public_doc.get("model") not in _NON_SECRET_MODELS:
        public_doc.pop("api_key", None)
    return public_doc


def _mask_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if len(key) <= 4:
        return "****"
    return f"{'*' * (len(key) - 4)}{key[-4:]}"


def _to_response(integration: Dict[str, Any], *, mask_key: bool = False) -> Dict[str, Any]:
    api_key = integration.get("api_key", "")
    return {
        "org_id": integration.get("org_id"),
        "model": integration.get("model"),
        "api_key": _mask_api_key(api_key) if mask_key else api_key,
        "created_at": integration.get("created_at"),
        "updated_at": integration.get("updated_at"),
    }


def create_integration(integration_data: IntegrationCreate) -> Dict[str, Any]:
    """
    Create or update an integration for a given org and model.
    Uses upsert to allow updating existing integrations.
    
    Args:
        integration_data: Integration creation data
        
    Returns:
        Dict with status and message
    """
    try:
        db = get_database()
        integration_table = db["Integrations"]
        
        now = datetime.now().isoformat()
        
        # Check if integration already exists
        existing = integration_table.find_one({
            "org_id": integration_data.org_id,
            "model": integration_data.model
        })
        
        if existing:
            # Update existing integration
            result = integration_table.update_one(
                {
                    "org_id": integration_data.org_id,
                    "model": integration_data.model
                },
                {
                    "$set": {
                        "api_key": integration_data.api_key,
                        "updated_at": now
                    }
                }
            )
            logger.info(f"Integration updated for org: {integration_data.org_id}, model: {integration_data.model}")
            return {"status": "success", "message": "Integration updated successfully"}
        else:
            # Create new integration
            integration_doc = {
                "org_id": integration_data.org_id,
                "model": integration_data.model,
                "api_key": integration_data.api_key,
                "created_at": now,
                "updated_at": now
            }
            integration_table.insert_one(integration_doc)
            logger.info(f"Integration created for org: {integration_data.org_id}, model: {integration_data.model}")
            return {"status": "success", "message": "Integration created successfully"}
        
    except Exception as e:
        logger.error(f"Error creating/updating integration: {str(e)}")
        return {"status": "fail", "message": f"Error creating/updating integration: {str(e)}"}


def get_integration(org_id: str, model: str) -> Optional[Dict[str, Any]]:
    """
    Fetch integration by org_id and model.
    
    Args:
        org_id: Organization ID
        model: Model identifier (e.g., 'openai', 'anthropic')
        
    Returns:
        Integration document or None
    """
    try:
        db = get_database()
        integration_table = db["Integrations"]
        
        integration = integration_table.find_one({
            "org_id": org_id,
            "model": model
        })
        
        if integration:
            integration.pop("_id", None)
            return integration

        return None

    except Exception as e:
        logger.error(f"Error fetching integration: {str(e)}")
        return None


def get_integration_response(
    org_id: str, model: str, *, mask_key: bool = False
) -> Optional[Dict[str, Any]]:
    """Fetch integration formatted for API response."""
    integration = get_integration(org_id, model)
    if not integration:
        return None
    return _to_response(integration, mask_key=mask_key)


def get_openai_api_key_for_org(org_id: str) -> Optional[str]:
    """
    Return the OpenAI API key for an org from Integrations.

    The dashboard stores provider display names (e.g. 'OpenAI'), not lowercase ids
    like 'openai', so we try several keys and a case-insensitive fallback scan.
    """
    for model in ("OpenAI", "openai"):
        doc = get_integration(org_id, model)
        if doc and doc.get("api_key"):
            key = str(doc["api_key"]).strip()
            if key:
                return key
    try:
        db = get_database()
        integration_table = db["Integrations"]
        for doc in integration_table.find({"org_id": org_id}):
            raw = (doc.get("model") or "").strip().lower().replace(" ", "")
            if raw == "openai" and doc.get("api_key"):
                key = str(doc["api_key"]).strip()
                if key:
                    return key
    except Exception as e:
        logger.error("Error scanning integrations for OpenAI key: %s", e)
    return None


def get_integration_public(org_id: str, model: str) -> Optional[Dict[str, Any]]:
    integration = get_integration(org_id, model)
    return _to_public_doc(integration) if integration else None


def get_integrations_by_org_public(org_id: str) -> List[Dict[str, Any]]:
    return [_to_public_doc(doc) for doc in get_integrations_by_org(org_id)]


def get_integrations_by_org(org_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all integrations for a given organization.

    Args:
        org_id: Organization ID

    Returns:
        List of integration documents
    """
    try:
        db = get_database()
        integration_table = db["Integrations"]

        integrations = list(integration_table.find({"org_id": org_id}))

        result = []
        for integration in integrations:
            # Remove MongoDB _id field
            integration.pop("_id", None)
            result.append(integration)

        return result

    except Exception as e:
        logger.error(f"Error fetching integrations: {str(e)}")
        return []


def delete_integration(org_id: str, model: str) -> Dict[str, Any]:
    """
    Delete an integration by org_id and model.
    
    Args:
        org_id: Organization ID
        model: Model identifier
        
    Returns:
        Dict with status and message
    """
    try:
        db = get_database()
        integration_table = db["Integrations"]
        
        result = integration_table.delete_one({
            "org_id": org_id,
            "model": model
        })
        
        if result.deleted_count == 0:
            return {"status": "fail", "message": "Integration not found"}
        
        logger.info(f"Integration deleted for org: {org_id}, model: {model}")
        return {"status": "success", "message": "Integration deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting integration: {str(e)}")
        return {"status": "fail", "message": f"Error deleting integration: {str(e)}"}
