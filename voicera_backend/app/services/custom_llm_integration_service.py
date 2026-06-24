"""
Custom LLM integration service for OpenAI-compatible chat completion endpoints.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from bson import ObjectId
from bson.errors import InvalidId

from app.database import get_database
from app.models.schemas import CustomLLMIntegrationCreate, CustomLLMIntegrationUpdate

logger = logging.getLogger(__name__)

_COLLECTION = "CustomLLMIntegrations"


def normalize_base_url(url: str) -> str:
    """Normalize a user-entered URL to an OpenAI-compatible base URL ending in /v1."""
    raw = (url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("Endpoint URL is required")

    if raw.endswith("/chat/completions"):
        raw = raw[: -len("/chat/completions")].rstrip("/")

    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Endpoint URL must be a valid http(s) URL")

    if not raw.endswith("/v1"):
        raw = f"{raw}/v1"

    return raw


def _mask_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if len(key) <= 4:
        return "****"
    return f"{'*' * (len(key) - 4)}{key[-4:]}"


def _doc_to_response(doc: Dict[str, Any], *, mask_key: bool = False) -> Dict[str, Any]:
    api_key = doc.get("api_key", "")
    return {
        "id": str(doc["_id"]),
        "org_id": doc["org_id"],
        "name": doc["name"],
        "base_url": doc["base_url"],
        "model": doc["model"],
        "api_key": _mask_api_key(api_key) if mask_key else api_key,
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def _parse_object_id(custom_llm_id: str) -> ObjectId:
    try:
        return ObjectId(custom_llm_id)
    except (InvalidId, TypeError) as exc:
        raise ValueError("Invalid custom LLM id") from exc


def create_custom_llm_integration(
    integration_data: CustomLLMIntegrationCreate,
) -> Dict[str, Any]:
    try:
        db = get_database()
        table = db[_COLLECTION]
        now = datetime.now().isoformat()
        base_url = normalize_base_url(integration_data.base_url)

        doc = {
            "org_id": integration_data.org_id,
            "name": integration_data.name.strip(),
            "base_url": base_url,
            "api_key": integration_data.api_key.strip(),
            "model": integration_data.model.strip(),
            "created_at": now,
            "updated_at": now,
        }
        result = table.insert_one(doc)
        doc["_id"] = result.inserted_id
        logger.info(
            "Custom LLM integration created for org=%s name=%s",
            integration_data.org_id,
            integration_data.name,
        )
        return {"status": "success", "integration": _doc_to_response(doc)}
    except ValueError as exc:
        return {"status": "fail", "message": str(exc)}
    except Exception as exc:
        logger.error("Error creating custom LLM integration: %s", exc)
        return {"status": "fail", "message": f"Error creating custom LLM integration: {exc}"}


def get_custom_llm_integration(org_id: str, custom_llm_id: str) -> Optional[Dict[str, Any]]:
    try:
        db = get_database()
        table = db[_COLLECTION]
        doc = table.find_one({"_id": _parse_object_id(custom_llm_id), "org_id": org_id})
        if not doc:
            return None
        return _doc_to_response(doc)
    except ValueError:
        return None
    except Exception as exc:
        logger.error("Error fetching custom LLM integration: %s", exc)
        return None


def get_custom_llm_integration_for_bot(
    org_id: str,
    custom_llm_id: str,
) -> Optional[Dict[str, Any]]:
    """Return full config including unmasked api_key for voice server."""
    try:
        db = get_database()
        table = db[_COLLECTION]
        doc = table.find_one({"_id": _parse_object_id(custom_llm_id), "org_id": org_id})
        if not doc:
            return None
        return _doc_to_response(doc, mask_key=False)
    except ValueError:
        return None
    except Exception as exc:
        logger.error("Error fetching custom LLM integration for bot: %s", exc)
        return None


def get_custom_llm_integrations_by_org(org_id: str) -> List[Dict[str, Any]]:
    try:
        db = get_database()
        table = db[_COLLECTION]
        docs = list(table.find({"org_id": org_id}).sort("created_at", 1))
        return [_doc_to_response(doc) for doc in docs]
    except Exception as exc:
        logger.error("Error listing custom LLM integrations: %s", exc)
        return []


def update_custom_llm_integration(
    org_id: str,
    custom_llm_id: str,
    update_data: CustomLLMIntegrationUpdate,
) -> Dict[str, Any]:
    try:
        db = get_database()
        table = db[_COLLECTION]
        oid = _parse_object_id(custom_llm_id)
        existing = table.find_one({"_id": oid, "org_id": org_id})
        if not existing:
            return {"status": "fail", "message": "Custom LLM integration not found"}

        updates: Dict[str, Any] = {}
        if update_data.name is not None:
            updates["name"] = update_data.name.strip()
        if update_data.base_url is not None:
            updates["base_url"] = normalize_base_url(update_data.base_url)
        if update_data.api_key is not None:
            updates["api_key"] = update_data.api_key.strip()
        if update_data.model is not None:
            updates["model"] = update_data.model.strip()

        if not updates:
            return {"status": "fail", "message": "No fields to update"}

        updates["updated_at"] = datetime.now().isoformat()
        table.update_one({"_id": oid, "org_id": org_id}, {"$set": updates})
        updated = table.find_one({"_id": oid, "org_id": org_id})
        return {"status": "success", "integration": _doc_to_response(updated)}
    except ValueError as exc:
        return {"status": "fail", "message": str(exc)}
    except Exception as exc:
        logger.error("Error updating custom LLM integration: %s", exc)
        return {"status": "fail", "message": f"Error updating custom LLM integration: {exc}"}


def delete_custom_llm_integration(org_id: str, custom_llm_id: str) -> Dict[str, Any]:
    try:
        db = get_database()
        table = db[_COLLECTION]
        result = table.delete_one(
            {"_id": _parse_object_id(custom_llm_id), "org_id": org_id}
        )
        if result.deleted_count == 0:
            return {"status": "fail", "message": "Custom LLM integration not found"}
        logger.info("Custom LLM integration deleted for org=%s id=%s", org_id, custom_llm_id)
        return {"status": "success", "message": "Custom LLM integration deleted successfully"}
    except ValueError as exc:
        return {"status": "fail", "message": str(exc)}
    except Exception as exc:
        logger.error("Error deleting custom LLM integration: %s", exc)
        return {"status": "fail", "message": f"Error deleting custom LLM integration: {exc}"}
