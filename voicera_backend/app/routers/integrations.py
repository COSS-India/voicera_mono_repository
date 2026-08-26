"""
Integration API routes.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from app.models.schemas import (
    IntegrationCreate, IntegrationResponse, IntegrationPublicResponse, IntegrationBotRequest,
    SuccessResponse, ErrorResponse
)
from app.services import integration_service
from app.services.member_service import is_org_owner
from app.auth import get_current_user, verify_api_key
from typing import Dict, Any, List

router = APIRouter(prefix="/integrations", tags=["integrations"])


def _require_org_owner(current_user: Dict[str, Any]) -> str:
    org_id = current_user["org_id"]
    if not is_org_owner(current_user["email"], org_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only organization owners can manage integration credentials",
        )
    return org_id


# ============================================================================
# Bot Endpoint (API Key Authentication)
# ============================================================================

@router.post("/bot/get-api-key", response_model=IntegrationResponse)
async def get_integration_for_bot(
    request: IntegrationBotRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Get integration API key for bot (bot endpoint).
    
    Requires X-API-Key header for authentication.
    Used by bot.py to retrieve API keys for LLM providers.
    """
    integration = integration_service.get_integration_response(
        request.org_id, request.model, mask_key=False
    )
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Integration not found for model: {request.model}",
        )
    return integration


# ============================================================================
# Frontend Endpoints (User JWT Authentication)
# ============================================================================

@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_integration(
    integration_data: IntegrationCreate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create or update an integration (protected endpoint, owners only).

    If an integration for the same org_id and model already exists, it will be updated.
    """
    if integration_data.org_id != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to create integrations for this organization"
        )

    _require_org_owner(current_user)
    
    result = integration_service.create_integration(integration_data)
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result


@router.get("/{model}", response_model=IntegrationPublicResponse, response_model_exclude_unset=True)
async def get_integration(
    model: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get integration by model for the current user's organization (protected endpoint).

    The api_key is never returned to the frontend.
    """
    org_id = current_user["org_id"]
    integration = integration_service.get_integration_public(org_id, model)

    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Integration not found for model: {model}"
        )

    return integration


@router.get("", response_model=List[IntegrationPublicResponse], response_model_exclude_unset=True)
async def get_all_integrations(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all integrations for the current user's organization (protected endpoint).

    The api_key is never returned to the frontend.
    """
    org_id = current_user["org_id"]
    integrations = integration_service.get_integrations_by_org_public(org_id)
    return integrations


@router.delete("/{model}", response_model=Dict[str, Any])
async def delete_integration(
    model: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete an integration by model (protected endpoint, owners only).
    """
    org_id = _require_org_owner(current_user)
    
    # Check if integration exists
    integration = integration_service.get_integration(org_id, model)
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Integration not found for model: {model}"
        )
    
    result = integration_service.delete_integration(org_id, model)
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result
