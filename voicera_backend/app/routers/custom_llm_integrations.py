"""
Custom LLM integration API routes.
"""
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import get_current_user, verify_api_key
from app.models.schemas import (
    CustomLLMBotRequest,
    CustomLLMBotResponse,
    CustomLLMIntegrationCreate,
    CustomLLMIntegrationResponse,
    CustomLLMIntegrationUpdate,
)
from app.services import custom_llm_integration_service

router = APIRouter(prefix="/custom-llm-integrations", tags=["custom-llm-integrations"])


@router.post("/bot/get-config", response_model=CustomLLMBotResponse)
async def get_custom_llm_config_for_bot(
    request: CustomLLMBotRequest,
    _: bool = Depends(verify_api_key),
):
    """Return full custom LLM config for the voice server."""
    integration = custom_llm_integration_service.get_custom_llm_integration_for_bot(
        request.org_id,
        request.custom_llm_id,
    )
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom LLM integration not found",
        )
    return integration


@router.get("", response_model=List[CustomLLMIntegrationResponse])
async def list_custom_llm_integrations(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    org_id = current_user["org_id"]
    return custom_llm_integration_service.get_custom_llm_integrations_by_org(org_id)


@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_custom_llm_integration(
    integration_data: CustomLLMIntegrationCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    if integration_data.org_id != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to create integrations for this organization",
        )

    result = custom_llm_integration_service.create_custom_llm_integration(integration_data)
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"],
        )
    return result


@router.put("/{custom_llm_id}", response_model=Dict[str, Any])
async def update_custom_llm_integration(
    custom_llm_id: str,
    update_data: CustomLLMIntegrationUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    org_id = current_user["org_id"]
    result = custom_llm_integration_service.update_custom_llm_integration(
        org_id,
        custom_llm_id,
        update_data,
    )
    if result["status"] == "fail":
        status_code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in result["message"].lower()
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=status_code, detail=result["message"])
    return result


@router.delete("/{custom_llm_id}", response_model=Dict[str, Any])
async def delete_custom_llm_integration(
    custom_llm_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    org_id = current_user["org_id"]
    result = custom_llm_integration_service.delete_custom_llm_integration(org_id, custom_llm_id)
    if result["status"] == "fail":
        status_code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in result["message"].lower()
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=status_code, detail=result["message"])
    return result
