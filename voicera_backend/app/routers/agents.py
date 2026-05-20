"""
Agent API routes.
"""
from fastapi import APIRouter, HTTPException, status, Depends, Query
from app.models.schemas import (
    AgentConfigCreate, AgentConfigResponse, AgentConfigUpdate,
    SuccessResponse, ErrorResponse
)
from app.services import agent_service, vobiz
from app.auth import get_current_user, verify_api_key
from typing import Dict, Any, List

router = APIRouter(prefix="/agents", tags=["agents"])


# ============================================================================
# Bot Endpoints (API Key Authentication)
# ============================================================================

@router.get("/config/{agent_type}", response_model=AgentConfigResponse)
async def get_agent_config_for_bot(
    agent_type: str,
    _: bool = Depends(verify_api_key)
):
    """
    Get agent configuration by agent_type (bot endpoint).
    
    Requires X-API-Key header for authentication.
    """
    agent = agent_service.fetch_agent_config(agent_type)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent type not found"
        )
    return agent


@router.get("/config/id/{agent_id}", response_model=AgentConfigResponse)
async def get_agent_config_by_id_for_bot(
    agent_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Get agent configuration by agent_id (bot endpoint).
    
    Requires X-API-Key header for authentication.
    """
    agent = agent_service.fetch_agent_config_by_id(agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent ID not found"
        )
    return agent


@router.get("/by-phone/{phone_number}", response_model=AgentConfigResponse)
async def get_agent_by_phone_number(
    phone_number: str,
    _: bool = Depends(verify_api_key)
):
    """
    Get agent configuration by phone number (bot endpoint).
    
    Requires X-API-Key header for authentication.
    Phone number format: +918071387434
    """
    # URL decode the phone number (+ becomes %2B in URLs)
    from urllib.parse import unquote
    decoded_phone = unquote(phone_number)
    
    # Use phone number as-is (format: +918071387434)
    agent = agent_service.fetch_agent_by_phone_number(decoded_phone)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No agent found for this phone number"
        )
    return agent


# ============================================================================
# Frontend Endpoints (User JWT Authentication)
# ============================================================================

@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_data: AgentConfigCreate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new agent configuration (protected endpoint).
    """
    if agent_data.org_id != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to create agents for this organization"
        )
    
    result = agent_service.create_agent(agent_data)
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result


@router.get("/org/{org_id}", response_model=List[AgentConfigResponse])
async def get_agents_by_org(
    org_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all agents for a given organization (protected endpoint).
    """
    if org_id != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this organization's agents"
        )
    
    agents = agent_service.fetch_agents_of_org(org_id)
    return agents


@router.get("/{agent_type}", response_model=AgentConfigResponse)
async def get_agent_config(
    agent_type: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get agent configuration by agent_type (protected endpoint).
    """
    agent = agent_service.fetch_agent_config(agent_type)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent type not found"
        )
    
    if agent.get("org_id") != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this agent"
        )
    
    return agent


@router.put("/{agent_type}", response_model=Dict[str, Any])
async def update_agent_config(
    agent_type: str,
    agent_data: AgentConfigUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update agent configuration (protected endpoint).
    """
    agent = agent_service.fetch_agent_config(agent_type)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent type not found"
        )
    
    if agent.get("org_id") != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this agent"
        )
    
    new_agent_type = (agent_data.agent_type or agent_type).strip()
    if (
        new_agent_type != agent_type
        and agent.get("telephony_provider") == "Vobiz"
        and agent.get("vobiz_app_id")
    ):
        vobiz_result = await vobiz.update_vobiz_application_name(
            current_user["org_id"],
            str(agent["vobiz_app_id"]),
            new_agent_type,
        )
        if vobiz_result["status"] == "fail":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to rename Vobiz application: {vobiz_result['message']}",
            )

    result = agent_service.update_agent_config(agent_type, agent_data, current_user["org_id"])
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result


@router.delete("/{agent_type}", response_model=Dict[str, Any])
async def delete_agent(
    agent_type: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete an agent configuration (protected endpoint).
    """
    agent = agent_service.fetch_agent_config_for_org(agent_type, current_user["org_id"])
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent type not found"
        )
    
    if agent.get("org_id") != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this agent"
        )
    
    result = agent_service.delete_agent(agent_type, current_user["org_id"])
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result


@router.delete("", response_model=Dict[str, Any])
async def delete_agent_by_query(
    agent_type: str = Query(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete an agent configuration by query param (safe for '/' in agent_type).
    """
    normalized_agent_type = agent_type.strip()
    if not normalized_agent_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="agent_type is required"
        )

    agent = agent_service.fetch_agent_config_for_org(normalized_agent_type, current_user["org_id"])
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent type not found"
        )

    result = agent_service.delete_agent(normalized_agent_type, current_user["org_id"])
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result
