"""
Agent API routes.
"""
from datetime import timedelta
from fastapi import APIRouter, HTTPException, status, Depends, Query, Body
from pydantic import BaseModel
from app.models.schemas import (
    AgentConfigCreate, AgentConfigResponse, AgentConfigUpdate,
    PublicAgentResponse, BroadcastTokenResponse,
    SuccessResponse, ErrorResponse
)
from app.services import agent_service, vobiz
from app.auth import get_current_user, verify_api_key, create_access_token, verify_token
from typing import Dict, Any, List

router = APIRouter(prefix="/agents", tags=["agents"])

# Host (presenter) broadcast tokens are short-lived: they only need to survive
# the moment between "Start broadcasting" and the publish WebSocket opening.
BROADCAST_TOKEN_EXPIRE_MINUTES = 10


class BroadcastResolveRequest(BaseModel):
    """Voice-server → backend: resolve a host broadcast token."""
    token: str


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


@router.get("/public/by-token/{share_token}", response_model=PublicAgentResponse)
async def get_public_agent_by_token_for_bot(
    share_token: str,
    _: bool = Depends(verify_api_key),
):
    """Resolve a share_token to a secret-stripped agent (voice-server endpoint).

    Includes agent_id because the voice server needs it to locate the room; the
    unauthenticated /public variant deliberately omits it.
    """
    agent = agent_service.fetch_agent_by_share_token(share_token)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found or disabled",
        )
    return agent_service.build_public_agent_projection(agent, include_agent_id=True)


@router.post("/broadcast/resolve", response_model=Dict[str, Any])
async def resolve_broadcast_token_for_bot(
    payload: BroadcastResolveRequest,
    _: bool = Depends(verify_api_key),
):
    """Verify a host broadcast token and return its agent_id/org_id.

    Keeps all JWT logic in the backend; the voice server only makes an
    X-API-Key HTTP call to authorise a presenter.
    """
    claims = verify_token(payload.token)
    if not claims or claims.get("role") != "host" or not claims.get("agent_id"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired broadcast token",
        )
    return {"agent_id": claims["agent_id"], "org_id": claims.get("org_id")}


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

    # Lazy backfill: orgs created before the demo-agent feature get one on first listing
    agent_service.ensure_default_agent_seeded(org_id)
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
    agent = agent_service.fetch_agent_config_for_org(agent_type, current_user["org_id"])
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
    agent = agent_service.fetch_agent_config_for_org(agent_type, current_user["org_id"])
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


@router.post("/{agent_type}/broadcast-token", response_model=BroadcastTokenResponse)
async def create_broadcast_token(
    agent_type: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Mint a short-lived host token so the owner can open the publish leg."""
    agent = agent_service.fetch_agent_config_for_org(agent_type, current_user["org_id"])
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent type not found",
        )
    agent_id = agent.get("agent_id")
    expires = timedelta(minutes=BROADCAST_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        {
            "sub": current_user["email"],
            "org_id": current_user["org_id"],
            "agent_id": agent_id,
            "role": "host",
        },
        expires_delta=expires,
    )
    return BroadcastTokenResponse(
        token=token,
        agent_id=agent_id,
        expires_in=int(expires.total_seconds()),
    )


@router.post("/{agent_type}/share/rotate", response_model=Dict[str, Any])
async def rotate_agent_share_token(
    agent_type: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Regenerate the public share_token, invalidating previously shared links."""
    agent = agent_service.fetch_agent_config_for_org(agent_type, current_user["org_id"])
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent type not found",
        )
    result = agent_service.rotate_share_token(agent_type, current_user["org_id"])
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"],
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
