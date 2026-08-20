"""
Public, unauthenticated routes.

Everything here is reachable without a JWT and must therefore only ever return
secret-stripped data. Access is gated by a per-agent opt-in flag and a random
share_token, so nothing is exposed unless an owner explicitly enables sharing.
"""
from fastapi import APIRouter, HTTPException, status

from app.models.schemas import PublicAgentResponse
from app.services import agent_service

router = APIRouter(prefix="/public", tags=["public"])


@router.get("/agents/{share_token}", response_model=PublicAgentResponse)
async def get_public_agent(share_token: str):
    """Resolve a share_token to a public agent projection for the share page."""
    agent = agent_service.fetch_agent_by_share_token(share_token)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found or disabled",
        )
    return agent_service.build_public_agent_projection(agent)
