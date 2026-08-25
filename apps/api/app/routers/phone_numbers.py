"""
Phone number API routes.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from app.models.schemas import (
    PhoneNumberAttachRequest, PhoneNumberDetachRequest, PhoneNumberResponse
)
from app.services import phone_number, agent_service
from app.auth import get_current_user
from typing import List, Dict, Any

router = APIRouter(prefix="/phone-numbers", tags=["phone-numbers"])


@router.get("/org/{org_id}", response_model=List[PhoneNumberResponse])
async def get_all_phone_numbers_by_org(
    org_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all phone numbers for a given organization (protected endpoint).
    """
    if org_id != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this organization's phone numbers"
        )
    
    phone_numbers = phone_number.get_all_phone_numbers_by_org(org_id)
    return phone_numbers


@router.get("/agent/{agent_type}", response_model=PhoneNumberResponse)
async def get_phone_number_by_agent_type(
    agent_type: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get the phone number attached to an agent by agent_type (protected endpoint).
    """
    # Validate that the agent belongs to the user's organization
    agent = agent_service.fetch_agent_config(agent_type)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent type not found"
        )
    
    if agent.get("org_id") != current_user["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this agent's phone number"
        )
    
    phone_number_doc = phone_number.get_phone_number_by_agent_type(
        agent_type,
        current_user["org_id"]
    )
    
    if not phone_number_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No phone number attached to this agent"
        )
    
    return phone_number_doc


@router.post("/attach", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def attach_phone_number_to_agent(
    request: PhoneNumberAttachRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Attach a phone number to an agent by agent_type (protected endpoint).
    If agent_type is provided, validates that the agent belongs to the user's organization.
    """
    # If agent_type is provided, validate that the agent belongs to the user's organization
    if request.agent_type:
        agent = agent_service.fetch_agent_config(request.agent_type)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent type not found"
            )
        
        if agent.get("org_id") != current_user["org_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to attach phone numbers to this agent"
            )
    
    result = phone_number.attach_phone_number_to_agent(
        request.phone_number,
        request.provider,
        request.agent_type,
        current_user["org_id"],
        current_user.get("email"),
    )
    
    if result["status"] == "fail":
        if "not found" in result["message"].lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["message"]
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    
    return result


@router.delete("/detach", response_model=Dict[str, Any])
async def detach_phone_number(
    request: PhoneNumberDetachRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Detach a phone number from an agent (protected endpoint).
    """
    result = phone_number.detach_phone_number(
        request.phone_number,
        current_user["org_id"],
        current_user.get("email"),
    )
    
    if result["status"] == "fail":
        if "not found" in result["message"].lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["message"]
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    
    return result
