"""
Member API routes.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from app.models.schemas import (
    MemberCreate, MemberResponse, MemberDelete, TransferOwnership,
    SuccessResponse, ErrorResponse
)
from app.services import member_service
from app.auth import get_current_user
from typing import Dict, Any, List

router = APIRouter(prefix="/members", tags=["members"])


@router.post("/add-member", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def add_member(member_data: MemberCreate):
    """
    Add a new member to an organization (public endpoint).
    This is used for invite links where new users don't have authentication yet.
    The org_id in the request body determines which organization the member joins.
    """
    result = member_service.add_member(member_data)
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result


@router.get("/{org_id}", response_model=Dict[str, Any])
async def get_members(
    org_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all members for an organization (protected endpoint).
    """
    # Optionally verify user belongs to this org
    if current_user.get("org_id") != org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access members of this organization"
        )
    
    result = member_service.get_members_by_org(org_id)
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
    return result


@router.post("/delete-member", response_model=Dict[str, Any])
async def delete_member(
    member_data: MemberDelete,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete a member from an organization (protected endpoint).
    """
    # Optionally verify user belongs to this org
    if current_user.get("org_id") != member_data.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete members from this organization"
        )
    
    result = member_service.delete_member(member_data, current_user["email"])
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result


@router.post("/transfer-ownership", response_model=Dict[str, Any])
async def transfer_ownership(
    transfer_data: TransferOwnership,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Transfer organization ownership to another member (owner only)."""
    if current_user.get("org_id") != transfer_data.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to transfer ownership for this organization",
        )

    result = member_service.transfer_ownership(
        transfer_data.org_id,
        current_user["email"],
        transfer_data.email,
    )
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"],
        )
    return result
