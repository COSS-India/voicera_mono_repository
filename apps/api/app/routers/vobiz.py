"""
Vobiz API routes.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from app.models.schemas import (
    VobizApplicationCreate, VobizApplicationResponse,
    VobizNumberLink, VobizNumberUnlink
)
from app.services import vobiz, agent_service
from app.auth import get_current_user
from typing import Dict, Any

router = APIRouter(prefix="/vobiz", tags=["vobiz"])


@router.post("/application", response_model=VobizApplicationResponse, status_code=status.HTTP_201_CREATED)
async def create_vobiz_application_endpoint(
    request: VobizApplicationCreate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a Vobiz application for an agent (protected endpoint).
    
    Validates that the agent belongs to the user's organization before creating the application.
    """
    try:
        result = await vobiz.create_vobiz_application(
            current_user["org_id"],
            request.agent_type,
            request.answer_url,
        )
        
        if result["status"] == "fail":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating Vobiz application: {str(e)}"
        )
    
    return result


@router.get("/numbers", response_model=Dict[str, Any])
async def get_vobiz_numbers(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get phone numbers from Vobiz API (protected endpoint).
    Returns a list of e164 phone numbers. Uses org Vobiz credentials from Integrations.
    """
    result = await vobiz.get_vobiz_numbers(current_user["org_id"])
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Failed to fetch Vobiz numbers"),
        )
    return {"status": "success", "numbers": result.get("numbers", [])}


@router.delete("/application/{application_id}", response_model=Dict[str, Any])
async def delete_vobiz_application_endpoint(
    application_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete a Vobiz application (protected endpoint).
    """
    try:
        result = await vobiz.delete_vobiz_application(
            current_user["org_id"],
            application_id,
        )
        
        if result["status"] == "fail":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting Vobiz application: {str(e)}"
        )
    
    return result


@router.post("/numbers/link", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def link_number_to_application_endpoint(
    request: VobizNumberLink,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Link a phone number to a Vobiz application (protected endpoint).
    """
    try:
        result = await vobiz.link_number_to_application(
            current_user["org_id"],
            request.phone_number,
            request.application_id,
        )
        
        if result["status"] == "fail":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error linking phone number to application: {str(e)}"
        )
    
    return result


@router.delete("/numbers/unlink", response_model=Dict[str, Any])
async def unlink_number_from_application_endpoint(
    request: VobizNumberUnlink,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Unlink a phone number from a Vobiz application (protected endpoint).
    """
    try:
        result = await vobiz.unlink_number_from_application(
            current_user["org_id"],
            request.phone_number,
        )
        
        if result["status"] == "fail":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error unlinking phone number from application: {str(e)}"
        )
    
    return result
