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
from app.config import settings
from typing import Dict, Any
import httpx

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
            request.agent_type,
            request.answer_url
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
    Returns a list of e164 phone numbers.
    """
    # Validate that Vobiz credentials are configured
    if not settings.VOBIZ_ACCOUNT_ID or not settings.VOBIZ_AUTH_ID or not settings.VOBIZ_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vobiz API credentials are not configured"
        )
    
    # Construct the Vobiz API URL - use VOBIZ_ACCOUNT_ID in the path
    url = f"{settings.VOBIZ_API_BASE_URL}/account/{settings.VOBIZ_AUTH_ID}/numbers"
    
    # Prepare headers
    headers = {
        "X-Auth-ID": settings.VOBIZ_AUTH_ID,
        "X-Auth-Token": settings.VOBIZ_AUTH_TOKEN
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            e164_numbers = [item.get("e164") for item in data.get("items", []) if item.get("e164")]
            return {"status": "success", "numbers": e164_numbers}
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Vobiz API error: {e.response.text}"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to Vobiz API: {str(e)}"
        )


@router.delete("/application/{application_id}", response_model=Dict[str, Any])
async def delete_vobiz_application_endpoint(
    application_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete a Vobiz application (protected endpoint).
    """
    try:
        result = await vobiz.delete_vobiz_application(application_id)
        
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
            request.phone_number,
            request.application_id
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
        result = await vobiz.unlink_number_from_application(request.phone_number)
        
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
