"""
User API routes.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from app.models.schemas import (
    UserCreate, UserResponse, UserLogin, UserLoginResponse,
    ForgotPasswordRequest, ResetPasswordRequest,
    SuccessResponse, ErrorResponse
)
from app.services import user_service
from app.auth import get_current_user
from typing import Dict, Any

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/signup", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def sign_up(user_data: UserCreate):
    """
    Create a new user account (public endpoint).
    Generates a new org_id for the user.
    """
    result = user_service.sign_up_user(user_data)
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result


@router.post("/login", response_model=UserLoginResponse)
async def login(credentials: UserLogin):
    """
    Authenticate user and get JWT access token (public endpoint).
    Token expires in 30 minutes.
    """
    result = user_service.validate_user_and_get_token(
        credentials.email,
        credentials.password
    )
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=result["message"]
        )
    return result

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current authenticated user's information (protected endpoint).
    Checks both UserTable and Members table.
    """
    user = user_service.get_user_by_email(current_user["email"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    user["is_owner"] = not user.get("is_member", False)
    return user


@router.get("/{email}", response_model=UserResponse)
async def get_user(email: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get user details by email (protected endpoint).
    """
    # Users can only access their own data or admins can access any
    if current_user["email"] != email:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this user's data"
        )
    
    user = user_service.get_user_by_email(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@router.post("/forgot-password", response_model=Dict[str, Any])
async def forgot_password(request: ForgotPasswordRequest):
    """
    Request password reset (public endpoint).
    Sends reset password email via Mailtrap.
    """
    result = user_service.request_password_reset(request.email)
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result

@router.post("/reset-password", response_model=Dict[str, Any])
async def reset_password(request: ResetPasswordRequest):
    """
    Reset password using reset token (public endpoint).
    """
    result = user_service.reset_password_with_token(
        request.token,
        request.new_password
    )
    if result["status"] == "fail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result

