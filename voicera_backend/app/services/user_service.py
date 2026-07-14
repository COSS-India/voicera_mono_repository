"""
User service for handling user-related database operations.
"""
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import os
import uuid
from app.database import get_database
from app.models.schemas import UserCreate, UserResponse, UserLoginResponse
from app.auth import create_access_token, get_password_hash, verify_password
from app.services.email_service import send_password_reset_email
from app.config import settings
import logging

logger = logging.getLogger(__name__)

def sign_up_user(user_data: UserCreate, client_ip: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new user. If org_id is provided (from invite link), user joins that org as member.
    Otherwise, a new org is created and user becomes the owner.

    Args:
        user_data: User creation data (optionally includes org_id for invite flow)
        client_ip: Requester IP, used to cap new-org creation per network (MAX_ORGS_PER_IP).
            Not enforced for the invite-join flow, so it's fine to omit for internal callers.

    Returns:
        Dict with status and message
    """
    try:
        db = get_database()
        user_table = db["UserTable"]
        members_table = db["Members"]
        
        # Check if user already exists in UserTable
        existing_user = user_table.find_one({"email": user_data.email})
        if existing_user:
            return {"status": "fail", "message": "User with this email already exists"}
        
        # Determine if this is a new org owner or a member joining existing org
        is_member = user_data.org_id is not None
        
        if is_member:
            # User is joining an existing org via invite link
            org_id = user_data.org_id
            
            # Verify the org exists (check if there's an owner with this org_id)
            org_exists = user_table.find_one({"org_id": org_id})
            if not org_exists:
                return {"status": "fail", "message": "Organization not found"}
            
            # Check if user is already a member of this org
            existing_member = members_table.find_one({"email": user_data.email, "org_id": org_id})
            if existing_member:
                return {"status": "fail", "message": "User is already a member of this organization"}
        else:
            # New user creating their own org — cap org creations per requester IP
            max_orgs_per_ip = int(os.getenv("MAX_ORGS_PER_IP", "1"))
            existing_count = db["OrgSignupsByIp"].count_documents({"ip": client_ip})
            if existing_count >= max_orgs_per_ip:
                return {"status": "fail", "message": "Org creation limit reached for this network"}

            org_id = str(uuid.uuid4()).replace('-', '')[:6]
        
        # Hash password before storing
        hashed_password = get_password_hash(user_data.password)
        
        user_doc = {
            "email": user_data.email,
            "password": hashed_password,
            "name": user_data.name,
            "org_id": org_id,
            "company_name": user_data.company_name,
            "is_member": is_member,  # True if joining existing org, False if creating new org
            "created_at": datetime.now().isoformat()
        }
        
        user_table.insert_one(user_doc)
        
        # If joining existing org, also add entry to Members table for easy lookup
        if is_member:
            member_mapping = {
                "email": user_data.email,
                "org_id": org_id,
                "created_at": datetime.now().isoformat()
            }
            members_table.insert_one(member_mapping)
            logger.info(f"Member created successfully: {user_data.email} in org: {org_id}")
        else:
            db["OrgSignupsByIp"].insert_one({
                "ip": client_ip,
                "org_id": org_id,
                "created_at": datetime.now().isoformat(),
            })
            logger.info(f"User (org owner) created successfully: {user_data.email}")
        
        return {"status": "success", "message": "User created successfully", "org_id": org_id}
        
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        return {"status": "fail", "message": f"Error creating user: {str(e)}"}


def validate_user_and_get_token(email: str, password: str) -> Dict[str, Any]:
    """
    Validate user credentials and return JWT access token.
    All users (both org owners and members) are in UserTable.
    Falls back to old Members table for backward compatibility.
    
    Args:
        email: User email
        password: User password
        
    Returns:
        Dict with status, message, access_token, token_type, and org_id if valid
    """
    try:
        db = get_database()
        user_table = db["UserTable"]
        
        user = user_table.find_one({"email": email})
        
        if not user:
            # For backward compatibility: check old Members table if user not in UserTable
            from app.services.member_service import validate_member_and_get_token
            member_result = validate_member_and_get_token(email, password)
            
            if member_result is None:
                return {"status": "fail", "message": "User not found"}
            
            return member_result
        
        # Verify password using bcrypt
        stored_password = user.get("password")
        if not verify_password(password, stored_password):
            return {"status": "fail", "message": "Invalid password"}
        
        # Create JWT token
        org_id = user.get("org_id")
        is_member = user.get("is_member", False)  # True for invited members, False for org owners
        token_data = {
            "sub": email,
            "org_id": org_id,
            "email": email,
            "is_member": is_member
        }
        access_token = create_access_token(data=token_data)
        
        return {
            "status": "success",
            "message": "User authenticated successfully",
            "access_token": access_token,
            "token_type": "bearer",
            "org_id": org_id
        }
            
    except Exception as e:
        logger.error(f"Error validating user: {str(e)}")
        return {"status": "fail", "message": f"Error validating user: {str(e)}"}

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """
    Get user by email. Checks UserTable first, then Members table.
    
    Args:
        email: User email
        
    Returns:
        User document or None
    """
    try:
        db = get_database()
        
        # First check UserTable (org owners)
        user_table = db["UserTable"]
        user = user_table.find_one({"email": email})
        
        if user:
            # Remove password and _id from response
            user.pop("password", None)
            user.pop("_id", None)
            return user
        
        # If not found, check Members table
        members_table = db["Members"]
        member = members_table.find_one({"email": email})
        
        if member:
            # Remove password and _id from response
            member.pop("password", None)
            member.pop("_id", None)
            return member
            
        return None
    except Exception as e:
        logger.error(f"Error fetching user: {str(e)}")
        return None

def request_password_reset(email: str) -> Dict[str, Any]:
    """
    Request password reset - generates token and sends email.
    
    Args:
        email: User email address
        
    Returns:
        Dict with status and message
    """
    try:
        db = get_database()
        user_table = db["UserTable"]
        
        user = user_table.find_one({"email": email})
        if not user:
            # Don't reveal if user exists for security
            return {"status": "success", "message": "If user exists, password reset email has been sent"}
        
        # Generate reset token
        reset_token = str(uuid.uuid4())
        expires_at = (datetime.now() + timedelta(hours=1)).isoformat()
        
        # Store token in user document
        user_table.update_one(
            {"email": email},
            {"$set": {
                "reset_token": reset_token,
                "reset_token_expires": expires_at,
                "reset_token_used": False
            }}
        )
        
        # Create reset URL
        reset_url = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
        
        # Send email
        email_sent = send_password_reset_email(email, reset_token, reset_url)
        
        if email_sent:
            logger.info(f"Password reset email sent to: {email}")
            return {"status": "success", "message": "Password reset email has been sent"}
        else:
            logger.warning(f"Failed to send password reset email to: {email}")
            return {"status": "fail", "message": "Failed to send password reset email. Please try again."}
            
    except Exception as e:
        logger.error(f"Error generating reset token: {str(e)}")
        return {"status": "fail", "message": f"Error: {str(e)}"}

def reset_password_with_token(token: str, new_password: str) -> Dict[str, Any]:
    """
    Reset password using reset token.
    
    Args:
        token: Password reset token
        new_password: New password to set
        
    Returns:
        Dict with status and message
    """
    try:
        db = get_database()
        user_table = db["UserTable"]
        
        # Find user with this token
        user = user_table.find_one({
            "reset_token": token,
            "reset_token_used": False
        })
        
        if not user:
            return {"status": "fail", "message": "Invalid or expired reset token"}
        
        # Check if token expired
        expires_at = datetime.fromisoformat(user.get("reset_token_expires"))
        if datetime.now() > expires_at:
            return {"status": "fail", "message": "Reset token has expired"}
        
        # Hash new password before storing
        hashed_password = get_password_hash(new_password)
        
        # Update password (stored as hashed)
        user_table.update_one(
            {"email": user.get("email")},
            {"$set": {
                "password": hashed_password,  # Store as hashed
                "reset_token_used": True
            }}
        )
        
        logger.info(f"Password reset successfully for: {user.get('email')}")
        return {"status": "success", "message": "Password reset successfully"}
        
    except Exception as e:
        logger.error(f"Error resetting password: {str(e)}")
        return {"status": "fail", "message": f"Error: {str(e)}"}

