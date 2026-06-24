"""
Member service for handling member-related database operations.
Members table stores {email, org_id} mappings for easy org membership lookup.
Actual user data (password, name, etc.) is stored in UserTable.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.database import get_database
from app.models.schemas import MemberCreate, MemberResponse, MemberDelete, UserCreate
from app.auth import get_password_hash, verify_password, create_access_token
import logging

logger = logging.getLogger(__name__)


def is_org_owner(email: str, org_id: str) -> bool:
    """Return True if the user is the organization owner."""
    try:
        db = get_database()
        user = db["UserTable"].find_one({"email": email, "org_id": org_id})
        return bool(user and not user.get("is_member", False))
    except Exception as e:
        logger.error(f"Error checking org owner: {str(e)}")
        return False


def add_member(member_data: MemberCreate) -> Dict[str, Any]:
    """
    Add a new member to an organization.
    Creates user in UserTable and adds membership mapping to Members table.
    
    Args:
        member_data: Member creation data including org_id
        
    Returns:
        Dict with status and message
    """
    # Use the user signup flow with org_id provided (member joining existing org)
    from app.services.user_service import sign_up_user
    
    user_create_data = UserCreate(
        email=member_data.email,
        password=member_data.password,
        name=member_data.name,
        company_name=member_data.company_name,
        org_id=member_data.org_id  # This triggers the "join existing org" flow
    )
    
    result = sign_up_user(user_create_data)
    
    if result["status"] == "success":
        logger.info(f"Member created successfully: {member_data.email} in org: {member_data.org_id}")
    
    return result


def get_members_by_org(org_id: str) -> Dict[str, Any]:
    """
    Get all members for an organization from UserTable.
    All users (both org owner and invited members) are in UserTable.
    
    Args:
        org_id: Organization ID
        
    Returns:
        Dict with status and list of members
    """
    try:
        db = get_database()
        users_table = db["UserTable"]
        
        members = []
        
        # Get all users with this org_id from UserTable
        users_cursor = users_table.find({"org_id": org_id})
        
        for user in users_cursor:
            member = {
                "email": user.get("email"),
                "name": user.get("name"),
                "org_id": user.get("org_id"),
                "company_name": user.get("company_name"),
                "created_at": user.get("created_at"),
                "is_owner": not user.get("is_member", False)  # Owner if is_member is False/missing
            }
            members.append(member)
        
        return {"status": "success", "members": members, "count": len(members)}
        
    except Exception as e:
        logger.error(f"Error fetching members: {str(e)}")
        return {"status": "fail", "message": f"Error fetching members: {str(e)}"}


def delete_member(member_data: MemberDelete, caller_email: str) -> Dict[str, Any]:
    """
    Delete a member from an organization.
    Removes user from both UserTable and Members mapping table.
    Only the org owner can remove members. Org owners cannot be deleted.
    
    Args:
        member_data: Member deletion data (email and org_id)
        caller_email: Email of the user performing the deletion
        
    Returns:
        Dict with status and message
    """
    try:
        if not is_org_owner(caller_email, member_data.org_id):
            return {
                "status": "fail",
                "message": "Only the organization owner can remove members",
            }

        db = get_database()
        users_table = db["UserTable"]
        members_table = db["Members"]
        
        # Check if user exists in UserTable with this org_id
        existing_user = users_table.find_one({
            "email": member_data.email,
            "org_id": member_data.org_id
        })
        
        if not existing_user:
            return {"status": "fail", "message": "Member not found in this organization"}
        
        # Don't allow deleting the org owner (is_member must be True)
        if not existing_user.get("is_member", False):
            return {"status": "fail", "message": "Cannot delete the organization owner"}
        
        # Delete from UserTable - only if is_member is True (extra safety)
        user_result = users_table.delete_one({
            "email": member_data.email,
            "org_id": member_data.org_id,
            "is_member": True
        })
        
        # Also delete from Members mapping table
        members_table.delete_one({
            "email": member_data.email,
            "org_id": member_data.org_id
        })
        
        if user_result.deleted_count > 0:
            logger.info(f"Member deleted successfully: {member_data.email} from org: {member_data.org_id}")
            return {"status": "success", "message": "Member deleted successfully"}
        else:
            return {"status": "fail", "message": "Failed to delete member"}
        
    except Exception as e:
        logger.error(f"Error deleting member: {str(e)}")
        return {"status": "fail", "message": f"Error deleting member: {str(e)}"}


def transfer_ownership(org_id: str, caller_email: str, new_owner_email: str) -> Dict[str, Any]:
    """
    Transfer organization ownership to another member.
    Only the current owner can transfer. Caller becomes a regular member.
    """
    try:
        if caller_email == new_owner_email:
            return {"status": "fail", "message": "Cannot transfer ownership to yourself"}

        if not is_org_owner(caller_email, org_id):
            return {
                "status": "fail",
                "message": "Only the organization owner can transfer ownership",
            }

        db = get_database()
        users_table = db["UserTable"]

        new_owner = users_table.find_one({"email": new_owner_email, "org_id": org_id})
        if not new_owner:
            return {"status": "fail", "message": "Member not found in this organization"}
        if not new_owner.get("is_member", False):
            return {"status": "fail", "message": "This user is already the organization owner"}

        caller = users_table.find_one({"email": caller_email, "org_id": org_id})
        if not caller:
            return {"status": "fail", "message": "Caller not found in this organization"}

        users_table.update_one(
            {"email": caller_email, "org_id": org_id},
            {"$set": {"is_member": True}},
        )
        users_table.update_one(
            {"email": new_owner_email, "org_id": org_id},
            {"$set": {"is_member": False}},
        )

        logger.info(
            f"Ownership transferred in org {org_id}: {caller_email} -> {new_owner_email}"
        )
        return {"status": "success", "message": "Ownership transferred successfully"}

    except Exception as e:
        logger.error(f"Error transferring ownership: {str(e)}")
        return {"status": "fail", "message": f"Error transferring ownership: {str(e)}"}


def validate_member_and_get_token(email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Validate member credentials and return JWT access token.
    This is called from user_service when user is not found in UserTable.
    
    Args:
        email: Member email
        password: Member password
        
    Returns:
        Dict with status, message, access_token, token_type, and org_id if valid
        None if member not found
    """
    try:
        db = get_database()
        members_table = db["Members"]
        
        # Find member by email (there could be multiple with same email in different orgs)
        # For login, we find the first match - user should use org-specific login if in multiple orgs
        member = members_table.find_one({"email": email})
        
        if not member:
            return None
        
        # Verify password using bcrypt
        stored_password = member.get("password")
        if not verify_password(password, stored_password):
            return {"status": "fail", "message": "Invalid password"}
        
        # Create JWT token
        org_id = member.get("org_id")
        token_data = {
            "sub": email,
            "org_id": org_id,
            "email": email,
            "is_member": True
        }
        access_token = create_access_token(data=token_data)
        
        return {
            "status": "success",
            "message": "Member authenticated successfully",
            "access_token": access_token,
            "token_type": "bearer",
            "org_id": org_id
        }
            
    except Exception as e:
        logger.error(f"Error validating member: {str(e)}")
        return {"status": "fail", "message": f"Error validating member: {str(e)}"}
