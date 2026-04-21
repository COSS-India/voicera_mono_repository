"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

# User Models
class UserCreate(BaseModel):
    """Schema for creating a new user."""
    email: EmailStr
    password: str
    name: str
    company_name: str
    org_id: Optional[str] = None  # If provided (from invite link), user joins existing org as member

class UserResponse(BaseModel):
    """Schema for user response."""
    email: str
    name: str
    org_id: str
    company_name: str
    created_at: Optional[str] = None

class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str

class UserLoginResponse(BaseModel):
    """Schema for login response."""
    status: str
    message: str
    access_token: Optional[str] = None
    token_type: Optional[str] = None
    org_id: Optional[str] = None

class ForgotPasswordRequest(BaseModel):
    """Schema for forgot password request."""
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    """Schema for password reset with token."""
    token: str
    new_password: str

# Agent Models
class AgentConfigCreate(BaseModel):
    """Schema for creating agent config."""
    agent_type: str
    agent_id: str
    agent_config: Dict[str, Any]
    org_id: str
    agent_category: Optional[str] = None
    phone_number: Optional[str] = None
    app_id: Optional[str] = None
    greeting_message: Optional[str] = None
    telephony_provider: Optional[str] = None
    vobiz_app_id: Optional[str] = None
    vobiz_answer_url: Optional[str] = None

class AgentConfigResponse(BaseModel):
    """Schema for agent config response."""
    agent_type: str
    agent_id: str
    agent_config: Dict[str, Any]
    org_id: str
    agent_category: Optional[str] = None
    phone_number: Optional[str] = None
    app_id: Optional[str] = None
    telephony_provider: Optional[str] = None
    vobiz_app_id: Optional[str] = None
    vobiz_answer_url: Optional[str] = None
    updated_at: Optional[str] = None

class AgentConfigUpdate(BaseModel):
    """Schema for updating agent config."""
    agent_config: Dict[str, Any]
    agent_category: Optional[str] = None
    phone_number: Optional[str] = None
    app_id: Optional[str] = None
    greeting_message: Optional[str] = None
    telephony_provider: Optional[str] = None
    vobiz_app_id: Optional[str] = None
    vobiz_answer_url: Optional[str] = None

# Meeting Models
class MeetingCreate(BaseModel):
    meeting_id: str
    agent_type: str
    org_id: Optional[str] = None  # Make sure this field exists!
    start_time_utc: Optional[str] = None
    end_time_utc: Optional[str] = None
    inbound: Optional[bool] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    created_at: Optional[str] = None
    call_busy: Optional[bool] = None
    
    class Config:
        populate_by_name = True  # Allow both "from"/"from_number" and "to"/"to_number"

class MeetingResponse(BaseModel):
    """Schema for meeting response."""
    meeting_id: str
    agent_type: str
    org_id: Optional[str] = None
    agent_category: Optional[str] = None
    agent_config: Optional[Dict[str, Any]] = None
    inbound: Optional[bool] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    created_at: Optional[str] = None
    start_time_utc: Optional[str] = None
    end_time_utc: Optional[str] = None
    duration: Optional[float] = None
    recording_url: Optional[str] = None
    transcript_url: Optional[str] = None
    transcript_content: Optional[str] = None
    transcript: Optional[List[Dict[str, Any]]] = None
    call_busy: Optional[bool] = None    

class MeetingUpdate(BaseModel):
    """Schema for updating a meeting (e.g., when call ends)."""
    end_time_utc: str

# Campaign Models
class CampaignCreate(BaseModel):
    """Schema for creating a campaign."""
    campaign_name: str
    org_id: Optional[str] = None
    agent_type: Optional[str] = None
    status: Optional[str] = "active"
    campaign_information: Optional[Dict[str, Any]] = None

class CampaignResponse(BaseModel):
    """Schema for campaign response."""
    campaign_name: str
    org_id: Optional[str] = None
    agent_type: Optional[str] = None
    status: Optional[str] = None
    campaign_information: Optional[Dict[str, Any]] = None

# Audience Models
class AudienceCreate(BaseModel):
    """Schema for creating audience."""
    audience_name: str
    phone_number: str
    parameters: Optional[Dict[str, Any]] = None

class AudienceResponse(BaseModel):
    """Schema for audience response."""
    audience_name: str
    phone_number: str
    parameters: Optional[Dict[str, Any]] = None

# CallLog Models
class CallLogCreate(BaseModel):
    """Schema for creating call log."""
    meeting_id: str
    org_id: Optional[str] = None
    agent_type: Optional[str] = None
    phone_number: Optional[str] = None
    call_type: Optional[str] = None
    status: Optional[str] = None
    duration: Optional[float] = None
    price: Optional[float] = None

class CallLogResponse(BaseModel):
    """Schema for call log response."""
    meeting_id: str
    org_id: Optional[str] = None
    agent_type: Optional[str] = None
    phone_number: Optional[str] = None
    call_type: Optional[str] = None
    status: Optional[str] = None
    duration: Optional[float] = None
    price: Optional[float] = None
    created_at: Optional[str] = None

# Call Recording Models
class CallRecordingCreate(BaseModel):
    """Schema for creating/updating call recording data."""
    call_sid: str
    recording_url: str
    transcript_url: str
    transcript_content: Optional[str] = None
    agent_type: str
    call_duration: Optional[float] = None
    end_time_utc: Optional[str] = None
    org_id: Optional[str] = None

class CallRecordingResponse(BaseModel):
    """Schema for call recording response."""
    call_sid: str
    recording_url: Optional[str] = None
    transcript_url: Optional[str] = None
    transcript_content: Optional[str] = None
    agent_type: Optional[str] = None
    call_duration: Optional[float] = None
    start_time_utc: Optional[str] = None
    end_time_utc: Optional[str] = None
    org_id: Optional[str] = None

# Phone Number Models
class PhoneNumberAttachRequest(BaseModel):
    """Schema for attaching phone number to agent."""
    phone_number: str
    provider: str
    agent_type: Optional[str] = None

class PhoneNumberDetachRequest(BaseModel):
    """Schema for detaching phone number from agent."""
    phone_number: str

class PhoneNumberResponse(BaseModel):
    """Schema for phone number response."""
    phone_number: str
    provider: str
    agent_type: Optional[str] = None
    org_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

# Vobiz Models
class VobizApplicationCreate(BaseModel):
    """Schema for creating Vobiz application."""
    agent_type: str
    answer_url: str

class VobizApplicationResponse(BaseModel):
    """Schema for Vobiz application response."""
    status: str
    message: str
    app_id: Optional[str] = None

class VobizNumberLink(BaseModel):
    """Schema for linking phone number to Vobiz application."""
    phone_number: str
    application_id: str

class VobizNumberUnlink(BaseModel):
    """Schema for unlinking phone number from Vobiz application."""
    phone_number: str

# Generic Response Models
class SuccessResponse(BaseModel):
    """Generic success response."""
    status: str = "success"
    message: str

class ErrorResponse(BaseModel):
    """Generic error response."""
    status: str = "fail"
    message: str

# Analytics Models
class AgentBreakdown(BaseModel):
    """Schema for agent breakdown in analytics."""
    agent_type: str
    call_count: int

class AnalyticsResponse(BaseModel):
    """Schema for analytics response."""
    org_id: str
    calls_attempted: int
    calls_connected: int
    average_call_duration: float
    total_minutes_connected: float
    most_used_agent: Optional[str] = None
    most_used_agent_count: int = 0
    agent_breakdown: List[AgentBreakdown] = []
    calculated_at: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

# Integration Models
class IntegrationCreate(BaseModel):
    """Schema for creating/updating an integration."""
    org_id: str
    model: str
    api_key: str

class IntegrationResponse(BaseModel):
    """Schema for integration response."""
    org_id: str
    model: str
    api_key: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class IntegrationBotRequest(BaseModel):
    """Schema for bot requesting integration API key."""
    org_id: str
    model: str

# Member Models
class MemberCreate(BaseModel):
    """Schema for creating a new member (user) in an existing organization."""
    email: EmailStr
    password: str
    name: str
    company_name: str
    org_id: str  # Pre-filled from URL params

class MemberResponse(BaseModel):
    """Schema for member response."""
    email: str
    name: str
    org_id: str
    company_name: str
    created_at: Optional[str] = None

class MemberDelete(BaseModel):
    """Schema for deleting a member from an organization."""
    email: EmailStr
    org_id: str

# Knowledge base (org-scoped PDF ingest)
class KnowledgeDocumentResponse(BaseModel):
    """A knowledge PDF document and ingest status."""
    document_id: str
    org_id: str
    original_filename: str
    status: str  # processing | ready | failed
    chunk_count: Optional[int] = None
    embedding_model: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class KnowledgeUploadResponse(BaseModel):
    """Response after scheduling PDF ingest."""
    document_id: str
    org_id: str
    original_filename: str
    status: str


class KnowledgeDeleteResponse(BaseModel):
    """Response after deleting a knowledge document."""
    deleted: bool


class KnowledgeRetrieveRequest(BaseModel):
    """Request payload for retrieving knowledge chunks for a query."""
    org_id: str
    question: str
    top_k: int = 3
    document_ids: Optional[List[str]] = None


class KnowledgeRetrievedChunk(BaseModel):
    """One retrieved chunk from Chroma."""
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    source_filename: Optional[str] = None
    text: str
    distance: Optional[float] = None


class KnowledgeRetrieveResponse(BaseModel):
    """Response payload for knowledge retrieval."""
    chunks: List[KnowledgeRetrievedChunk]


# Batch CSV Models
class BatchResponse(BaseModel):
    """Batch metadata returned to dashboard."""
    batch_id: str
    org_id: str
    batch_name: str
    agent_type: str
    concurrency: int = 5
    original_filename: str
    status: str  # uploaded | ready_for_processing | processing | completed | failed
    execution_status: str  # not_started | queued | running | paused | completed | failed
    total_contacts: int
    valid_contacts: int
    invalid_contacts: int
    attempted_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    error_message: Optional[str] = None
    schedule_mode: str = "run_now"  # run_now | scheduled
    scheduled_at_utc: Optional[str] = None
    scheduled_timezone: Optional[str] = None
    scheduled_status: str = "none"  # none | scheduled | triggered | canceled
    scheduled_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class BatchUploadResponse(BaseModel):
    """Response after uploading and parsing a batch CSV."""
    batch_id: str
    org_id: str
    batch_name: str
    agent_type: str
    concurrency: int = 5
    original_filename: str
    status: str
    total_contacts: int
    valid_contacts: int
    invalid_contacts: int
    schedule_mode: str = "run_now"
    scheduled_status: str = "none"
    created_at: Optional[str] = None


class BatchDeleteResponse(BaseModel):
    """Response after deleting a batch and its stored contacts/file."""
    deleted: bool


class BatchRunRequest(BaseModel):
    """Optional runtime config when starting a batch."""
    agent_type: Optional[str] = None
    concurrency: Optional[int] = Field(default=None, ge=1, le=20)


class BatchScheduleRequest(BaseModel):
    """One-time scheduling payload for batch execution."""
    scheduled_at_local: str
    timezone: str
    agent_type: Optional[str] = None
    concurrency: Optional[int] = Field(default=None, ge=1, le=20)
