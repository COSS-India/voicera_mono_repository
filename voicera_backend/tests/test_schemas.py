"""
Pydantic schema validation tests.

Ensures request/response models enforce their constraints correctly
so that invalid data never reaches the service layer.
"""

import os
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-voicera-tests!")

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    UserCreate,
    UserLogin,
    AgentConfigCreate,
    AgentConfigUpdate,
    CampaignCreate,
    MeetingCreate,
    BatchScheduleRequest,
    BatchRunRequest,
    KnowledgeRetrieveRequest,
    PhoneNumberAttachRequest,
    ResetPasswordRequest,
)


# ── UserCreate ─────────────────────────────────────────────────────────────

class TestUserCreate:
    def test_valid_user(self):
        u = UserCreate(
            email="user@example.com",
            password="secret",
            name="Alice",
            company_name="ACME",
        )
        assert u.email == "user@example.com"

    def test_invalid_email_raises(self):
        with pytest.raises(ValidationError):
            UserCreate(email="not-an-email", password="x", name="A", company_name="B")

    def test_missing_email_raises(self):
        with pytest.raises(ValidationError):
            UserCreate(password="x", name="A", company_name="B")

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            UserCreate(email="a@b.com", password="x", company_name="B")

    def test_optional_org_id_defaults_none(self):
        u = UserCreate(email="a@b.com", password="x", name="A", company_name="B")
        assert u.org_id is None

    def test_org_id_accepted_when_provided(self):
        u = UserCreate(
            email="a@b.com", password="x", name="A", company_name="B", org_id="myorg"
        )
        assert u.org_id == "myorg"


# ── UserLogin ──────────────────────────────────────────────────────────────

class TestUserLogin:
    def test_valid_login(self):
        ul = UserLogin(email="user@example.com", password="pass")
        assert ul.email == "user@example.com"

    def test_invalid_email_raises(self):
        with pytest.raises(ValidationError):
            UserLogin(email="bad-email", password="pass")


# ── AgentConfigCreate ──────────────────────────────────────────────────────

class TestAgentConfigCreate:
    def test_valid_minimal(self):
        a = AgentConfigCreate(
            agent_type="sales",
            agent_id="agent-001",
            agent_config={"prompt": "hello"},
            org_id="org1",
        )
        assert a.agent_type == "sales"

    def test_missing_agent_type_raises(self):
        with pytest.raises(ValidationError):
            AgentConfigCreate(agent_id="x", agent_config={}, org_id="o")

    def test_optional_fields_default_none(self):
        a = AgentConfigCreate(
            agent_type="t", agent_id="i", agent_config={}, org_id="o"
        )
        assert a.phone_number is None
        assert a.telephony_provider is None
        assert a.greeting_message is None

    def test_full_fields_accepted(self):
        a = AgentConfigCreate(
            agent_type="support",
            agent_id="a-002",
            agent_config={"key": "val"},
            org_id="org2",
            agent_category="inbound",
            phone_number="+911234567890",
            telephony_provider="Plivo",
            greeting_message="Hello there",
        )
        assert a.phone_number == "+911234567890"
        assert a.telephony_provider == "Plivo"


# ── AgentConfigUpdate ──────────────────────────────────────────────────────

class TestAgentConfigUpdate:
    def test_agent_config_required(self):
        with pytest.raises(ValidationError):
            AgentConfigUpdate()

    def test_agent_config_dict_accepted(self):
        u = AgentConfigUpdate(agent_config={"prompt": "new prompt"})
        assert u.agent_config == {"prompt": "new prompt"}

    def test_all_optional_fields_none_by_default(self):
        u = AgentConfigUpdate(agent_config={})
        assert u.agent_type is None
        assert u.phone_number is None


# ── CampaignCreate ─────────────────────────────────────────────────────────

class TestCampaignCreate:
    def test_valid_minimal(self):
        c = CampaignCreate(campaign_name="Q1 Outreach")
        assert c.campaign_name == "Q1 Outreach"
        assert c.status == "active"

    def test_missing_campaign_name_raises(self):
        with pytest.raises(ValidationError):
            CampaignCreate()

    def test_custom_status(self):
        c = CampaignCreate(campaign_name="c", status="paused")
        assert c.status == "paused"


# ── MeetingCreate ──────────────────────────────────────────────────────────

class TestMeetingCreate:
    def test_valid_minimal(self):
        m = MeetingCreate(meeting_id="m-001", agent_type="sales")
        assert m.meeting_id == "m-001"

    def test_missing_meeting_id_raises(self):
        with pytest.raises(ValidationError):
            MeetingCreate(agent_type="sales")

    def test_optional_fields_accepted(self):
        m = MeetingCreate(
            meeting_id="m-001",
            agent_type="sales",
            org_id="org1",
            inbound=True,
            from_number="+910000000001",
            to_number="+910000000002",
        )
        assert m.inbound is True
        assert m.from_number == "+910000000001"


# ── BatchScheduleRequest ───────────────────────────────────────────────────

class TestBatchScheduleRequest:
    def test_valid_schedule(self):
        b = BatchScheduleRequest(
            scheduled_at_local="2024-06-01T09:00:00",
            timezone="Asia/Kolkata",
        )
        assert b.timezone == "Asia/Kolkata"

    def test_concurrency_below_minimum_raises(self):
        with pytest.raises(ValidationError):
            BatchScheduleRequest(
                scheduled_at_local="2024-06-01T09:00:00",
                timezone="UTC",
                concurrency=0,
            )

    def test_concurrency_above_maximum_raises(self):
        with pytest.raises(ValidationError):
            BatchScheduleRequest(
                scheduled_at_local="2024-06-01T09:00:00",
                timezone="UTC",
                concurrency=21,
            )

    def test_concurrency_at_bounds_accepted(self):
        low = BatchScheduleRequest(
            scheduled_at_local="2024-06-01T09:00:00", timezone="UTC", concurrency=1
        )
        high = BatchScheduleRequest(
            scheduled_at_local="2024-06-01T09:00:00", timezone="UTC", concurrency=20
        )
        assert low.concurrency == 1
        assert high.concurrency == 20


# ── BatchRunRequest ────────────────────────────────────────────────────────

class TestBatchRunRequest:
    def test_empty_request_valid(self):
        b = BatchRunRequest()
        assert b.concurrency is None

    def test_concurrency_bounds_enforced(self):
        with pytest.raises(ValidationError):
            BatchRunRequest(concurrency=0)
        with pytest.raises(ValidationError):
            BatchRunRequest(concurrency=25)


# ── KnowledgeRetrieveRequest ───────────────────────────────────────────────

class TestKnowledgeRetrieveRequest:
    def test_valid_request(self):
        r = KnowledgeRetrieveRequest(org_id="org1", question="What is X?")
        assert r.top_k == 3

    def test_custom_top_k(self):
        r = KnowledgeRetrieveRequest(org_id="o", question="q", top_k=5)
        assert r.top_k == 5

    def test_missing_question_raises(self):
        with pytest.raises(ValidationError):
            KnowledgeRetrieveRequest(org_id="o")


# ── PhoneNumberAttachRequest ───────────────────────────────────────────────

class TestPhoneNumberAttachRequest:
    def test_valid(self):
        r = PhoneNumberAttachRequest(phone_number="+911234567890", provider="Plivo")
        assert r.provider == "Plivo"

    def test_missing_provider_raises(self):
        with pytest.raises(ValidationError):
            PhoneNumberAttachRequest(phone_number="+91000")


# ── ResetPasswordRequest ───────────────────────────────────────────────────

class TestResetPasswordRequest:
    def test_valid(self):
        r = ResetPasswordRequest(token="abc123", new_password="newpass")
        assert r.token == "abc123"

    def test_missing_fields_raise(self):
        with pytest.raises(ValidationError):
            ResetPasswordRequest(token="abc")
