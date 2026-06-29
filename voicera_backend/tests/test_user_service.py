"""
Unit tests for app.services.user_service.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from app.services.user_service import (
    sign_up_user,
    validate_user_and_get_token,
    get_user_by_email,
    request_password_reset,
    reset_password_with_token,
)
from app.models.schemas import UserCreate

# ── Sample data ───────────────────────────────────────────────────────────────

ORG_ID = "testorg1"
EMAIL = "alice@example.com"
PASSWORD = "secret123"
HASHED = "hashed_secret"
TOKEN = "jwt-access-token"
RESET_TOKEN = "reset-uuid-token"

USER_DOC = {
    "email": EMAIL,
    "password": HASHED,
    "name": "Alice",
    "org_id": ORG_ID,
    "company_name": "ACME",
    "is_member": False,
    "created_at": "2024-01-01T00:00:00",
}

MEMBER_DOC = {**USER_DOC, "is_member": True}


def _make_db(users_coll=None, members_coll=None):
    users_coll = users_coll or MagicMock()
    members_coll = members_coll or MagicMock()
    db = MagicMock()
    db.__getitem__.side_effect = lambda k: (
        users_coll if k == "UserTable" else
        members_coll if k == "Members" else
        MagicMock()
    )
    return db, users_coll, members_coll


def _user_create(org_id=None):
    return UserCreate(
        email=EMAIL, password=PASSWORD,
        name="Alice", company_name="ACME", org_id=org_id
    )


# ── TestSignUpUser ────────────────────────────────────────────────────────

class TestSignUpUser:
    def test_owner_signup_creates_new_org_id(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = None
        db, users_coll, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db), \
             patch("app.services.user_service.get_password_hash", return_value=HASHED):
            result = sign_up_user(_user_create(org_id=None))
        assert result["status"] == "success"
        assert "org_id" in result
        assert result["org_id"] is not None

    def test_member_signup_uses_provided_org_id(self):
        users_coll = MagicMock()
        # find_one calls: 1=email dup check, 2=org exists check
        users_coll.find_one.side_effect = [None, USER_DOC, None]
        members_coll = MagicMock()
        members_coll.find_one.return_value = None
        db, users_coll, members_coll = _make_db(users_coll=users_coll, members_coll=members_coll)
        with patch("app.services.user_service.get_database", return_value=db), \
             patch("app.services.user_service.get_password_hash", return_value=HASHED):
            result = sign_up_user(_user_create(org_id=ORG_ID))
        assert result["status"] == "success"
        assert result["org_id"] == ORG_ID

    def test_duplicate_email_returns_fail(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = USER_DOC  # duplicate
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db):
            result = sign_up_user(_user_create())
        assert result["status"] == "fail"
        assert "already exists" in result["message"]

    def test_org_not_found_returns_fail(self):
        users_coll = MagicMock()
        # 1=email check (None), 2=org check (None = not found)
        users_coll.find_one.side_effect = [None, None]
        members_coll = MagicMock()
        members_coll.find_one.return_value = None
        db, _, _ = _make_db(users_coll=users_coll, members_coll=members_coll)
        with patch("app.services.user_service.get_database", return_value=db):
            result = sign_up_user(_user_create(org_id="nonexistent-org"))
        assert result["status"] == "fail"
        assert "Organization not found" in result["message"]

    def test_already_member_returns_fail(self):
        users_coll = MagicMock()
        users_coll.find_one.side_effect = [None, USER_DOC]  # email=None, org=found
        members_coll = MagicMock()
        members_coll.find_one.return_value = MEMBER_DOC  # already a member
        db, _, _ = _make_db(users_coll=users_coll, members_coll=members_coll)
        with patch("app.services.user_service.get_database", return_value=db):
            result = sign_up_user(_user_create(org_id=ORG_ID))
        assert result["status"] == "fail"
        assert "already a member" in result["message"]


# ── TestValidateUserAndGetToken ───────────────────────────────────────────

class TestValidateUserAndGetToken:
    def test_valid_credentials_returns_token(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = USER_DOC
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db), \
             patch("app.services.user_service.verify_password", return_value=True), \
             patch("app.services.user_service.create_access_token", return_value=TOKEN):
            result = validate_user_and_get_token(EMAIL, PASSWORD)
        assert result["status"] == "success"
        assert result["access_token"] == TOKEN

    def test_wrong_password_returns_fail(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = USER_DOC
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db), \
             patch("app.services.user_service.verify_password", return_value=False):
            result = validate_user_and_get_token(EMAIL, "wrong-pass")
        assert result["status"] == "fail"
        assert "Invalid password" in result["message"]

    def test_user_not_found_in_usertable_falls_back_to_member_service(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = None
        db, _, _ = _make_db(users_coll=users_coll)
        member_result = {"status": "success", "access_token": "member-token"}
        with patch("app.services.user_service.get_database", return_value=db), \
             patch("app.services.member_service.validate_member_and_get_token",
                   return_value=member_result):
            result = validate_user_and_get_token(EMAIL, PASSWORD)
        assert result["access_token"] == "member-token"

    def test_both_tables_miss_returns_not_found(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = None
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db), \
             patch("app.services.member_service.validate_member_and_get_token",
                   return_value=None):
            result = validate_user_and_get_token(EMAIL, PASSWORD)
        assert result["status"] == "fail"
        assert "not found" in result["message"].lower()


# ── TestGetUserByEmail ────────────────────────────────────────────────────

class TestGetUserByEmail:
    def test_found_in_usertable_returns_without_password(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = {**USER_DOC}
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db):
            result = get_user_by_email(EMAIL)
        assert result is not None
        assert "password" not in result

    def test_returns_none_when_not_found_in_either_table(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = None
        members_coll = MagicMock()
        members_coll.find_one.return_value = None
        db, _, _ = _make_db(users_coll=users_coll, members_coll=members_coll)
        with patch("app.services.user_service.get_database", return_value=db):
            result = get_user_by_email("ghost@example.com")
        assert result is None

    def test_returns_none_on_exception(self):
        users_coll = MagicMock()
        users_coll.find_one.side_effect = Exception("DB error")
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db):
            result = get_user_by_email(EMAIL)
        assert result is None


# ── TestRequestPasswordReset ──────────────────────────────────────────────

class TestRequestPasswordReset:
    def test_user_not_found_returns_success_silently(self):
        """Security: must not reveal whether the email exists."""
        users_coll = MagicMock()
        users_coll.find_one.return_value = None
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db):
            result = request_password_reset("unknown@example.com")
        assert result["status"] == "success"

    def test_user_found_sends_email_and_returns_success(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = USER_DOC
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db), \
             patch("app.services.user_service.send_password_reset_email", return_value=True):
            result = request_password_reset(EMAIL)
        assert result["status"] == "success"
        users_coll.update_one.assert_called_once()

    def test_email_send_failure_returns_fail(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = USER_DOC
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db), \
             patch("app.services.user_service.send_password_reset_email", return_value=False):
            result = request_password_reset(EMAIL)
        assert result["status"] == "fail"

    def test_reset_token_stored_in_db(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = USER_DOC
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db), \
             patch("app.services.user_service.send_password_reset_email", return_value=True):
            request_password_reset(EMAIL)
        update_set = users_coll.update_one.call_args[0][1]["$set"]
        assert "reset_token" in update_set
        assert "reset_token_expires" in update_set


# ── TestResetPasswordWithToken ────────────────────────────────────────────

class TestResetPasswordWithToken:
    def _future_iso(self, hours=2):
        return (datetime.now() + timedelta(hours=hours)).isoformat()

    def _past_iso(self, hours=2):
        return (datetime.now() - timedelta(hours=hours)).isoformat()

    def test_valid_token_updates_password(self):
        user_with_token = {
            **USER_DOC,
            "reset_token": RESET_TOKEN,
            "reset_token_used": False,
            "reset_token_expires": self._future_iso(),
        }
        users_coll = MagicMock()
        users_coll.find_one.return_value = user_with_token
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db), \
             patch("app.services.user_service.get_password_hash", return_value="new_hash"):
            result = reset_password_with_token(RESET_TOKEN, "new_password")
        assert result["status"] == "success"
        users_coll.update_one.assert_called_once()

    def test_invalid_token_returns_fail(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = None  # token not found
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db):
            result = reset_password_with_token("invalid-token", "new_pass")
        assert result["status"] == "fail"
        assert "Invalid" in result["message"]

    def test_expired_token_returns_fail(self):
        user_with_expired_token = {
            **USER_DOC,
            "reset_token": RESET_TOKEN,
            "reset_token_used": False,
            "reset_token_expires": self._past_iso(),
        }
        users_coll = MagicMock()
        users_coll.find_one.return_value = user_with_expired_token
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.user_service.get_database", return_value=db):
            result = reset_password_with_token(RESET_TOKEN, "new_pass")
        assert result["status"] == "fail"
        assert "expired" in result["message"].lower()
