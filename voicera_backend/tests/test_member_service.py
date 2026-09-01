"""
Unit tests for app.services.member_service.
"""
import pytest
from unittest.mock import MagicMock, patch

from app.services.member_service import (
    is_org_owner,
    add_member,
    get_members_by_org,
    delete_member,
    transfer_ownership,
    validate_member_and_get_token,
)
from app.models.schemas import MemberCreate, MemberDelete
from tests.helpers import make_mock_db as _make_mock_db_base

# ── Sample data ───────────────────────────────────────────────────────────────

ORG_ID = "testorg1"
OWNER_EMAIL = "owner@example.com"
MEMBER_EMAIL = "member@example.com"
HASHED = "hashed_password"
TOKEN = "jwt-token"

OWNER_DOC = {
    "email": OWNER_EMAIL,
    "password": HASHED,
    "org_id": ORG_ID,
    "is_member": False,  # owner
}

MEMBER_DOC = {
    "email": MEMBER_EMAIL,
    "password": HASHED,
    "org_id": ORG_ID,
    "is_member": True,  # member
}

LEGACY_MEMBER_DOC = {
    "email": MEMBER_EMAIL,
    "org_id": ORG_ID,
}


def _make_db(users_coll=None, members_coll=None):
    users_coll = users_coll or MagicMock()
    members_coll = members_coll or MagicMock()
    db = _make_mock_db_base(UserTable=users_coll, Members=members_coll)
    return db, users_coll, members_coll


# ── TestIsOrgOwner ────────────────────────────────────────────────────────

class TestIsOrgOwner:
    def test_returns_true_when_user_is_owner(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = OWNER_DOC
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            assert is_org_owner(OWNER_EMAIL, ORG_ID) is True

    def test_returns_false_when_user_is_member(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = MEMBER_DOC
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            assert is_org_owner(MEMBER_EMAIL, ORG_ID) is False

    def test_returns_false_when_user_not_found(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = None
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            assert is_org_owner("ghost@example.com", ORG_ID) is False

    def test_returns_false_on_exception(self):
        users_coll = MagicMock()
        users_coll.find_one.side_effect = Exception("DB error")
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            assert is_org_owner(OWNER_EMAIL, ORG_ID) is False


# ── TestAddMember ─────────────────────────────────────────────────────────

class TestAddMember:
    def test_delegates_to_sign_up_user_and_returns_success(self):
        ok = {"status": "success", "message": "User created", "org_id": ORG_ID}
        data = MemberCreate(
            email=MEMBER_EMAIL, password="pass", name="Member",
            company_name="ACME", org_id=ORG_ID
        )
        with patch("app.services.user_service.sign_up_user", return_value=ok):
            result = add_member(data)
        assert result["status"] == "success"

    def test_propagates_fail_from_sign_up(self):
        fail = {"status": "fail", "message": "Email already exists"}
        data = MemberCreate(
            email=MEMBER_EMAIL, password="pass", name="Member",
            company_name="ACME", org_id=ORG_ID
        )
        with patch("app.services.user_service.sign_up_user", return_value=fail):
            result = add_member(data)
        assert result["status"] == "fail"


# ── TestGetMembersByOrg ───────────────────────────────────────────────────

class TestGetMembersByOrg:
    def test_returns_members_with_is_owner_flag(self):
        users_coll = MagicMock()
        users_coll.find.return_value = [OWNER_DOC, MEMBER_DOC]
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            result = get_members_by_org(ORG_ID)
        assert result["status"] == "success"
        members = result["members"]
        owner = next(m for m in members if m["email"] == OWNER_EMAIL)
        member = next(m for m in members if m["email"] == MEMBER_EMAIL)
        assert owner["is_owner"] is True
        assert member["is_owner"] is False

    def test_exception_returns_fail(self):
        users_coll = MagicMock()
        users_coll.find.side_effect = Exception("DB error")
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            result = get_members_by_org(ORG_ID)
        assert result["status"] == "fail"


# ── TestDeleteMember ──────────────────────────────────────────────────────

class TestDeleteMember:
    def _delete_data(self, email=MEMBER_EMAIL):
        return MemberDelete(email=email, org_id=ORG_ID)

    def test_success_deletes_from_both_tables(self):
        users_coll = MagicMock()
        # Caller lookup (owner check): return owner, member lookup: return member
        users_coll.find_one.side_effect = [OWNER_DOC, MEMBER_DOC]
        users_coll.delete_one.return_value.deleted_count = 1
        members_coll = MagicMock()
        db, _, _ = _make_db(users_coll=users_coll, members_coll=members_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            result = delete_member(self._delete_data(), OWNER_EMAIL)
        assert result["status"] == "success"
        members_coll.delete_one.assert_called()

    def test_non_owner_caller_returns_fail(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = MEMBER_DOC  # caller is member, not owner
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            result = delete_member(self._delete_data(), MEMBER_EMAIL)
        assert result["status"] == "fail"
        assert "owner" in result["message"].lower()

    def test_member_not_found_returns_fail(self):
        users_coll = MagicMock()
        # Owner check returns owner; member lookup returns None
        users_coll.find_one.side_effect = [OWNER_DOC, None]
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            result = delete_member(self._delete_data(), OWNER_EMAIL)
        assert result["status"] == "fail"
        assert "not found" in result["message"].lower()

    def test_cannot_delete_owner_returns_fail(self):
        users_coll = MagicMock()
        # Caller is owner; target is also owner (is_member=False)
        users_coll.find_one.side_effect = [OWNER_DOC, OWNER_DOC]
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            result = delete_member(self._delete_data(OWNER_EMAIL), OWNER_EMAIL)
        assert result["status"] == "fail"
        assert "owner" in result["message"].lower()


# ── TestTransferOwnership ─────────────────────────────────────────────────

class TestTransferOwnership:
    def test_success_flips_is_member_flags(self):
        # transfer_ownership calls is_org_owner (1 find_one) + new-owner lookup (1 find_one)
        # Use a side_effect function to handle any call order robustly.
        users_coll = MagicMock()
        members_coll = MagicMock()

        def find_one_side_effect(query):
            email = query.get("email")
            if email == OWNER_EMAIL:
                return OWNER_DOC
            if email == MEMBER_EMAIL:
                return MEMBER_DOC
            return None

        users_coll.find_one.side_effect = find_one_side_effect
        db, _, _ = _make_db(users_coll=users_coll, members_coll=members_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            result = transfer_ownership(ORG_ID, OWNER_EMAIL, MEMBER_EMAIL)
        assert result["status"] == "success"
        assert users_coll.update_one.call_count == 2

    def test_self_transfer_returns_fail(self):
        db, _, _ = _make_db()
        with patch("app.services.member_service.get_database", return_value=db):
            result = transfer_ownership(ORG_ID, OWNER_EMAIL, OWNER_EMAIL)
        assert result["status"] == "fail"
        assert "yourself" in result["message"].lower()

    def test_non_owner_caller_returns_fail(self):
        users_coll = MagicMock()
        users_coll.find_one.return_value = MEMBER_DOC  # caller is not owner
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            result = transfer_ownership(ORG_ID, MEMBER_EMAIL, OWNER_EMAIL)
        assert result["status"] == "fail"
        assert "owner" in result["message"].lower()

    def test_new_owner_not_in_org_returns_fail(self):
        users_coll = MagicMock()
        users_coll.find_one.side_effect = [OWNER_DOC, None]  # new owner not found
        db, _, _ = _make_db(users_coll=users_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            result = transfer_ownership(ORG_ID, OWNER_EMAIL, "unknown@example.com")
        assert result["status"] == "fail"
        assert "not found" in result["message"].lower()


# ── TestValidateMemberAndGetToken ─────────────────────────────────────────

class TestValidateMemberAndGetToken:
    def test_valid_credentials_returns_token(self):
        members_coll = MagicMock()
        members_coll.find_one.return_value = LEGACY_MEMBER_DOC
        # UserTable is also queried for the user doc with password
        users_coll = MagicMock()
        users_coll.find_one.return_value = {**MEMBER_DOC}
        db, _, _ = _make_db(users_coll=users_coll, members_coll=members_coll)
        with patch("app.services.member_service.get_database", return_value=db), \
             patch("app.services.member_service.verify_password", return_value=True), \
             patch("app.services.member_service.create_access_token", return_value=TOKEN):
            result = validate_member_and_get_token(MEMBER_EMAIL, "pass")
        assert result is not None

    def test_member_not_found_returns_none(self):
        members_coll = MagicMock()
        members_coll.find_one.return_value = None
        db, _, _ = _make_db(members_coll=members_coll)
        with patch("app.services.member_service.get_database", return_value=db):
            result = validate_member_and_get_token("ghost@example.com", "pass")
        assert result is None
