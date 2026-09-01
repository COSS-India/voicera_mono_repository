"""
Integration tests for /api/v1/members endpoints.
"""
import pytest
from unittest.mock import patch

BASE = "/api/v1/members"

MEMBER_CREATE = {
    "email": "newmember@example.com",
    "password": "pass123",
    "name": "New Member",
    "company_name": "ACME Corp",
    "org_id": "testorg1",
}

MEMBERS_RESPONSE = {
    "status": "success",
    "members": [
        {"email": "owner@example.com", "org_id": "testorg1", "is_owner": True},
        {"email": "member@example.com", "org_id": "testorg1", "is_owner": False},
    ],
}

DELETE_BODY = {"email": "member@example.com", "org_id": "testorg1"}
TRANSFER_BODY = {"org_id": "testorg1", "email": "member@example.com"}


# ── POST /members/add-member (public endpoint) ────────────────────────────

class TestAddMember:
    def test_success_returns_201(self, client):
        ok = {"status": "success", "message": "User created successfully", "org_id": "testorg1"}
        with patch("app.services.member_service.add_member", return_value=ok):
            resp = client.post(f"{BASE}/add-member", json=MEMBER_CREATE)
        assert resp.status_code == 201

    def test_duplicate_email_returns_400(self, client):
        fail = {"status": "fail", "message": "User with this email already exists"}
        with patch("app.services.member_service.add_member", return_value=fail):
            resp = client.post(f"{BASE}/add-member", json=MEMBER_CREATE)
        assert resp.status_code == 400

    def test_missing_required_fields_returns_422(self, client):
        resp = client.post(f"{BASE}/add-member", json={"email": "x@x.com"})
        assert resp.status_code == 422


# ── GET /members/{org_id} ─────────────────────────────────────────────────

class TestGetMembers:
    def test_success_returns_members(self, client):
        with patch("app.services.member_service.get_members_by_org",
                   return_value=MEMBERS_RESPONSE):
            resp = client.get(f"{BASE}/testorg1")
        assert resp.status_code == 200

    def test_wrong_org_returns_403(self, client):
        resp = client.get(f"{BASE}/otherorg9")
        assert resp.status_code == 403

    def test_service_fail_returns_500(self, client):
        fail = {"status": "fail", "message": "DB error"}
        with patch("app.services.member_service.get_members_by_org", return_value=fail):
            resp = client.get(f"{BASE}/testorg1")
        assert resp.status_code == 500


# ── POST /members/delete-member ───────────────────────────────────────────

class TestDeleteMember:
    def test_success_returns_200(self, client):
        ok = {"status": "success", "message": "Member deleted"}
        with patch("app.services.member_service.delete_member", return_value=ok):
            resp = client.post(f"{BASE}/delete-member", json=DELETE_BODY)
        assert resp.status_code == 200

    def test_wrong_org_returns_403(self, client):
        resp = client.post(f"{BASE}/delete-member",
                           json={"email": "m@m.com", "org_id": "otherorg9"})
        assert resp.status_code == 403

    def test_not_owner_returns_400(self, client):
        fail = {"status": "fail", "message": "Only the organization owner can remove members"}
        with patch("app.services.member_service.delete_member", return_value=fail):
            resp = client.post(f"{BASE}/delete-member", json=DELETE_BODY)
        assert resp.status_code == 400


# ── POST /members/transfer-ownership ─────────────────────────────────────

class TestTransferOwnership:
    def test_success_returns_200(self, client):
        ok = {"status": "success", "message": "Ownership transferred"}
        with patch("app.services.member_service.transfer_ownership", return_value=ok):
            resp = client.post(f"{BASE}/transfer-ownership", json=TRANSFER_BODY)
        assert resp.status_code == 200

    def test_wrong_org_returns_403(self, client):
        resp = client.post(
            f"{BASE}/transfer-ownership",
            json={"org_id": "otherorg9", "email": "m@m.com"},
        )
        assert resp.status_code == 403

    def test_service_fail_returns_400(self, client):
        fail = {"status": "fail", "message": "Cannot transfer to yourself"}
        with patch("app.services.member_service.transfer_ownership", return_value=fail):
            resp = client.post(f"{BASE}/transfer-ownership", json=TRANSFER_BODY)
        assert resp.status_code == 400
