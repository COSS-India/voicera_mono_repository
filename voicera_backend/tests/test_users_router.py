"""
Integration tests for /api/v1/users endpoints.

Service functions are mocked so these tests verify:
  - Correct HTTP status codes
  - Request body parsing and validation
  - Authorization enforcement (own email only, etc.)
  - Error propagation from services to HTTP responses
"""

import pytest
from unittest.mock import patch

BASE = "/api/v1/users"

VALID_SIGNUP = {
    "email": "newuser@example.com",
    "password": "secret123",
    "name": "New User",
    "company_name": "ACME",
}


# ── POST /signup ───────────────────────────────────────────────────────────

class TestSignup:
    def test_signup_success_returns_201(self, client):
        payload = {"status": "success", "message": "User created successfully", "org_id": "abc123"}
        with patch("app.services.user_service.sign_up_user", return_value=payload):
            resp = client.post(f"{BASE}/signup", json=VALID_SIGNUP)
        assert resp.status_code == 201
        assert resp.json()["org_id"] == "abc123"

    def test_signup_duplicate_email_returns_400(self, client):
        error = {"status": "fail", "message": "User with this email already exists"}
        with patch("app.services.user_service.sign_up_user", return_value=error):
            resp = client.post(f"{BASE}/signup", json=VALID_SIGNUP)
        assert resp.status_code == 400
        assert "already exists" in resp.json()["detail"]

    def test_signup_member_join_existing_org(self, client):
        payload = {"status": "success", "message": "User created successfully", "org_id": "existingorg"}
        data = {**VALID_SIGNUP, "org_id": "existingorg"}
        with patch("app.services.user_service.sign_up_user", return_value=payload):
            resp = client.post(f"{BASE}/signup", json=data)
        assert resp.status_code == 201

    def test_signup_org_not_found_returns_400(self, client):
        error = {"status": "fail", "message": "Organization not found"}
        with patch("app.services.user_service.sign_up_user", return_value=error):
            resp = client.post(f"{BASE}/signup", json={**VALID_SIGNUP, "org_id": "ghost"})
        assert resp.status_code == 400

    def test_signup_invalid_email_returns_422(self, client):
        resp = client.post(f"{BASE}/signup", json={**VALID_SIGNUP, "email": "not-an-email"})
        assert resp.status_code == 422

    def test_signup_missing_required_field_returns_422(self, client):
        resp = client.post(f"{BASE}/signup", json={"email": "a@b.com", "password": "x"})
        assert resp.status_code == 422


# ── POST /login ────────────────────────────────────────────────────────────

class TestLogin:
    def test_login_success_returns_200_with_token(self, client):
        payload = {
            "status": "success",
            "message": "Authenticated",
            "access_token": "tok123",
            "token_type": "bearer",
            "org_id": "org1",
        }
        with patch("app.services.user_service.validate_user_and_get_token", return_value=payload):
            resp = client.post(f"{BASE}/login", json={"email": "a@b.com", "password": "p"})
        assert resp.status_code == 200
        assert resp.json()["access_token"] == "tok123"
        assert resp.json()["token_type"] == "bearer"

    def test_login_wrong_password_returns_401(self, client):
        error = {"status": "fail", "message": "Invalid password"}
        with patch("app.services.user_service.validate_user_and_get_token", return_value=error):
            resp = client.post(f"{BASE}/login", json={"email": "a@b.com", "password": "wrong"})
        assert resp.status_code == 401

    def test_login_user_not_found_returns_401(self, client):
        error = {"status": "fail", "message": "User not found"}
        with patch("app.services.user_service.validate_user_and_get_token", return_value=error):
            resp = client.post(f"{BASE}/login", json={"email": "ghost@b.com", "password": "x"})
        assert resp.status_code == 401

    def test_login_invalid_email_format_returns_422(self, client):
        resp = client.post(f"{BASE}/login", json={"email": "bad-email", "password": "p"})
        assert resp.status_code == 422


# ── GET /me ────────────────────────────────────────────────────────────────

class TestGetCurrentUserInfo:
    def test_get_me_success(self, client):
        user = {
            "email": "test@example.com",
            "name": "Test User",
            "org_id": "testorg1",
            "company_name": "Test Co",
        }
        with patch("app.services.user_service.get_user_by_email", return_value=user):
            resp = client.get(f"{BASE}/me")
        assert resp.status_code == 200
        assert resp.json()["email"] == "test@example.com"
        assert resp.json()["org_id"] == "testorg1"

    def test_get_me_user_not_found_returns_404(self, client):
        with patch("app.services.user_service.get_user_by_email", return_value=None):
            resp = client.get(f"{BASE}/me")
        assert resp.status_code == 404


# ── GET /{email} ───────────────────────────────────────────────────────────

class TestGetUserByEmail:
    def test_get_own_email_success(self, client):
        user = {
            "email": "test@example.com",
            "name": "Test",
            "org_id": "testorg1",
            "company_name": "Co",
        }
        with patch("app.services.user_service.get_user_by_email", return_value=user):
            resp = client.get(f"{BASE}/test@example.com")
        assert resp.status_code == 200

    def test_get_other_user_email_returns_403(self, client):
        resp = client.get(f"{BASE}/other@example.com")
        assert resp.status_code == 403

    def test_get_own_email_not_found_returns_404(self, client):
        with patch("app.services.user_service.get_user_by_email", return_value=None):
            resp = client.get(f"{BASE}/test@example.com")
        assert resp.status_code == 404


# ── POST /forgot-password ──────────────────────────────────────────────────

class TestForgotPassword:
    def test_success_returns_200(self, client):
        payload = {"status": "success", "message": "Email sent"}
        with patch("app.services.user_service.request_password_reset", return_value=payload):
            resp = client.post(f"{BASE}/forgot-password", json={"email": "a@b.com"})
        assert resp.status_code == 200

    def test_email_not_found_still_returns_200(self, client):
        """
        Security: must not reveal whether the email exists.
        Service returns success regardless.
        """
        payload = {"status": "success", "message": "If user exists, email sent"}
        with patch("app.services.user_service.request_password_reset", return_value=payload):
            resp = client.post(f"{BASE}/forgot-password", json={"email": "ghost@b.com"})
        assert resp.status_code == 200

    def test_service_failure_returns_400(self, client):
        error = {"status": "fail", "message": "Failed to send email"}
        with patch("app.services.user_service.request_password_reset", return_value=error):
            resp = client.post(f"{BASE}/forgot-password", json={"email": "a@b.com"})
        assert resp.status_code == 400

    def test_invalid_email_returns_422(self, client):
        resp = client.post(f"{BASE}/forgot-password", json={"email": "bad"})
        assert resp.status_code == 422


# ── POST /reset-password ───────────────────────────────────────────────────

class TestResetPassword:
    def test_success_returns_200(self, client):
        payload = {"status": "success", "message": "Password reset successfully"}
        with patch("app.services.user_service.reset_password_with_token", return_value=payload):
            resp = client.post(
                f"{BASE}/reset-password",
                json={"token": "valid-token", "new_password": "newpass123"},
            )
        assert resp.status_code == 200

    def test_invalid_token_returns_400(self, client):
        error = {"status": "fail", "message": "Invalid or expired reset token"}
        with patch("app.services.user_service.reset_password_with_token", return_value=error):
            resp = client.post(
                f"{BASE}/reset-password",
                json={"token": "bad-token", "new_password": "newpass"},
            )
        assert resp.status_code == 400

    def test_expired_token_returns_400(self, client):
        error = {"status": "fail", "message": "Reset token has expired"}
        with patch("app.services.user_service.reset_password_with_token", return_value=error):
            resp = client.post(
                f"{BASE}/reset-password",
                json={"token": "expired", "new_password": "newpass"},
            )
        assert resp.status_code == 400

    def test_missing_token_returns_422(self, client):
        resp = client.post(f"{BASE}/reset-password", json={"new_password": "newpass"})
        assert resp.status_code == 422
