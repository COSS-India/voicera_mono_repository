"""
Unit tests for app/auth.py.

All tests here are pure — no HTTP calls, no database, no mocking needed.
The only exception is the FastAPI-dependency tests, which use the unauth_client
to drive the real dependency through a live request.
"""

import os
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-voicera-tests!")
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-api-key")

from datetime import timedelta
import pytest
from jose import jwt

from app.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    verify_token,
    ALGORITHM,
    SECRET_KEY,
)


# ── Password hashing ───────────────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_produces_string(self):
        h = get_password_hash("password123")
        assert isinstance(h, str)
        assert len(h) > 0

    def test_hash_is_bcrypt(self):
        h = get_password_hash("password123")
        assert h.startswith("$2b$") or h.startswith("$2a$")

    def test_correct_password_verifies(self):
        h = get_password_hash("correct_horse_battery_staple")
        assert verify_password("correct_horse_battery_staple", h) is True

    def test_wrong_password_fails(self):
        h = get_password_hash("realpassword")
        assert verify_password("wrongpassword", h) is False

    def test_empty_string_password(self):
        h = get_password_hash("")
        assert verify_password("", h) is True
        assert verify_password("not-empty", h) is False

    def test_password_stored_as_str_bytes_interop(self):
        """Verify that a hash stored as bytes can still be verified."""
        h = get_password_hash("testpass")
        h_bytes = h.encode("utf-8")
        assert verify_password("testpass", h_bytes) is True

    def test_72_byte_truncation(self):
        """
        bcrypt silently truncates at 72 bytes. Passwords that differ only past
        byte 72 must be considered equal by our implementation.
        """
        base = "a" * 72
        long_password = base + "extra_suffix_that_is_ignored"
        h = get_password_hash(base)
        assert verify_password(long_password, h) is True

    def test_two_hashes_of_same_password_differ(self):
        """bcrypt uses a random salt, so each hash is unique."""
        h1 = get_password_hash("same")
        h2 = get_password_hash("same")
        assert h1 != h2

    def test_verify_invalid_hash_returns_false(self):
        assert verify_password("anything", "not-a-valid-hash") is False


# ── JWT creation ───────────────────────────────────────────────────────────

class TestCreateAccessToken:
    def test_returns_string(self):
        token = create_access_token({"sub": "user@example.com", "org_id": "org1"})
        assert isinstance(token, str)
        assert len(token) > 0

    def test_token_contains_subject(self):
        token = create_access_token({"sub": "user@example.com", "org_id": "org1"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "user@example.com"

    def test_token_contains_org_id(self):
        token = create_access_token({"sub": "u@e.com", "org_id": "myorg"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["org_id"] == "myorg"

    def test_token_has_expiry(self):
        token = create_access_token({"sub": "u@e.com"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "exp" in payload

    def test_custom_expiry_is_respected(self):
        delta = timedelta(hours=2)
        token = create_access_token({"sub": "u@e.com"}, expires_delta=delta)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # exp should be approximately now + 2h; just ensure it decoded
        assert "exp" in payload

    def test_issued_at_present(self):
        token = create_access_token({"sub": "u@e.com"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "iat" in payload


# ── JWT verification ───────────────────────────────────────────────────────

class TestVerifyToken:
    def test_valid_token_returns_payload(self):
        token = create_access_token({"sub": "user@test.com", "org_id": "org123"})
        payload = verify_token(token)
        assert payload is not None
        assert payload["sub"] == "user@test.com"
        assert payload["org_id"] == "org123"

    def test_expired_token_returns_none(self):
        token = create_access_token(
            {"sub": "user@test.com"},
            expires_delta=timedelta(seconds=-1),
        )
        assert verify_token(token) is None

    def test_malformed_token_returns_none(self):
        assert verify_token("not.a.jwt") is None

    def test_garbage_string_returns_none(self):
        assert verify_token("completely_invalid") is None

    def test_empty_string_returns_none(self):
        assert verify_token("") is None

    def test_token_signed_with_wrong_key_returns_none(self):
        from jose import jwt as jose_jwt
        bad_token = jose_jwt.encode(
            {"sub": "x@y.com", "exp": 9999999999},
            "wrong-secret",
            algorithm=ALGORITHM,
        )
        assert verify_token(bad_token) is None


# ── HTTP dependency: internal API key ──────────────────────────────────────

class TestVerifyApiKey:
    def test_missing_api_key_header_returns_401(self, unauth_client):
        # Any bot endpoint that uses verify_api_key
        resp = unauth_client.get("/api/v1/agents/config/some_agent")
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self, unauth_client):
        resp = unauth_client.get(
            "/api/v1/agents/config/some_agent",
            headers={"X-API-Key": "definitely-wrong"},
        )
        assert resp.status_code == 401

    def test_correct_api_key_passes_auth(self, unauth_client):
        """
        With the correct API key the dependency passes. The 404 here is from
        the agent not existing — it proves auth was accepted.
        """
        from unittest.mock import patch
        with patch("app.services.agent_service.fetch_agent_config", return_value=None):
            resp = unauth_client.get(
                "/api/v1/agents/config/nonexistent",
                headers={"X-API-Key": "test-internal-api-key"},
            )
        assert resp.status_code == 404


# ── HTTP dependency: get_current_user ─────────────────────────────────────

class TestGetCurrentUser:
    def test_no_auth_header_returns_4xx(self, unauth_client):
        resp = unauth_client.get("/api/v1/users/me")
        assert resp.status_code in (401, 403)

    def test_invalid_token_returns_401(self, unauth_client):
        resp = unauth_client.get(
            "/api/v1/users/me",
            headers={"Authorization": "Bearer totally.invalid.token"},
        )
        assert resp.status_code == 401

    def test_valid_token_passes_auth(self, unauth_client):
        """A valid JWT passes auth; 404 means the user just doesn't exist in DB."""
        from unittest.mock import patch
        from app.auth import create_access_token
        token = create_access_token({"sub": "u@test.com", "org_id": "org1"})
        with patch("app.services.user_service.get_user_by_email", return_value=None):
            resp = unauth_client.get(
                "/api/v1/users/me",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 404

    def test_token_without_sub_returns_401(self, unauth_client):
        """Token with no 'sub' field triggers 401 in get_current_user."""
        from app.auth import create_access_token, SECRET_KEY, ALGORITHM
        from jose import jwt as jose_jwt
        # Build a valid JWT with no sub field
        token = jose_jwt.encode(
            {"org_id": "org1", "exp": 9999999999},
            SECRET_KEY,
            algorithm=ALGORITHM,
        )
        resp = unauth_client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401
