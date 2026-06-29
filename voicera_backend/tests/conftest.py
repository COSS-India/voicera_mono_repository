"""
Shared test fixtures for the voicera_backend test suite.

Strategy
--------
- MongoDB startup/shutdown are no-ops; mongodb.database is pre-set to a
  MagicMock so get_database() never tries to reconnect during tests.
- Individual router tests mock the service layer independently.
- `client` overrides JWT auth and API-key checks (always passes).
- `unauth_client` is function-scoped: it temporarily clears those overrides
  on the same shared TestClient so real auth enforcement runs, then restores.
"""

import os

# Must be set before any app code is imported so Settings picks them up.
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-voicera-tests!")
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-api-key")
os.environ.setdefault("MONGODB_HOST", "localhost")
os.environ.setdefault("MAILTRAP_API_TOKEN", "test-mailtrap-token")
os.environ.setdefault("FRONTEND_URL", "http://localhost:3000")

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from tests.helpers import make_mock_db  # noqa: F401 — re-exported for convenience

from app.main import app
from app.auth import get_current_user, verify_api_key, create_access_token
from app.database import mongodb

# ── Shared identity used across the suite ─────────────────────────────────
TEST_ORG_ID = "testorg1"
TEST_USER_EMAIL = "test@example.com"
TEST_USER = {"email": TEST_USER_EMAIL, "org_id": TEST_ORG_ID}
OTHER_ORG_ID = "otherorg9"


@pytest.fixture(scope="session")
def auth_token():
    """A valid JWT for TEST_USER, useful for Authorization headers."""
    return create_access_token({"sub": TEST_USER_EMAIL, "org_id": TEST_ORG_ID})


@pytest.fixture(scope="session")
def client():
    """
    Session-wide TestClient.

    - MongoDB startup/shutdown are no-ops; mongodb.database is a MagicMock so
      any service that reaches get_database() gets a safe stub rather than
      trying to dial the real MongoDB.
    - JWT auth and internal API-key checks always pass.
    """
    # Pre-wire a mock database so get_database() never falls back to the real
    # connect_to_mongo() during test runs.
    mock_client = MagicMock()
    mock_client.admin.command.return_value = {"ok": 1}
    mongodb.client = mock_client
    mongodb.database = MagicMock()

    with patch("app.main.connect_to_mongo"), \
         patch("app.main.close_mongo_connection"), \
         patch("app.database_init.initialize_database"), \
         patch("app.main.start_batch_scheduler"), \
         patch("app.main.stop_batch_scheduler"):

        app.dependency_overrides[get_current_user] = lambda: TEST_USER
        app.dependency_overrides[verify_api_key] = lambda: True

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    app.dependency_overrides.clear()
    mongodb.client = None
    mongodb.database = None


@pytest.fixture
def unauth_client(client):
    """
    Function-scoped fixture. Temporarily clears all dependency overrides on the
    shared TestClient so that real JWT / API-key enforcement runs. Overrides are
    restored after each test.

    Use this fixture to verify 401 / 403 responses when credentials are
    absent or wrong.
    """
    saved = dict(app.dependency_overrides)
    app.dependency_overrides.clear()
    yield client
    app.dependency_overrides.update(saved)
