"""Tests for app/main.py root and health endpoints."""

from unittest.mock import MagicMock


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "version" in data


class TestHealthEndpoint:
    def test_healthy_when_client_connected(self, client):
        from app.database import mongodb
        mock_client = MagicMock()
        mock_client.admin.command.return_value = {"ok": 1}
        original = mongodb.client
        mongodb.client = mock_client
        try:
            resp = client.get("/health")
        finally:
            mongodb.client = original
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_unhealthy_when_no_client(self, client):
        from app.database import mongodb
        original = mongodb.client
        mongodb.client = None
        try:
            resp = client.get("/health")
        finally:
            mongodb.client = original
        assert resp.status_code == 200
        assert resp.json()["status"] == "unhealthy"

    def test_unhealthy_when_ping_raises(self, client):
        from app.database import mongodb
        mock_client = MagicMock()
        mock_client.admin.command.side_effect = Exception("Connection refused")
        original = mongodb.client
        mongodb.client = mock_client
        try:
            resp = client.get("/health")
        finally:
            mongodb.client = original
        assert resp.status_code == 200
        assert resp.json()["status"] == "unhealthy"
        assert "Connection refused" in resp.json()["error"]
