"""
Integration tests for GET /api/v1/analytics.

The analytics service is mocked so that router logic, query-parameter
wiring, and the AnalyticsResponse schema are exercised without a database.
"""

import pytest
from unittest.mock import patch
from datetime import datetime, timezone

BASE = "/api/v1/analytics"

ANALYTICS_PAYLOAD = {
    "org_id": "testorg1",
    "calls_attempted": 10,
    "calls_connected": 7,
    "average_call_duration": 3.5,
    "total_minutes_connected": 24.5,
    "most_used_agent": "sales_bot",
    "most_used_agent_count": 5,
    "agent_breakdown": [{"agent_type": "sales_bot", "call_count": 5}],
    "calculated_at": datetime.now(timezone.utc).isoformat(),
}

EMPTY_ANALYTICS = {
    "org_id": "testorg1",
    "calls_attempted": 0,
    "calls_connected": 0,
    "average_call_duration": 0.0,
    "total_minutes_connected": 0.0,
    "most_used_agent": None,
    "most_used_agent_count": 0,
    "agent_breakdown": [],
    "calculated_at": datetime.now(timezone.utc).isoformat(),
}


class TestGetAnalytics:
    def test_success_returns_200_with_expected_shape(self, client):
        with patch("app.services.analytics_service.get_analytics", return_value=ANALYTICS_PAYLOAD):
            resp = client.get(BASE)
        assert resp.status_code == 200
        data = resp.json()
        assert data["org_id"] == "testorg1"
        assert data["calls_attempted"] == 10
        assert data["calls_connected"] == 7
        assert data["average_call_duration"] == 3.5
        assert "calculated_at" in data

    def test_empty_org_returns_zero_metrics(self, client):
        with patch("app.services.analytics_service.get_analytics", return_value=EMPTY_ANALYTICS):
            resp = client.get(BASE)
        assert resp.status_code == 200
        assert resp.json()["calls_attempted"] == 0
        assert resp.json()["most_used_agent"] is None
        assert resp.json()["agent_breakdown"] == []

    def test_agent_type_filter_routed_correctly(self, client):
        """Router must forward agent_type query param to the service."""
        with patch("app.services.analytics_service.get_analytics", return_value=ANALYTICS_PAYLOAD) as mock_svc:
            resp = client.get(f"{BASE}?agent_type=sales_bot")
        assert resp.status_code == 200
        mock_svc.assert_called_once()
        _, kwargs = mock_svc.call_args
        assert kwargs.get("agent_type") == "sales_bot"

    def test_phone_number_filter_routed_correctly(self, client):
        with patch("app.services.analytics_service.get_analytics", return_value=ANALYTICS_PAYLOAD) as mock_svc:
            resp = client.get(f"{BASE}?phone_number=%2B911234567890")
        assert resp.status_code == 200
        _, kwargs = mock_svc.call_args
        assert kwargs.get("phone_number") == "+911234567890"

    def test_date_range_routes_to_date_range_function(self, client):
        """When start_date or end_date is provided, the router uses the
        get_analytics_by_date_range service function, not get_analytics."""
        with patch("app.services.analytics_service.get_analytics_by_date_range", return_value=ANALYTICS_PAYLOAD) as mock_dr, \
             patch("app.services.analytics_service.get_analytics") as mock_plain:
            resp = client.get(f"{BASE}?start_date=2024-01-01&end_date=2024-01-31")
        assert resp.status_code == 200
        mock_dr.assert_called_once()
        mock_plain.assert_not_called()

    def test_only_start_date_triggers_date_range_function(self, client):
        with patch("app.services.analytics_service.get_analytics_by_date_range", return_value=ANALYTICS_PAYLOAD) as mock_dr:
            resp = client.get(f"{BASE}?start_date=2024-01-01")
        assert resp.status_code == 200
        mock_dr.assert_called_once()

    def test_date_range_passes_dates_to_service(self, client):
        with patch("app.services.analytics_service.get_analytics_by_date_range", return_value=ANALYTICS_PAYLOAD) as mock_dr:
            client.get(f"{BASE}?start_date=2024-01-01&end_date=2024-01-31")
        _, kwargs = mock_dr.call_args
        assert kwargs.get("start_date") == "2024-01-01"
        assert kwargs.get("end_date") == "2024-01-31"

    def test_service_exception_returns_500(self, client):
        with patch("app.services.analytics_service.get_analytics", side_effect=RuntimeError("DB down")):
            resp = client.get(BASE)
        assert resp.status_code == 500

    def test_org_id_comes_from_jwt_not_query_params(self, client):
        """org_id must always come from the token, not a user-supplied query param."""
        with patch("app.services.analytics_service.get_analytics", return_value=ANALYTICS_PAYLOAD) as mock_svc:
            client.get(BASE)
        _, kwargs = mock_svc.call_args
        assert kwargs.get("org_id") == "testorg1"

    def test_unauthenticated_returns_4xx(self, unauth_client):
        resp = unauth_client.get(BASE)
        assert resp.status_code in (401, 403)
