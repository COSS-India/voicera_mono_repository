"""
Unit tests for app/services/analytics_service.py.

The pure helper functions (calculate_duration_in_minutes, is_call_connected)
need no mocking. The DB-backed functions mock get_database() so no running
MongoDB instance is required.
"""

import os
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-voicera-tests!")

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from app.services.analytics_service import (
    calculate_duration_in_minutes,
    is_call_connected,
    get_analytics,
    get_analytics_by_date_range,
)


# ── calculate_duration_in_minutes ──────────────────────────────────────────

class TestCalculateDuration:
    def test_from_seconds(self):
        assert calculate_duration_in_minutes(None, None, 120) == pytest.approx(2.0)

    def test_from_seconds_less_than_a_minute(self):
        assert calculate_duration_in_minutes(None, None, 30) == pytest.approx(0.5)

    def test_from_timestamps(self):
        start = "2024-01-01T10:00:00"
        end = "2024-01-01T10:02:00"  # 2 minutes later
        result = calculate_duration_in_minutes(start, end, None)
        assert result == pytest.approx(2.0)

    def test_seconds_takes_priority_over_timestamps(self):
        start = "2024-01-01T10:00:00"
        end = "2024-01-01T10:10:00"  # 10-min gap
        result = calculate_duration_in_minutes(start, end, 120)  # 2-min seconds
        assert result == pytest.approx(2.0)

    def test_zero_seconds_falls_back_to_timestamps(self):
        start = "2024-01-01T10:00:00"
        end = "2024-01-01T10:03:00"
        result = calculate_duration_in_minutes(start, end, 0)
        assert result == pytest.approx(3.0)

    def test_returns_none_when_no_data(self):
        assert calculate_duration_in_minutes(None, None, None) is None

    def test_returns_none_with_only_start_time(self):
        assert calculate_duration_in_minutes("2024-01-01T10:00:00", None, None) is None

    def test_returns_none_when_end_before_start(self):
        start = "2024-01-01T10:05:00"
        end = "2024-01-01T10:00:00"
        assert calculate_duration_in_minutes(start, end, None) is None

    def test_handles_z_suffix_in_iso_string(self):
        start = "2024-01-01T10:00:00Z"
        end = "2024-01-01T10:01:00Z"
        result = calculate_duration_in_minutes(start, end, None)
        assert result == pytest.approx(1.0)

    def test_handles_invalid_timestamp_gracefully(self):
        result = calculate_duration_in_minutes("not-a-date", "also-not", None)
        assert result is None


# ── is_call_connected ──────────────────────────────────────────────────────

class TestIsCallConnected:
    def test_connected_when_has_end_time(self):
        call = {"end_time_utc": "2024-01-01T10:05:00", "call_busy": False}
        assert is_call_connected(call) is True

    def test_connected_when_has_positive_duration(self):
        call = {"duration": 120.0, "call_busy": False}
        assert is_call_connected(call) is True

    def test_not_connected_when_call_busy(self):
        call = {"end_time_utc": "2024-01-01T10:05:00", "call_busy": True}
        assert is_call_connected(call) is False

    def test_not_connected_when_no_end_time_or_duration(self):
        assert is_call_connected({}) is False

    def test_not_connected_when_duration_is_zero(self):
        call = {"duration": 0}
        assert is_call_connected(call) is False

    def test_not_connected_when_duration_is_none(self):
        call = {"duration": None}
        assert is_call_connected(call) is False

    def test_call_busy_none_treated_as_not_busy(self):
        call = {"end_time_utc": "2024-01-01T10:05:00", "call_busy": None}
        assert is_call_connected(call) is True


# ── get_analytics ──────────────────────────────────────────────────────────

def _make_mock_db(call_logs):
    """Return a MagicMock DB where CallLogs.find returns call_logs."""
    mock_db = MagicMock()
    mock_db.__getitem__.return_value.find.return_value = call_logs
    return mock_db


class TestGetAnalytics:
    def test_empty_org_returns_zero_metrics(self):
        mock_db = _make_mock_db([])
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            result = get_analytics("emptyorg")
        assert result["calls_attempted"] == 0
        assert result["calls_connected"] == 0
        assert result["average_call_duration"] == 0.0
        assert result["total_minutes_connected"] == 0.0
        assert result["most_used_agent"] is None
        assert result["org_id"] == "emptyorg"
        assert "calculated_at" in result

    def test_counts_all_calls_as_attempted(self):
        calls = [
            {"agent_type": "sales", "call_busy": True},
            {"agent_type": "sales", "call_busy": False, "end_time_utc": "2024-01-01T10:01:00"},
        ]
        mock_db = _make_mock_db(calls)
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            result = get_analytics("org1")
        assert result["calls_attempted"] == 2

    def test_only_connected_calls_counted(self):
        calls = [
            {"agent_type": "sales", "call_busy": True},  # not connected
            {"agent_type": "sales", "end_time_utc": "2024-01-01T10:01:00"},  # connected
            {"agent_type": "sales", "duration": 60.0},  # connected
        ]
        mock_db = _make_mock_db(calls)
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            result = get_analytics("org1")
        assert result["calls_attempted"] == 3
        assert result["calls_connected"] == 2

    def test_average_duration_calculated_correctly(self):
        calls = [
            {"agent_type": "a", "duration": 120.0, "end_time_utc": "x"},  # 2 min
            {"agent_type": "a", "duration": 60.0, "end_time_utc": "x"},   # 1 min
        ]
        mock_db = _make_mock_db(calls)
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            result = get_analytics("org1")
        assert result["average_call_duration"] == pytest.approx(1.5)
        assert result["total_minutes_connected"] == pytest.approx(3.0)

    def test_most_used_agent_identified(self):
        calls = [
            {"agent_type": "sales"},
            {"agent_type": "sales"},
            {"agent_type": "support"},
        ]
        mock_db = _make_mock_db(calls)
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            result = get_analytics("org1")
        assert result["most_used_agent"] == "sales"
        assert result["most_used_agent_count"] == 2

    def test_agent_breakdown_sorted_descending(self):
        calls = [
            {"agent_type": "a"},
            {"agent_type": "b"},
            {"agent_type": "b"},
            {"agent_type": "c"},
            {"agent_type": "c"},
            {"agent_type": "c"},
        ]
        mock_db = _make_mock_db(calls)
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            result = get_analytics("org1")
        breakdown = result["agent_breakdown"]
        counts = [item["call_count"] for item in breakdown]
        assert counts == sorted(counts, reverse=True)

    def test_agent_filter_applied(self):
        """Verify the query filter is forwarded to MongoDB find()."""
        mock_db = _make_mock_db([])
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            get_analytics("org1", agent_type="sales")
        call_args = mock_db.__getitem__.return_value.find.call_args[0][0]
        assert call_args["agent_type"] == "sales"

    def test_phone_number_filter_applied(self):
        mock_db = _make_mock_db([])
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            get_analytics("org1", phone_number="+911234567890")
        call_args = mock_db.__getitem__.return_value.find.call_args[0][0]
        assert call_args["phone_number"] == "+911234567890"

    def test_calls_without_agent_type_skipped_in_breakdown(self):
        calls = [{"agent_type": None}, {"agent_type": "sales"}]
        mock_db = _make_mock_db(calls)
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            result = get_analytics("org1")
        agents_in_breakdown = [b["agent_type"] for b in result["agent_breakdown"]]
        assert None not in agents_in_breakdown


# ── get_analytics_by_date_range ────────────────────────────────────────────

class TestGetAnalyticsByDateRange:
    def test_basic_date_range_returns_expected_shape(self):
        mock_db = _make_mock_db([])
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            result = get_analytics_by_date_range(
                "org1", start_date="2024-01-01", end_date="2024-01-31"
            )
        assert result["org_id"] == "org1"
        assert "calls_attempted" in result
        assert "calculated_at" in result

    def test_date_filter_added_to_query(self):
        mock_db = _make_mock_db([])
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            get_analytics_by_date_range("org1", start_date="2024-01-01")
        call_args = mock_db.__getitem__.return_value.find.call_args[0][0]
        assert "created_at" in call_args
        assert "$gte" in call_args["created_at"]

    def test_end_date_filter_added_to_query(self):
        mock_db = _make_mock_db([])
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            get_analytics_by_date_range("org1", end_date="2024-01-31")
        call_args = mock_db.__getitem__.return_value.find.call_args[0][0]
        assert "$lte" in call_args["created_at"]

    def test_invalid_start_date_ignored_gracefully(self):
        mock_db = _make_mock_db([])
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            result = get_analytics_by_date_range("org1", start_date="not-a-date")
        assert "calls_attempted" in result

    def test_no_date_range_returns_results(self):
        mock_db = _make_mock_db([{"agent_type": "sales", "duration": 60.0}])
        with patch("app.services.analytics_service.get_database", return_value=mock_db):
            result = get_analytics_by_date_range("org1")
        assert result["calls_attempted"] == 1
