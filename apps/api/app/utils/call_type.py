"""Helpers for call direction / channel type on CallLogs documents."""
from typing import Any, Dict, Optional

VALID_CALL_TYPES = frozenset({"inbound", "outbound", "web"})


def is_browser_meeting_id(meeting_id: Optional[str]) -> bool:
    return bool(meeting_id and str(meeting_id).startswith("browser-"))


def normalize_call_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in VALID_CALL_TYPES else None


def resolve_call_type(doc: Dict[str, Any]) -> str:
    stored = normalize_call_type(doc.get("call_type"))
    if stored:
        return stored
    meeting_id = doc.get("meeting_id")
    if is_browser_meeting_id(meeting_id):
        return "web"
    if doc.get("inbound") is True:
        return "inbound"
    return "outbound"


def call_type_filter(call_type: str) -> Dict[str, Any]:
    """MongoDB filter for a canonical call_type, including legacy documents."""
    normalized = normalize_call_type(call_type)
    if normalized == "web":
        return {
            "$or": [
                {"call_type": "web"},
                {"meeting_id": {"$regex": r"^browser-"}},
            ]
        }
    if normalized == "inbound":
        return {
            "$or": [
                {"call_type": "inbound"},
                {
                    "$and": [
                        {"$or": [{"call_type": {"$exists": False}}, {"call_type": None}, {"call_type": ""}]},
                        {"inbound": True},
                    ]
                },
            ]
        }
    if normalized == "outbound":
        return {
            "$and": [
                {
                    "$or": [
                        {"call_type": "outbound"},
                        {
                            "$and": [
                                {"$or": [{"call_type": {"$exists": False}}, {"call_type": None}, {"call_type": ""}]},
                                {"inbound": False},
                            ]
                        },
                    ]
                },
                {"meeting_id": {"$not": {"$regex": r"^browser-"}}},
            ]
        }
    return {}
