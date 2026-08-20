"""Parse VI CPaaS recording webhook payloads (inbound push from flow-builder ApiCall)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

RECORDING_URL_KEYS = ("recording_url", "RecordVoice", "record_voice")
CALL_ID_KEYS = ("callid", "call_id", "CallId", "CallUUID", "call_sid")


def _coerce_url(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _find_url_in_mapping(mapping: Dict[str, Any]) -> Optional[str]:
    for key in RECORDING_URL_KEYS:
        if key not in mapping:
            continue
        url = _coerce_url(mapping.get(key))
        if url:
            return url
    return None


def extract_call_id_from_recording_url(recording_url: str) -> Optional[str]:
    """Extract VI call id from fetch-voice-recording query string (callid=...)."""
    if not recording_url:
        return None
    try:
        query = parse_qs(urlparse(recording_url).query)
        for key in CALL_ID_KEYS:
            values = query.get(key)
            if values:
                call_id = _coerce_url(values[0])
                if call_id:
                    return call_id
    except Exception:
        return None
    return None


def extract_call_id(payload: Any, recording_url: Optional[str] = None) -> Optional[str]:
    """Resolve call id from webhook payload and/or recording URL query params."""
    if recording_url:
        from_url = extract_call_id_from_recording_url(recording_url)
        if from_url:
            return from_url

    if isinstance(payload, dict):
        for key in CALL_ID_KEYS:
            call_id = _coerce_url(payload.get(key))
            if call_id:
                return call_id
        for value in payload.values():
            if isinstance(value, dict):
                for key in CALL_ID_KEYS:
                    call_id = _coerce_url(value.get(key))
                    if call_id:
                        return call_id
    return None


def extract_recording_url(payload: Any) -> Optional[str]:
    """Extract recording URL from parsed webhook body (top-level or one level nested)."""
    if payload is None:
        return None

    if isinstance(payload, dict):
        direct = _find_url_in_mapping(payload)
        if direct:
            return direct
        for value in payload.values():
            if isinstance(value, dict):
                nested = _find_url_in_mapping(value)
                if nested:
                    return nested
        return None

    if isinstance(payload, list):
        for item in payload:
            url = extract_recording_url(item)
            if url:
                return url
        return None

    return None


def _form_bytes_to_dict(raw: bytes) -> Dict[str, Any]:
    text = raw.decode("utf-8", errors="replace")
    parsed = parse_qs(text, keep_blank_values=True)
    return {key: values[0] if len(values) == 1 else values for key, values in parsed.items()}


def parse_webhook_body(raw: bytes, content_type: Optional[str] = None) -> Tuple[Optional[dict], str]:
    """
    Parse VI recording webhook body.

    Returns (parsed_dict_or_none, parse_mode) where parse_mode is json, form, or raw.
    """
    ctype = (content_type or "").lower()
    stripped = raw.strip()

    if stripped:
        try:
            data = json.loads(stripped.decode("utf-8"))
            if isinstance(data, dict):
                return data, "json"
            return {"_value": data}, "json"
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    if "application/x-www-form-urlencoded" in ctype or "multipart/form-data" in ctype:
        try:
            return _form_bytes_to_dict(raw), "form"
        except Exception:
            return None, "raw"

    # Unknown content-type: try form decode as a secondary fallback.
    if stripped and b"=" in raw and b"&" in raw:
        try:
            return _form_bytes_to_dict(raw), "form"
        except Exception:
            pass

    return None, "raw"


def format_raw_body_for_log(raw: bytes, max_preview: int = 512) -> str:
    """Format raw body for logging (UTF-8 with replacement, plus hex preview if needed)."""
    if not raw:
        return "(empty body)"
    text = raw.decode("utf-8", errors="replace")
    if text.isprintable() or all(ch in "\r\n\t" or ch.isprintable() for ch in text):
        if len(text) > max_preview:
            return f"{text[:max_preview]}... ({len(raw)} bytes)"
        return text
    preview = raw[:64].hex()
    return f"<non-text {len(raw)} bytes> hex[:64]={preview}"
