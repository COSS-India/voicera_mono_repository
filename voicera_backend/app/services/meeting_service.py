"""
Meeting service for handling meeting-related database operations.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from app.database import get_database
from app.models.schemas import MeetingCreate
from app.services.agent_service import fetch_agent_config
import logging
import re

logger = logging.getLogger(__name__)


def _serialize_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Convert MongoDB document to JSON-serializable format.
    Removes or converts ObjectId fields.
    """
    if doc is None:
        return None
    
    result = {}
    for key, value in doc.items():
        if key == "_id":
            result["id"] = str(value)
        else:
            result[key] = value
    return result


def _serialize_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert list of MongoDB documents to JSON-serializable format."""
    return [_serialize_doc(doc) for doc in docs if doc is not None]


def setup_meeting_id(meeting_data: MeetingCreate) -> Dict[str, Any]:
    """
    Set up a meeting with the specified ID and time information.
    
    Args:
        meeting_data: Meeting creation data
        
    Returns:
        Meeting document with agent details
    """
    try:
        db = get_database()
        meeting_table = db["CallLogs"]
        
        current_utc = datetime.now(timezone.utc).isoformat()
        
        is_update_only = (
            meeting_data.end_time_utc is not None and
            meeting_data.start_time_utc is None and
            meeting_data.inbound is None and
            meeting_data.from_number is None and
            meeting_data.to_number is None
        )
        
        meeting_doc = {
            "meeting_id": meeting_data.meeting_id,
            "agent_type": meeting_data.agent_type
        }
        
        # Set org_id from request if provided
        if meeting_data.org_id:
            meeting_doc["org_id"] = meeting_data.org_id
        
        # Set call_busy if provided (should be set regardless of update_only status)
        if meeting_data.call_busy is not None:
            meeting_doc["call_busy"] = meeting_data.call_busy
            logger.info(f"Setting call_busy={meeting_data.call_busy} for meeting {meeting_data.meeting_id}")
        
        if not is_update_only:
            if meeting_data.inbound is not None:
                meeting_doc["inbound"] = meeting_data.inbound
            if meeting_data.from_number:
                meeting_doc["from_number"] = meeting_data.from_number
            if meeting_data.to_number:
                meeting_doc["to_number"] = meeting_data.to_number
            
            meeting_doc["created_at"] = meeting_data.created_at or current_utc
            meeting_doc["start_time_utc"] = meeting_data.start_time_utc or current_utc
        
        if meeting_data.end_time_utc:
            meeting_doc["end_time_utc"] = meeting_data.end_time_utc
        
        # Fetch agent details (org_id from agent_config only if not already set)
        try:
            agent_config = fetch_agent_config(meeting_data.agent_type)
            if agent_config:
                for field in ['agent_category', 'agent_config']:
                    if field in agent_config and agent_config[field] is not None:
                        meeting_doc[field] = agent_config[field]
                # Only use org_id from agent_config if not provided in request
                if 'org_id' not in meeting_doc and agent_config.get('org_id'):
                    meeting_doc['org_id'] = agent_config['org_id']
        except Exception as e:
            logger.warning(f"Error fetching agent details: {str(e)}")
        
        logger.info(f"Meeting document to save: {meeting_doc}")
        result = meeting_table.update_one(
            {"meeting_id": meeting_data.meeting_id},
            {"$set": meeting_doc},
            upsert=True
        )
        logger.info(f"Update result - matched: {result.matched_count}, modified: {result.modified_count}, upserted_id: {result.upserted_id}")
        
        logger.info(f"Meeting setup successfully: {meeting_data.meeting_id}")
        return meeting_doc
        
    except Exception as e:
        logger.error(f"Error setting up meeting: {str(e)}")
        return {"status": "fail", "message": f"Error setting up meeting: {str(e)}"}


def fetch_meeting_details(meeting_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch meeting details by meeting_id.
    
    Args:
        meeting_id: Meeting identifier
        
    Returns:
        Meeting document or None
    """
    try:
        db = get_database()
        meeting_table = db["CallLogs"]
        meeting = meeting_table.find_one({"meeting_id": meeting_id})
        return _serialize_doc(meeting)
    except Exception as e:
        logger.error(f"Error fetching meeting details: {str(e)}")
        return None


def _build_meetings_query(
    org_id: str,
    agent_type: Optional[str] = None,
    from_number: Optional[str] = None,
    to_number: Optional[str] = None,
    inbound: Optional[bool] = None,
    call_status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    has_latency_metrics: Optional[bool] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """Build MongoDB query mirroring History tab client filters."""
    conditions: List[Dict[str, Any]] = [{"org_id": org_id}]

    if agent_type:
        conditions.append({"agent_type": agent_type})
    if from_number:
        conditions.append({"from_number": from_number})
    if to_number:
        conditions.append({"to_number": to_number})
    if inbound is not None:
        conditions.append({"inbound": inbound})

    if call_status:
        status_lower = call_status.strip().lower()
        if status_lower == "busy":
            conditions.append({"call_busy": True})
        elif status_lower == "completed":
            conditions.append({
                "$and": [
                    {"$or": [{"call_busy": {"$ne": True}}, {"call_busy": {"$exists": False}}]},
                    {"end_time_utc": {"$exists": True, "$nin": [None, ""]}},
                ]
            })
        elif status_lower == "in progress":
            conditions.append({
                "$and": [
                    {"$or": [{"call_busy": {"$ne": True}}, {"call_busy": {"$exists": False}}]},
                    {
                        "$or": [
                            {"end_time_utc": {"$exists": False}},
                            {"end_time_utc": None},
                            {"end_time_utc": ""},
                        ]
                    },
                ]
            })

    if date_from or date_to:
        expr_parts: List[Dict[str, Any]] = []
        coalesced = {"$ifNull": ["$start_time_utc", "$created_at"]}
        if date_from:
            expr_parts.append({"$gte": [coalesced, date_from]})
        if date_to:
            expr_parts.append({"$lte": [coalesced, date_to]})
        if expr_parts:
            conditions.append({"$expr": {"$and": expr_parts}})

    if has_latency_metrics:
        conditions.append({"latency_metrics.turns.0": {"$exists": True}})

    if search:
        term = search.strip()
        if term:
            regex = {"$regex": re.escape(term), "$options": "i"}
            search_clauses: List[Dict[str, Any]] = [
                {"meeting_id": regex},
                {"agent_type": regex},
                {"from_number": regex},
                {"to_number": regex},
            ]
            term_lower = term.lower()
            if term_lower == "inbound":
                search_clauses.append({"inbound": True})
            elif term_lower == "outbound":
                search_clauses.append({"inbound": False})
            conditions.append({"$or": search_clauses})

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _meetings_sort(
    date_sort_order: str = "latest",
    duration_sort_order: Optional[str] = None,
) -> List[tuple]:
    if duration_sort_order == "longest":
        return [("duration", -1), ("created_at", -1)]
    if duration_sort_order == "shortest":
        return [("duration", 1), ("created_at", -1)]
    if date_sort_order == "oldest":
        return [("created_at", 1)]
    return [("created_at", -1)]


def fetch_meetings_paginated(
    org_id: str,
    page: int = 1,
    limit: int = 50,
    agent_type: Optional[str] = None,
    from_number: Optional[str] = None,
    to_number: Optional[str] = None,
    inbound: Optional[bool] = None,
    call_status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    date_sort_order: str = "latest",
    duration_sort_order: Optional[str] = None,
    has_latency_metrics: Optional[bool] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch a page of meetings for an org with filters and sort applied server-side.
    """
    try:
        db = get_database()
        meeting_table = db["CallLogs"]
        query = _build_meetings_query(
            org_id=org_id,
            agent_type=agent_type,
            from_number=from_number,
            to_number=to_number,
            inbound=inbound,
            call_status=call_status,
            date_from=date_from,
            date_to=date_to,
            has_latency_metrics=has_latency_metrics,
            search=search,
        )
        total = meeting_table.count_documents(query)
        skip = (page - 1) * limit
        sort_spec = _meetings_sort(date_sort_order, duration_sort_order)
        cursor = (
            meeting_table.find(query)
            .sort(sort_spec)
            .skip(skip)
            .limit(limit)
        )
        meetings = list(cursor)
        serialized = _serialize_docs(meetings)
        items = transform_meetings_for_frontend(serialized)
        return {
            "items": items,
            "total": total,
            "page": page,
            "limit": limit,
        }
    except Exception as e:
        logger.error(f"Error fetching paginated meetings: {str(e)}")
        return {"items": [], "total": 0, "page": page, "limit": limit}


def fetch_meeting_filter_options(org_id: str) -> Dict[str, List[str]]:
    """Distinct values for History filter dropdowns."""
    try:
        db = get_database()
        meeting_table = db["CallLogs"]
        base = {"org_id": org_id}

        def _distinct(field: str) -> List[str]:
            values = meeting_table.distinct(field, base)
            return sorted(
                v for v in values
                if v is not None and str(v).strip() != ""
            )

        return {
            "agent_types": _distinct("agent_type"),
            "from_numbers": _distinct("from_number"),
            "to_numbers": _distinct("to_number"),
        }
    except Exception as e:
        logger.error(f"Error fetching meeting filter options: {str(e)}")
        return {"agent_types": [], "from_numbers": [], "to_numbers": []}


def fetch_meetings_of_org(org_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all meetings for a given org.
    
    Args:
        org_id: Organization ID
        
    Returns:
        List of meeting documents (transformed for frontend)
    """
    try:
        db = get_database()
        meeting_table = db["CallLogs"]
        meetings = list(meeting_table.find({"org_id": org_id}).sort("created_at", -1))
        serialized = _serialize_docs(meetings)
        return transform_meetings_for_frontend(serialized)
    except Exception as e:
        logger.error(f"Error fetching meetings: {str(e)}")
        return []


def fetch_meetings_by_org_and_agent(org_id: str, agent_type: str) -> List[Dict[str, Any]]:
    """
    Fetch all meetings for a given org and agent type.
    
    Args:
        org_id: Organization ID
        agent_type: Agent type identifier
        
    Returns:
        List of meeting documents (transformed for frontend)
    """
    try:
        db = get_database()
        meeting_table = db["CallLogs"]
        meetings = list(meeting_table.find({
            "org_id": org_id,
            "agent_type": agent_type
        }).sort("created_at", -1))
        serialized = _serialize_docs(meetings)
        return transform_meetings_for_frontend(serialized)
    except Exception as e:
        logger.error(f"Error fetching meetings: {str(e)}")
        return []


def update_meeting_end_time(meeting_id: str, end_time_utc: str) -> Dict[str, Any]:
    """
    Update a meeting with end time when call ends.
    
    Args:
        meeting_id: Meeting identifier (call_sid)
        end_time_utc: End time in UTC ISO format
        
    Returns:
        Updated meeting document or error dict
    """
    try:
        db = get_database()
        meeting_table = db["CallLogs"]
        
        existing = meeting_table.find_one({"meeting_id": meeting_id})
        if not existing:
            return {"status": "fail", "message": f"Meeting not found: {meeting_id}"}
        
        result = meeting_table.update_one(
            {"meeting_id": meeting_id},
            {"$set": {"end_time_utc": end_time_utc}}
        )
        
        if result.modified_count == 0:
            logger.warning(f"Meeting not modified: {meeting_id}")
        
        logger.info(f"Meeting end time updated: {meeting_id}")
        
        updated = meeting_table.find_one({"meeting_id": meeting_id})
        return _serialize_doc(updated)
        
    except Exception as e:
        logger.error(f"Error updating meeting end time: {str(e)}")
        return {"status": "fail", "message": f"Error updating meeting: {str(e)}"}


def parse_transcript(transcript_content: str) -> List[Dict[str, Any]]:
    """
    Parse transcript content into structured messages.
    
    Expected format:
        [timestamp] user: message
        [timestamp] assistant: message
    
    Args:
        transcript_content: Raw transcript text
        
    Returns:
        List of message objects with role, content, and timestamp
    """
    if not transcript_content:
        return []
    
    messages: List[Dict[str, Any]] = []
    lines = transcript_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Match pattern: [timestamp] role: message
        # Supports: user, assistant, agent, human, bot
        match = re.match(r'^\[([^\]]+)\]\s*(user|assistant|agent|human|bot):\s*(.+)$', line, re.IGNORECASE)
        
        if match:
            timestamp, role, message = match.groups()
            # Normalize role to 'user' or 'agent'
            normalized_role = 'user' if role.lower() in ['user', 'human'] else 'agent'
            
            messages.append({
                'role': normalized_role,
                'content': message.strip(),
                'timestamp': timestamp.strip()
            })
        else:
            # If no timestamp/role prefix, try to infer from common patterns
            if line.lower().startswith('user:') or line.lower().startswith('human:'):
                content = re.sub(r'^(user|human):\s*', '', line, flags=re.IGNORECASE).strip()
                messages.append({
                    'role': 'user',
                    'content': content,
                })
            elif line.lower().startswith('agent:') or line.lower().startswith('assistant:') or line.lower().startswith('bot:'):
                content = re.sub(r'^(agent|assistant|bot):\s*', '', line, flags=re.IGNORECASE).strip()
                messages.append({
                    'role': 'agent',
                    'content': content,
                })
            else:
                # Default: alternate between user and agent if we have previous messages
                # Otherwise default to agent
                if messages:
                    last_role = messages[-1]['role']
                    next_role = 'user' if last_role == 'agent' else 'agent'
                else:
                    next_role = 'agent'
                
                messages.append({
                    'role': next_role,
                    'content': line,
                })
    
    return messages


def transform_recording_url(recording_url: str, meeting_id: str, base_url: str = "/api/v1") -> str:
    """
    Transform minio:// URLs to proxy endpoint URLs.
    
    Args:
        recording_url: Original URL (minio:// or http://)
        meeting_id: Meeting ID for the proxy endpoint
        base_url: Base API URL path (for backend, use /api/v1; frontend will use /api)
        
    Returns:
        Transformed URL pointing to proxy endpoint
        For frontend consumption, returns /api/meetings/{id}/recording
    """
    if not recording_url:
        return None
    
    # If already HTTP/HTTPS, return as-is
    if recording_url.startswith('http://') or recording_url.startswith('https://'):
        return recording_url
    
    # Convert minio:// URLs to proxy endpoint
    # Use /api/meetings/... for frontend (Next.js API route)
    # The frontend will proxy this to the backend
    if recording_url.startswith('minio://'):
        return f"/api/meetings/{meeting_id}/recording"
    
    # Return as-is if unknown format
    return recording_url


def transform_meeting_for_frontend(meeting: Dict[str, Any], base_url: str = None) -> Dict[str, Any]:
    """
    Transform meeting data for frontend consumption.
    
    - Converts minio:// recording URLs to proxy endpoints
    - Parses transcript_content into structured transcript array
    - Preserves all other fields
    
    Args:
        meeting: Raw meeting document from database
        base_url: Base API URL path (not used, kept for backward compatibility)
        
    Returns:
        Transformed meeting document
    """
    if not meeting:
        return meeting
    
    result = meeting.copy()
    meeting_id = meeting.get('meeting_id', '')
    
    if 'recording_url' in result and result['recording_url']:
        result['recording_url'] = transform_recording_url(
            result['recording_url'],
            meeting_id
        )
    
    # Parse transcript if present in database
    transcript_content = result.get('transcript_content')
    if transcript_content:
        result['transcript'] = parse_transcript(transcript_content)
    else:
        result['transcript'] = []
    
    return result


def transform_meetings_for_frontend(meetings: List[Dict[str, Any]], base_url: str = None) -> List[Dict[str, Any]]:
    """
    Transform a list of meetings for frontend consumption.
    
    Args:
        meetings: List of meeting documents
        base_url: Base API URL path (not used, kept for backward compatibility)
        
    Returns:
        List of transformed meeting documents
    """
    return [transform_meeting_for_frontend(meeting, base_url) for meeting in meetings]