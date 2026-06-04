"""
Call recording service for handling call recording-related database operations.
"""
from typing import Optional, Dict, Any
from app.database import get_database
from app.models.schemas import CallRecordingCreate
from app.utils.mongo_utils import prepare_mongo_response
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def save_call_recording(recording_data: CallRecordingCreate) -> Dict[str, Any]:
    """
    Save or update call recording data in the database.
    
    This function updates the existing meeting record (identified by call_sid/meeting_id)
    with recording URLs, transcript content, and call duration information.
    
    Args:
        recording_data: Call recording data including URLs and transcript
        
    Returns:
        Updated meeting document or error response
    """
    try:
        db = get_database()
        meeting_table = db["CallLogs"]
        
        # Build update document
        update_doc = {
            "recording_url": recording_data.recording_url,
            "transcript_url": recording_data.transcript_url,
            "agent_type": recording_data.agent_type,
        }
        
        # Add optional fields if provided
        if recording_data.transcript_content:
            update_doc["transcript_content"] = recording_data.transcript_content
        
        if recording_data.call_duration is not None:
            update_doc["duration"] = recording_data.call_duration
        
        if recording_data.end_time_utc:
            update_doc["end_time_utc"] = recording_data.end_time_utc
        
        if recording_data.org_id:
            update_doc["org_id"] = recording_data.org_id

        if recording_data.latency_metrics:
            update_doc["latency_metrics"] = recording_data.latency_metrics

        # Update or insert meeting record
        # Use call_sid as meeting_id for lookup
        result = meeting_table.update_one(
            {"meeting_id": recording_data.call_sid},
            {
                "$set": update_doc,
                "$setOnInsert": {
                    "meeting_id": recording_data.call_sid,
                    "created_at": datetime.utcnow().isoformat()
                }
            },
            upsert=True
        )
        
        # Fetch and return the updated document
        updated_meeting = meeting_table.find_one({"meeting_id": recording_data.call_sid})
        
        if updated_meeting:
            logger.info(f"Call recording saved successfully: {recording_data.call_sid}")
            # Convert ObjectId to string for JSON serialization
            return prepare_mongo_response(updated_meeting)
        else:
            logger.warning(f"Call recording saved but document not found: {recording_data.call_sid}")
            return {"status": "fail", "message": "Recording saved but document not found"}
        
    except Exception as e:
        logger.error(f"Error saving call recording: {str(e)}")
        return {"status": "fail", "message": f"Error saving call recording: {str(e)}"}
