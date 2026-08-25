"""
Analytics service for calculating call metrics on-demand.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from app.database import get_database
import logging

logger = logging.getLogger(__name__)


def calculate_duration_in_minutes(start_time: Optional[str], end_time: Optional[str], duration_seconds: Optional[float]) -> Optional[float]:
    """
    Calculate call duration in minutes.
    
    Priority:
    1. Use duration_seconds if available
    2. Calculate from start_time_utc and end_time_utc if both exist
    3. Return None if insufficient data
    
    Args:
        start_time: ISO format start time string
        end_time: ISO format end time string
        duration_seconds: Duration in seconds (if available)
        
    Returns:
        Duration in minutes, or None if cannot be calculated
    """
    # First, try to use the duration field if available
    if duration_seconds is not None and duration_seconds > 0:
        return duration_seconds / 60.0
    
    # Otherwise, calculate from start and end times
    if start_time and end_time:
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            duration_seconds = (end_dt - start_dt).total_seconds()
            if duration_seconds > 0:
                return duration_seconds / 60.0
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error parsing dates for duration calculation: {e}")
    
    return None


def is_call_connected(call_log: Dict[str, Any]) -> bool:
    """
    Determine if a call was successfully connected.
    
    A call is considered connected if:
    - It has an end_time_utc (call completed)
    - OR it has a duration > 0
    - AND call_busy is not True (call wasn't busy/failed)
    
    Args:
        call_log: Call log document from CallLogs collection
        
    Returns:
        True if call was connected, False otherwise
    """
    # Check if call was busy/failed
    if call_log.get("call_busy") is True:
        return False
    
    # Check if call has end time or duration
    has_end_time = bool(call_log.get("end_time_utc"))
    has_duration = call_log.get("duration") is not None and call_log.get("duration", 0) > 0
    
    return has_end_time or has_duration


def get_analytics(org_id: str, agent_type: Optional[str] = None, phone_number: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate analytics metrics on-demand for a given organization.
    
    Metrics calculated:
    - Calls Attempted: Total number of call records
    - Calls Connected: Number of calls that successfully connected
    - Average Call Duration: Average duration of connected calls (in minutes)
    - Total Minutes Connected: Sum of all connected call durations (in minutes)
    - Most Used Agent: Agent type with the highest call count
    
    Args:
        org_id: Organization ID
        agent_type: Optional filter by agent type
        phone_number: Optional filter by phone number
        
    Returns:
        Dictionary containing all analytics metrics
    """
    try:
        db = get_database()
        call_logs_table = db["CallLogs"]
        
        # Build query filter
        query_filter = {"org_id": org_id}
        if agent_type:
            query_filter["agent_type"] = agent_type
        if phone_number:
            query_filter["phone_number"] = phone_number
        
        # Get all call logs for the org
        all_calls = list(call_logs_table.find(query_filter))
        
        # Calculate metrics
        calls_attempted = len(all_calls)
        
        # Filter connected calls
        connected_calls = [call for call in all_calls if is_call_connected(call)]
        calls_connected = len(connected_calls)
        
        # Calculate duration metrics
        durations = []
        total_minutes = 0.0
        
        for call in connected_calls:
            duration_minutes = calculate_duration_in_minutes(
                call.get("start_time_utc"),
                call.get("end_time_utc"),
                call.get("duration")
            )
            if duration_minutes is not None:
                durations.append(duration_minutes)
                total_minutes += duration_minutes
        
        # Calculate average duration
        average_call_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Find most used agent
        agent_counts: Dict[str, int] = {}
        for call in all_calls:
            agent = call.get("agent_type")
            if agent:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        most_used_agent = max(agent_counts.items(), key=lambda x: x[1])[0] if agent_counts else None
        most_used_agent_count = agent_counts.get(most_used_agent, 0) if most_used_agent else 0
        
        # Build agent breakdown (optional detailed view)
        agent_breakdown = [
            {"agent_type": agent, "call_count": count}
            for agent, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return {
            "org_id": org_id,
            "calls_attempted": calls_attempted,
            "calls_connected": calls_connected,
            "average_call_duration": round(average_call_duration, 2),
            "total_minutes_connected": round(total_minutes, 2),
            "most_used_agent": most_used_agent,
            "most_used_agent_count": most_used_agent_count,
            "agent_breakdown": agent_breakdown,
            "calculated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating analytics for org {org_id}: {str(e)}")
        raise


def get_analytics_by_date_range(
    org_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    agent_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate analytics metrics for a specific date range.
    
    Args:
        org_id: Organization ID
        start_date: ISO format start date (optional)
        end_date: ISO format end date (optional)
        agent_type: Optional filter by agent type
        
    Returns:
        Dictionary containing all analytics metrics for the date range
    """
    try:
        db = get_database()
        call_logs_table = db["CallLogs"]
        
        # Build query filter
        query_filter = {"org_id": org_id}
        if agent_type:
            query_filter["agent_type"] = agent_type
        
        # Add date range filter if provided
        # Filter by created_at (when call was initiated)
        if start_date or end_date:
            date_filter = {}
            if start_date:
                try:
                    # Handle both date-only and datetime formats
                    if 'T' not in start_date:
                        start_date = f"{start_date}T00:00:00"
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    date_filter["$gte"] = start_dt.isoformat()
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Invalid start_date format: {start_date}, error: {e}")
            if end_date:
                try:
                    # Handle both date-only and datetime formats
                    if 'T' not in end_date:
                        end_date = f"{end_date}T23:59:59"
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    date_filter["$lte"] = end_dt.isoformat()
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Invalid end_date format: {end_date}, error: {e}")
            
            if date_filter:
                query_filter["created_at"] = date_filter
        
        # Get all call logs for the org and date range
        all_calls = list(call_logs_table.find(query_filter))
        
        # Calculate metrics (same logic as get_analytics)
        calls_attempted = len(all_calls)
        
        connected_calls = [call for call in all_calls if is_call_connected(call)]
        calls_connected = len(connected_calls)
        
        durations = []
        total_minutes = 0.0
        
        for call in connected_calls:
            duration_minutes = calculate_duration_in_minutes(
                call.get("start_time_utc"),
                call.get("end_time_utc"),
                call.get("duration")
            )
            if duration_minutes is not None:
                durations.append(duration_minutes)
                total_minutes += duration_minutes
        
        average_call_duration = sum(durations) / len(durations) if durations else 0.0
        
        agent_counts: Dict[str, int] = {}
        for call in all_calls:
            agent = call.get("agent_type")
            if agent:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        most_used_agent = max(agent_counts.items(), key=lambda x: x[1])[0] if agent_counts else None
        most_used_agent_count = agent_counts.get(most_used_agent, 0) if most_used_agent else 0
        
        agent_breakdown = [
            {"agent_type": agent, "call_count": count}
            for agent, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return {
            "org_id": org_id,
            "start_date": start_date,
            "end_date": end_date,
            "calls_attempted": calls_attempted,
            "calls_connected": calls_connected,
            "average_call_duration": round(average_call_duration, 2),
            "total_minutes_connected": round(total_minutes, 2),
            "most_used_agent": most_used_agent,
            "most_used_agent_count": most_used_agent_count,
            "agent_breakdown": agent_breakdown,
            "calculated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating analytics for org {org_id} with date range: {str(e)}")
        raise
