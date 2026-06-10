"""Utilities for submitting call recording data to the backend API."""

import os
import time
import traceback
from datetime import datetime
from typing import Optional

from loguru import logger
import requests
from storage.minio_client import MinIOStorage


async def submit_call_recording(
    call_sid: str,
    agent_type: str,
    agent_config: dict,
    storage: MinIOStorage,
    call_start_time: float,
    latency_metrics: Optional[dict] = None,
    recording_url: Optional[str] = None,
    omit_recording_url: bool = False,
) -> None:
    """
    Submit call recording data to the backend API after a call ends.
    
    This function reads the transcript from MinIO, builds the recording URLs,
    and sends all call metadata to the backend API endpoint.
    
    Args:
        call_sid: Call identifier (same as meeting_id)
        agent_type: Type of agent used for the call
        agent_config: Agent configuration dictionary
        storage: MinIOStorage instance for accessing stored files
        call_start_time: Monotonic time when call started
    """
    try:
        logger.info(f"Submitting call recording data to backend after call ends: {call_sid}")
        call_end_time = time.monotonic()
        call_duration = call_end_time - call_start_time
        end_time_utc = datetime.utcnow().isoformat()
        
        transcript_url = f"minio://transcripts/{call_sid}.txt"
        
        transcript_content = None
        try:
            response = await storage.get_object("transcripts", f"{call_sid}.txt")
            transcript_content = response.read().decode("utf-8")
            response.close()
            response.release_conn()
        except Exception as e:
            logger.warning(f"⚠️ Could not read transcript: {e}")
        
        # Get backend API URL from environment
        backend_url = os.getenv("VOICERA_BACKEND_URL", "http://localhost:8000")
        api_endpoint = f"{backend_url}/api/v1/call-recordings"
        
        # Prepare payload
        payload = {
            "call_sid": call_sid,
            "transcript_url": transcript_url,
            "transcript_content": transcript_content,
            "agent_type": agent_type,
            "call_duration": call_duration,
            "end_time_utc": end_time_utc,
        }
        
        if not omit_recording_url:
            payload["recording_url"] = recording_url or f"minio://recordings/{call_sid}.wav"

        # Add org_id if available in agent config
        if "org_id" in agent_config:
            payload["org_id"] = agent_config["org_id"]

        if latency_metrics and latency_metrics.get("turns"):
            payload["latency_metrics"] = latency_metrics

        # Send to backend API
        logger.info(f"📤 Sending call recording data to backend: {call_sid}")
        response = requests.post(
            api_endpoint,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        logger.info(f"✅ Call recording data saved successfully: {call_sid}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to send call recording data: {e}")
    except Exception as e:
        logger.error(f"❌ Error processing call recording data: {e}")
        logger.debug(traceback.format_exc())
