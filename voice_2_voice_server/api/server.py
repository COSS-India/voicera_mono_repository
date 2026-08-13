"""FastAPI server for telephony integration with optimized TCP settings."""

import os
import socket
import json
import traceback
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from loguru import logger
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

from .bot import bot
from utils.telemetry import router as telemetry_router
from utils.backend_utils import (
    create_meeting_in_backend,
    update_meeting_end_time,
    fetch_agent_config_from_backend,
    fetch_integration_key,
)
from utils.batching import create_batch_router


load_dotenv()

# Constants
AGENT_CONFIGS_DIR = Path("agent_configs")


# === TCP_NODELAY WebSocket Protocol ===

def create_nodelay_websocket_protocol():
    """Create a WebSocket protocol class with TCP_NODELAY enabled.
    
    This disables Nagle's algorithm for lower latency on small packets,
    which is critical for real-time voice applications.
    """
    try:
        from uvicorn.protocols.websockets.websockets_impl import WebSocketProtocol

        class NoDelayWebSocketProtocol(WebSocketProtocol):
            def connection_made(self, transport):
                # Set TCP_NODELAY before calling parent
                try:
                    sock = transport.get_extra_info("socket")
                    if sock is not None:
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        logger.debug("TCP_NODELAY enabled on WebSocket connection")
                except Exception as e:
                    logger.warning(f"Failed to set TCP_NODELAY: {e}")
                
                super().connection_made(transport)

        return NoDelayWebSocketProtocol
    
    except ImportError:
        logger.warning("Could not import WebSocketProtocol from uvicorn, TCP_NODELAY not available")
        return None


# === Pydantic Models ===

class OutboundCallRequest(BaseModel):
    """Request model for initiating outbound calls."""
    customer_number: str
    agent_id: str
    custom_field: Optional[str] = None
    caller_id: Optional[str] = None


class ViPostStreamRequest(BaseModel):
    """VI post-stream callback payload (Option 2 in VI user manual)."""
    Callid: str
    dni: str
    cli: str


class ViPostStreamResponse(BaseModel):
    """VI post-stream callback response."""
    callstatus: str
    rm_number: str = ""


# === Helper Functions ===

def _get_env_or_raise(key: str) -> str:
    """Get environment variable or raise ValueError."""
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


async def make_outbound_call_vobiz(
    customer_number: str,
    agent_id: str,
    caller_id: Optional[str] = None,
) -> dict:
    """Make an outbound call using Vobiz API.

    Vobiz Auth ID and Auth Token are loaded from the backend Integrations collection
    for the agent's organization (models VobizAuthId and VobizAuthToken), not from .env.

    Args:
        customer_number: Phone number to call
        agent_id: Agent ID to use
        caller_id: Optional caller ID (defaults to VOBIZ_CALLER_ID env var)

    Returns:
        Vobiz API response dictionary

    Raises:
        ValueError: If required credentials are missing
        requests.HTTPError: If API call fails
    """
    agent_config = await fetch_agent_config_from_backend(agent_id)
    if not agent_config:
        raise ValueError(f"Could not load agent config for agent_id={agent_id}")
    org_id = agent_config.get("org_id")
    if not org_id:
        raise ValueError("Agent has no org_id; cannot resolve Vobiz credentials from Integrations")

    auth_id = fetch_integration_key(org_id, "VobizAuthId")
    auth_token = fetch_integration_key(org_id, "VobizAuthToken")
    if not auth_id or not auth_token:
        raise ValueError(
            "Vobiz Auth ID and Auth Token must be configured in Integrations (Telephony) for this organization."
        )

    server_url = _get_env_or_raise("JOHNAIC_SERVER_URL")
    vobiz_api_base_url = _get_env_or_raise("VOBIZ_API_BASE")

    from_number = caller_id or os.environ.get("VOBIZ_CALLER_ID")
    if not from_number:
        raise ValueError("No caller_id provided and VOBIZ_CALLER_ID not set")

    headers = {
        "X-Auth-ID": auth_id,
        "X-Auth-Token": auth_token,
        "Content-Type": "application/json",
    }
    payload = {
        "from": from_number,
        "to": customer_number,
        "answer_url": f"{server_url}/answer?agent_id={agent_id}",
        "answer_method": "POST",
    }

    logger.info(f"📞 Outbound call: {from_number} → {customer_number} (agent: {agent_id})")
    
    vobiz_api_url = f"{vobiz_api_base_url}/Account/{auth_id}/Call/"
    response = requests.post(vobiz_api_url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()

    result = response.json()
    logger.info(f"✅ Call initiated: {result.get('call_uuid', 'unknown')}")
    return result


async def make_outbound_call_plivo(
    customer_number: str,
    agent_id: str,
    caller_id: Optional[str] = None,
) -> dict:
    """Make an outbound call using Plivo API with org-scoped Integration credentials."""
    agent_config = await fetch_agent_config_from_backend(agent_id)
    if not agent_config:
        raise ValueError(f"Could not load agent config for agent_id={agent_id}")
    org_id = agent_config.get("org_id")
    if not org_id:
        raise ValueError("Agent has no org_id; cannot resolve Plivo credentials from Integrations")

    auth_id = fetch_integration_key(org_id, "PlivoAuthId")
    auth_token = fetch_integration_key(org_id, "PlivoAuthToken")
    if not auth_id or not auth_token:
        raise ValueError(
            "Plivo Auth ID and Auth Token must be configured in Integrations (Telephony) for this organization."
        )

    server_url = _get_env_or_raise("JOHNAIC_SERVER_URL")
    plivo_api_base_url = os.environ.get("PLIVO_API_BASE", "https://api.plivo.com/v1")

    from_number = caller_id or os.environ.get("PLIVO_CALLER_ID")
    if not from_number:
        raise ValueError("No caller_id provided and PLIVO_CALLER_ID not set")

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "from": from_number,
        "to": customer_number,
        "answer_url": f"{server_url}/plivo/answer?agent_id={agent_id}",
        "answer_method": "POST",
        "hangup_url": f"{server_url}/plivo/hangup?agent_id={agent_id}",
        "hangup_method": "POST",
    }

    logger.info(f"📞 Outbound Plivo call: {from_number} → {customer_number} (agent: {agent_id})")
    plivo_api_url = f"{plivo_api_base_url}/Account/{auth_id}/Call/"
    response = requests.post(
        plivo_api_url,
        json=payload,
        headers=headers,
        auth=(auth_id, auth_token),
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    logger.info(f"✅ Plivo call initiated: {result.get('request_uuid', 'unknown')}")
    return result


async def make_outbound_call_provider(
    customer_number: str,
    agent_id: str,
    caller_id: Optional[str] = None,
) -> dict:
    """Dispatch outbound call based on agent telephony provider."""
    agent_config = await fetch_agent_config_from_backend(agent_id)
    if not agent_config:
        raise ValueError(f"Could not load agent config for agent_id={agent_id}")
    provider = (agent_config.get("telephony_provider") or "Vobiz").strip().lower()
    logger.info(f"Outbound provider={provider} agent_id={agent_id}")
    if provider == "vi":
        raise ValueError(
            "VI outbound calls are initiated via VI CPaaS APIs and DIY call flows, "
            "not through this endpoint."
        )
    if provider == "plivo":
        return await make_outbound_call_plivo(customer_number, agent_id, caller_id)
    return await make_outbound_call_vobiz(customer_number, agent_id, caller_id)


def _build_vi_websocket_url(agent_id: str) -> str:
    """Build the WSS URL VI configures in their Streaming Object."""
    websocket_prefix = os.environ.get("JOHNAIC_WEBSOCKET_URL", "").rstrip("/")
    return f"{websocket_prefix}/vi/agent/{agent_id}"


def _build_vi_post_stream_url() -> str:
    """Build the HTTPS callback URL for VI post-stream routing (Option 2)."""
    server_url = os.environ.get("JOHNAIC_SERVER_URL", "").rstrip("/")
    return f"{server_url}/vi/post-stream"


VI_START_TIMEOUT_SECS = 15.0
VI_START_MAX_MESSAGES = 20


async def _read_vi_start_message(websocket: WebSocket) -> tuple[dict, str, str]:
    """Read VI WebSocket messages until the start event is received."""
    for _ in range(VI_START_MAX_MESSAGES):
        message = await asyncio.wait_for(
            websocket.receive_text(), timeout=VI_START_TIMEOUT_SECS
        )
        data = json.loads(message)
        event = data.get("event")
        if event == "connected":
            logger.info("VI WebSocket connected event received")
            continue
        if event == "start":
            start_info = data.get("start", {}) or {}
            call_sid = (
                start_info.get("call_id")
                or data.get("call_id")
                or start_info.get("callSid")
                or "unknown"
            )
            stream_sid = (
                data.get("room_id")
                or start_info.get("room_id")
                or start_info.get("streamSid")
                or "unknown"
            )
            return data, str(call_sid), str(stream_sid)
        logger.warning(f"VI WebSocket expected connected/start, got event={event}")
    raise TimeoutError("VI start event not received")


def _resolve_vi_agent_id(path_agent_id: Optional[str], start_info: Dict[str, Any]) -> Optional[str]:
    """Resolve which agent to run for a VI stream.

    VI's proxy may not preserve the URL path, so fall back to the custom
    parameters configured on the Streaming object, then to a server default.
    """
    if path_agent_id:
        return path_agent_id

    custom_parameters = start_info.get("custom_parameters") or {}
    for key in ("agent_id", "agentId", "agent"):
        value = custom_parameters.get(key)
        if value:
            return str(value).strip()

    return (os.environ.get("VI_DEFAULT_AGENT_ID") or "").strip() or None


def _apply_vi_custom_parameters(
    agent_config: Dict[str, Any], custom_parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Overlay VI custom parameters (e.g. upstream language choice) on the agent."""
    if not custom_parameters:
        return agent_config

    agent_config["vi_custom_parameters"] = dict(custom_parameters)

    language = custom_parameters.get("language") or custom_parameters.get("lang")
    if language:
        language = str(language).strip()
        agent_config["language"] = language
        for model_key in ("stt_model", "tts_model"):
            model = agent_config.get(model_key)
            if isinstance(model, dict):
                model["language"] = language
        logger.info(f"VI custom parameter set language={language}")

    return agent_config


def _build_stream_xml(websocket_url: str) -> str:
    """Build Vobiz XML response for WebSocket streaming."""
    sample_rate = int(os.environ.get("SAMPLE_RATE", "8000"))

    # Use L16 for 16kHz per Vobiz spec (μ-law is 8kHz only)
    if sample_rate == 16000:
        content_type = "audio/x-l16;rate=16000"
    else:
        content_type = f"audio/x-mulaw;rate={sample_rate}"

    logger.info(f"Sending XML with contentType: {content_type}")
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" keepCallAlive="true" contentType="{content_type}">
        {websocket_url}
    </Stream>
</Response>'''


def _resolve_call_identifier(payload: Dict[str, Any]) -> str:
    """Resolve canonical provider call identifier for meeting_id."""
    if not isinstance(payload, dict):
        return "unknown"

    candidate_paths = [
        ("CallUUID",),            # Vobiz webhook
        ("call_uuid",),
        ("call_id",),
        ("callId",),              # Plivo/ws variants
        ("callSid",),
        ("CallSid",),
        ("request_uuid",),        # outbound response (fallback only)
        ("start", "callId"),      # nested websocket start object
        ("start", "callSid"),
    ]

    for path in candidate_paths:
        value: Any = payload
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value:
            resolved = str(value).strip()
            if resolved:
                return resolved
    return "unknown"


# === FastAPI App ===

app = FastAPI(
    title="Telephony Agent API",
    description="Voice bot API for telephony integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(telemetry_router)
app.include_router(create_batch_router(make_outbound_call_provider))


# === Routes ===

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"service": "Telephony Server", "status": "running"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {"status": "healthy"}


@app.post("/outbound/call/")
async def make_outbound_call(request: OutboundCallRequest):
    """Initiate an outbound call.

    Args:
        request: Outbound call parameters

    Returns:
        Call initiation result
    """
    try:
        result = await make_outbound_call_provider(
            request.customer_number,
            request.agent_id,
            request.caller_id,
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Outbound call initiated",
                "customer_number": request.customer_number,
                "agent_id": request.agent_id,
                "caller_id": request.caller_id,
                "result": result,
            },
        )
    except ValueError as e:
        logger.error(f"❌ Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Outbound call failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def log_meeting(agent_id: str, form_data_dict: dict):
    """Log meeting/call data to backend."""
    try:
        agent_config = await fetch_agent_config_from_backend(agent_id)
        agent_type = agent_config.get("agent_type")
        org_id = agent_config.get("org_id")

        direction = form_data_dict.get("Direction", "outbound")
        hangup_cause = str(form_data_dict.get("HangupCause", "")).upper()
        call_status = str(form_data_dict.get("CallStatus", "")).lower()
        is_busy = hangup_cause == "USER_BUSY" or call_status in {"busy", "no-answer", "failed"}
        start_time_utc = datetime.now(timezone.utc).isoformat()
        end_time_utc = start_time_utc if is_busy else ""

        meeting_data = {
            "meeting_id": _resolve_call_identifier(form_data_dict),
            "agent_type": agent_type,
            "org_id": org_id,
            "start_time_utc": start_time_utc,
            "end_time_utc": end_time_utc,
            "inbound": direction == "inbound",
            "from_number": form_data_dict.get("From", "unknown"),
            "to_number": form_data_dict.get("To", "unknown"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "call_busy": is_busy,
        }
        logger.info(f"Meeting data: {meeting_data}")
        await create_meeting_in_backend(meeting_data)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Meeting log failed: {e}")
        return {"status": "error", "message": str(e)}


@app.api_route("/answer", methods=["GET", "POST"])
async def vobiz_answer_webhook(request: Request):
    """Vobiz answer webhook - returns XML with WebSocket URL.

    This endpoint is called by Vobiz when a call is answered.
    It returns XML instructing Vobiz to connect to our WebSocket.
    """
    agent_id = request.query_params.get("agent_id")
    form_data = await request.form()
    form_data_dict = dict(form_data)
    event = form_data_dict.get("Event", "unknown")
    hangup_cause = form_data_dict.get("HangupCause", "USER_BUSY")

    if event == "StartApp":
        await log_meeting(agent_id, form_data_dict)
        websocket_prefix = os.environ.get("JOHNAIC_WEBSOCKET_URL", "")
        websocket_url = f"{websocket_prefix}/agent/{agent_id}"
        return Response(
            content=_build_stream_xml(websocket_url),
            media_type="application/xml",
        )
    elif event == "Hangup" and hangup_cause == "USER_BUSY":
        logger.info("User hung up the call")
        await log_meeting(agent_id, form_data_dict)
    else:
        logger.info("Hang URL Event Sent")


@app.websocket("/agent/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for Vobiz audio streaming.

    Args:
        websocket: WebSocket connection
        agent_id: Agent ID to use
    """
    await websocket.accept()
    logger.info(f"🔌 WebSocket connected: agent={agent_id}")

    call_sid = None
    stream_sid = None

    try:
        # Load agent configuration
        agent_config = await fetch_agent_config_from_backend(agent_id)
        agent_type = agent_config.get("agent_type")

        logger.info(f"📥 Agent config: {agent_config}")
        if not agent_config:
            logger.error(f"❌ Failed to fetch agent config from backend: {agent_id}")
            return

        # Wait for start event with call metadata
        first_message = await websocket.receive_text()
        data = json.loads(first_message)

        if data.get("event") != "start":
            logger.warning(f"⚠️ Expected 'start' event, got: {data.get('event')}")
            return

        start_info = data.get("start", {})
        call_sid = start_info.get("callSid") or start_info.get("callId", "unknown")
        stream_sid = start_info.get("streamSid") or start_info.get("streamId", "unknown")

        logger.info(f"📞 Call started: call_sid={call_sid}, stream_sid={stream_sid}")
        logger.debug(f"📋 Start info: {start_info}")

        await bot(
            websocket,
            stream_sid,
            call_sid,
            agent_type,
            agent_config,
            provider="vobiz",
        )

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        await websocket.close(code=1008, reason="Agent config not found")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info(f"🔌 WebSocket closed: call_sid={call_sid}")


@app.api_route("/plivo/answer", methods=["GET", "POST"])
async def plivo_answer_webhook(request: Request):
    """Plivo answer webhook - returns XML with WebSocket URL."""
    agent_id = request.query_params.get("agent_id")
    form_data = await request.form()
    form_data_dict = dict(form_data)
    await log_meeting(agent_id, form_data_dict)

    websocket_prefix = os.environ.get("JOHNAIC_WEBSOCKET_URL", "")
    websocket_url = f"{websocket_prefix}/plivo/agent/{agent_id}"
    return Response(
        content=_build_stream_xml(websocket_url),
        media_type="application/xml",
    )


@app.api_route("/plivo/hangup", methods=["GET", "POST"])
async def plivo_hangup_webhook(request: Request):
    """Plivo hangup callback for provider-native call completion events."""
    agent_id = request.query_params.get("agent_id")
    form_data = await request.form()
    form_data_dict = dict(form_data)
    await log_meeting(agent_id, form_data_dict)
    return Response(status_code=200)


@app.websocket("/plivo/agent/{agent_id}")
async def plivo_websocket_endpoint(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for Plivo audio streaming."""
    logger.info(f"🔌 Plivo WebSocket connected: agent={agent_id}")

    call_sid = None
    try:
        agent_config = await fetch_agent_config_from_backend(agent_id)
        if not agent_config:
            logger.error(f"❌ Failed to fetch agent config from backend: {agent_id}")
            return
        agent_type = agent_config.get("agent_type")

        call_sid = await bot(
            websocket,
            stream_sid=None,
            call_sid=None,
            agent_type=agent_type,
            agent_config=agent_config,
            provider="plivo",
        )
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        await websocket.close(code=1008, reason="Agent config not found")
    except Exception as e:
        logger.error(f"❌ Plivo WebSocket error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info(f"🔌 Plivo WebSocket closed: call_sid={call_sid}")


async def _run_vi_session(websocket: WebSocket, path_agent_id: Optional[str] = None):
    """Handle one VI bidirectional streaming session."""
    await websocket.accept()
    logger.info(f"🔌 VI WebSocket connected (path agent={path_agent_id or '-'})")

    call_sid = None
    stream_sid = None

    try:
        # The agent may be identified by the URL path or by custom parameters
        # inside `start`, so read the handshake before resolving the config.
        start_data, call_sid, stream_sid = await _read_vi_start_message(websocket)
        start_info = start_data.get("start", {}) or {}
        custom_parameters = start_info.get("custom_parameters") or {}

        agent_id = _resolve_vi_agent_id(path_agent_id, start_info)
        if not agent_id:
            logger.error(
                "❌ VI stream has no agent_id (URL path, custom_parameters, "
                "and VI_DEFAULT_AGENT_ID are all empty)"
            )
            await websocket.close(code=1008, reason="Missing agent_id")
            return

        agent_config = await fetch_agent_config_from_backend(agent_id)
        if not agent_config:
            logger.error(f"❌ Failed to fetch agent config from backend: {agent_id}")
            await websocket.close(code=1008, reason="Agent config not found")
            return

        agent_config = _apply_vi_custom_parameters(agent_config, custom_parameters)
        agent_type = agent_config.get("agent_type")

        meeting_payload = {
            "CallUUID": call_sid,
            "Direction": "inbound",
            "From": start_info.get("cli", "unknown"),
            "To": start_info.get("dni", "unknown"),
            "Event": "StartApp",
        }
        await log_meeting(agent_id, meeting_payload)

        logger.info(
            f"📞 VI call started: agent={agent_id} call_id={call_sid} room_id={stream_sid} "
            f"cli={start_info.get('cli')} dni={start_info.get('dni')}"
        )

        await bot(
            websocket,
            stream_sid,
            call_sid,
            agent_type,
            agent_config,
            provider="vi",
            sample_rate=8000,
        )

    except (asyncio.TimeoutError, TimeoutError):
        logger.error("❌ VI stream did not send a start event in time")
        await websocket.close(code=1008, reason="No start event")
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        await websocket.close(code=1008, reason="Agent config not found")
    except Exception as e:
        logger.error(f"❌ VI WebSocket error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info(f"🔌 VI WebSocket closed: call_id={call_sid}")


@app.websocket("/vi/agent/{agent_id}")
async def vi_websocket_endpoint(websocket: WebSocket, agent_id: str):
    """VI streaming endpoint with the agent id in the URL path."""
    await _run_vi_session(websocket, path_agent_id=agent_id)


@app.websocket("/vi/stream")
async def vi_stream_websocket_endpoint(websocket: WebSocket):
    """Static VI streaming endpoint; agent resolved from custom parameters."""
    await _run_vi_session(websocket)


@app.websocket("/")
async def vi_root_websocket_endpoint(websocket: WebSocket):
    """Root VI endpoint for proxies that strip the URL path."""
    await _run_vi_session(websocket)


@app.get("/vi/integration-info/{agent_id}")
async def vi_integration_info(agent_id: str):
    """Return VI integration URLs for an agent (for dashboard / onboarding)."""
    agent_config = await fetch_agent_config_from_backend(agent_id)
    if not agent_config:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    websocket_prefix = os.environ.get("JOHNAIC_WEBSOCKET_URL", "").rstrip("/")
    return {
        "agent_id": agent_id,
        "telephony_provider": agent_config.get("telephony_provider"),
        "websocket_url": _build_vi_websocket_url(agent_id),
        "websocket_url_fallbacks": [
            f"{websocket_prefix}/vi/stream",
            websocket_prefix,
        ],
        "custom_parameters": {"agent_id": agent_id},
        "post_stream_callback_url": _build_vi_post_stream_url(),
        "protocol": "VI Voice Streaming (bidirectional, 8kHz PCM16, JSON over WSS)",
        "custom_parameters_supported": 3,
    }


@app.post("/vi/post-stream")
async def vi_post_stream_callback(request: ViPostStreamRequest):
    """VI post-stream routing callback (Option 2 in the VI user manual).

    Default behaviour disconnects the call. Extend this to look up agent/session
    state and return callstatus=1 with rm_number for agent transfer when needed.
    """
    logger.info(
        "VI post-stream callback: Callid={} cli={} dni={}",
        request.Callid,
        request.cli,
        request.dni,
    )
    return ViPostStreamResponse(callstatus="0", rm_number="")


@app.websocket("/browser/agent/{agent_id}")
async def browser_websocket_endpoint(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for browser testing with live transcript events."""
    await websocket.accept()
    logger.info(f"🔌 Browser WebSocket connected: agent={agent_id}")

    call_sid = None
    stream_sid = None

    try:
        agent_config = await fetch_agent_config_from_backend(agent_id)
        agent_type = agent_config.get("agent_type")

        if not agent_config:
            logger.error(f"❌ Failed to fetch agent config from backend: {agent_id}")
            return

        first_message = await websocket.receive_text()
        data = json.loads(first_message)
        if data.get("event") != "start":
            logger.warning(f"⚠️ Expected 'start' event, got: {data.get('event')}")
            return

        start_info = data.get("start", {})
        call_sid = start_info.get("callSid") or start_info.get("callId", "unknown")
        stream_sid = start_info.get("streamSid") or start_info.get("streamId", "unknown")

        async def send_transcript(role: str, content: str, timestamp: Optional[str]):
            await websocket.send_text(
                json.dumps(
                    {
                        "event": "transcript",
                        "role": role,
                        "content": content,
                        "timestamp": timestamp,
                    }
                )
            )

        # Browser client streams 16 kHz L16 PCM (see test-browser-dialog.tsx).
        await bot(
            websocket,
            stream_sid,
            call_sid,
            agent_type,
            agent_config,
            provider="browser",
            transcript_callback=send_transcript,
            sample_rate=16000,
        )

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        await websocket.close(code=1008, reason="Agent config not found")
    except Exception as e:
        logger.error(f"❌ Browser WebSocket error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info(f"🔌 Browser WebSocket closed: call_sid={call_sid}")

def run_server(host: str = "0.0.0.0", port: int = 7860, log_level: str = "info"):
    """Run the server with optimized settings for low-latency voice applications.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        log_level: Logging level
    """
    import uvicorn

    # Create config with optimized settings
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        # Use uvloop for better async performance (if available)
        loop="auto",
        # HTTP/1.1 settings
        http="auto",
        # WebSocket settings
        ws="websockets",
    )

    # Set custom WebSocket protocol with TCP_NODELAY
    nodelay_protocol = create_nodelay_websocket_protocol()
    if nodelay_protocol:
        config.ws_protocol_class = nodelay_protocol
        logger.info("✅ TCP_NODELAY enabled for WebSocket connections (Nagle's algorithm disabled)")
    else:
        logger.warning("⚠️ Could not enable TCP_NODELAY, latency may be affected")

    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    run_server(host="0.0.0.0", port=7860, log_level="info")
