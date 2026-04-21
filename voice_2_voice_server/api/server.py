"""FastAPI server for Vobiz telephony integration with optimized TCP settings."""

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

from .bot import bot, ubona_bot, telnyx_bot
from .telemetry import router as telemetry_router
from .backend_utils import (
    create_meeting_in_backend,
    update_meeting_end_time,
    fetch_agent_config_from_backend,
    fetch_integration_key,
)
from .batching import create_batch_router


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
    answer_url = f"{server_url}/answer?agent_id={agent_id}"
    payload = {
        "from": from_number,
        "to": customer_number,
        "answer_url": answer_url,
        "answer_method": "POST",
    }

    ws_url = os.environ.get("JOHNAIC_WEBSOCKET_URL", "<not set>")
    logger.info(
        "📞 Outbound call: {} → {} agent={} | answer_url={} | websocket_base={}",
        from_number,
        customer_number,
        agent_id,
        answer_url,
        ws_url,
    )
    
    vobiz_api_url = f"{vobiz_api_base_url}/Account/{auth_id}/Call/"
    response = requests.post(vobiz_api_url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()

    result = response.json()
    logger.info(f"✅ Call initiated: {result.get('call_uuid', 'unknown')}")
    return result


async def make_outbound_call_telnyx(
    customer_number: str,
    agent_id: str,
    caller_id: Optional[str] = None,
) -> dict:
    """Initiate an outbound call via the Telnyx Call Control API.

    Credentials are read from environment variables (not the backend
    Integrations collection) because the Telnyx integration is currently
    configured at the server level. If we later want per-org Telnyx
    credentials we can mirror the pattern used in make_outbound_call_vobiz().

    Args:
        customer_number: E.164 destination number (for example "+14155550123").
        agent_id: Voicera agent ID used in the TeXML webhook query string.
        caller_id: Optional E.164 override for the From number. Falls back to
            TELNYX_CALLER_ID when not provided.

    Returns:
        Parsed JSON response from POST https://api.telnyx.com/v2/calls.

    Raises:
        ValueError: If any required environment variable is missing.
        httpx.HTTPStatusError: If the Telnyx API returns a non-2xx status.
    """
    import httpx  # fastapi[all] brings httpx in; avoid pulling telnyx SDK.

    api_key = _get_env_or_raise("TELNYX_API_KEY")
    connection_id = _get_env_or_raise("TELNYX_CONNECTION_ID")
    server_url = _get_env_or_raise("JOHNAIC_SERVER_URL")
    from_number = caller_id or _get_env_or_raise("TELNYX_CALLER_ID")

    webhook_url = f"{server_url}/telnyx/answer?agent_id={agent_id}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "connection_id": connection_id,
        "to": customer_number,
        "from": from_number,
        "webhook_url": webhook_url,
        "webhook_url_method": "POST",
    }

    ws_url = os.environ.get("JOHNAIC_WEBSOCKET_URL", "<not set>")
    logger.info(
        "📞 Telnyx outbound call: {} → {} agent={} | webhook_url={} | websocket_base={}",
        from_number,
        customer_number,
        agent_id,
        webhook_url,
        ws_url,
    )

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.telnyx.com/v2/calls",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

    # Telnyx wraps the resource in a "data" key.
    data = result.get("data", {}) if isinstance(result, dict) else {}
    call_control_id = data.get("call_control_id", "unknown")
    logger.info(f"✅ Telnyx call initiated: call_control_id={call_control_id}")
    return result


def _build_telnyx_stream_xml(websocket_url: str) -> str:
    """Build the TeXML response that tells Telnyx to start a bidirectional stream.

    Telnyx's TeXML <Stream> element is compatible with Twilio's and accepts the
    audio/x-l16;rate=16000 contentType when we want 16kHz PCM. Default remains
    mu-law 8kHz so a fresh deploy works out of the box.
    """
    sample_rate = int(os.environ.get("SAMPLE_RATE", "8000"))
    if sample_rate == 16000:
        content_type = "audio/x-l16;rate=16000"
    else:
        content_type = "audio/x-mulaw;rate=8000"

    logger.info(f"Sending Telnyx TeXML with contentType: {content_type}")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        f'    <Stream url="{websocket_url}" bidirectional="true" '
        f'contentType="{content_type}" />\n'
        "</Response>"
    )


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


# === FastAPI App ===

app = FastAPI(
    title="Vobiz Telephony Agent API",
    description="Voice bot API for Vobiz telephony integration",
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
app.include_router(create_batch_router(make_outbound_call_vobiz))


@app.on_event("startup")
async def _log_voice_urls_at_startup():
    """Log URLs at startup so we can verify they are set and public."""
    server_url = os.environ.get("JOHNAIC_SERVER_URL", "")
    ws_url = os.environ.get("JOHNAIC_WEBSOCKET_URL", "")
    logger.info(
        "📞 Voice server startup: JOHNAIC_SERVER_URL={} JOHNAIC_WEBSOCKET_URL={} (Vobiz must reach these)",
        server_url or "<not set>",
        ws_url or "<not set>",
    )
    if not ws_url:
        logger.warning("📞 JOHNAIC_WEBSOCKET_URL not set - WebSocket for call audio will not work")


# === Routes ===

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"service": "Vobiz Telephony Server", "status": "running"}


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
        result = await make_outbound_call_vobiz(
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
        is_busy = form_data_dict.get("HangupCause", "unknown") == "USER_BUSY"
        start_time_utc = datetime.now(timezone.utc).isoformat()
        end_time_utc = start_time_utc if is_busy else ""

        meeting_data = {
            "meeting_id": form_data_dict.get("CallUUID", "unknown"),
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

    logger.info(
        "📞 Answer webhook: method={} agent_id={} Event={} HangupCause={} form_keys={}",
        request.method,
        agent_id,
        event,
        hangup_cause,
        list(form_data_dict.keys()),
    )

    if event == "StartApp":
        await log_meeting(agent_id, form_data_dict)
        websocket_prefix = os.environ.get("JOHNAIC_WEBSOCKET_URL", "")
        websocket_url = f"{websocket_prefix}/agent/{agent_id}"
        xml_body = _build_stream_xml(websocket_url)
        logger.info(
            "📞 Answer webhook: returning Stream XML | websocket_url={} | Vobiz should connect here for audio",
            websocket_url,
        )
        logger.info(
            "📞 Answer webhook: XML length={} (contentType=audio/x-mulaw;rate=8000 or L16)",
            len(xml_body),
        )
        return Response(
            content=xml_body,
            media_type="application/xml",
        )
    elif event == "Hangup" and hangup_cause == "USER_BUSY":
        logger.info("📞 Answer webhook: Hangup USER_BUSY - user hung up")
        await log_meeting(agent_id, form_data_dict)
    else:
        logger.info("📞 Answer webhook: Event={} - no Stream XML returned", event)


@app.websocket("/agent/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for Vobiz audio streaming.

    Args:
        websocket: WebSocket connection
        agent_id: Agent ID to use
    """
    # Log immediately when connection attempt is received (before accept)
    logger.info(
        "🔌 WebSocket CONNECTION ATTEMPT: /agent/{} - if you see this, Vobiz connected to this pod",
        agent_id,
    )
    await websocket.accept()
    logger.info(
        "🔌 WebSocket ACCEPTED: agent_id={} - call audio stream started, next: wait for 'start' message",
        agent_id,
    )

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

        # Wait for start event with call metadata (Vobiz may send "connected" first)
        first_message = await websocket.receive_text()
        data = json.loads(first_message)
        logger.info(f"📨 First WebSocket message: {first_message}")
        first_event = data.get("event", "<no event key>")
        logger.info(f"📨 First WebSocket message: event={first_event}, keys={list(data.keys())}")

        if first_event == "connected":
            logger.info("📨 Skipping 'connected' event, waiting for 'start'")
            first_message = await websocket.receive_text()
            data = json.loads(first_message)
            first_event = data.get("event", "<no event key>")
            logger.info(f"📨 Second WebSocket message: event={first_event}, keys={list(data.keys())}")

        if first_event != "start":
            logger.warning(f"⚠️ Expected 'start' event, got: {first_event}. Full message keys: {list(data.keys())}")
            return

        start_info = data.get("start", {})
        call_sid = start_info.get("callSid") or start_info.get("callId", "unknown")
        stream_sid = start_info.get("streamSid") or start_info.get("streamId", "unknown")

        logger.info(f"📞 Call started: call_sid={call_sid}, stream_sid={stream_sid}")
        logger.debug(f"📋 Start info: {start_info}")

        await bot(websocket, stream_sid, call_sid, agent_type, agent_config)

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        await websocket.close(code=1008, reason="Agent config not found")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info(f"🔌 WebSocket closed: call_sid={call_sid}")


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

        await bot(
            websocket,
            stream_sid,
            call_sid,
            agent_type,
            agent_config,
            transcript_callback=send_transcript,
        )

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        await websocket.close(code=1008, reason="Agent config not found")
    except Exception as e:
        logger.error(f"❌ Browser WebSocket error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info(f"🔌 Browser WebSocket closed: call_sid={call_sid}")

# Ubona: hardcoded to Mahavistaar agent; answer URL is https://vobiz.johnaic.com/ubona
UBONA_AGENT_TYPE = "Mahavistaar"
UBONA_STREAM_PATH_ID = "mahavistaar"


def _fetch_mahavistaar_config() -> Optional[dict]:
    """Fetch Mahavistaar agent config from backend (inline, no backend_utils)."""
    backend_url = os.environ.get("VOICERA_BACKEND_URL", "http://localhost:8000")
    api_key = os.environ.get("INTERNAL_API_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    endpoint = f"{backend_url}/api/v1/agents/config/{UBONA_AGENT_TYPE}"
    try:
        response = requests.get(endpoint, headers=headers, timeout=10)
        response.raise_for_status()
        agent_data = response.json()
        agent_config = agent_data.get("agent_config", {})
        if "org_id" in agent_data:
            agent_config["org_id"] = agent_data["org_id"]
        if "agent_type" in agent_data:
            agent_config["agent_type"] = agent_data["agent_type"]
        if "greeting_message" in agent_data:
            agent_config["greeting_message"] = agent_data["greeting_message"]
        return agent_config
    except Exception as e:
        logger.error(f"Failed to fetch Mahavistaar config: {e}")
        return None


@app.api_route("/ubona", methods=["GET", "POST"])
async def ubona_answer(request: Request):
    """Ubona answer webhook - returns XML with WebSocket URL. Hardcoded to Mahavistaar."""
    form_data = dict(await request.form())
    event = form_data.get("Event", "unknown")

    if event == "StartApp":
        await log_meeting(UBONA_STREAM_PATH_ID, form_data)
        ws_prefix = os.environ.get("JOHNAIC_WEBSOCKET_URL", "https://vobiz.johnaic.com")
        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">
        {ws_prefix}/ubona/stream/{UBONA_STREAM_PATH_ID}
    </Stream>
</Response>'''
        return Response(content=xml, media_type="application/xml")
    elif event == "Hangup":
        await log_meeting(UBONA_STREAM_PATH_ID, form_data)

    return Response(status_code=200)


@app.websocket("/ubona/stream/{agent_id}")
async def ubona_stream(websocket: WebSocket, agent_id: str):
    """Ubona WebSocket endpoint for audio streaming. Hardcoded to Mahavistaar."""
    await websocket.accept()
    logger.info(f"🔌 Ubona WS connected: agent={agent_id}")

    call_id = stream_id = None

    try:
        agent_config = _fetch_mahavistaar_config()
        if not agent_config:
            logger.error(f"❌ No config for {UBONA_AGENT_TYPE}")
            return
        agent_type = agent_config.get("agent_type", UBONA_AGENT_TYPE)

        # Handle connected event (optional)
        msg = json.loads(await websocket.receive_text())
        if msg.get("event") == "connected":
            msg = json.loads(await websocket.receive_text())

        if msg.get("event") != "start":
            logger.warning(f"⚠️ Expected 'start', got: {msg.get('event')}")
            return

        # Spec: callId/streamId at top level or under start (Ubona Media Stream v1.0.0)
        start_obj = msg.get("start", {})
        call_id = msg.get("callId") or start_obj.get("callId", "unknown")
        stream_id = msg.get("streamId") or start_obj.get("streamId", "unknown")

        logger.info(f"📞 Ubona call: call_id={call_id}, stream_id={stream_id}")

        await ubona_bot(websocket, stream_id, call_id, agent_type, agent_config)

    except Exception as e:
        logger.error(f"❌ Ubona WS error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info(f"🔌 Ubona WS closed: call_id={call_id}")

# === Telnyx routes ===

class TelnyxOutboundCallRequest(BaseModel):
    """Request model for initiating outbound Telnyx calls."""
    customer_number: str
    agent_id: str
    caller_id: Optional[str] = None


@app.post("/telnyx/outbound/call/")
async def make_telnyx_outbound_call(request: TelnyxOutboundCallRequest):
    """Initiate an outbound call via Telnyx Call Control.

    Mirrors /outbound/call/ for Vobiz but routes through Telnyx.
    """
    try:
        result = await make_outbound_call_telnyx(
            request.customer_number,
            request.agent_id,
            request.caller_id,
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Telnyx outbound call initiated",
                "customer_number": request.customer_number,
                "agent_id": request.agent_id,
                "caller_id": request.caller_id,
                "result": result,
            },
        )
    except ValueError as e:
        logger.error(f"❌ Invalid Telnyx request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Telnyx outbound call failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/telnyx/answer", methods=["GET", "POST"])
async def telnyx_answer_webhook(request: Request):
    """TeXML answer webhook.

    Telnyx hits this URL when the outbound (or inbound) call connects. We
    respond with a <Stream> TeXML instruction pointing at our WebSocket, and
    Telnyx then opens a bidirectional media stream there.
    """
    agent_id = request.query_params.get("agent_id")

    # Telnyx may send form-encoded (TeXML webhooks) or JSON (Call Control
    # webhooks). Read both defensively so we don't crash either way.
    form_data_dict: Dict[str, Any] = {}
    try:
        form_data = await request.form()
        form_data_dict = dict(form_data)
    except Exception:
        form_data_dict = {}

    if not form_data_dict:
        try:
            body = await request.json()
            if isinstance(body, dict):
                form_data_dict = body
        except Exception:
            form_data_dict = {}

    logger.info(
        "📞 Telnyx answer webhook: method={} agent_id={} keys={}",
        request.method,
        agent_id,
        list(form_data_dict.keys()),
    )

    ws_prefix = os.environ.get("JOHNAIC_WEBSOCKET_URL", "")
    if not ws_prefix:
        logger.warning(
            "📞 Telnyx answer webhook: JOHNAIC_WEBSOCKET_URL not set; returning empty TeXML"
        )
    websocket_url = f"{ws_prefix}/telnyx/stream/{agent_id}"
    xml_body = _build_telnyx_stream_xml(websocket_url)

    logger.info(
        "📞 Telnyx answer webhook: returning Stream TeXML | websocket_url={}",
        websocket_url,
    )
    return Response(content=xml_body, media_type="application/xml")


@app.websocket("/telnyx/stream/{agent_id}")
async def telnyx_websocket_endpoint(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for Telnyx Media Streaming.

    Telnyx frames follow the connected → start → media → stop sequence.
    The "start" event carries call_control_id and stream_id that we need
    to forward to the pipeline (so the serializer can hang the call up on
    EndFrame via the Call Control API).
    """
    logger.info(
        "🔌 Telnyx WebSocket CONNECTION ATTEMPT: /telnyx/stream/{}", agent_id
    )
    await websocket.accept()
    logger.info(
        "🔌 Telnyx WebSocket ACCEPTED: agent_id={} - waiting for 'start' event",
        agent_id,
    )

    call_control_id: Optional[str] = None
    stream_id: Optional[str] = None

    try:
        agent_config = await fetch_agent_config_from_backend(agent_id)
        if not agent_config:
            logger.error(f"❌ Failed to fetch agent config from backend: {agent_id}")
            await websocket.close(code=1008, reason="Agent config not found")
            return
        agent_type = agent_config.get("agent_type")
        logger.info(f"📥 Telnyx agent config: {agent_config}")

        # Skip optional "connected" preamble, same pattern as Vobiz.
        first_message = await websocket.receive_text()
        data = json.loads(first_message)
        first_event = data.get("event", "<no event key>")
        logger.info(
            f"📨 Telnyx first WS message: event={first_event}, keys={list(data.keys())}"
        )

        if first_event == "connected":
            logger.info("📨 Telnyx: skipping 'connected' event, waiting for 'start'")
            first_message = await websocket.receive_text()
            data = json.loads(first_message)
            first_event = data.get("event", "<no event key>")
            logger.info(
                f"📨 Telnyx second WS message: event={first_event}, keys={list(data.keys())}"
            )

        if first_event != "start":
            logger.warning(
                f"⚠️ Telnyx: expected 'start' event, got: {first_event}. keys={list(data.keys())}"
            )
            return

        start_info = data.get("start", {})
        # Telnyx places ids under the "start" object, but we also check the top
        # level to be forgiving across protocol variants.
        call_control_id = (
            start_info.get("call_control_id")
            or data.get("call_control_id")
            or "unknown"
        )
        stream_id = (
            start_info.get("stream_id")
            or data.get("stream_id")
            or data.get("streamId")
            or "unknown"
        )

        logger.info(
            f"📞 Telnyx call started: call_control_id={call_control_id}, stream_id={stream_id}"
        )

        await telnyx_bot(websocket, stream_id, call_control_id, agent_type, agent_config)

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        await websocket.close(code=1008, reason="Agent config not found")
    except Exception as e:
        logger.error(f"❌ Telnyx WebSocket error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info(f"🔌 Telnyx WebSocket closed: call_control_id={call_control_id}")


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
