# VI Multi-Agent Routing

One VI Streaming Object serves all VoicERA agents. Inbound calls are routed by **DNI** (the VI number called) to whichever agent has that number attached on the **Numbers** page.

## One-time VI portal setup

Configure **once** in the VI CPaaS DIY Flow Streaming Object:

| Setting | Value |
|---------|--------|
| WebSocket URL | `wss://<your-host>/vi/stream` |
| Streaming mode | Bidirectional, Foreground |
| Custom parameters | **None** (remove any fixed `agent_id`) |
| Post-stream callback | Optional — not required for routing |
| Domain whitelist | Your public host (must not be a `.ai` domain) |

Do **not** use per-agent URLs like `/vi/agent/mahavistaar` in the VI portal.

## VoicERA workflow

1. Create an agent with telephony provider **Vodafone Idea (VI)**.
2. On **Numbers**, add a VI DNI and attach it to that agent.
3. Inbound calls to that DNI → VoicERA loads the linked agent automatically.
4. Outbound test calls / batches use the VI OBD API (separate from the WSS URL above).

## Routing order (voice server)

When a VI WebSocket `start` event arrives:

1. URL path `/vi/agent/{id}` (legacy)
2. `custom_parameters.agent_id` (optional override)
3. **`start.dni`** → agent with matching `phone_number`
4. `VI_DEFAULT_AGENT_ID` env fallback (dev only)

## Environment

| Variable | Purpose |
|----------|---------|
| `JOHNAIC_WEBSOCKET_URL` | Base URL; VI connects to `{prefix}/vi/stream` |
| `VI_DEFAULT_AGENT_ID` | Fallback agent if DNI is not attached |

OBD outbound uses `VI_OBD_*`, `VI_DNI`, and `VI_FLOW_ID` — see `voice_2_voice_server/.env.example`.
