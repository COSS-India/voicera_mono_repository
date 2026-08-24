# Broadcast (Live Translation) — Architecture Diagram

One presenter speaks → many listeners hear a live translation in their chosen
language. Cost scales with **distinct active languages**, not listener count
(N listeners on one language share one translate→TTS stream).

## Component / dataflow

```mermaid
flowchart TB
    subgraph Client["Browser clients"]
        Host["Presenter (host)<br/>mic capture, no playback<br/>broadcast-dialog.tsx"]
        L1["Listener A · Hindi"]
        L2["Listener B · Hindi"]
        L3["Listener C · Tamil"]
        Live["Public page<br/>app/live/[token]<br/>lang picker + playback"]
    end

    subgraph FE["Next.js BFF (voicera_frontend)"]
        BT["POST /api/agents/broadcast-token"]
        PUB["GET /api/public/agents/[token]"]
    end

    subgraph BE["Backend — FastAPI + MongoDB (voicera_backend)"]
        MintTok["POST /agents/{type}/broadcast-token<br/>(JWT) → short-lived HS256 host token"]
        Resolve["POST /agents/broadcast/resolve<br/>(X-API-Key) verify host token"]
        ByToken["GET /agents/public/by-token/{share_token}<br/>(X-API-Key)"]
        PublicR["GET /public/agents/{share_token}<br/>(no auth, secret-stripped)"]
        Cfg["GET /agents/config/id/{agent_id}<br/>(X-API-Key)"]
        DB[("AgentConfig<br/>interaction_mode=translation<br/>source_language, target_languages,<br/>target_voices, public_share_enabled,<br/>share_token")]
    end

    subgraph VS["voice_2_voice_server — FastAPI + Pipecat (process-local _ROOMS)"]
        PubWS["WS /translate/publish/{agent_id}<br/>?token=host_token"]
        LisWS["WS /translate/listen/{share_token}?lang="]
        subgraph Room["TranslationRoom (per agent_id)"]
            Pipe["Publisher pipeline<br/>transport.input → SileroVAD → STT<br/>→ TranscriptCollector<br/>(audio_out_enabled=False)"]
            WHi["LangWorker[Hindi]<br/>translate → TTS → FanOutSink"]
            WTa["LangWorker[Tamil]<br/>translate → TTS → FanOutSink"]
        end
        OAI["OpenAI chat.completions<br/>translate-only prompt + agent guidance"]
    end

    Host -->|1 mint| BT --> MintTok --> DB
    Host -->|2 WS + token| PubWS
    PubWS -->|verify token| Resolve --> DB
    PubWS -->|fetch config| Cfg --> DB

    Live -->|load| PUB --> PublicR --> DB
    L1 & L2 & L3 -->|WS lang| LisWS
    LisWS -->|resolve token| ByToken --> DB
    LisWS -->|fetch config| Cfg

    PubWS --> Pipe
    Pipe -->|final transcript<br/>on_source_text| WHi
    Pipe -->|enqueue per worker<br/>bounded, drop-oldest| WTa
    WHi -->|chat.completions| OAI
    WTa -->|chat.completions| OAI
    LisWS --> Room
    WHi -->|fan-out playAudio| L1 & L2
    WTa -->|fan-out playAudio| L3
```

## Sequence (happy path)

```mermaid
sequenceDiagram
    participant O as Owner/Host
    participant FE as Next.js BFF
    participant BE as Backend
    participant VS as voice_server
    participant OAI as OpenAI
    participant A as Listener A (Hindi)
    participant C as Listener C (Tamil)

    O->>BE: create translation agent (source=EN, targets=[HI,TA]), enable share
    BE-->>O: share_token
    O->>FE: POST /api/agents/broadcast-token
    FE->>BE: mint host token (JWT)
    BE-->>O: host token
    O->>VS: WS /translate/publish/{agent_id}?token=host
    VS->>BE: broadcast/resolve + config (X-API-Key)
    VS->>VS: claim publisher, build VAD+STT pipeline

    A->>VS: WS /translate/listen/{token}?lang=Hindi
    VS->>BE: by-token + config
    VS->>VS: start LangWorker[Hindi], subscribe A
    C->>VS: WS listen ?lang=Tamil
    VS->>VS: start LangWorker[Tamil], subscribe C

    O->>VS: speaks "Good morning"
    VS->>VS: STT final → on_source_text → enqueue(HI), enqueue(TA)
    VS->>OAI: translate EN→HI
    OAI-->>VS: "सुप्रभात"
    VS->>VS: TTS → FanOutSink
    VS-->>A: playAudio (HI)
    VS->>OAI: translate EN→TA
    OAI-->>VS: TTS → fan-out
    VS-->>C: playAudio (TA)

    C->>VS: disconnect → subscribers[TA] empty → stop LangWorker[Tamil]
    O->>VS: end broadcast → end_broadcast(): notify+close listeners, stop all workers, tear down room
```

## Key facts

- **Auth split:** public `share_token` = listen only. Publishing needs a
  short-lived HS256 **host token** (JWT-authed mint). `agent_id` never exposed
  publicly (would be a de-facto key on the unauthenticated WS routes).
- **Mute-bot is structural:** publisher pipeline has no LLM/TTS/output
  (`audio_out_enabled=False`), so the presenter inherently hears nothing back.
- **Lazy workers:** `LangWorker` starts on first listener of a language, stops
  when its last listener leaves. Capacity = 1 slot presenter + 1 per active
  language (not per listener).
- **Backpressure:** per-worker bounded `asyncio.Queue` (`MAX_SEGMENT_BACKLOG=50`),
  drop-oldest so a slow language stays near live speech, never back-pressures others.
- **Multi-worker caveat:** `_ROOMS` is process-local. With
  `VOICE_SERVER_NUM_WORKERS > 1`, presenter and listeners must land on the same
  worker — single-worker deploy or sticky routing on `/translate/*`.
```
