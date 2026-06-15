# Changelog

All notable changes to VoicERA are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Open-source community files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, GitHub issue/PR templates, and `AGENTS.md` for coding agents.
- README overhaul with browser-first onboarding, provider matrix, and accurate
  quick-start instructions.

### Planned (not yet implemented)

These items appear in the original product vision but **do not exist in the
codebase today**. They are tracked here as roadmap gaps, not as shipped features:

- Unified `make demo`, `make setup`, `make doctor`, `make test`, and `make lint`
  targets.
- Example-agent library and in-dashboard template selector.
- Skill / tool-calling framework and cross-call memory.
- LLM providers: Gemini, OpenRouter. STT/TTS: AssemblyAI.

## [0.1.0] - 2026-06-04

Initial open-source snapshot of the working VoicERA stack.

### Added

- **Browser testing** — talk to agents from the dashboard via
  `WS /browser/agent/{agent_id}` without telephony credentials.
- **Telephony** — Plivo and Vobiz support with inbound/outbound calling.
- **Vobiz native call recording** — uses Vobiz recording APIs; Pipecat recording
  retained for other providers ([#21](https://github.com/COSS-India/voicera_mono_repository/pull/21)).
- **Call latency telemetry** — metrics captured in the voice pipeline and
  surfaced in the dashboard with CSV export
  ([#19](https://github.com/COSS-India/voicera_mono_repository/pull/19)).
- **STT/TTS language selection** — per-agent language configuration, including
  Bhili model support for AI4Bharat servers
  ([#20](https://github.com/COSS-India/voicera_mono_repository/pull/20)).
- **Knowledge base (RAG)** — document upload, chunking, embedding, and
  retrieval-augmented LLM responses at call time.
- **Multi-provider AI stack** — LLM (OpenAI, Anthropic, Grok, Qwen/vLLM,
  Kenpath), STT (Deepgram, Google, OpenAI, ElevenLabs, Sarvam, AI4Bharat,
  Bhashini), TTS (ElevenLabs, Cartesia, OpenAI, Google, Deepgram, Sarvam,
  AI4Bharat, Bhashini).
- **Docker Compose deployment** — full stack with MongoDB, MinIO, nginx, and
  optional AI4Bharat STT/TTS services.
- **Dashboard** — agent configuration, meetings history, call latency views,
  integrations management, and multi-tenant org scoping.

### Changed

- Corrected README license statement to MIT (was incorrectly marked proprietary).
- Barge-in / interruption processing and greeting filters in the voice pipeline.
- Bhashini STT service metrics processing.

[Unreleased]: https://github.com/COSS-India/voicera_mono_repository/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/COSS-India/voicera_mono_repository/releases/tag/v0.1.0
