---
name: Bug report
about: Report something that isn't working as expected
title: "[Bug]: "
labels: bug
assignees: ''
---

## Describe the bug

A clear, concise description of what went wrong.

## Steps to reproduce

1. Go to '...'
2. Click on '...'
3. See error

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened (error message, wrong UI state, silent failure, etc.).

## Environment

- **VoicERA commit / version:** (e.g. `main` @ `abc1234`, or Docker image tag)
- **Deployment:** Docker Compose / local dev / other
- **OS:** (e.g. macOS 15, Ubuntu 22.04)
- **Browser** (if dashboard-related): (e.g. Chrome 125)
- **Telephony provider** (if call-related): None (browser test) / Plivo / Vobiz
- **AI providers used** (LLM / STT / TTS):

## Logs / screenshots

Paste relevant logs (`docker compose logs backend voice_server`) or attach
screenshots. **Redact API keys, tokens, and phone numbers.**

## Additional context

Anything else that might help us reproduce or fix the issue.
