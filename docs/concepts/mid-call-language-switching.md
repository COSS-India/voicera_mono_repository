---
description: Runtime language switching for AI4Bharat STT and TTS services during an active voice call without restarting the session.
---

# Mid-Call Language Switching

**Scope:** AI4Bharat STT and TTS — 23 Indic languages

## Overview

Mid-call language switching allows users to change the conversation language during an active voice call without restarting the session.

The implementation introduces runtime language state management across the voice pipeline, enabling both STT and TTS services to switch languages dynamically while preserving the existing call context.

## How It Works

A dedicated `switch_language` tool is exposed to the LLM with a predefined set of supported language codes.

When a user requests a language change:

1. The LLM detects the language switch intent and invokes the `switch_language` tool.
2. The requested language is normalized to a canonical language code.
3. STT and TTS language states are updated at runtime.
4. Subsequent transcription and synthesis requests use the updated language.

This approach enables seamless language transitions without rebuilding the pipeline, reconnecting services, or restarting the call.

## Flow

```text
User Speech → AI4Bharat STT (current language)
           → LLM (detects switch intent)
           → switch_language("ta") tool call
           → stt.set_language("ta") + tts.set_language("ta")
           → Tool returns { "success": true, "language": "ta" }
           → LLM continues in new language
           → AI4Bharat TTS (switched language)
           → Audio Output
```

## Supported Languages

The implementation supports 23 Indic languages including:

- Hindi, Tamil, Telugu, Kannada, Malayalam
- Bengali, Marathi, Gujarati, Punjabi, Urdu
- And others

## Supported Scope

Runtime language switching is supported for AI4Bharat STT and TTS services, covering 23 Indic languages.
