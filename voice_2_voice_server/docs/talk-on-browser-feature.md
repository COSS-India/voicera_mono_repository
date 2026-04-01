# Talk on Browser Feature

This document explains how the **Talk on Browser** feature is implemented in the Voicera frontend and how audio flows between browser and agent runtime.

## 1) What this feature does

Users can test an agent directly from the browser without receiving a phone call.

The flow is:
1. User clicks **Test on Browser** on an agent card.
2. A popup opens.
3. Browser microphone audio is captured.
4. Audio is streamed to the existing voice server WebSocket endpoint.
5. Agent audio is streamed back and played live.
6. Orb animation reacts to live input/output audio levels.

## 2) Code changes summary

### Frontend UI entry points

- `voicera_frontend/components/assistants/agent-card.tsx`
  - Added new callback prop: `onTestBrowser`.
  - Added new button label: **Test on Browser**.

- `voicera_frontend/app/(dashboard)/assistants/page.tsx`
  - Imported and rendered `TestBrowserDialog`.
  - Added state: `isTestBrowserDialogOpen`.
  - Added handler: `handleTestBrowser(agent)`.

### Browser test dialog

- `voicera_frontend/components/assistants/test-browser-dialog.tsx`
  - Implements the full browser voice session lifecycle:
    - open/close handling
    - microphone capture
    - PCM conversion and downsampling
    - WebSocket send/receive
    - playback scheduling
    - mute and teardown
  - Integrates ElevenLabs Orb and feeds it live manual volume refs.

### Orb component (official ElevenLabs UI component)

- `voicera_frontend/components/ui/orb.tsx`
  - Added from ElevenLabs registry (`https://ui.elevenlabs.io/r/orb.json`).
  - Uses React Three Fiber + Three.js shader-based animation.

### Frontend dependencies

- `voicera_frontend/package.json` (+ lockfile)
  - Added:
    - `@react-three/fiber`
    - `@react-three/drei`
    - `three`
    - `@types/three`

## 3) Runtime architecture

The feature reuses the existing voice server transport protocol (same one used for telephony stream handling):

1. **Client capture**
   - `navigator.mediaDevices.getUserMedia({ audio: ... })` starts mic capture.

2. **Audio graph**
   - `AudioContext` + `createMediaStreamSource` + `ScriptProcessorNode` reads PCM frames.

3. **Input preprocessing**
   - Input float PCM is downsampled to 16kHz (`downsampleTo16k`).
   - Converted to signed 16-bit PCM.
   - Base64 encoded.

4. **WebSocket uplink**
   - Client opens WS to:
     - `NEXT_PUBLIC_JOHNAIC_WEBSOCKET_URL` if set, else
     - derived from `NEXT_PUBLIC_JOHNAIC_SERVER_URL`, else
     - `ws://localhost:7860`
   - Final path: `/agent/{agent_id}`.

5. **Session start frame**
   - Client sends:
   - `{"event":"start","start":{"callSid":"...","streamSid":"..."}}`

6. **Media frames uplink**
   - Client sends:
   - `{"event":"media","media":{"contentType":"audio/x-l16","sampleRate":16000,"payload":"..."}}`

7. **Agent audio downlink**
   - Server sends `playAudio` frames with payload.
   - Client decodes payload to PCM and schedules playback with small buffer lead time.

8. **Playback smoothing**
   - `playbackTimeRef` is used so chunks are queued continuously and avoid jitter/dropouts.

## 4) Orb voice reactivity

The Orb is used in manual volume mode.

Implementation details:
1. Input RMS is computed from mic frames before send.
2. Output RMS is computed from received assistant audio before playback.
3. Normalized values are written to refs:
   - `inputVolumeRef`
   - `outputVolumeRef`
4. Orb receives:
   - `volumeMode="manual"`
   - `inputVolumeRef={inputVolumeRef}`
   - `outputVolumeRef={outputVolumeRef}`
5. UI also sets Orb state:
   - `listening` when connected and no active assistant output burst
   - `talking` when output energy passes threshold
   - `null` when disconnected

## 5) Session lifecycle and cleanup

Cleanup is critical to avoid dangling mic/WS resources.

On close/end/unmount:
1. WebSocket handlers removed and socket closed.
2. Processor/source nodes disconnected.
3. Mic tracks stopped.
4. AudioContext closed.
5. Refs/state reset (`isConnected`, volumes, orb state, etc.).

## 6) Why no backend requirement changes were needed

- This feature and Orb are frontend-side additions.
- The existing backend WebSocket pipeline was already compatible.
- No Python package additions were required.
- Only frontend npm dependencies were added.

## 7) Environment/config notes

Recommended env vars in frontend:
1. `NEXT_PUBLIC_JOHNAIC_WEBSOCKET_URL=wss://<voice-server-host>`
2. Fallback: `NEXT_PUBLIC_JOHNAIC_SERVER_URL` (converted to ws/wss)
3. Final fallback for local dev: `ws://localhost:7860`

## 8) Troubleshooting

1. Orb not visible
- Check that `three`, `@react-three/fiber`, and `@react-three/drei` are installed.
- Ensure popup container has explicit width/height for Orb.

2. Connection fails
- Verify WS URL and TLS (`wss://` for HTTPS sites).
- Check CORS/proxy/network path to voice server.

3. No mic capture
- Confirm browser mic permission granted.
- Check OS/browser input device settings.

4. Choppy playback
- Verify stable network.
- Confirm sample rate and payload format are consistent (`audio/x-l16`, 16kHz).

## 9) Main files to inspect

1. `voicera_frontend/components/assistants/agent-card.tsx`
2. `voicera_frontend/app/(dashboard)/assistants/page.tsx`
3. `voicera_frontend/components/assistants/test-browser-dialog.tsx`
4. `voicera_frontend/components/ui/orb.tsx`
