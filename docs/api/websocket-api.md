# WebSocket API Documentation

Real-time communication protocol for VoiceERA Voice Server.

## Connection

```javascript
// Client-side (JavaScript)
const socket = new WebSocket('ws://localhost:7860/voice');

// Or with authentication
const token = localStorage.getItem('token');
const socket = new WebSocket(
  `ws://localhost:7860/voice?token=${token}`
);
```

## Message Format

All messages are JSON:

```json
{
  "type": "message_type",
  "session_id": "session-uuid",
  "timestamp": 1674003600000,
  "payload": {}
}
```

---

## Authentication

### Client: Send Auth Message

```json
{
  "type": "auth",
  "token": "jwt-token-here",
  "agent_id": "agent-uuid",
  "campaign_id": "campaign-uuid"
}
```

### Server: Auth Response

```json
{
  "type": "auth_response",
  "status": "success",
  "session_id": "session-uuid",
  "message": "Authentication successful"
}
```

Or on failure:

```json
{
  "type": "auth_response",
  "status": "error",
  "message": "Invalid token"
}
```

---

## Call Control

### Client: Start Call

After authentication, server automatically signals readiness:

```json
{
  "type": "ready",
  "session_id": "session-uuid",
  "message": "Ready to receive audio"
}
```

### Client: Send Audio

```json
{
  "type": "audio",
  "session_id": "session-uuid",
  "sequence": 1,
  "format": "pcm_16k",
  "data": "base64-encoded-audio-data"
}
```

**Audio Format Details:**
- Format: PCM 16-bit signed
- Sample Rate: 16kHz (mono)
- Duration: 100ms chunks
- Encoding: base64 for JSON

### Server: Send Response Audio

```json
{
  "type": "audio",
  "session_id": "session-uuid",
  "sequence": 1,
  "format": "pcm_16k",
  "data": "base64-encoded-response-audio"
}
```

### Client: Control Messages

```json
{
  "type": "control",
  "action": "pause|resume|end",
  "session_id": "session-uuid"
}
```

### Server: Status Updates

```json
{
  "type": "status",
  "status": "transcribing|processing|synthesizing",
  "session_id": "session-uuid",
  "message": "Processing your request..."
}
```

---

## Metadata & Events

### Client: Send Metadata

```json
{
  "type": "metadata",
  "session_id": "session-uuid",
  "phone_number": "+1234567890",
  "caller_id": "caller-name",
  "agent_config": {
    "language": "en",
    "emotions_enabled": true
  }
}
```

### Server: Transcript Update

Real-time transcript as user speaks:

```json
{
  "type": "transcript",
  "session_id": "session-uuid",
  "partial": "Hello, I'd like to",
  "complete": false,
  "confidence": 0.95
}
```

Final transcript:

```json
{
  "type": "transcript",
  "session_id": "session-uuid",
  "partial": null,
  "complete": "Hello, I'd like to know about your services",
  "confidence": 0.98,
  "final": true
}
```

### Server: Agent Response

```json
{
  "type": "response",
  "session_id": "session-uuid",
  "text": "Sure! I'd be happy to help. We offer...",
  "agent_id": "agent-uuid",
  "emotion": "helpful",
  "sentiment": "positive"
}
```

### Server: Error Messages

```json
{
  "type": "error",
  "session_id": "session-uuid",
  "error_code": "STT_FAILED",
  "message": "Failed to transcribe audio",
  "details": "Please try again"
}
```

---

## Session Lifecycle

### 1. Connection Established

```
Client connects → Server accepts → WebSocket open
```

### 2. Authentication

```
Client sends auth message
      ↓
Server validates JWT
      ↓
Server sends auth_response (success)
      ↓
Server sends ready message
```

### 3. Call Processing

```
Client sends audio chunk
      ↓
Server transcribes (STT)
      ↓
Server generates response (LLM)
      ↓
Server sends response audio (TTS)
      ↓
Client plays audio
      ↓
(Loop continues)
```

### 4. Call Termination

```
Client sends control message (action: "end")
      ↓
Server stops audio stream
      ↓
Server saves recording
      ↓
Server sends final transcript & analytics
      ↓
Server closes connection
```

---

## Example: JavaScript Client

```javascript
class VoiceClient {
  constructor(serverUrl, token) {
    this.socket = null;
    this.serverUrl = serverUrl;
    this.token = token;
    this.sessionId = null;
    this.mediaRecorder = null;
  }

  async connect(agentId) {
    return new Promise((resolve, reject) => {
      this.socket = new WebSocket(this.serverUrl);

      this.socket.onopen = () => {
        // Send authentication
        this.socket.send(JSON.stringify({
          type: 'auth',
          token: this.token,
          agent_id: agentId
        }));
      };

      this.socket.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.type === 'auth_response') {
          if (message.status === 'success') {
            this.sessionId = message.session_id;
            resolve(this.sessionId);
          } else {
            reject(new Error(message.message));
          }
        }

        if (message.type === 'ready') {
          this.startAudioCapture();
        }

        if (message.type === 'audio') {
          this.playAudio(message.data);
        }

        if (message.type === 'transcript') {
          console.log('Transcript:', message.partial || message.complete);
        }

        if (message.type === 'response') {
          console.log('Agent:', message.text);
        }

        if (message.type === 'error') {
          console.error('Error:', message.message);
        }
      };

      this.socket.onerror = (error) => {
        reject(error);
      };
    });
  }

  async startAudioCapture() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true
    });

    this.mediaRecorder = new MediaRecorder(stream);

    this.mediaRecorder.ondataavailable = (event) => {
      const audioBlob = event.data;
      const reader = new FileReader();

      reader.onload = () => {
        const arrayBuffer = reader.result;
        const base64 = btoa(
          String.fromCharCode(...new Uint8Array(arrayBuffer))
        );

        this.socket.send(JSON.stringify({
          type: 'audio',
          session_id: this.sessionId,
          data: base64,
          format: 'pcm_16k'
        }));
      };

      reader.readAsArrayBuffer(audioBlob);
    };

    this.mediaRecorder.start(100); // Send chunks every 100ms
  }

  playAudio(base64Data) {
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);

    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = audioContext.createBuffer(1, bytes.length, 16000);
    const channelData = audioBuffer.getChannelData(0);

    for (let i = 0; i < bytes.length; i++) {
      channelData[i] = (bytes[i] - 128) / 128;
    }

    const audioSource = audioContext.createBufferSource();
    audioSource.buffer = audioBuffer;
    audioSource.connect(audioContext.destination);
    audioSource.start(0);
  }

  end() {
    this.mediaRecorder.stop();

    this.socket.send(JSON.stringify({
      type: 'control',
      action: 'end',
      session_id: this.sessionId
    }));

    this.socket.close();
  }

  pause() {
    this.mediaRecorder.pause();
    this.socket.send(JSON.stringify({
      type: 'control',
      action: 'pause',
      session_id: this.sessionId
    }));
  }

  resume() {
    this.mediaRecorder.resume();
    this.socket.send(JSON.stringify({
      type: 'control',
      action: 'resume',
      session_id: this.sessionId
    }));
  }
}

// Usage
const client = new VoiceClient(
  'ws://localhost:7860/voice',
  'jwt-token'
);

client.connect('agent-uuid').then((sessionId) => {
  console.log('Connected:', sessionId);
  // Client will automatically start recording
});
```

---

## Example: Python Client

```python
import json
import asyncio
import aiohttp
import pyaudio
import numpy as np
from base64 import b64encode, b64decode

class VoiceClient:
    def __init__(self, server_url, token):
        self.server_url = server_url
        self.token = token
        self.session_id = None
        self.websocket = None

    async def connect(self, agent_id):
        """Establish WebSocket connection"""
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.server_url) as ws:
                self.websocket = ws

                # Send authentication
                await ws.send_json({
                    'type': 'auth',
                    'token': self.token,
                    'agent_id': agent_id
                })

                # Handle authentication response
                async for message in ws:
                    data = json.loads(message.data)

                    if data['type'] == 'auth_response':
                        if data['status'] == 'success':
                            self.session_id = data['session_id']
                            await self.start_audio_capture()
                        break

    async def start_audio_capture(self):
        """Capture microphone input and send to server"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )

        try:
            while True:
                data = stream.read(1024)
                base64_data = b64encode(data).decode()

                await self.websocket.send_json({
                    'type': 'audio',
                    'session_id': self.session_id,
                    'data': base64_data,
                    'format': 'pcm_16k'
                })

                # Process server responses
                async for message in self.websocket:
                    server_data = json.loads(message.data)

                    if server_data['type'] == 'transcript':
                        if server_data.get('final'):
                            print(f"Transcript: {server_data['complete']}")

                    elif server_data['type'] == 'response':
                        print(f"Agent: {server_data['text']}")

                    elif server_data['type'] == 'audio':
                        await self.play_audio(server_data['data'])

                    break

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def play_audio(self, base64_data):
        """Decode and play audio response"""
        audio_bytes = b64decode(base64_data)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        # Play audio (simplified)
        # In production, use sounddevice or similar
        print(f"Playing {len(audio_array)} samples")

    async def end(self):
        """End the call"""
        await self.websocket.send_json({
            'type': 'control',
            'action': 'end',
            'session_id': self.session_id
        })
        await self.websocket.close()

# Usage
async def main():
    client = VoiceClient('ws://localhost:7860/voice', 'jwt-token')
    await client.connect('agent-uuid')

asyncio.run(main())
```

---

## Error Handling

Common WebSocket errors:

| Code | Message | Action |
|------|---------|--------|
| 1000 | Normal closure | Connection ended normally |
| 1002 | Protocol error | Invalid message format |
| 1003 | Unsupported data | Invalid message type |
| 1008 | Policy violation | Invalid token |
| 1011 | Server error | Internal server error |

---

## Best Practices

1. **Audio Quality:** Send 16kHz mono PCM for best compatibility
2. **Chunk Size:** 100ms chunks (1600 samples at 16kHz)
3. **Error Handling:** Implement retry logic for network failures
4. **Resource Cleanup:** Always close WebSocket and stop audio capture
5. **Token Refresh:** Refresh JWT before it expires
6. **Logging:** Log all errors for debugging

---

## Next Steps

- **[REST API](rest-api.md)** - HTTP endpoints
- **[Voice Server](../services/voice-server.md)** - Server documentation
- **[Quick Start](../getting-started/quickstart.md)** - Get started
