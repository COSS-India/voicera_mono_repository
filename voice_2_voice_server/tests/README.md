# Voice server tests

Manual, run-against-a-live-service checks. Nothing here is part of a CI suite yet.

## Orpheus TTS

Two scripts, deliberately split by what they need installed.

| Script | Needs | Checks |
|---|---|---|
| `orpheus_protocol_check.py` | `aiohttp`, `numpy` | The WebSocket wire contract: frame format, int16 PCM, styles, voice/language resolution, the error path. No Pipecat. |
| `orpheus_tts_smoke.py` | the full voice-server venv | The real `OrpheusTTSService`, end to end: the Pipecat frame sequence and per-frame sample rate. |

Start the server first (`make start-orpheus-tts && make wait-orpheus-tts` from the
repo root), then:

```bash
cd voice_2_voice_server
export ORPHEUS_TTS_SERVER_URL=ws://localhost:8004

python tests/orpheus_protocol_check.py
python tests/orpheus_tts_smoke.py
python tests/orpheus_tts_smoke.py --voice Anitha --language ta --style NEWS
python tests/orpheus_tts_smoke.py --fragment          # the live-call text cadence
python tests/orpheus_tts_smoke.py --out /tmp/o.pcm    # then listen to it
```

The protocol check runs without a host-side `aiohttp` by borrowing the Orpheus
container's own Python:

```bash
docker cp tests/orpheus_protocol_check.py voicera_orpheus_tts:/tmp/c.py
docker exec voicera_orpheus_tts python /tmp/c.py
```

**Listen to the audio at least once.** An int16/float32 mix-up in a TTS adapter
produces confident-sounding white noise rather than an error, and no automated
check downstream of the frame would catch it.

## AI4Bharat Parler TTS / Conformer STT

See `ai4bharat_tts_server/tests/` and `ai4bharat_stt_server/`.
