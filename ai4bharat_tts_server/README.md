# AI4Bharat TTS server

Optional **on-premises Indic text-to-speech** (Parler) over WebSocket. Required only when agents use `indic-parler-tts`.

## When you need it

| Scenario | Need this server? |
|----------|-------------------|
| Cloud TTS only | No |
| Local Indic TTS / Bhili | Yes (+ GPU recommended) |

## Run

```bash
cd ai4bharat_tts_server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

Default port **8002**. Monorepo: `make start-voice-only-services`.

## Dependencies

- Voice server: `AI4BHARAT_TTS_URL` or `INDIC_TTS_SERVER_URL` when agents use local TTS

## Protocol

WebSocket JSON per utterance (`prompt`, `description`, `language`); server streams float32 PCM at 44.1 kHz.

Full detail: [docs/services/ai4bharat-tts.md](../docs/services/ai4bharat-tts.md)

## GPU / VRAM

- **Production:** NVIDIA GPU expected for Parler inference.
- **Recommended:** **RTX 4000 series and newer** — e.g. RTX 5090–5060, RTX 4090–4060 (incl. 4060 Ti 16GB), RTX 6000/5000/4500/4000 Ada, L40S, L40, H100, H200. Full list: [docs/services/ai4bharat-tts.md#gpu-vram](../docs/services/ai4bharat-tts.md#gpu-vram).
- **Pinned GB figures:** Deferred — VRAM depends on model weights and concurrent utterances; size on staging with `nvidia-smi`.

## Documentation

- [AI4Bharat TTS (MkDocs)](../docs/services/ai4bharat-tts.md)
- [Source brief A5](../docs/source-briefs/A5-ai4bharat-servers.md)