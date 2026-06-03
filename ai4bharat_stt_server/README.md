# AI4Bharat STT server

Optional **on-premises Indic speech-to-text** (NeMo). Required only when agents use the `indic-conformer-stt` provider.

## When you need it

| Scenario | Need this server? |
|----------|-------------------|
| Cloud STT only (Deepgram, Bhashini, etc.) | No |
| Local Indic STT / Bhili (`bhb`) | Yes (+ GPU recommended) |

## Run

```bash
cd ai4bharat_stt_server
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python server.py --port 8001
```

Docker / monorepo: optional service on port **8001**; `make start-voice-only-services` for local dev.

## Dependencies

- Other services: none for the STT process itself
- Voice server must set `AI4BHARAT_STT_URL` or `INDIC_STT_SERVER_URL` when agents use local STT

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Health |
| `POST /transcribe` | Indic Conformer |
| `POST /transcribe/bhili` | Bhili (agent language `bhb`) |

Full detail: [docs/services/ai4bharat-stt.md](../docs/services/ai4bharat-stt.md)

## Configuration

Set `INDIC_NEMO_PATH`, optional `BHILI_NEMO_PATH`, `BHILI_ENABLE`, `HF_TOKEN`, `PORT` — see `.env.example`.

## GPU / VRAM

- **Production:** NVIDIA GPU strongly recommended (`cuda:0` when available).
- **Development:** CPU mode is supported but slow.
- **Pinned GB figures:** Not in repo yet — VRAM depends on your NeMo checkpoint files and batch settings. Size with your hosting partner or a staging `nvidia-smi` load test. See [docs/services/ai4bharat-stt.md](../docs/services/ai4bharat-stt.md#gpu-vram).

## Documentation

- [AI4Bharat STT (MkDocs)](../docs/services/ai4bharat-stt.md)
- [Source brief A5](../docs/source-briefs/A5-ai4bharat-servers.md)