# MinIO Storage Setup

Local S3-compatible storage for recordings and transcripts.

## Quick Start

### First Time Setup (only once)
```bash
docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -v minio_data:/data \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

### Daily Commands

| Action | Command |
|--------|---------|
| Start | `docker start minio` |
| Stop | `docker stop minio` |
| Status | `docker ps` |
| Logs | `docker logs minio` |

## Access Files

### Web Console

Open `http://localhost:9001`
- Username: `minioadmin`
- Password: `minioadmin`

### Python
```python
from storage.minio_client import MinIOStorage

storage = MinIOStorage()

# List files
storage.list_recordings()
storage.list_transcripts()

# Read transcript
text = storage.get_transcript("call_123")

# Download recording
storage.download_recording("call_123", "./call_123.wav")
```

## Buckets

- `recordings/` - WAV audio files
- `transcripts/` - TXT transcript files

## Environment Variables
```env
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```