# Installation Guide

This guide will help you install and set up VoiceERA on your system.

## System Requirements

### Minimum Requirements
- **OS:** Linux, macOS, or Windows (WSL2)
- **RAM:** 8GB
- **Disk Space:** 50GB (including Docker images)
- **Docker:** 20.10+
- **Docker Compose:** 1.29+

### Recommended Requirements
- **OS:** Linux (Ubuntu 20.04 LTS or newer)
- **RAM:** 16GB+
- **CPU Cores:** 4+
- **GPU:** NVIDIA GPU with CUDA support (for local AI4Bharat services)
- **SSD:** 100GB+ NVMe SSD

## Install Prerequisites

### Docker & Docker Compose

=== "Ubuntu/Debian"
    ```bash
    # Install Docker
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose
    
    # Add your user to docker group (optional, avoids sudo)
    sudo usermod -aG docker $USER
    newgrp docker
    ```

=== "macOS"
    ```bash
    # Using Homebrew
    brew install docker docker-compose
    
    # Or download Docker Desktop from docker.com
    ```

=== "Windows"
    1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
    2. Install with WSL2 backend
    3. Enable WSL2 in Windows Features
    4. Restart your computer

### Node.js (Optional - for local frontend development)

```bash
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18

# Or using package manager
# Ubuntu/Debian
sudo apt-get install nodejs npm

# macOS
brew install node

# Windows
# Download from https://nodejs.org/
```

### Python 3.10+ (Optional - for local development)

```bash
# Ubuntu/Debian
sudo apt-get install python3.10 python3-pip

# macOS
brew install python@3.10

# Windows
# Download from https://python.org/
```

### Make (Optional - for using Makefile)

=== "Ubuntu/Debian"
    ```bash
    sudo apt-get install make
    ```

=== "macOS"
    ```bash
    # Included with Xcode Command Line Tools
    xcode-select --install
    ```

=== "Windows"
    ```bash
    # Download from https://gnuwin32.sourceforge.net/packages/make.htm
    # Or use WSL2 with Linux installation
    ```

## Clone the Repository

```bash
# Clone the repository
git clone https://github.com/voicera/voicera.git
cd voicera_mono_repository

# Verify structure
ls -la  # Check for docker-compose.yml, Makefile, etc.
```

## Environment Setup

### 1. Backend Configuration

```bash
# Copy example environment file
cp voicera_backend/env.example voicera_backend/.env

# Edit with your configuration
nano voicera_backend/.env  # or use your preferred editor
```

Key environment variables to configure:

```env
# Database
MONGODB_HOST=mongodb
MONGODB_PORT=27017
MONGODB_USER=admin
MONGODB_PASSWORD=admin123

# Storage
MINIO_HOST=minio
MINIO_PORT=9000
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# JWT
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
```

### 2. Voice Server Configuration

```bash
# Copy example environment file
cp voice_2_voice_server/.env.example voice_2_voice_server/.env

# Edit with your API keys
nano voice_2_voice_server/.env
```

Required API keys:

```env
# Vobiz Telephony
VOBIZ_API_BASE=https://api.vobiz.ai/api/v1
VOBIZ_AUTH_ID=your_auth_id
VOBIZ_AUTH_PASSWORD=your_auth_password

# LLM Provider (choose one)
OPENAI_API_KEY=sk-...  # For OpenAI

# STT Provider (choose one)
DEEPGRAM_API_KEY=...   # For Deepgram

# TTS Provider (choose one)
CARTESIA_API_KEY=...   # For Cartesia
```

### Optional: Ngrok (Expose Local Voice Server)

If you need to expose your local `voice_2_voice_server` to the public internet (for testing webhooks, external integrations or remote callbacks), you can use Ngrok to create a public URL that tunnels to your local port.

```bash
# Ngrok setup:

# Install pyngrok (optional helper) or the ngrok CLI
# Python helper (optional):
pip install pyngrok

# Or download the ngrok CLI from https://ngrok.com and install it

# Authenticate ngrok with your token (get it from https://ngrok.com)
ngrok config add-authtoken <YOUR_AUTH_TOKEN>

# Start your local server and forward the port (example uses 7860)
ngrok http 7860

# Ngrok will print a public URL like:
# https://abcd-12-34-56-78.ngrok-free.app
```

Copy the generated public URL and add it to your `voice_2_voice_server/.env` as follows:

```env
# Johnaic / external webhook (example)
JOHNAIC_SERVER_URL="https://abcd-12-34-56-78.ngrok-free.app"
JOHNAIC_WEBSOCKET_URL="ws://abcd-12-34-56-78.ngrok-free.app"
```

If ngrok provides a secure websocket endpoint use `wss://` for `JOHNAIC_WEBSOCKET_URL` (recommended).

Note: Keep these URLs private; they allow external access to your local server while the tunnel is active.


### 3. Frontend Configuration

```bash
# Copy example environment file
cp voicera_frontend/.env.example voicera_frontend/.env.local

# Edit configuration
nano voicera_frontend/.env.local
```

```env
# API Configuration
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_VOICE_SERVER_URL=http://localhost:7860

# Authentication
NEXT_PUBLIC_AUTH_ENABLED=true
```

## Verify Installation

### Check Docker Installation

```bash
# Verify Docker version
docker --version
# Output: Docker version 20.10.x, build ...

# Verify Docker Compose
docker-compose --version
# Output: Docker Compose version 1.29.x, ...

# Test Docker
docker run hello-world
```

### Check Python Installation (Optional)

```bash
python3 --version
# Output: Python 3.10.x
```

### Check Node.js Installation (Optional)

```bash
node --version
# Output: v18.x.x

npm --version
# Output: 9.x.x
```

## Build Docker Images

```bash
# Build all services
make build-all-services

# Or manually
docker-compose build
```

This will build images for:
- Frontend (Next.js)
- Backend (FastAPI)
- Voice Server (Pipecat)
- AI4Bharat STT (optional)
- AI4Bharat TTS (optional)

## Verify Build

```bash
# List Docker images
docker images | grep voicera

# Expected output:
# voicera_mono_repository_frontend   latest    ...
# voicera_mono_repository_backend    latest    ...
# voicera_mono_repository_voice_server latest ...
```

## Next Steps

1. **[Quick Start](quickstart.md)** - Start all services and access the application
2. **[Configuration Guide](configuration.md)** - Learn about all configuration options
3. **[Architecture Overview](../architecture/overview.md)** - Understand the system

## Troubleshooting

### Docker daemon not running

=== "Linux"
    ```bash
    sudo systemctl start docker
    sudo systemctl enable docker  # Auto-start on boot
    ```

=== "macOS"
    Open Docker Desktop from Applications

=== "Windows"
    Open Docker Desktop from Start Menu

### Permission denied errors

```bash
# Add current user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Log out and back in for changes to take effect
```

### Port already in use

```bash
# Find and kill process using port (example: 8000)
lsof -i :8000
kill -9 <PID>

# Or modify port in docker-compose.yml
```

### Out of disk space

```bash
# Clean up Docker resources
docker system prune -a

# Remove unused volumes
docker volume prune
```

## Getting Help

- Check [Troubleshooting Guide](../troubleshooting.md)
- Read [Docker Documentation](https://docs.docker.com/)
- Visit [GitHub Discussions](https://github.com/voicera/voicera/discussions)
