#!/bin/bash
# =============================================================================
# VoicEra — Full Setup & Go Live
# One command to go from fresh EC2 to running application.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/PRANABraight/voicera_mono_repository/dev/setup.sh -o /tmp/setup.sh && bash /tmp/setup.sh
#
# Optional env vars:
#   HF_TOKEN=xxx           (required for TTS — gated model)
#   VOBIZ_AUTH_ID=xxx      (telephony)
#   VOBIZ_AUTH_TOKEN=xxx   (telephony)
#   ENABLE_STT=yes|no      (default: yes)
#   ENABLE_TTS=yes|no      (default: yes)
#   ENABLE_LLM=none|openai|grok|vllm  (default: none)
#   OPENAI_API_KEY=xxx
#   XAI_API_KEY=xxx
# =============================================================================
set -e

NGROK_TOKEN="38n07ZPka4Ra3hBJBnu5w45AMGr_2frPwYhkYHXDsuEE9WF6Q"
HF_TOKEN="${HF_TOKEN:-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}"
ENABLE_STT="${ENABLE_STT:-yes}"
ENABLE_TTS="${ENABLE_TTS:-yes}"
ENABLE_LLM="${ENABLE_LLM:-none}"
VOBIZ_AUTH_ID="${VOBIZ_AUTH_ID:-PLACEHOLDER}"
VOBIZ_AUTH_TOKEN="${VOBIZ_AUTH_TOKEN:-PLACEHOLDER}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
XAI_API_KEY="${XAI_API_KEY:-}"

log()  { echo -e "\n\033[1;32m[VoicEra]\033[0m $1"; }
ok()   { echo -e "\033[1;34m  ✓\033[0m $1"; }
err()  { echo -e "\033[1;31m[ERROR]\033[0m $1"; exit 1; }
ask()  { read -r -p "  $1: " "$2"; }

show_banner() {
  local C1="\033[1;36m" C2="\033[0;36m" C3="\033[1;34m"
  local C4="\033[0;34m" C5="\033[34m"
  local G="\033[1;32m" Y="\033[1;33m" W="\033[1;37m"
  local DIM="\033[2m" BOLD="\033[1m" NC="\033[0m"
  clear
  echo ""
  echo -e "${C1}    ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗██████╗  █████╗ ${NC}"
  echo -e "${C2}    ██║   ██║██╔═══██╗██║██╔════╝██╔════╝██╔══██╗██╔══██╗${NC}"
  echo -e "${C3}    ██║   ██║██║   ██║██║██║     █████╗  ██████╔╝███████║${NC}"
  echo -e "${C4}    ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  ██╔══██╗██╔══██║${NC}"
  echo -e "${C5}     ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗██║  ██║██║  ██║${NC}"
  echo -e "${C5}      ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝${NC}"
  echo ""
  echo -e "  ${DIM}───────────────────────────────────────────────────────${NC}"
  echo -e "  ${C3}${BOLD} Voice AI for Every Language${NC}  ${DIM}│${NC}  ${W}Built by COSS India${NC}"
  echo -e "  ${DIM}───────────────────────────────────────────────────────${NC}"
  echo ""
}

show_banner

echo -e "\033[1;37m  Configure Services\033[0m"
echo -e "\033[2m  ─────────────────────────────────────────────────────\033[0m"
read -r -p "  Enable STT? [yes/no, default: yes]: " _stt
[ "$_stt" = "y" ] && _stt="yes"; [ "$_stt" = "n" ] && _stt="no"
[ -n "$_stt" ] && ENABLE_STT="$_stt"

read -r -p "  Enable TTS? [yes/no, default: yes]: " _tts
[ "$_tts" = "y" ] && _tts="yes"; [ "$_tts" = "n" ] && _tts="no"
[ -n "$_tts" ] && ENABLE_TTS="$_tts"

echo "  LLM: none | openai | grok | vllm"
read -r -p "  LLM provider [default: none]: " _llm
[ -n "$_llm" ] && ENABLE_LLM="$_llm"
[ "$ENABLE_LLM" = "openai" ] && [ -z "$OPENAI_API_KEY" ] && ask "  OpenAI API key" OPENAI_API_KEY
[ "$ENABLE_LLM" = "grok"   ] && [ -z "$XAI_API_KEY"   ] && ask "  xAI API key" XAI_API_KEY

echo ""
[ -z "${VOBIZ_AUTH_ID/PLACEHOLDER/}" ] && read -r -p "  Vobiz Auth ID [enter to skip]: " _vid && [ -n "$_vid" ] && VOBIZ_AUTH_ID="$_vid"
[ -z "${VOBIZ_AUTH_TOKEN/PLACEHOLDER/}" ] && read -r -p "  Vobiz Auth Token [enter to skip]: " _vtk && [ -n "$_vtk" ] && VOBIZ_AUTH_TOKEN="$_vtk"

# ── Domain check ─────────────────────────────────────────────────────────────
echo ""
echo -e "\033[2m  ─────────────────────────────────────────────────────\033[0m"
CUSTOM_V2V_DOMAIN=""
CUSTOM_APP_DOMAIN=""
read -r -p "  Do you have a domain already forwarding port 7860 (V2V)? [yes/no, default: no]: " _has_v2v_domain
if [ "$_has_v2v_domain" = "yes" ] || [ "$_has_v2v_domain" = "y" ]; then
  read -r -p "  Enter V2V domain (e.g. https://v2v.example.com): " CUSTOM_V2V_DOMAIN
fi
read -r -p "  Do you have a domain already forwarding port 3000 (App)? [yes/no, default: no]: " _has_app_domain
if [ "$_has_app_domain" = "yes" ] || [ "$_has_app_domain" = "y" ]; then
  read -r -p "  Enter App domain (e.g. https://app.example.com): " CUSTOM_APP_DOMAIN
fi
echo ""

echo ""
echo -e "  STT: ${ENABLE_STT}  |  TTS: ${ENABLE_TTS}  |  LLM: ${ENABLE_LLM}"
[ -n "$CUSTOM_V2V_DOMAIN" ] && echo -e "  V2V domain: ${CUSTOM_V2V_DOMAIN}" || echo -e "  V2V tunnel: ngrok (auto)"
[ -n "$CUSTOM_APP_DOMAIN" ] && echo -e "  App domain: ${CUSTOM_APP_DOMAIN}" || echo -e "  App tunnel: Cloudflare (auto)"
read -r -p "  Proceed? [Y/n]: " _ok
[ "$_ok" = "n" ] && exit 0

PRIVATE_IP=$(hostname -I | awk '{print $1}')
INTERNAL_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))" 2>/dev/null || echo "voicera-key-change-me")
REPO_DIR="$HOME/voicera_mono_repository"

# ── Phase 1: Instance Setup ──────────────────────────────────────────────────
log "Phase 1/3: Instance Setup"

# System packages
NEEDED=""
for pkg in git curl wget ffmpeg sox libsndfile1 gcc g++ make cargo rustc tmux nginx openssl ca-certificates; do
  dpkg -l "$pkg" 2>/dev/null | grep -q "^ii" || NEEDED="$NEEDED $pkg"
done
if [ -n "$NEEDED" ]; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y $NEEDED 2>&1 | tail -2
fi
ok "System packages ready"

# Node.js 20 (Next.js requires 18+; Ubuntu 22.04 ships 12)
NODE_VER=$(node --version 2>/dev/null | sed 's/v//' | cut -d. -f1 || echo "0")
if [ "$NODE_VER" -lt 18 ] 2>/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - 2>/dev/null
  sudo apt-get remove --purge -y libnode-dev libnode72 nodejs 2>/dev/null || true
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs 2>&1 | tail -2
  ok "Node.js $(node --version) installed"
fi

# Docker CE
if ! command -v docker &>/dev/null; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
  . /etc/os-release
  echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/docker.list
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io 2>&1 | tail -2
  sudo systemctl enable docker && sudo systemctl start docker
  sudo usermod -aG docker ubuntu
fi
ok "Docker ready"

# NVIDIA Container Toolkit
if ! dpkg -l 2>/dev/null | grep -q "nvidia-container-toolkit"; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit 2>&1 | tail -2
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi
ok "NVIDIA Container Toolkit ready"

# NVIDIA Driver
if ! ls /dev/nvidia0 &>/dev/null; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ubuntu-drivers-common 2>&1 | tail -1
  sudo ubuntu-drivers autoinstall 2>&1 | tail -3
  sudo modprobe nvidia 2>/dev/null || true
  sudo modprobe nvidia-uvm 2>/dev/null || true
  printf "nvidia\nnvidia-uvm\nnvidia-modeset\n" | sudo tee /etc/modules-load.d/nvidia.conf
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read l; do ok "GPU: $l"; done

# CUDA
if ! command -v nvcc &>/dev/null && ! /usr/local/cuda/bin/nvcc --version &>/dev/null; then
  UBUNTU_VER=$(. /etc/os-release && echo "$VERSION_ID" | tr -d '.')
  case "$UBUNTU_VER" in 2204|2404) CUDA_DISTRO="ubuntu${UBUNTU_VER}" ;; *) CUDA_DISTRO="ubuntu2204" ;; esac
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb && sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit 2>&1 | tail -2
fi
ok "CUDA ready"

# Python 3.12
if ! command -v "$HOME/.local/bin/uv" &>/dev/null; then
  (curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null) || true
fi
set +e
PY312=$("$HOME/.local/bin/uv" python find 3.12 2>/dev/null)
if [ -z "$PY312" ]; then
  "$HOME/.local/bin/uv" python install 3.12 2>&1 | tail -2
  PY312=$("$HOME/.local/bin/uv" python find 3.12 2>/dev/null)
fi
set -e
sudo ln -sf "$PY312" /usr/local/bin/python3.12 2>/dev/null || true
ok "Python 3.12: $PY312"

# ngrok + cloudflared (install unconditionally; only used as fallback if no custom domain)
ngrok config add-authtoken "$NGROK_TOKEN" 2>/dev/null || true
if ! command -v ngrok &>/dev/null; then
  curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
  echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ngrok 2>&1 | tail -2
fi
if ! command -v cloudflared &>/dev/null; then
  sudo curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
  sudo chmod +x /usr/local/bin/cloudflared
fi
ok "ngrok + cloudflared ready"

# ── Phase 2: Application Deploy ──────────────────────────────────────────────
log "Phase 2/3: Application Deploy"

# Clone repo
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone -b dev https://github.com/PRANABraight/voicera_mono_repository.git "$REPO_DIR"
fi

# MongoDB 7.0
if ! command -v mongod &>/dev/null; then
  UBUNTU_CODENAME=$(. /etc/os-release && echo "$UBUNTU_CODENAME")
  curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg
  echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu ${UBUNTU_CODENAME}/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y mongodb-org 2>&1 | tail -2
  sudo systemctl enable mongod
fi
sudo mkdir -p /var/log/mongodb /var/lib/mongodb && sudo chown -R mongodb:mongodb /var/log/mongodb /var/lib/mongodb 2>/dev/null || true
pgrep -x mongod &>/dev/null || sudo systemctl start mongod
sleep 2
ok "MongoDB running"

# STT
if [ "$ENABLE_STT" = "yes" ]; then
  STT_DIR="$REPO_DIR/ai4bharat_stt_server"
  [ -d "$STT_DIR/venv" ] || "$PY312" -m venv "$STT_DIR/venv"
  if [ ! -d "$HOME/ai4bharat_nemo" ]; then
    git clone --branch nemo-v2 --depth 1 https://github.com/AI4Bharat/NeMo.git "$HOME/ai4bharat_nemo"
  fi
  NEMO_EXP="$HOME/ai4bharat_nemo/nemo/utils/exp_manager.py"
  grep -q "NeptuneLogger" "$NEMO_EXP" && \
    sed -i 's/from pytorch_lightning.loggers import MLFlowLogger, NeptuneLogger, TensorBoardLogger, WandbLogger/from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger/' "$NEMO_EXP" && \
    sed -i 's/^    if create_neptune_logger:/    if False and create_neptune_logger:/' "$NEMO_EXP" || true
  grep -q "nemo_asr.models.ASRModel.restore_from" "$STT_DIR/server.py" && \
    sed -i 's/nemo_asr.models.ASRModel.restore_from(/EncDecHybridRNNTCTCBPEModel.restore_from(/' "$STT_DIR/server.py" || true
  if ! "$STT_DIR/venv/bin/python3" -c "import nemo" 2>/dev/null; then
    "$STT_DIR/venv/bin/pip" install -q -e "$HOME/ai4bharat_nemo[asr]" 2>&1 | tail -2
    "$STT_DIR/venv/bin/pip" install -q fastapi uvicorn pydantic torchaudio python-dotenv "onnxruntime-gpu>=1.24" "ml_dtypes>=0.5" --upgrade protobuf ml_dtypes 2>&1 | tail -2
  else
    ok "STT packages already installed"
  fi
  if [ ! -f "$STT_DIR/models/IndicConformer.nemo" ]; then
    mkdir -p "$STT_DIR/models"
    wget -q --show-progress "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo" -O "$STT_DIR/models/IndicConformer.nemo"
  fi
  cat > "$STT_DIR/.env" << ENVEOF
PORT=8001
BHILI_ENABLE=no
INDIC_NEMO_PATH=$STT_DIR/models/IndicConformer.nemo
HF_TOKEN=$HF_TOKEN
ENVEOF
  ok "STT ready"
fi

# TTS
if [ "$ENABLE_TTS" = "yes" ]; then
  TTS_DIR="$REPO_DIR/ai4bharat_tts_server"
  [ -d "$TTS_DIR/venv" ] || "$PY312" -m venv "$TTS_DIR/venv"
  if ! "$TTS_DIR/venv/bin/python3" -c "import flashinfer" 2>/dev/null; then
    "$TTS_DIR/venv/bin/pip" install -q "flashinfer-python==0.6.7" "flashinfer-cubin==0.6.7" 2>&1 | tail -2
    "$TTS_DIR/venv/bin/pip" install -q transformers sentencepiece protobuf scipy "websockets>=12.0" python-dotenv 2>&1 | tail -2
  else
    ok "TTS packages already installed"
  fi
  if [ ! -f "$TTS_DIR/checkpoints/model_step_ref.pt" ]; then
    "$TTS_DIR/venv/bin/pip" install -q gdown 2>&1 | tail -1
    mkdir -p "$TTS_DIR/checkpoints"
    "$TTS_DIR/venv/bin/python3" -m gdown --folder https://drive.google.com/drive/folders/1qrh56MWXboiBO38gaWEcWhFl0NzlDiaT -O "$TTS_DIR/checkpoints/" 2>&1 | tail -3
    [ -d "$TTS_DIR/checkpoints/checkpoints" ] && mv "$TTS_DIR/checkpoints/checkpoints/"* "$TTS_DIR/checkpoints/" && rmdir "$TTS_DIR/checkpoints/checkpoints" 2>/dev/null || true
  fi
  cat > "$TTS_DIR/.env" << ENVEOF
CHECKPOINT_PATH_DEFAULT=$TTS_DIR/checkpoints
BHILI_ENABLE=no
PORT=8002
HF_TOKEN=$HF_TOKEN
ENVEOF
  ok "TTS ready"
fi

# V2V
V2V_DIR="$REPO_DIR/voice_2_voice_server"
[ -d "$V2V_DIR/venv" ] || "$PY312" -m venv "$V2V_DIR/venv"
if ! "$V2V_DIR/venv/bin/python3" -c "import pipecat" 2>/dev/null; then
  "$V2V_DIR/venv/bin/pip" install -q python-dotenv==1.2.1 "fastapi[all]==0.121.3" uvicorn==0.40.0 pydantic==2.12.0 numpy loguru pyyaml requests "aiohttp==3.13.2" websockets minio "boto3==1.42.21" "python-socketio[asyncio_client]==5.11.0" PyJWT cryptography "protobuf~=5.29.5" "deepgram-sdk==4.7.0" "google-cloud-texttospeech==2.33.0" "google-cloud-speech==2.35.0" "cartesia==2.0.17" "sarvamai==0.1.21" 2>&1 | tail -2
  "$V2V_DIR/venv/bin/pip" install -q "pipecat-ai[silero,websocket,google,cartesia,openai,deepgram,sarvam,elevenlabs,anthropic,grok]==0.0.98" 2>&1 | tail -2
else
  ok "V2V packages already installed"
fi

# Backend
BACKEND_DIR="$REPO_DIR/voicera_backend"
[ -d "$BACKEND_DIR/venv" ] || "$PY312" -m venv "$BACKEND_DIR/venv"
if ! "$BACKEND_DIR/venv/bin/python3" -c "import fastapi, pymongo" 2>/dev/null; then
  "$BACKEND_DIR/venv/bin/pip" install -q -r "$BACKEND_DIR/requirements.txt" 2>&1 | tail -2
else
  ok "Backend packages already installed"
fi
python3 - "$BACKEND_DIR/app/config.py" << 'PYEOF'
import sys; path=sys.argv[1]; c=open(path).read()
if 'MONGODB_USER and self.MONGODB_PASSWORD' not in c:
    old=('    @property\n    def mongodb_uri(self) -> str:\n        """Build MongoDB connection URI."""\n        return (\n            f"mongodb://{self.MONGODB_USER}:{self.MONGODB_PASSWORD}"\n            f"@{self.MONGODB_HOST}:{self.MONGODB_PORT}/{self.MONGODB_DATABASE}"\n            f"?authSource={self.MONGODB_AUTH_SOURCE}"\n        )')
    new=('    @property\n    def mongodb_uri(self) -> str:\n        """Build MongoDB connection URI."""\n        if self.MONGODB_USER and self.MONGODB_PASSWORD:\n            return (\n                f"mongodb://{self.MONGODB_USER}:{self.MONGODB_PASSWORD}"\n                f"@{self.MONGODB_HOST}:{self.MONGODB_PORT}/{self.MONGODB_DATABASE}"\n                f"?authSource={self.MONGODB_AUTH_SOURCE}"\n            )\n        return f"mongodb://{self.MONGODB_HOST}:{self.MONGODB_PORT}/{self.MONGODB_DATABASE}"')
    open(path,'w').write(c.replace(old,new)) if old in c else None
PYEOF

SECRET_KEY=$(openssl rand -hex 32)
cat > "$BACKEND_DIR/.env" << ENVEOF
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USER=
MONGODB_PASSWORD=
MONGODB_DATABASE=voicera
MONGODB_AUTH_SOURCE=admin
DEBUG=True
SECRET_KEY=$SECRET_KEY
MAILTRAP_API_TOKEN=placeholder
MAILTRAP_FROM_EMAIL=noreply@voicera.com
MAILTRAP_FROM_NAME=Voicera
FRONTEND_URL=${CUSTOM_APP_DOMAIN:-https://PENDING}
INTERNAL_API_KEY=$INTERNAL_KEY
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
VOBIZ_API_BASE_URL=https://api.vobiz.in/v1
VOBIZ_AUTH_ID=$VOBIZ_AUTH_ID
VOBIZ_AUTH_TOKEN=$VOBIZ_AUTH_TOKEN
ENVEOF

# Frontend
if [ ! -d "$REPO_DIR/voicera_frontend/node_modules/.bin/next" ] && [ ! -f "$REPO_DIR/voicera_frontend/node_modules/.bin/next" ]; then
  cd "$REPO_DIR/voicera_frontend" && npm install --silent 2>&1 | tail -2
else
  ok "Frontend node_modules already installed"
fi
ok "All services installed"

# ── Phase 3: Go Live ─────────────────────────────────────────────────────────
log "Phase 3/3: Starting Services"

cat > "$HOME/start_voicera.sh" << STARTEOF
#!/bin/bash
REPO_DIR="$REPO_DIR"
BACKEND_DIR="$BACKEND_DIR"
STT_DIR="$REPO_DIR/ai4bharat_stt_server"
TTS_DIR="$REPO_DIR/ai4bharat_tts_server"
V2V_DIR="$REPO_DIR/voice_2_voice_server"
HF_TOKEN="$HF_TOKEN"
CUSTOM_V2V_DOMAIN="$CUSTOM_V2V_DOMAIN"
CUSTOM_APP_DOMAIN="$CUSTOM_APP_DOMAIN"
ls /dev/nvidia0 2>/dev/null || { sudo modprobe nvidia; sudo modprobe nvidia-uvm; }
pgrep -x mongod &>/dev/null || sudo systemctl start mongod
tmux kill-session -t voicera 2>/dev/null || true
tmux new-session -d -s voicera -n backend
tmux send-keys -t voicera:backend "cd \$BACKEND_DIR && source venv/bin/activate && python3 run.py" Enter
STARTEOF

[ "$ENABLE_STT" = "yes" ] && cat >> "$HOME/start_voicera.sh" << STARTEOF
tmux new-window -t voicera -n stt
tmux send-keys -t voicera:stt 'cd $STT_DIR && source venv/bin/activate && python3 server.py' Enter
STARTEOF

[ "$ENABLE_TTS" = "yes" ] && cat >> "$HOME/start_voicera.sh" << STARTEOF
tmux new-window -t voicera -n tts
tmux send-keys -t voicera:tts 'cd $TTS_DIR && HUGGING_FACE_HUB_TOKEN=$HF_TOKEN source venv/bin/activate && python3 server.py --port 8002' Enter
STARTEOF

[ "$ENABLE_LLM" = "vllm" ] && cat >> "$HOME/start_voicera.sh" << STARTEOF
tmux new-window -t voicera -n vllm
tmux send-keys -t voicera:vllm 'source $REPO_DIR/llm_server/venv/bin/activate && vllm serve Qwen/Qwen3-8B --port 8003 --host 0.0.0.0 --enable-auto-tool-choice --tool-call-parser hermes' Enter
STARTEOF

cat >> "$HOME/start_voicera.sh" << 'STARTEOF'
tmux new-window -t voicera -n v2v
tmux send-keys -t voicera:v2v "cd $V2V_DIR && source venv/bin/activate && python3 main.py" Enter
# Start ngrok for V2V only if no custom domain was provided
if [ -z "$CUSTOM_V2V_DOMAIN" ]; then
  tmux new-window -t voicera -n ngrok
  tmux send-keys -t voicera:ngrok 'ngrok http 7860' Enter
fi
tmux new-window -t voicera -n frontend
tmux send-keys -t voicera:frontend "cd \$REPO_DIR/voicera_frontend && npx next dev --port 3000" Enter
# Start Cloudflare tunnel for App only if no custom domain was provided
if [ -z "$CUSTOM_APP_DOMAIN" ]; then
  tmux new-window -t voicera -n cloudflare
  tmux send-keys -t voicera:cloudflare 'cloudflared tunnel --url http://localhost:3000 --logfile /tmp/cf.log' Enter
fi
# Wait for services to come up (STT loads 2.4GB model — takes 2-3 min)
echo "  Waiting for services to start (up to 3 min)..."
for i in $(seq 1 18); do
  sleep 10
  STT_OK=$(curl -s http://localhost:8001/health 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print('yes' if d.get('main_loaded') else 'no')" 2>/dev/null)
  V2V_OK=$(curl -s http://localhost:7860/health 2>/dev/null | python3 -c "import sys,json;print('yes' if json.load(sys.stdin).get('status')=='healthy' else 'no')" 2>/dev/null)
  API_OK=$(curl -s http://localhost:8000/health 2>/dev/null | python3 -c "import sys,json;print('yes' if json.load(sys.stdin).get('status')=='healthy' else 'no')" 2>/dev/null)
  [ "$STT_OK" = "yes" ] && [ "$V2V_OK" = "yes" ] && [ "$API_OK" = "yes" ] && break
  echo "  ...${i}0s elapsed (STT=${STT_OK:-loading} V2V=${V2V_OK:-loading} API=${API_OK:-loading})"
done
echo "  All services starting..."
STARTEOF
chmod +x "$HOME/start_voicera.sh"

# V2V + frontend .env
# Determine the V2V public URL: prefer custom domain, fall back to PENDING (ngrok resolves at runtime)
V2V_PUBLIC_URL="${CUSTOM_V2V_DOMAIN:-https://PENDING}"
V2V_PUBLIC_WSS=$(echo "$V2V_PUBLIC_URL" | sed 's|^https://|wss://|; s|^http://|ws://|')

cat > "$V2V_DIR/.env" << ENVEOF
VOBIZ_AUTH_ID=$VOBIZ_AUTH_ID
VOBIZ_AUTH_TOKEN=$VOBIZ_AUTH_TOKEN
VOBIZ_API_BASE=https://api.vobiz.in/v1
VOBIZ_CALLER_ID=+91XXXXXXXXXX
PLIVO_AUTH_ID=PLACEHOLDER
PLIVO_AUTH_TOKEN=PLACEHOLDER
JOHNAIC_SERVER_URL=$V2V_PUBLIC_URL
JOHNAIC_WEBSOCKET_URL=$V2V_PUBLIC_WSS
VOICERA_BACKEND_URL=http://localhost:8000
INTERNAL_API_KEY=$INTERNAL_KEY
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
BHASHINI_API_KEY=PLACEHOLDER
BHASHINI_SOCKET_URL=PLACEHOLDER
AI4BHARAT_STT_URL=http://$PRIVATE_IP:8001
AI4BHARAT_TTS_URL=ws://$PRIVATE_IP:8002
OPENAI_API_KEY=$OPENAI_API_KEY
XAI_API_KEY=$XAI_API_KEY
ENVEOF

APP_PUBLIC_URL="${CUSTOM_APP_DOMAIN:-https://PENDING}"
echo "NEXT_PUBLIC_JOHNAIC_SERVER_URL=\"$APP_PUBLIC_URL\"" > "$REPO_DIR/voicera_frontend/.env.local"

bash "$HOME/start_voicera.sh"

# ── Post-start service health check & auto-fix ────────────────────────────────
log "Verifying all services..."

# Check and restart Backend if down
if ! curl -s --max-time 3 http://localhost:8000/health | python3 -c "import sys,json;json.load(sys.stdin)" 2>/dev/null; then
  ok "Restarting Backend..."
  tmux send-keys -t voicera:backend "cd $BACKEND_DIR && source venv/bin/activate && python3 run.py" Enter
  sleep 5
fi

# Check and restart V2V if down
if ! curl -s --max-time 3 http://localhost:7860/health | python3 -c "import sys,json;json.load(sys.stdin)" 2>/dev/null; then
  ok "Restarting V2V..."
  tmux send-keys -t voicera:v2v "cd $V2V_DIR && source venv/bin/activate && python3 main.py" Enter
  sleep 5
fi

# Wait for Frontend on port 3000
echo "  Waiting for frontend..."
for i in $(seq 1 12); do
  ss -tlnp | grep -q ':3000' && break
  [ "$i" = "1" ] && tmux send-keys -t voicera:frontend "cd $REPO_DIR/voicera_frontend && node_modules/.bin/next dev --port 3000" Enter
  sleep 5
done
ss -tlnp | grep -q ':3000' && ok "Frontend ready on :3000" || ok "WARNING: Frontend may still be starting"

# Resolve tunnel URLs for services without a custom domain
NGROK_URL=""
CF_URL=""

if [ -z "$CUSTOM_V2V_DOMAIN" ]; then
  # Only start ngrok and wait for its URL if no custom V2V domain
  sleep 5
  NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["tunnels"][0]["public_url"])' 2>/dev/null || echo "")
fi

if [ -z "$CUSTOM_APP_DOMAIN" ]; then
  # Only start Cloudflare tunnel AFTER frontend is confirmed up
  pkill cloudflared 2>/dev/null; sleep 1
  tmux send-keys -t voicera:cloudflare "cloudflared tunnel --url http://localhost:3000 --logfile /tmp/cf.log" Enter
  sleep 12
  CF_URL=$(grep -o 'https://[^ |]*\.trycloudflare\.com' /tmp/cf.log 2>/dev/null | tail -1 || echo "")
  [ -n "$CF_URL" ] && sed -i "s|https://PENDING|$CF_URL|g" "$BACKEND_DIR/.env" 2>/dev/null || true
fi

echo ""
echo -e "[1;36m  ══════════════════════════════════════[0m"
echo -e "[1;32m         ✓  VoicEra is Live![0m"
echo -e "[1;36m  ══════════════════════════════════════[0m"
[ -n "$CUSTOM_V2V_DOMAIN" ] && echo -e "[0m  V2V    (custom):  $CUSTOM_V2V_DOMAIN[0m" || { [ -n "$NGROK_URL" ] && echo -e "[0m  V2V    (ngrok):   $NGROK_URL[0m"; }
[ -n "$CUSTOM_APP_DOMAIN" ] && echo -e "[0m  App    (custom):  $CUSTOM_APP_DOMAIN[0m" || { [ -n "$CF_URL"    ] && echo -e "[0m  App    (CF):      $CF_URL[0m"; }
echo ""
curl -s http://localhost:8001/health 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);s=d.get("status","?");m=d.get("main_loaded","?");print("  STT  : "+str(s)+" | model="+str(m))' 2>/dev/null || echo "  STT  : loading..."
curl -s http://localhost:7860/health 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  V2V  : "+str(d.get("status","?")))' 2>/dev/null || echo "  V2V  : loading..."
curl -s http://localhost:8000/health 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  API  : "+str(d.get("status","ok")))' 2>/dev/null || echo "  API  : loading..."
ss -tlnp | grep 8002 | grep -q LISTEN && echo "  TTS  : listening :8002" || echo "  TTS  : loading..."
echo ""
echo "  Attach:  tmux attach -t voicera"
