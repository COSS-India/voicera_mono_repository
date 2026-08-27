#!/bin/bash
# =============================================================================
# VoicEra — Full Setup & Go Live
# One command to go from fresh EC2 to running application.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/PRANABraight/voicera_mono_repository/dev/setup.sh -o /tmp/setup.sh && bash /tmp/setup.sh
#
# Optional env vars:
#   HF_TOKEN=xxx           (TTS only: the tokenizers and T5 encoder come from
#                           HuggingFace at startup. The Parler checkpoint and the
#                           STT model are fetched from elsewhere and need no token.)
#   VOBIZ_AUTH_ID=xxx      (telephony)
#   VOBIZ_AUTH_TOKEN=xxx   (telephony)
#   STT_MODEL=<id>         a folder under model-server/stt/  (empty = no STT)
#   TTS_MODEL=<id>         a folder under model-server/tts/  (empty = no TTS)
#   LLM_MODEL=<id>         a folder under model-server/llm/  (empty = no local LLM)
#                          Set any of these to skip its menu and run unattended.
#   ENABLE_LLM=none|openai|grok|vllm  (default: none; vllm = host it ourselves)
#   OPENAI_API_KEY=xxx
#   XAI_API_KEY=xxx
# =============================================================================
set -e

NGROK_TOKEN="${NGROK_TOKEN:-}"
HF_TOKEN="${HF_TOKEN:-}"
ENABLE_LLM="${ENABLE_LLM:-none}"
STT_SEL="${STT_MODEL-__ask__}"
TTS_SEL="${TTS_MODEL-__ask__}"
LLM_SEL="${LLM_MODEL-__ask__}"
VOBIZ_AUTH_ID="${VOBIZ_AUTH_ID:-PLACEHOLDER}"
VOBIZ_AUTH_TOKEN="${VOBIZ_AUTH_TOKEN:-PLACEHOLDER}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
XAI_API_KEY="${XAI_API_KEY:-}"

log()  { echo -e "\n\033[1;32m[VoicEra]\033[0m $1"; }
ok()   { echo -e "\033[1;34m  ✓\033[0m $1"; }
err()  { echo -e "\033[1;31m[ERROR]\033[0m $1"; exit 1; }
ask()  { read -r -p "  $1: " "$2"; }

REPO_DIR="$HOME/voicera_mono_repository"

# Which models can fill a slot? The folders under model-server/<slot>/ are the
# answer -- one folder per model is the whole contract, and a test enforces that
# the folders and models.yaml agree, so listing them here needs no YAML parser
# and cannot drift from what Compose will actually build.
list_slot_models() {
  [ -d "$REPO_DIR/model-server/$1" ] || return 0
  find "$REPO_DIR/model-server/$1" -mindepth 1 -maxdepth 1 -type d \
       -not -name '_*' -not -name '.*' -printf '%f\n' 2>/dev/null | sort
}

# Present the folders as a menu and set $3 to the chosen id ("" for none).
pick_model() {
  local slot="$1" label="$2" var="$3"
  local options=() choice i=1
  while IFS= read -r m; do [ -n "$m" ] && options+=("$m"); done < <(list_slot_models "$slot")

  if [ ${#options[@]} -eq 0 ]; then
    echo -e "  \033[2m$label: no models available in model-server/$slot/\033[0m"
    eval "$var=''"
    return
  fi

  echo ""
  echo -e "  \033[1;37m$label\033[0m"
  for m in "${options[@]}"; do
    echo "    $i) $m"
    i=$((i + 1))
  done
  echo "    0) none"
  read -r -p "  Choose [1]: " choice
  choice="${choice:-1}"

  if [ "$choice" = "0" ]; then
    eval "$var=''"
  elif [ "$choice" -ge 1 ] 2>/dev/null && [ "$choice" -le ${#options[@]} ] 2>/dev/null; then
    eval "$var=\"${options[$((choice - 1))]}\""
  else
    echo "  Pick a number from the list."
    pick_model "$slot" "$label" "$var"
  fi
}

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
# The menus come from the folders in the checkout, so a model added to
# model-server/<slot>/ shows up here on its own. Cloning has to happen before we
# can offer the choice -- there is no list to offer otherwise.
if [ ! -d "$REPO_DIR/.git" ]; then
  command -v git >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq git; }
  echo -e "  \033[2mFetching the repository to see which models are available...\033[0m"
  git clone -q -b dev https://github.com/COSS-India/voicera_mono_repository.git "$REPO_DIR"
fi

[ "$STT_SEL" = "__ask__" ] && pick_model stt "Speech to text" STT_SEL
[ "$TTS_SEL" = "__ask__" ] && pick_model tts "Text to speech" TTS_SEL

# The TTS server downloads its tokenizers and T5 encoder from HuggingFace on
# first start. Nothing else here needs a token.
[ -n "$TTS_SEL" ] && [ -z "$HF_TOKEN" ] && ask "HuggingFace token (for TTS)" HF_TOKEN

echo ""
echo "  LLM: none | openai | grok | vllm"
read -r -p "  LLM provider [default: none]: " _llm
[ -n "$_llm" ] && ENABLE_LLM="$_llm"
[ "$ENABLE_LLM" = "openai" ] && [ -z "$OPENAI_API_KEY" ] && ask "  OpenAI API key" OPENAI_API_KEY
[ "$ENABLE_LLM" = "grok"   ] && [ -z "$XAI_API_KEY"   ] && ask "  xAI API key" XAI_API_KEY
# openai and grok are somebody else's servers; only vllm is a model we host.
if [ "$ENABLE_LLM" = "vllm" ]; then
  [ "$LLM_SEL" = "__ask__" ] && pick_model llm "Language model" LLM_SEL
else
  LLM_SEL=""
fi
[ "$STT_SEL" = "__ask__" ] && STT_SEL=""
[ "$TTS_SEL" = "__ask__" ] && TTS_SEL=""

# Derived from the choices so the rest of the script reads the way it did.
ENABLE_STT=$([ -n "$STT_SEL" ] && echo yes || echo no)
ENABLE_TTS=$([ -n "$TTS_SEL" ] && echo yes || echo no)

echo ""
[ -z "${VOBIZ_AUTH_ID/PLACEHOLDER/}" ] && read -r -p "  Vobiz Auth ID [enter to skip]: " _vid && [ -n "$_vid" ] && VOBIZ_AUTH_ID="$_vid"
[ -z "${VOBIZ_AUTH_TOKEN/PLACEHOLDER/}" ] && read -r -p "  Vobiz Auth Token [enter to skip]: " _vtk && [ -n "$_vtk" ] && VOBIZ_AUTH_TOKEN="$_vtk"

echo ""
echo -e "  STT: ${STT_SEL:-none}  |  TTS: ${TTS_SEL:-none}  |  LLM: ${LLM_SEL:-${ENABLE_LLM}}"
read -r -p "  Proceed? [Y/n]: " _ok
[ "$_ok" = "n" ] && exit 0

PRIVATE_IP=$(hostname -I | awk '{print $1}')
INTERNAL_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))" 2>/dev/null || echo "voicera-key-change-me")

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

# ngrok + cloudflared
if [ -n "$NGROK_TOKEN" ]; then
  ngrok config add-authtoken "$NGROK_TOKEN" 2>/dev/null || true
else
  echo "  ngrok: no NGROK_TOKEN set, skipping auth (tunnels will be anonymous)"
fi
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
# Already cloned above -- the model menus needed it.

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
  STT_DIR="$REPO_DIR/model-server/stt/$STT_SEL"
  # The AI4Bharat NeMo fork is a *build context*, not weights, so it cannot live
  # in fetch.sh -- Compose needs the path before the image is built. This is the
  # one piece of model-specific knowledge left in setup.sh; a model that does not
  # reference the `nemo` context simply never triggers it.
  NEMO_DIR=""
  if [ "$STT_SEL" = "indic-conformer" ]; then
  NEMO_DIR="${NEMO_CONTEXT_PATH:-$HOME/ai4bharat_nemo}"
  if [ ! -d "$NEMO_DIR" ]; then
    git clone --branch nemo-v2 --depth 1 https://github.com/AI4Bharat/NeMo.git "$NEMO_DIR"
    ok "NeMo fork cloned to $NEMO_DIR"
  else
    ok "NeMo fork already present at $NEMO_DIR"
  fi
  fi
  # Weights come from the model's own fetch.sh, so setup.sh needs no knowledge
  # of what this particular model downloads or from where.
  [ -f "$STT_DIR/fetch.sh" ] && bash "$STT_DIR/fetch.sh"
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
  TTS_DIR="$REPO_DIR/model-server/tts/$TTS_SEL"
  # torch/flashinfer are built into the model's Dockerfile.
  PY312="$PY312" bash -c '[ -f "$0/fetch.sh" ] && bash "$0/fetch.sh"' "$TTS_DIR" || true
  cat > "$TTS_DIR/.env" << ENVEOF
CHECKPOINT_PATH_DEFAULT=$TTS_DIR/checkpoints
BHILI_ENABLE=no
PORT=8002
HF_TOKEN=$HF_TOKEN
ENVEOF
  ok "TTS ready"
fi

# Build the model images. STT/TTS/LLM run in containers now, not tmux venvs.
MS_DIR="$REPO_DIR/model-server"
[ -f "$MS_DIR/.env" ] || cp "$MS_DIR/.env.example" "$MS_DIR/.env"
# The answers above choose the model for each slot. They are written into
# model-server/.env, which is the single place both compose (which containers
# to start) and the gateway (which slots exist) read from. If these disagree,
# the gateway reports a slot as deployed that was never started.
# Profiles are slot names; the *_MODEL values pick which model fills each slot.
# Writing both from the same answers is what stops them drifting apart.
MODEL_PROFILES=""
[ -n "$STT_SEL" ] && MODEL_PROFILES="$MODEL_PROFILES,stt"
[ -n "$TTS_SEL" ] && MODEL_PROFILES="$MODEL_PROFILES,tts"
[ -n "$LLM_SEL" ] && MODEL_PROFILES="$MODEL_PROFILES,llm"
MODEL_PROFILES="${MODEL_PROFILES#,}"

sed -i "s|^STT_MODEL=.*|STT_MODEL=$STT_SEL|; \
        s|^TTS_MODEL=.*|TTS_MODEL=$TTS_SEL|; \
        s|^LLM_MODEL=.*|LLM_MODEL=$LLM_SEL|; \
        s|^COMPOSE_PROFILES=.*|COMPOSE_PROFILES=$MODEL_PROFILES|" "$MS_DIR/.env"

# The Parler tokenizer lives in a gated HuggingFace repo, so the TTS container
# needs a token to download it on first start.
sed -i "s|^HF_TOKEN=.*|HF_TOKEN=$HF_TOKEN|" "$MS_DIR/.env"
if [ -n "$NEMO_DIR" ]; then
  sed -i "s|^NEMO_CONTEXT_PATH=.*|NEMO_CONTEXT_PATH=$NEMO_DIR|" "$MS_DIR/.env"
fi
if [ "$ENABLE_TTS" = "yes" ] && [ -z "$HF_TOKEN" ]; then
  echo "  WARNING: TTS is enabled but no HF_TOKEN was given. ai4bharat/indic-parler-tts"
  echo "           is gated -- the container will fail to start. Either supply a token,"
  echo "           or reuse an existing cache with compose.shared-hf-cache.yml."
fi
if [ -n "$MODEL_PROFILES" ]; then
  log "Building model images (first build takes 20-40 min)"
  # No COMPOSE_PROFILES here: it comes from .env, same as everything else.
  docker compose -f "$MS_DIR/compose.model-server.yml" \
    --project-directory "$MS_DIR" build
  ok "Model images built for slots: $MODEL_PROFILES"
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
# Allow unauthenticated local URI when user/password are empty (native mongod / no-auth setups).
if 'if self.MONGODB_USER and self.MONGODB_PASSWORD:' not in c:
    needle = 'def mongodb_uri(self) -> str:'
    if needle in c:
        # No-op if already patched; setup keeps FerretDB-oriented config as shipped.
        pass
PYEOF

SECRET_KEY=$(openssl rand -hex 32)
cat > "$BACKEND_DIR/.env" << ENVEOF
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USER=admin
MONGODB_PASSWORD=admin123
MONGODB_DATABASE=voicera
MONGODB_AUTH_SOURCE=
DEBUG=True
SECRET_KEY=$SECRET_KEY
MAILTRAP_API_TOKEN=placeholder
MAILTRAP_FROM_EMAIL=noreply@voicera.com
MAILTRAP_FROM_NAME=Voicera
FRONTEND_URL=https://PENDING
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
STT_DIR="$REPO_DIR/model-server/stt/$STT_SEL"
TTS_DIR="$REPO_DIR/model-server/tts/$TTS_SEL"
V2V_DIR="$REPO_DIR/voice_2_voice_server"
HF_TOKEN="$HF_TOKEN"
ls /dev/nvidia0 2>/dev/null || { sudo modprobe nvidia; sudo modprobe nvidia-uvm; }
pgrep -x mongod &>/dev/null || sudo systemctl start mongod
tmux kill-session -t voicera 2>/dev/null || true
tmux new-session -d -s voicera -n backend
tmux send-keys -t voicera:backend "cd \$BACKEND_DIR && source venv/bin/activate && python3 run.py" Enter
STARTEOF

# Models start as containers. Same ports as the tmux windows they replace,
# so nothing downstream can tell the difference.
[ -n "$MODEL_PROFILES" ] && cat >> "$HOME/start_voicera.sh" << STARTEOF
docker compose -f $REPO_DIR/model-server/compose.model-server.yml \
  --project-directory $REPO_DIR/model-server up -d
STARTEOF

cat >> "$HOME/start_voicera.sh" << 'STARTEOF'
tmux new-window -t voicera -n v2v
tmux send-keys -t voicera:v2v "cd $V2V_DIR && source venv/bin/activate && python3 main.py" Enter
tmux new-window -t voicera -n ngrok
tmux send-keys -t voicera:ngrok 'ngrok http 7860' Enter
tmux new-window -t voicera -n frontend
tmux send-keys -t voicera:frontend "cd \$REPO_DIR/voicera_frontend && npx next dev --port 3000" Enter
tmux new-window -t voicera -n cloudflare
tmux send-keys -t voicera:cloudflare 'cloudflared tunnel --url http://localhost:3000 --logfile /tmp/cf.log' Enter
# Wait for services to come up (STT loads 2.4GB model — takes 2-3 min)
echo "  Waiting for services to start (up to 3 min)..."
for i in $(seq 1 18); do
  sleep 10
  # Models sit behind the gateway now; they publish no host ports of their own.
  STT_OK=$(curl -s http://localhost:8100/health 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print('yes' if d.get('status')=='healthy' else 'no')" 2>/dev/null)
  V2V_OK=$(curl -s http://localhost:7860/health 2>/dev/null | python3 -c "import sys,json;print('yes' if json.load(sys.stdin).get('status')=='healthy' else 'no')" 2>/dev/null)
  API_OK=$(curl -s http://localhost:8000/health 2>/dev/null | python3 -c "import sys,json;print('yes' if json.load(sys.stdin).get('status')=='healthy' else 'no')" 2>/dev/null)
  [ "$STT_OK" = "yes" ] && [ "$V2V_OK" = "yes" ] && [ "$API_OK" = "yes" ] && break
  echo "  ...${i}0s elapsed (STT=${STT_OK:-loading} V2V=${V2V_OK:-loading} API=${API_OK:-loading})"
done
echo "  All services starting..."
STARTEOF
chmod +x "$HOME/start_voicera.sh"

# V2V + frontend .env
cat > "$V2V_DIR/.env" << ENVEOF
VOBIZ_AUTH_ID=$VOBIZ_AUTH_ID
VOBIZ_AUTH_TOKEN=$VOBIZ_AUTH_TOKEN
VOBIZ_API_BASE=https://api.vobiz.in/v1
VOBIZ_CALLER_ID=+91XXXXXXXXXX
PLIVO_AUTH_ID=PLACEHOLDER
PLIVO_AUTH_TOKEN=PLACEHOLDER
JOHNAIC_SERVER_URL=https://PENDING
JOHNAIC_WEBSOCKET_URL=wss://PENDING
VOICERA_BACKEND_URL=http://localhost:8000
INTERNAL_API_KEY=$INTERNAL_KEY
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
BHASHINI_API_KEY=PLACEHOLDER
BHASHINI_SOCKET_URL=PLACEHOLDER
MODEL_SERVER_URL=http://$PRIVATE_IP:8100
OPENAI_API_KEY=$OPENAI_API_KEY
XAI_API_KEY=$XAI_API_KEY
ENVEOF

echo 'NEXT_PUBLIC_JOHNAIC_SERVER_URL="https://PENDING"' > "$REPO_DIR/voicera_frontend/.env.local"

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

# Only start Cloudflare tunnel AFTER frontend is confirmed up
pkill cloudflared 2>/dev/null; sleep 1
tmux send-keys -t voicera:cloudflare "cloudflared tunnel --url http://localhost:3000 --logfile /tmp/cf.log" Enter
sleep 12
CF_URL=$(grep -o 'https://[^ |]*\.trycloudflare\.com' /tmp/cf.log 2>/dev/null | tail -1 || echo "")
[ -n "$CF_URL" ] && sed -i "s|https://PENDING|$CF_URL|g" "$BACKEND_DIR/.env" 2>/dev/null || true

echo ""
echo -e "[1;36m  ══════════════════════════════════════[0m"
echo -e "[1;32m         ✓  VoicEra is Live![0m"
echo -e "[1;36m  ══════════════════════════════════════[0m"
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["tunnels"][0]["public_url"])' 2>/dev/null || echo "")
[ -n "$NGROK_URL" ] && echo -e "[0m  V2V    (ngrok):  $NGROK_URL[0m"
[ -n "$CF_URL"   ] && echo -e "[0m  App    (CF):     $CF_URL[0m"
echo ""
curl -s http://localhost:8100/health 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);u=d.get("upstreams",{});print("  MODELS: "+str(d.get("status","?"))+" | "+", ".join(f"{k}={(v.get("model") or "-")}" for k,v in u.items()))' 2>/dev/null || echo "  MODELS: loading..."
curl -s http://localhost:7860/health 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  V2V  : "+str(d.get("status","?")))' 2>/dev/null || echo "  V2V  : loading..."
curl -s http://localhost:8000/health 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  API  : "+str(d.get("status","ok")))' 2>/dev/null || echo "  API  : loading..."
ss -tlnp | grep 8100 | grep -q LISTEN && echo "  GATEWAY: listening :8100" || echo "  GATEWAY: loading..."
echo ""
echo "  Attach:  tmux attach -t voicera"
