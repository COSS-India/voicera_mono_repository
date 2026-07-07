# =============================================================================
# VoicEra — Windows Full Setup & Go Live
# One command to go from fresh Windows EC2 to running application.
#
# Usage (run as Administrator in PowerShell 7 — REQUIRED, the script hard-exits on PS5):
#   Set-ExecutionPolicy Bypass -Scope Process -Force; $s="$env:TEMP\voicera_setup.ps1"; Invoke-RestMethod https://raw.githubusercontent.com/COSS-India/voicera_mono_repository/dev/setup.ps1 -OutFile $s; &$s
#
# Optional env vars (set before running):
#   $env:NGROK_TOKEN        get from https://dashboard.ngrok.com/get-started/your-authtoken
#   $env:HF_TOKEN           (required for TTS — gated model) get from https://huggingface.co/settings/tokens
#   $env:VOBIZ_AUTH_ID      (telephony) get from https://www.vobiz.in dashboard
#   $env:VOBIZ_AUTH_TOKEN   (telephony) get from https://www.vobiz.in dashboard
#   $env:ENABLE_STT         yes|no  (default: yes)
#   $env:ENABLE_TTS         yes|no  (default: yes)
#   $env:ENABLE_LLM         none|openai|grok|vllm  (default: none)
#   $env:OPENAI_API_KEY
#   $env:XAI_API_KEY
# =============================================================================
#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"
try {
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force -ErrorAction Stop
} catch {
    Write-Host "  WARN Could not set execution policy (likely GPO-enforced at a stricter scope) — continuing, assuming Bypass at Process scope is already in effect." -ForegroundColor Yellow
}

# ── Defaults ──────────────────────────────────────────────────────────────────
$NGROK_TOKEN    = if ($env:NGROK_TOKEN)     { $env:NGROK_TOKEN }     else { "" } # get from https://dashboard.ngrok.com/get-started/your-authtoken
$HF_TOKEN       = if ($env:HF_TOKEN)        { $env:HF_TOKEN }        else { "" } # get from https://huggingface.co/settings/tokens
$ENABLE_STT     = if ($env:ENABLE_STT)      { $env:ENABLE_STT }      else { "yes" }
$ENABLE_TTS     = if ($env:ENABLE_TTS)      { $env:ENABLE_TTS }      else { "yes" }
$ENABLE_LLM     = if ($env:ENABLE_LLM)      { $env:ENABLE_LLM }      else { "none" }
$VOBIZ_AUTH_ID  = if ($env:VOBIZ_AUTH_ID)   { $env:VOBIZ_AUTH_ID }   else { "PLACEHOLDER" }
$VOBIZ_AUTH_TOKEN = if ($env:VOBIZ_AUTH_TOKEN) { $env:VOBIZ_AUTH_TOKEN } else { "PLACEHOLDER" }
$OPENAI_API_KEY = if ($env:OPENAI_API_KEY)  { $env:OPENAI_API_KEY }  else { "" }
$XAI_API_KEY    = if ($env:XAI_API_KEY)     { $env:XAI_API_KEY }     else { "" }
$REPO_DIR       = "C:\VoicEra"

# ── Helpers ───────────────────────────────────────────────────────────────────
function Timestamp { (Get-Date).ToString("HH:mm:ss") }
function log  { param($m) Write-Host "`n[$(Timestamp)] [VoicEra] $m" -ForegroundColor Green }
function ok   { param($m) Write-Host "[$(Timestamp)]   OK  $m" -ForegroundColor Cyan }
function warn { param($m) Write-Host "[$(Timestamp)]   WARN $m" -ForegroundColor Yellow }
function err  { param($m) Write-Host "[$(Timestamp)] [ERROR] $m" -ForegroundColor Red; exit 1 }
function step { param($m) Write-Host "[$(Timestamp)]  $m" -ForegroundColor DarkGray }

function Refresh-Path {
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH","User")
}

function Test-Port {
    param([int]$Port, [int]$Timeout = 3000)
    $tcp = New-Object System.Net.Sockets.TcpClient
    try {
        $r = $tcp.BeginConnect("localhost", $Port, $null, $null)
        $r.AsyncWaitHandle.WaitOne($Timeout) | Out-Null
        return $tcp.Connected
    } catch { return $false }
    finally { $tcp.Close() }
}

# ── Banner ────────────────────────────────────────────────────────────────────
function Show-Banner {
    Clear-Host
    Write-Host ""
    Write-Host "    ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗██████╗  █████╗ " -ForegroundColor Cyan
    Write-Host "    ██║   ██║██╔═══██╗██║██╔════╝██╔════╝██╔══██╗██╔══██╗" -ForegroundColor Cyan
    Write-Host "    ██║   ██║██║   ██║██║██║     █████╗  ██████╔╝███████║" -ForegroundColor Blue
    Write-Host "    ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  ██╔══██╗██╔══██║" -ForegroundColor DarkBlue
    Write-Host "     ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗██║  ██║██║  ██║" -ForegroundColor Blue
    Write-Host "      ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝" -ForegroundColor Blue
    Write-Host ""
    Write-Host "  ────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "   Voice AI for Every Language  │  Built by COSS India" -ForegroundColor White
    Write-Host "  ────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host ""
}

Show-Banner

# ── Interactive Config ────────────────────────────────────────────────────────
Write-Host "  Configure Services" -ForegroundColor White
Write-Host "  ─────────────────────────────────────────────────────" -ForegroundColor DarkGray

if ($NGROK_TOKEN -eq "") {
    warn "No ngrok token found. Get one at: https://dashboard.ngrok.com/get-started/your-authtoken"
    $NGROK_TOKEN = Read-Host "  ngrok authtoken"
}
if ($ENABLE_TTS -eq "yes" -and $HF_TOKEN -eq "") {
    warn "No Hugging Face token found. Get one at: https://huggingface.co/settings/tokens"
    $HF_TOKEN = Read-Host "  Hugging Face token (required for TTS gated model)"
}

$_stt = Read-Host "  Enable STT? [yes/no, default: $ENABLE_STT]"
if ($_stt -eq "y") { $_stt = "yes" } elseif ($_stt -eq "n") { $_stt = "no" }
if ($_stt -ne "") { $ENABLE_STT = $_stt }

$_tts = Read-Host "  Enable TTS? [yes/no, default: $ENABLE_TTS]"
if ($_tts -eq "y") { $_tts = "yes" } elseif ($_tts -eq "n") { $_tts = "no" }
if ($_tts -ne "") { $ENABLE_TTS = $_tts }

Write-Host "  LLM: none | openai | grok | vllm"
$_llm = Read-Host "  LLM provider [default: $ENABLE_LLM]"
if ($_llm -ne "") { $ENABLE_LLM = $_llm }

if ($ENABLE_LLM -eq "openai" -and $OPENAI_API_KEY -eq "") {
    warn "No OpenAI API key found. Get one at: https://platform.openai.com/api-keys"
    $OPENAI_API_KEY = Read-Host "  OpenAI API key"
}
if ($ENABLE_LLM -eq "grok" -and $XAI_API_KEY -eq "") {
    warn "No xAI API key found. Get one at: https://console.x.ai"
    $XAI_API_KEY = Read-Host "  xAI API key"
}

if ($VOBIZ_AUTH_ID -eq "PLACEHOLDER") {
    warn "No Vobiz credentials found. Get them at: https://www.vobiz.in dashboard"
    $v = Read-Host "  Vobiz Auth ID [Enter to skip]"
    if ($v -ne "") { $VOBIZ_AUTH_ID = $v }
}
if ($VOBIZ_AUTH_TOKEN -eq "PLACEHOLDER") {
    $v = Read-Host "  Vobiz Auth Token [Enter to skip]"
    if ($v -ne "") { $VOBIZ_AUTH_TOKEN = $v }
}

Write-Host ""
Write-Host "  STT: $ENABLE_STT  |  TTS: $ENABLE_TTS  |  LLM: $ENABLE_LLM" -ForegroundColor White
$ok = Read-Host "  Proceed? [Y/n]"
if ($ok -eq "n") { exit 0 }

$PRIVATE_IP = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -like '172.31.*' -or $_.IPAddress -like '10.*' } |
    Select-Object -First 1).IPAddress
if (-not $PRIVATE_IP) {
    $PRIVATE_IP = (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -ne '127.0.0.1' } |
        Select-Object -First 1).IPAddress
}
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$rngBytes = New-Object byte[] 32
$rng.GetBytes($rngBytes)
$INTERNAL_KEY = [System.Convert]::ToBase64String($rngBytes)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Instance Setup
# ═════════════════════════════════════════════════════════════════════════════
log "Phase 1/3: Instance Setup"

# ── winget packages ──
$pkgs = @(
    @{id="Git.Git";         check={ git --version 2>$null } },
    @{id="Python.Python.3.10"; check={ py -3.10 --version 2>$null } },
    @{id="OpenJS.NodeJS.LTS"; check={ node --version 2>$null } },
    @{id="Gyan.FFmpeg";     check={ ffmpeg -version 2>$null } },
    @{id="aria2.aria2";     check={ aria2c --version 2>$null } }
)
foreach ($pkg in $pkgs) {
    try { & $pkg.check | Out-Null; ok "$($pkg.id) already installed" }
    catch {
        step "Installing $($pkg.id)..."
        winget install --id $pkg.id -e --source winget --accept-package-agreements --accept-source-agreements --silent 2>&1 | Select-Object -Last 2
    }
}
Refresh-Path

# ── PowerShell 7 (install if running on PS5) ──
if ($PSVersionTable.PSVersion.Major -lt 7) {
    if (Get-Command pwsh -ErrorAction SilentlyContinue) {
        err "PowerShell 7 is already installed — relaunch this script with 'pwsh' (not the blue Windows PowerShell 5 window)."
    }
    warn "PowerShell 5 detected — installing PowerShell 7"
    winget install --id Microsoft.PowerShell --source winget --silent
    err "Relaunch this script in PowerShell 7 (search 'PowerShell 7' or run 'pwsh', not the blue Windows PowerShell 5 window) after install completes."
}

# ── GPU check ──
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    ok "GPU: $gpuInfo"
} catch {
    warn "nvidia-smi not found. Install NVIDIA driver before running AI services."
    warn "Download: https://www.nvidia.com/Download/index.aspx"
}

# ── ngrok ──
New-Item -ItemType Directory -Force -Path "C:\ngrok" | Out-Null
if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
    step "Installing ngrok..."
    Invoke-WebRequest "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip" -OutFile "$env:TEMP\ngrok.zip"
    Expand-Archive -Path "$env:TEMP\ngrok.zip" -DestinationPath "C:\ngrok\" -Force
    [System.Environment]::SetEnvironmentVariable('PATH', $env:PATH + ";C:\ngrok", [System.EnvironmentVariableTarget]::Machine)
    Refresh-Path
}
ngrok config add-authtoken $NGROK_TOKEN 2>$null | Out-Null
ok "ngrok ready"

# ── cloudflared ──
New-Item -ItemType Directory -Force -Path "C:\cloudflared" | Out-Null
if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    step "Installing cloudflared..."
    Invoke-WebRequest "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
        -OutFile "C:\cloudflared\cloudflared.exe"
    [System.Environment]::SetEnvironmentVariable('PATH', $env:PATH + ";C:\cloudflared", [System.EnvironmentVariableTarget]::Machine)
    Refresh-Path
}
ok "cloudflared ready"

# ── Clone repo ──
New-Item -ItemType Directory -Force -Path $REPO_DIR | Out-Null
if (-not (Test-Path "$REPO_DIR\.git")) {
    step "Cloning VoicEra repository..."
    git clone -b dev https://github.com/COSS-India/voicera_mono_repository.git $REPO_DIR
}
ok "Repository at $REPO_DIR"

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Application Deploy
# ═════════════════════════════════════════════════════════════════════════════
log "Phase 2/3: Application Deploy"

# ── MongoDB 9.0 nightly ──
$mongoPath = (Get-ChildItem "C:\mongodb9" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1)
if (-not $mongoPath -or -not (Get-Command mongod -ErrorAction SilentlyContinue)) {
    step "Downloading MongoDB 9.0 nightly..."
    $mongoZip = "$env:TEMP\mongodb.zip"
    Invoke-WebRequest "https://downloads.mongodb.org/windows/mongodb-windows-x86_64-latest.zip" -OutFile $mongoZip
    Expand-Archive -Path $mongoZip -DestinationPath "C:\mongodb9\" -Force
    $mongoBin = (Get-ChildItem "C:\mongodb9" -Directory | Select-Object -First 1).FullName + "\bin"
    [System.Environment]::SetEnvironmentVariable('PATH', $env:PATH + ";$mongoBin", [System.EnvironmentVariableTarget]::Machine)
    $env:PATH += ";$mongoBin"
    Refresh-Path
}

# Install mongosh
if (-not (Get-Command mongosh -ErrorAction SilentlyContinue)) {
    winget install MongoDB.Shell --source winget --silent --accept-package-agreements --accept-source-agreements 2>&1 | Select-Object -Last 2
    Refresh-Path
}

New-Item -ItemType Directory -Force -Path "C:\data\mongodb9" | Out-Null
New-Item -ItemType Directory -Force -Path "C:\logs\mongodb" | Out-Null

# Start MongoDB
if (-not (Test-Port 27017)) {
    Start-Process mongod -ArgumentList "--dbpath C:\data\mongodb9 --logpath C:\logs\mongodb\mongod.log --port 27017" -WindowStyle Hidden
}

# Wait for MongoDB to actually accept connections (WiredTiger init can outlast a fixed sleep)
$mongoReady = $false
for ($i = 0; $i -lt 30; $i++) {
    mongosh --quiet --eval "db.runCommand('ping')" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $mongoReady = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $mongoReady) { err "MongoDB did not become ready on port 27017 after 30s" }

# Create admin user — ok if it already exists, but a real failure must not proceed to --auth restart
$createUserOutput = mongosh admin --quiet --eval "db.createUser({user:'admin',pwd:'admin123',roles:[{role:'root',db:'admin'}]})" 2>&1
$userReady = ($LASTEXITCODE -eq 0) -or ($createUserOutput -match "already exists")
if (-not $userReady) { err "Failed to create MongoDB admin user: $createUserOutput" }

# Restart with --auth
Stop-Process -Name mongod -ErrorAction SilentlyContinue; Start-Sleep -Seconds 2
Start-Process mongod -ArgumentList "--dbpath C:\data\mongodb9 --logpath C:\logs\mongodb\mongod.log --port 27017 --auth" -WindowStyle Hidden
Start-Sleep -Seconds 4
ok "MongoDB ready"

# ── MinIO ──
New-Item -ItemType Directory -Force -Path "C:\minio" | Out-Null
New-Item -ItemType Directory -Force -Path "C:\minio-data" | Out-Null
if (-not (Test-Path "C:\minio\minio.exe")) {
    step "Downloading MinIO..."
    Invoke-WebRequest "https://dl.min.io/server/minio/release/windows-amd64/minio.exe" -OutFile "C:\minio\minio.exe"
}
ok "MinIO binary ready"

# ── STT venv ──
if ($ENABLE_STT -eq "yes") {
    $STT_DIR = "$REPO_DIR\ai4bharat_stt_server"
    $sttMarker = "$STT_DIR\venv\.install_complete"
    if (-not (Test-Path $sttMarker)) {
        step "Creating STT venv..."
        Set-Location $STT_DIR
        py -3.10 -m venv venv

        # Compatibility patch for NeMo/pytorch_lightning
        $serverPath = "$STT_DIR\server.py"
        $content = Get-Content $serverPath -Raw
        if ($content -notmatch "pytorch_lightning.loggers") {
            $patch = "import sys`nimport pytorch_lightning.loggers`nsys.modules['pytorch_lightning.loggers'].NeptuneLogger = None`n`n"
            Set-Content -Path $serverPath -Value ($patch + $content)
        }

        & "$STT_DIR\venv\Scripts\pip.exe" install -q --upgrade pip 2>&1 | Select-Object -Last 1
        & "$STT_DIR\venv\Scripts\pip.exe" install -q -r requirements.txt 2>&1 | Select-Object -Last 2
        & "$STT_DIR\venv\Scripts\pip.exe" install -q numba ruamel.yaml scikit-learn tensorboard text-unidecode 2>&1 | Select-Object -Last 1
        # nemo_toolkit installs --no-deps below, so pytorch-lightning (needed by server.py's compat patch) never lands unless installed explicitly
        & "$STT_DIR\venv\Scripts\pip.exe" install -q pytorch-lightning 2>&1 | Select-Object -Last 1
        & "$STT_DIR\venv\Scripts\pip.exe" install -q --no-deps "nemo_toolkit[asr] @ git+https://github.com/AI4Bharat/NeMo.git@nemo-v2" 2>&1 | Select-Object -Last 2
        if ($LASTEXITCODE -eq 0) {
            New-Item -ItemType File -Force -Path $sttMarker | Out-Null
        } else {
            warn "STT venv install did not complete cleanly — will retry on next run"
        }
    }

    # Download STT checkpoint
    New-Item -ItemType Directory -Force -Path "$STT_DIR\checkpoints" | Out-Null
    if (-not (Test-Path "$STT_DIR\checkpoints\indic_conformer.nemo")) {
        step "Downloading STT checkpoint (~2.4 GB)..."
        aria2c -x 16 -s 16 -k 1M `
            "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo" `
            -d "$STT_DIR\checkpoints" -o "indic_conformer.nemo"
    }

    Set-Content -Path "$STT_DIR\.env" -Value @"
PORT=8001
BHILI_ENABLE=no
INDIC_NEMO_PATH=$STT_DIR\checkpoints\indic_conformer.nemo
HF_TOKEN=$HF_TOKEN
"@
    ok "STT ready"
}

# ── TTS venv ──
if ($ENABLE_TTS -eq "yes") {
    $TTS_DIR = "$REPO_DIR\ai4bharat_tts_server"
    $ttsMarker = "$TTS_DIR\venv\.install_complete"
    if (-not (Test-Path $ttsMarker)) {
        step "Creating TTS venv..."
        Set-Location $TTS_DIR
        py -3.10 -m venv venv
        & "$TTS_DIR\venv\Scripts\pip.exe" install -q --upgrade pip 2>&1 | Select-Object -Last 1
        & "$TTS_DIR\venv\Scripts\pip.exe" install -q "flashinfer-python==0.6.7" "flashinfer-cubin==0.6.7" 2>&1 | Select-Object -Last 2
        if ($LASTEXITCODE -ne 0) {
            warn "flashinfer install failed — likely no Windows wheel for this CUDA-dependent package. TTS server needs flashinfer for paged-attention decode; consider running ai4bharat_tts_server under WSL2 (same approach used for vLLM) if native install doesn't work."
        }
        & "$TTS_DIR\venv\Scripts\pip.exe" install -q torch transformers==4.46.1 sentencepiece protobuf scipy websockets python-dotenv numpy 2>&1 | Select-Object -Last 2
        & "$TTS_DIR\venv\Scripts\pip.exe" install -q gdown 2>&1 | Select-Object -Last 1
        if ($LASTEXITCODE -eq 0) {
            New-Item -ItemType File -Force -Path $ttsMarker | Out-Null
        } else {
            warn "TTS venv install did not complete cleanly — will retry on next run"
        }
    }

    # Download TTS checkpoints
    New-Item -ItemType Directory -Force -Path "$TTS_DIR\checkpoints" | Out-Null
    $ckptFiles = Get-ChildItem "$TTS_DIR\checkpoints" -ErrorAction SilentlyContinue | Where-Object { $_.Length -gt 100MB }
    if (-not $ckptFiles) {
        step "Downloading TTS checkpoints from Google Drive..."
        # --fuzzy is not supported together with --folder; drop it
        & "$TTS_DIR\venv\Scripts\python.exe" -m gdown --folder `
            "https://drive.google.com/drive/folders/1qrh56MWXboiBO38gaWEcWhFl0NzlDiaT" `
            -O "$TTS_DIR\checkpoints" 2>&1 | Select-Object -Last 3
        # Flatten nested folder if gdown created one
        $nested = "$TTS_DIR\checkpoints\checkpoints"
        if (Test-Path $nested) {
            Get-ChildItem $nested | Move-Item -Destination "$TTS_DIR\checkpoints\"
            Remove-Item $nested -Force
        }
        $ckptFiles = Get-ChildItem "$TTS_DIR\checkpoints" -ErrorAction SilentlyContinue | Where-Object { $_.Length -gt 100MB }
    }

    Set-Content -Path "$TTS_DIR\.env" -Value @"
CHECKPOINT_PATH_DEFAULT=$TTS_DIR\checkpoints
BHILI_ENABLE=no
PORT=8002
HF_TOKEN=$HF_TOKEN
"@
    if ($ckptFiles) {
        ok "TTS ready"
    } else {
        warn "TTS checkpoint download failed — no checkpoint file found in $TTS_DIR\checkpoints"
    }
}

# ── V2V venv ──
$V2V_DIR = "$REPO_DIR\voice_2_voice_server"
$v2vMarker = "$V2V_DIR\venv\.install_complete"
if (-not (Test-Path $v2vMarker)) {
    step "Creating V2V venv..."
    Set-Location $V2V_DIR
    py -3.10 -m venv venv
    & "$V2V_DIR\venv\Scripts\pip.exe" install -q --upgrade pip 2>&1 | Select-Object -Last 1
    & "$V2V_DIR\venv\Scripts\pip.exe" install -q -r requirements.txt 2>&1 | Select-Object -Last 2
    if ($LASTEXITCODE -eq 0) {
        New-Item -ItemType File -Force -Path $v2vMarker | Out-Null
    } else {
        warn "V2V venv install did not complete cleanly — will retry on next run"
    }
}
ok "V2V venv ready"

# ── Backend venv ──
$BACKEND_DIR = "$REPO_DIR\voicera_backend"
$backendMarker = "$BACKEND_DIR\venv\.install_complete"
if (-not (Test-Path $backendMarker)) {
    step "Creating Backend venv..."
    Set-Location $BACKEND_DIR
    py -3.10 -m venv venv
    & "$BACKEND_DIR\venv\Scripts\pip.exe" install -q --upgrade pip 2>&1 | Select-Object -Last 1
    & "$BACKEND_DIR\venv\Scripts\pip.exe" install -q -r requirements.txt 2>&1 | Select-Object -Last 2
    if ($LASTEXITCODE -eq 0) {
        New-Item -ItemType File -Force -Path $backendMarker | Out-Null
    } else {
        warn "Backend venv install did not complete cleanly — will retry on next run"
    }
}

# Backend .env
$SECRET_KEY = & "$BACKEND_DIR\venv\Scripts\python.exe" -c "import secrets; print(secrets.token_urlsafe(32))"
Set-Content -Path "$BACKEND_DIR\.env" -Value @"
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USER=admin
MONGODB_PASSWORD=admin123
MONGODB_DATABASE=voicera
MONGODB_AUTH_SOURCE=admin
DEBUG=True
SECRET_KEY=$SECRET_KEY
INTERNAL_API_KEY=$INTERNAL_KEY
MAILTRAP_API_TOKEN=placeholder
MAILTRAP_FROM_EMAIL=noreply@voicera.com
MAILTRAP_FROM_NAME=VoicEra
FRONTEND_URL=https://PENDING
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
VOBIZ_API_BASE_URL=https://api.vobiz.in/v1
VOBIZ_AUTH_ID=$VOBIZ_AUTH_ID
VOBIZ_AUTH_TOKEN=$VOBIZ_AUTH_TOKEN
"@
ok "Backend ready"

# ── Frontend npm install ──
$FRONTEND_DIR = "$REPO_DIR\voicera_frontend"

# package.json's "dev" script uses bash-style inline env var syntax, which cmd.exe can't parse — patch it to use cross-env
$pkgJsonPath = "$FRONTEND_DIR\package.json"
$pkgJson = Get-Content $pkgJsonPath -Raw
if ($pkgJson -notmatch "cross-env WATCHPACK_POLLING") {
    $pkgJson = $pkgJson -replace '"dev":\s*"WATCHPACK_POLLING=true next dev --webpack"', '"dev": "cross-env WATCHPACK_POLLING=true next dev --webpack"'
    Set-Content -Path $pkgJsonPath -Value $pkgJson
}

$frontendMarker = "$FRONTEND_DIR\node_modules\.install_complete"
if (-not (Test-Path $frontendMarker)) {
    step "Installing frontend node_modules..."
    Set-Location $FRONTEND_DIR
    npm install --silent 2>&1 | Select-Object -Last 3
    if ($LASTEXITCODE -eq 0) {
        New-Item -ItemType File -Force -Path $frontendMarker | Out-Null
    } else {
        warn "Frontend npm install did not complete cleanly — will retry on next run"
    }
}

# "npm run dev" resolves cross-env from node_modules/.bin — ensure it's present even on reruns where node_modules predates this fix
if (-not (Test-Path "$FRONTEND_DIR\node_modules\cross-env")) {
    & npm install --prefix $FRONTEND_DIR cross-env --save-dev --silent 2>&1 | Select-Object -Last 2
}
Set-Content -Path "$FRONTEND_DIR\.env.local" -Value 'NEXT_PUBLIC_JOHNAIC_SERVER_URL="https://PENDING"'
ok "Frontend ready"

# ── V2V .env ──
Set-Content -Path "$V2V_DIR\.env" -Value @"
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
MINIO_SECURE=false
BHASHINI_API_KEY=PLACEHOLDER
BHASHINI_SOCKET_URL=PLACEHOLDER
AI4BHARAT_STT_URL=http://${PRIVATE_IP}:8001
AI4BHARAT_TTS_URL=ws://${PRIVATE_IP}:8002
OPENAI_API_KEY=$OPENAI_API_KEY
XAI_API_KEY=$XAI_API_KEY
"@
ok "V2V .env written"

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Go Live
# ═════════════════════════════════════════════════════════════════════════════
log "Phase 3/3: Starting Services"

New-Item -ItemType Directory -Force -Path "C:\logs\voicera" | Out-Null

# Helper: launch a service in a new visible PowerShell window
function Start-Service-Window {
    param([string]$Title, [string]$WorkDir, [string]$Command)
    $args = "-NoExit -Command `"Set-Location '$WorkDir'; $Command`""
    Start-Process powershell -ArgumentList $args -WindowStyle Normal
}

# ── MinIO ──
if (-not (Test-Port 9000)) {
    Start-Process -FilePath "C:\minio\minio.exe" `
        -ArgumentList "server C:\minio-data --console-address :9001" `
        -RedirectStandardOutput "C:\logs\voicera\minio.log" -WindowStyle Minimized
    Start-Sleep -Seconds 3
}
ok "MinIO started (port 9000)"

# ── Backend ──
if (-not (Test-Port 8000)) {
    Start-Service-Window "VoicEra Backend" $BACKEND_DIR `
        "$BACKEND_DIR\venv\Scripts\python.exe run.py"
    Start-Sleep -Seconds 8
}
ok "Backend started (port 8000)"

# ── STT ──
if ($ENABLE_STT -eq "yes" -and -not (Test-Port 8001)) {
    Start-Service-Window "VoicEra STT" "$REPO_DIR\ai4bharat_stt_server" `
        ".\venv\Scripts\python.exe server.py"
    ok "STT started (port 8001) — loading model, takes ~2 min"
}

# ── TTS ──
if ($ENABLE_TTS -eq "yes" -and -not (Test-Port 8002)) {
    Start-Service-Window "VoicEra TTS" "$REPO_DIR\ai4bharat_tts_server" `
        ".\venv\Scripts\python.exe server.py"
    ok "TTS started (port 8002)"
}

# ── vLLM via WSL2 ──
if ($ENABLE_LLM -eq "vllm") {
    Start-Process wsl -ArgumentList "-e bash -c 'source /mnt/c/VoicEra/voice_2_voice_server/venv_vllm/bin/activate && python /mnt/c/VoicEra/llm_server/server.py'" -WindowStyle Normal
    ok "vLLM started via WSL2 (port 8003)"
}

# ── Voice2Voice ──
if (-not (Test-Port 7860)) {
    Start-Service-Window "VoicEra V2V" $V2V_DIR `
        "$V2V_DIR\venv\Scripts\python.exe main.py"
    Start-Sleep -Seconds 5
}
ok "V2V started (port 7860)"

# ── Frontend ──
if (-not (Test-Port 3000)) {
    Start-Service-Window "VoicEra Frontend" $FRONTEND_DIR `
        "npm run dev -- --port 3000"
    Start-Sleep -Seconds 8
}
ok "Frontend started (port 3000)"

# ── ngrok ──
if (-not (Test-Port 4040)) {
    Start-Process ngrok -ArgumentList "http 7860" -WindowStyle Minimized
    Start-Sleep -Seconds 5
}
$NGROK_URL = ""
try {
    $tunnels = (Invoke-RestMethod "http://localhost:4040/api/tunnels").tunnels
    $NGROK_URL = ($tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1).public_url
    if (-not $NGROK_URL) { $NGROK_URL = $tunnels[0].public_url }
} catch {}
if ($NGROK_URL) { ok "ngrok started: $NGROK_URL" } else { ok "ngrok started (URL pending)" }

# ── Cloudflare tunnel ──
$CF_LOG = "$env:TEMP\voicera_cf.log"
Remove-Item $CF_LOG -ErrorAction SilentlyContinue
Start-Process cloudflared -ArgumentList "tunnel --url http://localhost:3000 --logfile $CF_LOG" -WindowStyle Minimized
Write-Host "  Waiting for Cloudflare tunnel..." -ForegroundColor DarkGray
$CF_URL = ""
for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Seconds 3
    if (Test-Path $CF_LOG) {
        $CF_URL = Select-String -Path $CF_LOG -Pattern 'https://[^ |]+\.trycloudflare\.com' |
            Select-Object -Last 1 | ForEach-Object { $_.Matches[0].Value }
        if ($CF_URL) { break }
    }
}

# ── Update .env files with real tunnel URLs ──
if ($NGROK_URL) {
    $WS_URL = $NGROK_URL -replace "^https://","wss://" -replace "^http://","ws://"
    (Get-Content "$V2V_DIR\.env") -replace "https://PENDING",$NGROK_URL -replace "wss://PENDING",$WS_URL |
        Set-Content "$V2V_DIR\.env"
    (Get-Content "$FRONTEND_DIR\.env.local") -replace "https://PENDING",$NGROK_URL |
        Set-Content "$FRONTEND_DIR\.env.local"
}
if ($CF_URL) {
    (Get-Content "$BACKEND_DIR\.env") -replace "https://PENDING",$CF_URL |
        Set-Content "$BACKEND_DIR\.env"
}

# ── Wait for services to come up (STT loads 2.4 GB model) ──
Write-Host ""
Write-Host "  Waiting for services to come up (STT loads ~2.4 GB model, allow up to 3 min)..." -ForegroundColor DarkGray
for ($i = 1; $i -le 18; $i++) {
    Start-Sleep -Seconds 10
    $sttOk = $false; $v2vOk = $false; $apiOk = $false
    try {
        $sttHealth = Invoke-RestMethod "http://localhost:8001/health" -TimeoutSec 3 -ErrorAction Stop
        $sttOk = $sttHealth.main_loaded -eq $true
    } catch {}
    try {
        $v2vHealth = Invoke-RestMethod "http://localhost:7860/health" -TimeoutSec 3 -ErrorAction Stop
        $v2vOk = $v2vHealth.status -eq "healthy"
    } catch {}
    try {
        $apiHealth = Invoke-RestMethod "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction Stop
        $apiOk = $true
    } catch {}

    $sttSkip = $ENABLE_STT -ne "yes"
    if (($sttOk -or $sttSkip) -and $v2vOk -and $apiOk) { break }
    Write-Host "  ...${i}0s (STT=$(if($sttSkip){'skip'}elseif($sttOk){'ok'}else{'loading'}) V2V=$(if($v2vOk){'ok'}else{'loading'}) API=$(if($apiOk){'ok'}else{'loading'}))" -ForegroundColor DarkGray
}

# ── Final summary ──
Write-Host ""
Write-Host "  ══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "            VoicEra is Live!" -ForegroundColor Green
Write-Host "  ══════════════════════════════════════════════════════" -ForegroundColor Cyan

if ($NGROK_URL)  { Write-Host "  V2V (ngrok):      $NGROK_URL" -ForegroundColor White }
if ($CF_URL)     { Write-Host "  App (Cloudflare): $CF_URL"    -ForegroundColor White }

Write-Host ""
try {
    $h = Invoke-RestMethod "http://localhost:8001/health" -TimeoutSec 3
    Write-Host "  STT  : $($h.status) | model=$($h.main_loaded)" -ForegroundColor Cyan
} catch { Write-Host "  STT  : loading..." -ForegroundColor DarkGray }

try {
    $h = Invoke-RestMethod "http://localhost:7860/health" -TimeoutSec 3
    Write-Host "  V2V  : $($h.status)" -ForegroundColor Cyan
} catch { Write-Host "  V2V  : loading..." -ForegroundColor DarkGray }

try {
    $h = Invoke-RestMethod "http://localhost:8000/health" -TimeoutSec 3
    Write-Host "  API  : ok" -ForegroundColor Cyan
} catch { Write-Host "  API  : loading..." -ForegroundColor DarkGray }

if (Test-Port 8002) {
    Write-Host "  TTS  : listening :8002" -ForegroundColor Cyan
} else {
    Write-Host "  TTS  : loading..." -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "  NOTE: First login requires email verification bypass." -ForegroundColor Yellow
Write-Host "  Run after signup:" -ForegroundColor Yellow
Write-Host '  mongosh "mongodb://admin:admin123@localhost:27017/voicera?authSource=admin" --quiet --eval "db.users.updateOne({email:''your@email.com''},{$set:{is_verified:true}})"' -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Logs: C:\logs\voicera\" -ForegroundColor DarkGray
Write-Host "  ngrok dashboard: http://localhost:4040" -ForegroundColor DarkGray
Write-Host "  MinIO console:   http://localhost:9001  (minioadmin / minioadmin)" -ForegroundColor DarkGray
Write-Host ""
