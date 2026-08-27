#!/bin/bash
# Fetch this model's weights into its own folder.
#
# setup.sh runs <slot>/<model>/fetch.sh if it exists, so a model that needs
# weights brings its own download step and setup.sh needs to know nothing about
# it. Safe to re-run: it skips work already done.
#
# Note: the checkpoint comes from Drive, but the tokenizer and T5 encoder are
# pulled from the gated ai4bharat/indic-parler-tts repo when the container
# starts -- so this finishing cleanly is not the same as the model being able
# to run. That needs HF_TOKEN, or the shared-cache overlay.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT_DIR="$HERE/checkpoints"
FOLDER="https://drive.google.com/drive/folders/1qrh56MWXboiBO38gaWEcWhFl0NzlDiaT"

if [ -f "$CKPT_DIR/model_step_ref.pt" ]; then
  echo "  Parler checkpoints already present"
  exit 0
fi

# gdown lives in a throwaway venv so it never lands in a model's runtime image.
DL_VENV="${DL_VENV:-$HOME/.voicera_downloader}"
PY="${PY312:-python3}"
[ -d "$DL_VENV" ] || "$PY" -m venv "$DL_VENV"
"$DL_VENV/bin/pip" install -q gdown 2>&1 | tail -1

mkdir -p "$CKPT_DIR"
echo "  Downloading Parler checkpoints"
"$DL_VENV/bin/python3" -m gdown --folder "$FOLDER" -O "$CKPT_DIR/" 2>&1 | tail -3

# gdown nests the folder inside itself; flatten so the paths match the Dockerfile.
if [ -d "$CKPT_DIR/checkpoints" ]; then
  mv "$CKPT_DIR/checkpoints/"* "$CKPT_DIR/"
  rmdir "$CKPT_DIR/checkpoints" 2>/dev/null || true
fi
