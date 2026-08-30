#!/bin/sh
# Emit the -f flags needed to drive this stack, and nothing else.
#
# There are three things that decide the file list, and none of them are fixed:
# which model fills each slot, whether that model brings its own services, and
# whether this host has an MPS daemon. Anything that starts or stops the stack
# has to agree on all three -- and they did not. `make up` used only the base
# file, which was harmless while MPS lived in it and became silent breakage the
# moment MPS moved to an overlay: on a GPU in Exclusive Process mode the
# containers come up and cannot get a CUDA context.
#
# So the logic lives here once. setup.sh sources it, the Makefile calls it.
#
#   docker compose $(sh model-server/compose-files.sh) --project-directory model-server up -d
#
# Reads model-server/.env when present; falls back to the same defaults Compose
# would use, so it is still correct before setup.sh has ever run.

HERE=$(cd "$(dirname "$0")" && pwd)

if [ -f "$HERE/.env" ]; then
  # Only the keys we need, and only well-formed lines -- this is sourced-ish
  # input, so do not exec it.
  for key in STT_MODEL TTS_MODEL LLM_MODEL GPU_DEVICE_IDS MPS_PIPE_DIR; do
    val=$(grep -E "^${key}=" "$HERE/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"'\r')
    [ -n "$val" ] && eval "$key=\$val"
  done
fi

STT_MODEL=${STT_MODEL:-indic-conformer}
TTS_MODEL=${TTS_MODEL:-indic-parler}
LLM_MODEL=${LLM_MODEL:-}
GPU_DEVICE_IDS=${GPU_DEVICE_IDS:-1}
MPS_PIPE_DIR=${MPS_PIPE_DIR:-/tmp/nvidia-mps-gpu${GPU_DEVICE_IDS}}

FILES="-f $HERE/compose.model-server.yml"

# A model may bring services of its own.
for slot_model in "stt/$STT_MODEL" "tts/$TTS_MODEL" "llm/$LLM_MODEL"; do
  case "$slot_model" in */) continue;; esac
  [ -f "$HERE/$slot_model/compose.extra.yml" ] && \
    FILES="$FILES -f $HERE/$slot_model/compose.extra.yml"
done

# MPS is a property of the host: attach only if a daemon is really there.
if [ -e "$MPS_PIPE_DIR/control" ] || pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; then
  FILES="$FILES -f $HERE/compose.mps.yml"
  for slot_model in "stt/$STT_MODEL" "tts/$TTS_MODEL" "llm/$LLM_MODEL"; do
    case "$slot_model" in */) continue;; esac
    [ -f "$HERE/$slot_model/compose.mps.yml" ] && \
      FILES="$FILES -f $HERE/$slot_model/compose.mps.yml"
  done
fi

echo "$FILES"
