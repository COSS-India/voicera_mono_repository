#!/usr/bin/env bash
# Prove this deployment, end to end. Run it after setup, and after any change to the engine.
#
# The gates are ordered so the cheap ones fail first: weights, then conversion, then streaming
# behaviour, then the failure path. Each prints PASS or FAIL and the script exits non-zero if
# any of them fails.
#
#   ./verify.sh              # everything except the conversion (assumes artifacts/ exists)
#   ./verify.sh --convert    # also re-run the HF -> NeMo conversion first (slow, ~4 GB write)
#   ./verify.sh --quick      # streaming gates only; skips anything that loads a second model
set -uo pipefail
cd "$(dirname "$0")"

CONVERT=0; QUICK=0
for a in "$@"; do
  case "$a" in
    --convert) CONVERT=1 ;;
    --quick)   QUICK=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

IMAGE=${IMAGE:-core-asr:latest}
MODELS=${MODELS:-$PWD/models/core}
CKPT=/artifacts/indic_transcribe_core.nemo
FAILED=()

# Two ways to run a container here, and the difference matters:
#   GPU_RUN  loads its own copy of the model -- isolated, needs --gpus, no server required.
#   NET_RUN  uses the HOST network to reach the published port. Deliberately not
#            `--network container:core-asr`: sharing the server's namespace means a server
#            restart destroys it and the client loses networking for good, turning every
#            later check into a spurious failure.
GPU_RUN=(docker run --rm --gpus all
         -v "$MODELS:/models/core:ro" -v "$PWD/artifacts:/artifacts"
         -v "$PWD/results:/results" -v "$PWD/corpus:/corpus:ro"
         -v "$PWD/tools:/app/tools:ro" -v "$PWD/bench:/app/bench:ro" "$IMAGE")
NET_RUN=(docker run --rm --network host
         -v "$PWD/results:/results" -v "$PWD/corpus:/corpus:ro"
         -v "$PWD/bench:/app/bench:ro" "$IMAGE")

step() {                       # step <name> <command...>
  local name=$1; shift
  printf '\n\033[1m== %s\033[0m\n' "$name"
  if "$@"; then
    printf '\033[32mPASS\033[0m %s\n' "$name"
  else
    printf '\033[31mFAIL\033[0m %s\n' "$name"
    FAILED+=("$name")
  fi
}

# ---------------------------------------------------------------------------- preconditions
printf '\033[1m== preconditions\033[0m\n'
[ -d "$MODELS" ] || { echo "no checkpoint at $MODELS — see SETUP.md" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "no image $IMAGE — docker compose build" >&2; exit 2; }
SERVER_UP=0
curl -sf localhost:"${CORE_HOST_PORT:-9002}"/health >/dev/null 2>&1 && SERVER_UP=1
echo "checkpoint: $MODELS"
echo "image:      $IMAGE"
echo "server:     $([ $SERVER_UP = 1 ] && echo 'up' || echo 'DOWN — streaming gates will be skipped')"

# ---------------------------------------------------------------------------- weights
if [ $QUICK = 0 ]; then
  # The oracle also asserts that the checkpoint's _init_weights is inert. That check is not
  # ceremony: an initialising body silently randomises the weights AFTER a load that reports
  # no missing, unexpected or mismatched keys.
  step "oracle: HF weights load and transcribe" \
    "${GPU_RUN[@]}" python /app/tools/transcribe_hf.py --verify-only \
      --model-dir /models/core --lang hi

  if [ $CONVERT = 1 ]; then
    step "convert: HF -> NeMo" \
      "${GPU_RUN[@]}" python /app/tools/hf_to_nemo.py --hf-dir /models/core --out "$CKPT"
  fi

  # The conversion gate compares TOKEN IDS, not text: the HF port's decode() ends in .strip()
  # while NeMo keeps the leading SentencePiece space, so identical models differ by one U+0020.
  CLIP=$(ls corpus/hi/medium/*.wav 2>/dev/null | head -1)
  if [ -n "${CLIP:-}" ]; then
    step "gate: converted checkpoint is byte-identical to the oracle" \
      "${GPU_RUN[@]}" sh -c "
        python /app/tools/transcribe_hf.py --model-dir /models/core --lang hi \
          --out /results/_verify_oracle.json /corpus/hi/medium/$(basename "$CLIP") >/dev/null &&
        IDS=\$(python -c \"import json;print(json.dumps(json.load(open('/results/_verify_oracle.json'))['results'][0]['token_ids']))\") &&
        python /app/tools/verify_nemo.py --ckpt $CKPT \
          --audio /corpus/hi/medium/$(basename "$CLIP") --lang hi --expect-ids \"\$IDS\""
  else
    echo "SKIP gate: no corpus clip (tools/make_corpus.py --langs hi)"
  fi
fi

# ---------------------------------------------------------------------------- streaming
if [ $SERVER_UP = 1 ]; then
  # Long-form: the regression guard for the stall that decoder-state rotation exists to prevent.
  step "streaming: long-form does not stall" \
    "${NET_RUN[@]}" python /app/bench/test_longform.py --url ws://localhost:9002/v1/asr/ws

  # Turns: the stream must survive a pause instead of closing on it.
  step "streaming: turn rollover (manual and auto)" \
    "${NET_RUN[@]}" python /app/bench/test_turns.py --url ws://localhost:9002/v1/asr/ws

  # Gap-free speech: natural pauses let the soft rotation trigger fire early and hide this case.
  step "streaming: 75 s of gap-free speech leaves the service healthy" \
    "${NET_RUN[@]}" python /app/bench/test_continuous.py --url ws://localhost:9002/v1/asr/ws
else
  echo "SKIP streaming gates — server not reachable (docker compose up -d)"
fi

# ---------------------------------------------------------------------------- failure path
# Injects the failure rather than waiting for the intermittent bug: a fatal CUDA fault must
# take the process down (exit 70) so compose replaces it, and /health must stop lying.
step "failure path: fatal CUDA error exits instead of spinning" \
  "${GPU_RUN[@]}" python /app/bench/test_fatal_path.py

# ---------------------------------------------------------------------------- verdict
rm -f results/_verify_oracle.json
printf '\n\033[1m== verdict\033[0m\n'
if [ ${#FAILED[@]} -eq 0 ]; then
  printf '\033[32mALL GATES PASS\033[0m\n'; exit 0
fi
printf '\033[31m%d FAILED:\033[0m %s\n' "${#FAILED[@]}" "${FAILED[*]}"; exit 1
