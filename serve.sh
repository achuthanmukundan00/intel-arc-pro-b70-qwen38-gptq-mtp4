#!/usr/bin/env bash
# One-command Qwen3.8-27B server for Intel Arc Pro B70.
# Usage: ./serve.sh [--ctx 161000] [--port 8000] [--gpu-mem 0.90] [--dry-run] [--stop] [--compat-flags]
set -euo pipefail

CTX="${CTX:-161000}"
PORT="${PORT:-8000}"
GPU_MEM="${GPU_MEM:-0.90}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}"
CONTAINER="${CONTAINER:-b70-qwen38}"
DRY_RUN=0; DO_STOP=0; COMPAT=0
for a in "$@"; do
  case "$a" in
    --ctx) CTX="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --gpu-mem) GPU_MEM="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --stop) DO_STOP=1; shift;;
    --compat-flags) COMPAT=1; shift;;
    *) echo "unknown flag: $a" >&2; exit 2;;
  esac
done

readonly IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
readonly MODEL_REPO='SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16'
readonly MODEL_REV='9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e'
readonly COOKBOOK_COMMIT='1378950ac875ab6962addd1ddaf44550d34c3103'
readonly COOKBOOK='https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook'
readonly PATCH_MTP_SHA='4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14'
readonly PATCH_BOUNDARY_SHA='41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50'

if [[ "$DO_STOP" == 1 ]]; then docker rm -f "$CONTAINER" 2>/dev/null || true; echo stopped; exit 0; fi

fail() { echo "prereq: $*" >&2; exit 1; }
command -v docker >/dev/null || fail "docker not installed"
docker info >/dev/null 2>&1 || fail "docker daemon unreachable (or no permission)"
[[ -e /dev/dri/renderD128 || -e /dev/dri/renderD129 ]] || fail "no B70 render node (/dev/dri/renderD*)"
RENDER_NODE="$(ls /dev/dri/renderD12[89] 2>/dev/null | head -n1)"
RENDER_GID="$(stat -c '%g' "$RENDER_NODE")"
free -g | awk '/^Mem:/{if ($2 < 16) exit 1}' || fail "need >=16GB host RAM"
avail_kb="$(df -k --output=avail "$HOME" | tail -n1)"
[[ "$avail_kb" -ge 94371840 ]] || fail "need ~90GB free disk"
echo "prereqs ok: node=$RENDER_NODE gid=$RENDER_GID"

PATCH_DIR="$HOME/.cache/b70-qwen38/patches"
mkdir -p "$PATCH_DIR"
for name in patch_mtp_nightly patch_mtp_boundary; do
  f="$PATCH_DIR/$name.py"
  if [[ ! -f "$f" ]]; then
    echo "fetching $name.py @ $COOKBOOK_COMMIT"
    curl -fsSL "$COOKBOOK/$COOKBOOK_COMMIT/patches/$name.py" -o "$f"
  fi
done
echo "$PATCH_MTP_SHA  $PATCH_DIR/patch_mtp_nightly.py" | sha256sum -c - || fail "patch_mtp hash mismatch"
echo "$PATCH_BOUNDARY_SHA  $PATCH_DIR/patch_mtp_boundary.py" | sha256sum -c - || fail "patch_boundary hash mismatch"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "downloading model $MODEL_REPO @ $MODEL_REV (~18GB)..."
  huggingface-cli download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$MODEL_DIR" \
    || fail "model download failed (pip install huggingface_hub?)"
fi
[[ -f "$MODEL_DIR/chat_template.jinja" ]] || fail "model dir incomplete: $MODEL_DIR"

[[ "$DRY_RUN" == 1 ]] && { echo "dry-run ok"; exit 0; }

EXTRA=()
if [[ "$COMPAT" == 1 ]]; then EXTRA+=(--cap-add SYS_PTRACE --security-opt seccomp=unconfined); fi
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" --restart unless-stopped \
  --device /dev/dri:/dev/dri -v /dev/dri:/dev/dri:ro --group-add "$RENDER_GID" \
  --network host --ipc host "${EXTRA[@]}" \
  -v "$MODEL_DIR:/model:ro" \
  -v "$PATCH_DIR/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -v "$PATCH_DIR/patch_mtp_boundary.py:/patch_boundary.py:ro" \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True -e VLLM_LOGGING_LEVEL=INFO \
  --entrypoint bash "$IMAGE" \
  -lc "python /patch_mtp.py && python /patch_boundary.py && exec vllm serve /model \
    --quantization gptq --dtype float16 --max-model-len $CTX \
    --gpu-memory-utilization $GPU_MEM --kv-cache-dtype fp8 --block-size 64 \
    --host 127.0.0.1 --port $PORT --max-num-seqs 1 --max-num-batched-tokens 8192 \
    --async-scheduling --enable-prefix-caching --enable-prompt-tokens-details \
    --served-model-name qwen38 --chat-template /model/chat_template.jinja \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}'"

echo "waiting for startup (several minutes: compile + graphs)..."
for _ in $(seq 1 100); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo READY; break; fi
  sleep 10
done
curl -fsS "http://127.0.0.1:$PORT/v1/models" | head -c 300; echo
PORT="$PORT" ./smoke.sh
