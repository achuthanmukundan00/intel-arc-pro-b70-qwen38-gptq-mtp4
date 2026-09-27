#!/usr/bin/env bash
# Health + models + tiny generation smoke test. Usage: [PORT=8000] ./smoke.sh
set -euo pipefail
PORT="${PORT:-8000}"
curl -fsS "http://127.0.0.1:$PORT/health" && echo " HEALTH OK"
curl -fsS "http://127.0.0.1:$PORT/v1/models" | grep -o '"id":"[^"]*"' | head -n 3
curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen38","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":32,"temperature":0,"stream":false}' \
  | python3 -c "import json,sys; u=json.load(sys.stdin); print('usage:', u.get('usage')); print('finish:', u['choices'][0].get('finish_reason'))"
echo SMOKE_OK
