#!/usr/bin/env bash
# Start Heyo's services natively (no Docker needed): Qdrant + the API server.
# Ollama is expected to run as a systemd service (scripts/setup.sh installs it).
set -euo pipefail
cd "$(dirname "$0")/.."

# Load .env (HF_TOKEN, HEYO_STT_MODEL, ...) into the environment for the server.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# Let CTranslate2 (faster-whisper) use the GPU: it needs libcublas.so.12 (Ollama
# already bundles it) + libcudnn.so.9 (the nvidia-cudnn-cu12 wheel, if installed).
# Harmless when absent — STT just runs on CPU. Build LD_LIBRARY_PATH from both.
_cuda_libs=""
for d in /usr/local/lib/ollama/cuda_v12 \
         "$PWD/.venv"/lib/python*/site-packages/nvidia/cudnn/lib \
         "$PWD/.venv"/lib/python*/site-packages/nvidia/cublas/lib; do
  [ -d "$d" ] && _cuda_libs="$_cuda_libs:$d"
done
[ -n "$_cuda_libs" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}$_cuda_libs"

QDRANT_BIN="${QDRANT_BIN:-$HOME/.local/qdrant/qdrant}"

if ! curl -s --max-time 2 http://localhost:11434/api/version >/dev/null; then
  echo "[run] ollama not responding — start it: sudo systemctl start ollama" >&2
  exit 1
fi

if ! curl -s --max-time 2 http://localhost:6333/healthz >/dev/null; then
  if [ -x "$QDRANT_BIN" ]; then
    echo "[run] starting qdrant (native binary)..."
    mkdir -p .qdrant_data
    QDRANT__STORAGE__STORAGE_PATH="$PWD/.qdrant_data/storage" \
    QDRANT__STORAGE__SNAPSHOTS_PATH="$PWD/.qdrant_data/snapshots" \
      nohup "$QDRANT_BIN" >/tmp/qdrant.log 2>&1 &
    sleep 3
  elif docker info >/dev/null 2>&1; then
    echo "[run] starting qdrant via docker compose..."
    docker compose up -d qdrant
  else
    echo "[run] no qdrant binary at $QDRANT_BIN and docker unavailable." >&2
    echo "[run] install: https://github.com/qdrant/qdrant/releases (musl build)" >&2
    exit 1
  fi
fi

echo "[run] starting Heyo API on http://localhost:8000 (Ctrl-C to stop)"
exec uv run heyo-api
