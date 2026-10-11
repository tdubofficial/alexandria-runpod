#!/usr/bin/env bash
# Alexandria on RunPod — one-time install onto the network volume (/workspace).
# Safe to run again: it updates ACE-Step and keeps your models, renders and password.
set -euo pipefail

W=/workspace
APP="$W/alexandria"
ACE="$W/ACE-Step-1.5"
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -d "$W" ] || { echo "No /workspace found. Attach a network volume when you create the pod."; exit 1; }

echo "==> System packages (git, ffmpeg)"
if command -v apt-get >/dev/null; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git ffmpeg curl ca-certificates >/dev/null
fi

echo "==> uv (Python package manager), kept on the volume"
export UV_INSTALL_DIR="$W/.uv/bin" UV_CACHE_DIR="$W/.uv/cache" UV_PYTHON_INSTALL_DIR="$W/.uv/python"
if [ ! -x "$UV_INSTALL_DIR/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"

echo "==> ACE-Step 1.5"
if [ -d "$ACE/.git" ]; then git -C "$ACE" pull --ff-only || echo "   (couldn't update ACE-Step; keeping the installed version)"
else git clone https://github.com/ACE-Step/ACE-Step-1.5.git "$ACE"; fi
cd "$ACE"
uv sync

# Pick the planner size from the GPU's memory (ACE-Step 1.5's own VRAM table):
# 40 GB+ cards -> 4B planner, 16-40 GB -> 1.7B, smaller -> 0.6B.
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
VRAM_MB=${VRAM_MB:-0}
if   [ "$VRAM_MB" -ge 40000 ]; then LM=acestep-5Hz-lm-4B
elif [ "$VRAM_MB" -ge 16000 ]; then LM=acestep-5Hz-lm-1.7B
else LM=acestep-5Hz-lm-0.6B; fi
echo "==> ACE-Step settings: GPU memory ${VRAM_MB} MB -> planner $LM (edit $ACE/.env to change)"
if [ ! -f "$ACE/.env" ] || grep -q "Written by Alexandria setup.sh" "$ACE/.env"; then
  cat > "$ACE/.env" <<ENV
# Written by Alexandria setup.sh (delete this line to keep your own edits)
ACESTEP_CONFIG_PATH=acestep-v15-turbo
ACESTEP_LM_MODEL_PATH=$LM
ACESTEP_INIT_LLM=true
ACESTEP_CHECKPOINTS_DIR=$W/ace-checkpoints
ENV
fi
mkdir -p "$W/ace-checkpoints"

echo "==> Alexandria studio and bridge"
mkdir -p "$APP/renders"
cp "$HERE/alexandria_bridge.py" "$HERE/alexandria.html" "$HERE/start.sh" "$APP/"
if [ -f "$HERE/alexandria_auth.json" ] && [ ! -f "$APP/alexandria_auth.json" ]; then
  cp "$HERE/alexandria_auth.json" "$APP/"
fi
chmod 600 "$APP/alexandria_auth.json" 2>/dev/null || true
chmod +x "$APP/start.sh"

echo
echo "Setup complete. Start Alexandria with:"
echo "    bash /workspace/alexandria/start.sh"
