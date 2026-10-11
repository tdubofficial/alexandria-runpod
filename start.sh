#!/usr/bin/env bash
# Alexandria on RunPod — run this each time the pod starts.
# Starts ACE-Step 1.5's API (private, 127.0.0.1:$ACE_PORT) and the password-protected studio on port 7870.
# RunPod images run nginx on several ports (8001 among them), so ACE-Step gets its own port.
set -uo pipefail

W=/workspace
APP="$W/alexandria"
ACE="$W/ACE-Step-1.5"
export UV_CACHE_DIR="$W/.uv/cache" UV_PYTHON_INSTALL_DIR="$W/.uv/python" PATH="$W/.uv/bin:$PATH"
export HF_HOME="$W/hf-cache"
ACE_PORT="${ALEX_ACE_PORT:-18011}"
ACE_URL="http://127.0.0.1:$ACE_PORT"

[ -d "$ACE" ] || { echo "ACE-Step isn't installed yet. Run setup.sh first."; exit 1; }
[ -f "$APP/alexandria_auth.json" ] || { echo "No password set. Run: python3 $APP/alexandria_bridge.py --set-password"; exit 1; }
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null; } || true

mkdir -p "$APP/logs"
if curl -fs "$ACE_URL/health" >/dev/null 2>&1; then
  echo "==> ACE-Step API already running"
else
  echo "==> Starting ACE-Step API (first start downloads the models; this can take a while)"
  (cd "$ACE" && nohup uv run acestep-api --host 127.0.0.1 --port "$ACE_PORT" > "$APP/logs/ace-step.log" 2>&1 &)
  for i in $(seq 1 180); do
    curl -fs "$ACE_URL/health" >/dev/null 2>&1 && break
    sleep 5
    [ $((i % 6)) -eq 0 ] && echo "   still starting… (tail -f $APP/logs/ace-step.log to watch)"
  done
  curl -fs "$ACE_URL/health" >/dev/null 2>&1 \
    && echo "   ACE-Step is up" \
    || echo "   ACE-Step hasn't answered yet; the studio will report it until it does. Check $APP/logs/ace-step.log"
fi

pkill -f "alexandria_bridge.py" 2>/dev/null || true
echo "==> Starting the studio on port 7870"
cd "$APP"
nohup python3 alexandria_bridge.py --host 0.0.0.0 --port 7870 --no-browser --ace-thinking --ace-api "$ACE_URL" --idle-stop-minutes "${ALEX_IDLE_STOP_MINUTES:-30}" --out "$APP/renders" \
  > "$APP/logs/bridge.log" 2>&1 &
sleep 2
cat "$APP/logs/bridge.log"
echo
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "Open:  https://${RUNPOD_POD_ID}-7870.proxy.runpod.net"
else
  echo "Open the pod's port 7870 link from the RunPod console (Connect -> HTTP Service)."
fi
