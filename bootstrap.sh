#!/usr/bin/env bash
# Alexandria on RunPod — runs automatically when the pod starts (set as the pod's start command).
# Downloads the latest Alexandria code, writes the password hash from the RunPod secret,
# installs/updates ACE-Step 1.5 on the network volume, and starts the studio on port 7870.
set -uo pipefail
W=/workspace
mkdir -p "$W/alexandria/logs"
exec >> "$W/alexandria/logs/bootstrap.log" 2>&1
echo "=== Alexandria bootstrap $(date -u +%FT%TZ) pod=${RUNPOD_POD_ID:-?}"

REPO="${ALEX_REPO:-tdubofficial/alexandria-runpod}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC=/tmp/alexandria-src
if [ -f "$HERE/setup.sh" ] && [ -f "$HERE/alexandria_bridge.py" ]; then
  SRC="$HERE"   # running from an unzipped package
elif ! { rm -rf "$SRC" && mkdir -p "$SRC"; } || ! curl -fsSL "https://codeload.github.com/$REPO/tar.gz/refs/heads/main" | tar xz -C "$SRC" --strip-components=1; then
  echo "Couldn't download $REPO; using the copy already on the volume if there is one."
  [ -f "$W/alexandria/start.sh" ] && exec bash "$W/alexandria/start.sh"
  exit 1
fi

if [ -n "${ALEX_AUTH_JSON:-}" ] && printf '%s' "$ALEX_AUTH_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["hash"] and d["salt"]' 2>/dev/null; then
  printf '%s' "$ALEX_AUTH_JSON" > "$W/alexandria/alexandria_auth.json"
  chmod 600 "$W/alexandria/alexandria_auth.json"
  echo "Password hash loaded from the RunPod secret."
else
  echo "ALEX_AUTH_JSON is missing or invalid; keeping any existing password file."
fi

bash "$SRC/setup.sh" && bash "$W/alexandria/start.sh"
echo "=== bootstrap finished $(date -u +%FT%TZ)"
