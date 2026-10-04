#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_dir="$root/.runtime"
mode="${1:-sql}"
model_health_url="${LUMEN_MODEL_HEALTH_URL:-http://127.0.0.1:8080/health}"
model_start_command="${LUMEN_MODEL_START_COMMAND:-}"
mkdir -p "$state_dir"
exec 9>"$state_dir/launcher.lock"
flock 9

ready() { curl -fsS --max-time 2 "$1" >/dev/null 2>&1; }
wait_for() {
  local url="$1" label="$2" attempts="$3"
  for _ in $(seq 1 "$attempts"); do
    ready "$url" && return 0
    sleep 1
  done
  echo "$label did not become ready. Check $state_dir" >&2
  return 1
}

if ! ready "$model_health_url"; then
  if [[ -z "$model_start_command" ]]; then
    echo "Model endpoint is unavailable. Start it first or set LUMEN_MODEL_START_COMMAND." >&2
    exit 1
  fi
  nohup bash -lc "$model_start_command" >"$state_dir/model.log" 2>&1 &
  echo "$!" >"$state_dir/model.pid"
fi

if ! ready http://127.0.0.1:8000/api/health; then
  (
    cd "$root/backend"
    nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 >"$state_dir/api.log" 2>&1 &
    echo "$!" >"$state_dir/api.pid"
  )
fi

if ! ready http://127.0.0.1:3000/; then
  (
    cd "$root/frontend"
    nohup env NEXT_DIST_DIR=.next-dev npm run dev -- --hostname 127.0.0.1 --port 3000 >"$state_dir/frontend.log" 2>&1 &
    echo "$!" >"$state_dir/frontend.pid"
  )
fi

wait_for http://127.0.0.1:8000/api/health "API" 60
wait_for http://127.0.0.1:3000/ "Frontend" 90
wait_for "$model_health_url" "Model endpoint" 240

case "$mode" in
  chat) url='http://127.0.0.1:3000/?mode=chat' ;;
  sql|ml) url="http://127.0.0.1:3000/?mode=$mode" ;;
  *) echo "usage: $0 {chat|sql|ml}" >&2; exit 2 ;;
esac

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$url" >/dev/null 2>&1 &
else
  echo "Lumen is ready at $url"
fi
