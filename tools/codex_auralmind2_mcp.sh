#!/usr/bin/env bash
set -euo pipefail

ROOT="${AURALMIND2_REPO:-/mnt/c/Users/goku/Documents/AuralMind2}"
FALLBACK_URL="${AURALMIND2_HTTP_FALLBACK_URL:-http://127.0.0.1:8080/mcp}"

to_python_target_path() {
  local python_bin="$1"
  local target_path="$2"

  if [[ "$python_bin" == *.exe ]]; then
    if command -v wslpath >/dev/null 2>&1; then
      wslpath -w "$target_path"
      return 0
    fi
    return 1
  fi

  printf '%s\n' "$target_path"
}

try_local() {
  local python_bin="$1"
  local server_target
  local bootstrap_code

  if [ -z "$python_bin" ] || [ ! -x "$python_bin" ]; then
    return 1
  fi

  if ! "$python_bin" -c "import fastmcp, mcp, dotenv" >/dev/null 2>&1; then
    return 1
  fi

  if ! server_target="$(to_python_target_path "$python_bin" "$ROOT/server.py")"; then
    return 1
  fi

  if [[ "$python_bin" == *.exe ]]; then
    bootstrap_code="import os, runpy; os.environ['ACTIVE_TRANSPORT']='stdio'; runpy.run_path(r'''$server_target''', run_name='__main__')"
    exec "$python_bin" -c "$bootstrap_code"
  fi

  export ACTIVE_TRANSPORT=stdio
  exec "$python_bin" "$server_target"
}

if [ -d "$ROOT" ] && [ -f "$ROOT/server.py" ]; then
  cd "$ROOT"
  try_local "${AURALMIND2_LOCAL_PYTHON:-}" || true
  try_local "$ROOT/.venv/Scripts/python.exe" || true
  try_local "$ROOT/.venv/bin/python" || true

  if command -v python3 >/dev/null 2>&1 && python3 -c "import fastmcp, mcp, dotenv" >/dev/null 2>&1; then
    export ACTIVE_TRANSPORT=stdio
    exec python3 "$ROOT/server.py"
  fi
fi

if command -v npx >/dev/null 2>&1; then
  exec npx -y mcp-remote "$FALLBACK_URL"
fi

printf 'AuralMind2 Codex launcher could not start local stdio or HTTP fallback.\n' >&2
exit 1
