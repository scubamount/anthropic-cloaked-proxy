#!/usr/bin/env bash
# Two-proxy chain restart: soul (8319) → cloaked (8318) → Anthropic
# v2: proactive Claude OAuth refresh, bounded health checks, clearer failures.

set -u

PROXY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOAKED_SCRIPT="$PROXY_DIR/cloaked-proxy.py"
SOUL_SCRIPT="$PROXY_DIR/soul-proxy.py"
CLOAKED_PORT=8318
SOUL_PORT=8319
PROXY_LOG="/tmp/proxy.log"
SOUL_LOG="/tmp/soul.log"
CRED_FILE="$HOME/.claude/.credentials.json"
REFRESH_MARGIN_MINUTES=10

log() { printf '[start_both] %s %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { log "ERROR: $*"; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

find_claude() {
  if [[ -n "${CLAUDE_BIN:-}" && -x "${CLAUDE_BIN:-}" ]]; then
    printf '%s\n' "$CLAUDE_BIN"
    return 0
  fi
  if command -v claude >/dev/null 2>&1; then
    command -v claude
    return 0
  fi
  if [[ -x "$HOME/.local/bin/claude" ]]; then
    printf '%s\n' "$HOME/.local/bin/claude"
    return 0
  fi
  return 1
}

port_listening() {
  local port="$1"
  # Linux has `ss` (iproute2); macOS has only `lsof`/`netstat`. Prefer `ss`
  # when present, fall back to `lsof` (the macOS default path).
  if command -v ss >/dev/null 2>&1; then
    [[ -n "$(ss -H -ltn "( sport = :$port )" 2>/dev/null || true)" ]]
  else
    [[ -n "$(lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)" ]]
  fi
}

wait_port() {
  local port="$1"
  local name="$2"
  local i
  for i in {1..30}; do
    if port_listening "$port"; then
      log "$name listening on :$port"
      return 0
    fi
    sleep 0.5
  done
  return 1
}

stop_proxies() {
  log "killing old proxies..."
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${SOUL_PORT}/tcp" >/dev/null 2>&1 || true
    fuser -k "${CLOAKED_PORT}/tcp" >/dev/null 2>&1 || true
  else
    pkill -f "$SOUL_SCRIPT --port ${SOUL_PORT}" >/dev/null 2>&1 || true
    pkill -f "$CLOAKED_SCRIPT --port ${CLOAKED_PORT}" >/dev/null 2>&1 || true
  fi
  sleep 1
}

token_minutes_left() {
  python3 - "$CRED_FILE" <<'PY'
import json, sys, time
path = sys.argv[1]
try:
    data = json.load(open(path))
    oauth = data.get('claudeAiOauth') or data.get('claude_ai_oauth') or {}
    exp = int(oauth.get('expiresAt') or oauth.get('expires_at') or 0)
    if not exp:
        print('UNKNOWN')
    else:
        print(int((exp - int(time.time() * 1000)) / 1000 / 60))
except Exception as exc:
    print('ERROR:' + str(exc))
PY
}

refresh_claude_token() {
  local reason="$1"
  log "refreshing Claude OAuth via Claude Code (${reason})..."
  python3 - "$CLAUDE_BIN_RESOLVED" <<'PY'
import os, subprocess, sys
claude = sys.argv[1]
model = os.environ.get('CLOAKED_REFRESH_MODEL', 'opus-4-8')
cmd = [
    claude,
    '-p',
    '--model', model,
    '--output-format', 'json',
    '--no-session-persistence',
    'Reply with exactly OK.',
]
try:
    result = subprocess.run(cmd, cwd=None, capture_output=True, text=True, timeout=150)
except subprocess.TimeoutExpired:
    print('Claude refresh timed out after 150s', file=sys.stderr)
    sys.exit(124)
if result.returncode != 0:
    print((result.stderr or result.stdout or '').strip()[:1000], file=sys.stderr)
    sys.exit(result.returncode)
print('Claude refresh command OK')
PY
}

check_auth_status() {
  "$CLAUDE_BIN_RESOLVED" auth status >/tmp/claude-auth-status.json 2>/tmp/claude-auth-status.err
}

maybe_refresh_before_start() {
  [[ -f "$CRED_FILE" ]] || die "credentials file not found: $CRED_FILE"

  if ! check_auth_status; then
    log "claude auth status failed; trying refresh once"
    refresh_claude_token "auth status failed" || {
      log "claude auth stderr: $(tr '\n' ' ' </tmp/claude-auth-status.err 2>/dev/null | cut -c1-500)"
      die "Claude OAuth refresh failed. Run 'claude auth login --claudeai' interactively."
    }
  fi

  local mins
  mins="$(token_minutes_left)"
  case "$mins" in
    ERROR:*) die "cannot parse Claude credentials: ${mins#ERROR:}" ;;
    UNKNOWN) log "token expiry unknown; continuing, proxy can retry on 401" ;;
    *)
      log "Claude OAuth token expires in ~${mins} minutes"
      if [[ "$mins" =~ ^-?[0-9]+$ ]] && (( mins < REFRESH_MARGIN_MINUTES )); then
        refresh_claude_token "expires in ${mins}m" || die "Claude OAuth refresh failed"
        mins="$(token_minutes_left)"
        log "after refresh, token expires in ~${mins} minutes"
      fi
      ;;
  esac
}

start_proxies() {
  : > "$PROXY_LOG"
  : > "$SOUL_LOG"

  log "starting cloaked proxy :${CLOAKED_PORT}..."
  nohup python3 "$CLOAKED_SCRIPT" --port "$CLOAKED_PORT" > "$PROXY_LOG" 2>&1 &
  CLOAKED_PID=$!
  wait_port "$CLOAKED_PORT" "cloaked proxy" || {
    log "cloaked proxy failed to listen; recent log follows"
    tail -40 "$PROXY_LOG" || true
    return 1
  }
  if ! kill -0 "$CLOAKED_PID" >/dev/null 2>&1; then
    log "cloaked proxy process exited; recent log follows"
    tail -40 "$PROXY_LOG" || true
    return 1
  fi

  log "starting soul proxy :${SOUL_PORT}..."
  nohup python3 "$SOUL_SCRIPT" --port "$SOUL_PORT" > "$SOUL_LOG" 2>&1 &
  SOUL_PID=$!
  wait_port "$SOUL_PORT" "soul proxy" || {
    log "soul proxy failed to listen; recent log follows"
    tail -40 "$SOUL_LOG" || true
    return 1
  }
  if ! kill -0 "$SOUL_PID" >/dev/null 2>&1; then
    log "soul proxy process exited; recent log follows"
    tail -40 "$SOUL_LOG" || true
    return 1
  fi
}

HEALTH_STATUS=0
HEALTH_BODY=""
health_check() {
  local out first rest
  out="$(python3 - <<'PY'
import http.client, json, time
payload = {
    'model': 'claude-opus-4-8',
    'max_tokens': 8,
    'messages': [{'role': 'user', 'content': 'Reply with exactly OK.'}],
    'stream': False,
}
conn = http.client.HTTPConnection('127.0.0.1', 8319, timeout=90)
try:
    t = time.time()
    conn.request('POST', '/v1/messages', body=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    resp = conn.getresponse()
    body = resp.read(2000).decode(errors='replace').replace('\n', ' ')
    print(f'HTTP_STATUS={resp.status};ELAPSED={time.time()-t:.1f}')
    print(body)
except Exception as exc:
    print('HTTP_STATUS=0;ELAPSED=0')
    print(f'{type(exc).__name__}: {exc}')
finally:
    conn.close()
PY
)"
  first="${out%%$'\n'*}"
  rest="${out#*$'\n'}"
  HEALTH_STATUS="${first#HTTP_STATUS=}"
  HEALTH_STATUS="${HEALTH_STATUS%%;*}"
  HEALTH_BODY="$rest"
  log "proxy health status=${HEALTH_STATUS} ${first#*;}"
  [[ "$HEALTH_STATUS" == "200" ]]
}

main() {
  need_cmd python3
  CLAUDE_BIN_RESOLVED="$(find_claude)" || die "Claude Code CLI not found"
  export CLAUDE_BIN_RESOLVED
  log "using Claude CLI: $CLAUDE_BIN_RESOLVED"

  maybe_refresh_before_start
  stop_proxies
  start_proxies || die "proxy startup failed"

  if ! health_check; then
    if [[ "$HEALTH_STATUS" == "401" ]] || [[ "$HEALTH_BODY" == *"authentication_error"* ]]; then
      log "health check shows auth failure; refreshing token and restarting once"
      refresh_claude_token "health check auth failure" || die "Claude OAuth refresh failed after health 401"
      stop_proxies
      start_proxies || die "proxy restart after refresh failed"
      health_check || die "proxy health failed after refresh: status=${HEALTH_STATUS} body=${HEALTH_BODY:0:500}"
    else
      die "proxy health failed: status=${HEALTH_STATUS} body=${HEALTH_BODY:0:500}"
    fi
  fi

  log "✓ Both proxies UP and authenticated"
  log "logs: $PROXY_LOG $SOUL_LOG"
}

main "$@"
