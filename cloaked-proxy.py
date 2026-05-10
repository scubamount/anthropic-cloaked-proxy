#!/usr/bin/env python3
"""Hermes → Anthropic OAuth Cloaking Proxy (v20 — namespaced tool cloak, no collisions)."""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

API = "https://api.anthropic.com"
CC_SYS = "You are Claude Code, Anthropic's official CLI for Claude."
CC_H = {
    "User-Agent": "claude-cli/2.1.133 (external, cli)",
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "oauth-2025-04-20,interleaved-thinking-2025-05-14,token-counting-2024-11-01",
    "x-app": "cli",
}
UPSTREAM_TIMEOUT = int(os.environ.get("CLOAKED_PROXY_UPSTREAM_TIMEOUT", "180"))
REFRESH_TIMEOUT = int(os.environ.get("CLOAKED_PROXY_REFRESH_TIMEOUT", "120"))
EXPIRY_REFRESH_MARGIN_SEC = int(os.environ.get("CLOAKED_PROXY_EXPIRY_MARGIN_SEC", "600"))

# Semantic grouping: 28 Hermes tools → 14 CC names
_MAPPING = [
    ("Bash",         r"^(terminal)$"),
    ("Bash",         r"^(execute_code)$"),
    ("Read",         r"^(read_file)$"),
    ("Write",        r"^(write_file)$"),
    ("Edit",         r"^(patch)$"),
    ("Grep",         r"^(search_files)$"),
    ("Task",         r"^(delegate_task)$"),
    ("TodoWrite",    r"^(todo)$"),
    ("NotebookEdit", r"^(clarify)$"),
    ("WebFetch",     r"^(web_extract)$"),
    ("WebSearch",    r"^(web_search)$"),
    ("WebSearch",    r"^(viking_search)$"),
    ("WebFetch",     r"^(viking_read)$"),
    ("TodoWrite",    r"^(viking_browse)$"),
    ("TodoWrite",    r"^(viking_remember)$"),
    ("WebFetch",     r"^(viking_add_resource)$"),
    ("Glob",         r"^(browser_)"),
    ("Glob",         r"^(vision_analyze)$"),
    ("Glob",         r"^(image_generate)$"),
    ("Glob",         r"^(text_to_speech)$"),
    ("BashOutput",   r"^(process)$"),
    ("KillShell",    r"^(cronjob)$"),
    ("Skill",        r"^(memory)$"),
    ("Skill",        r"^(session_search)$"),
    ("Skill",        r"^(send_message)$"),
    ("Skill",        r"^(skill_view)$"),
    ("Skill",        r"^(skill_manage)$"),
    ("Skill",        r"^(skills_list)$"),
    ("Skill",        r"^(mcp_)"),
]

CC_DESC = {
    "Bash":         "Execute shell commands and Python scripts (hermes_tools)",
    "Read":         "Read text files with line numbers, offset and limit controls",
    "Write":        "Create or overwrite files, auto-creating parent directories",
    "Edit":         "Apply targeted find-and-replace patches to files",
    "Grep":         "Search file contents with regex or find files by glob pattern",
    "Glob":         "Browser automation, vision analysis, image generation, and text-to-speech",
    "Task":         "Spawn subagents for parallel isolated work on complex tasks",
    "TodoWrite":    "Manage task list, browse knowledge base, store persistent facts",
    "WebFetch":     "Extract web pages and PDFs to markdown, read knowledge base, add resources",
    "WebSearch":    "Search the web and OpenViking knowledge base for information",
    "NotebookEdit": "Ask user clarifying questions, multiple choice or open-ended",
    "BashOutput":   "Manage background processes: poll, wait, kill, send stdin",
    "KillShell":    "Schedule recurring cron jobs with flexible delivery targets",
    "Skill":        "Manage skills, persistent memory, session search, messaging, and MCP tools",
}

# Model context limits (from platform.claude.com/docs/en/about-claude/models/overview)
MODEL_CONTEXT = {
    "claude-opus-4-7": 1000000,
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 1000000,
    "claude-sonnet-4-20250514": 200000,
    "claude-sonnet-4-5-20250929": 200000,
    "claude-haiku-4-5": 200000,
    "claude-haiku-4-5-20251001": 200000,
}

_CC_FOR_HS = {}

# --- v20: namespaced cloak ---------------------------------------------------
# Each Hermes tool gets a unique cloaked name of the form
# "<CC_NAME>__<hs_name>" so the upstream model sees the full toolset (no
# silent collisions) while every name still starts with one of the 14 CC
# tool names (preserves the cosmetic disguise).
_NAMESPACE_SEP = "__"
_CLOAKED_NAME_CACHE: dict = {}
_UNCLOAK_CACHE: dict = {}


def _validate_hs_name(hs_name: str) -> str:
    if _NAMESPACE_SEP in hs_name:
        raise ValueError(
            f"hs_name {hs_name!r} contains reserved separator {_NAMESPACE_SEP!r}"
        )
    return hs_name


def _cloaked_tool_name(hs_name: str) -> str:
    if hs_name in _CLOAKED_NAME_CACHE:
        return _CLOAKED_NAME_CACHE[hs_name]
    _validate_hs_name(hs_name)
    cc = _cc_for(hs_name)
    cloaked = f"{cc}{_NAMESPACE_SEP}{hs_name}"
    _CLOAKED_NAME_CACHE[hs_name] = cloaked
    _UNCLOAK_CACHE[cloaked] = hs_name
    return cloaked


def _uncloak_tool_name(cloaked: str) -> str:
    if cloaked in _UNCLOAK_CACHE:
        return _UNCLOAK_CACHE[cloaked]
    if _NAMESPACE_SEP in cloaked:
        prefix, _sep, hs = cloaked.partition(_NAMESPACE_SEP)
        if hs:
            _UNCLOAK_CACHE[cloaked] = hs
            _CLOAKED_NAME_CACHE.setdefault(hs, cloaked)
            return hs
    return cloaked


def log(msg: str) -> None:
    print(f"[cloaked-proxy v20] {msg}", file=sys.stderr, flush=True)


def _cc_for(hs_name: str) -> str:
    if hs_name in _CC_FOR_HS:
        return _CC_FOR_HS[hs_name]
    clean = hs_name.removeprefix("mcp_")
    for cc, pat in _MAPPING:
        if re.match(pat, clean):
            _CC_FOR_HS[hs_name] = cc
            return cc
    _CC_FOR_HS[hs_name] = "Skill"
    return "Skill"


CRED_FILE = Path.home() / ".claude" / ".credentials.json"
if not CRED_FILE.exists():
    CRED_FILE = Path.home() / ".claude.json"


class TokenManager:
    """Thread-safe Claude OAuth token loader/refresher.

    Claude Code owns the OAuth refresh flow. This proxy never attempts to mint
    tokens directly; it pokes Claude Code once, reloads the credentials file,
    and retries the failed Anthropic request once.
    """

    _state_lock = threading.RLock()
    _refresh_lock = threading.Lock()
    _token = None
    _expires_at_ms = 0
    _cred_mtime = 0.0
    _last_refresh_attempt = 0.0

    @classmethod
    def _claude_bin(cls) -> str:
        configured = os.environ.get("CLAUDE_BIN")
        if configured:
            return configured
        found = shutil.which("claude")
        if found:
            return found
        return str(Path.home() / ".local" / "bin" / "claude")

    @classmethod
    def _read_credentials(cls) -> tuple[str, int, float]:
        now_ms = int(time.time() * 1000)

        # Collect candidate tokens from all sources with their expiry.
        candidates: list[tuple[str, int]] = [] # (token, expires_at_ms)

        # Source 1: credentials file (may not exist — Claude Code /login can
        # delete it and move to Keychain-only storage)
        f_token, f_expires = "", 0
        if CRED_FILE.exists():
            try:
                data = json.loads(CRED_FILE.read_text())
                oauth = data.get("claudeAiOauth") or data.get("claude_ai_oauth") or {}
                f_token = oauth.get("accessToken") or oauth.get("access_token") or ""
                f_expires = int(oauth.get("expiresAt") or oauth.get("expires_at") or 0)
                if f_token:
                    candidates.append((f_token, f_expires))
            except Exception as exc:
                log(f"credential file read failed: {type(exc).__name__}: {exc}")

        # Source 2: macOS Keychain
        k_token, k_expires = "", 0
        if sys.platform == "darwin":
            try:
                result = subprocess.run(
                    ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                keychain_data = json.loads(result.stdout)
                k_oauth = keychain_data.get("claudeAiOauth") or keychain_data.get("claude_ai_oauth") or {}
                k_token = k_oauth.get("accessToken") or k_oauth.get("access_token") or ""
                k_expires = int(k_oauth.get("expiresAt") or k_oauth.get("expires_at") or 0)
                if k_token:
                    candidates.append((k_token, k_expires))
            except Exception:
                pass # Keychain unavailable is non-fatal

        if not candidates:
            raise RuntimeError(
                f"no OAuth token found in {CRED_FILE} or macOS Keychain — "
                f"run `claude /login` to refresh credentials"
            )

        # Prefer the token with the latest future expiry.
        # This handles the case where one source has a stale expired token
        # while the other has a freshly refreshed one.
        def _score(tok_exp: tuple[str, int]) -> float:
            tok, exp = tok_exp
            if not exp:
                return 0.0
            remaining = (exp - now_ms) / 1000 # seconds left
            return remaining

        best_token, best_expires = max(candidates, key=_score)
        if not best_token:
            raise RuntimeError(
                f"no valid OAuth token found in {CRED_FILE} or macOS Keychain"
            )

        src_label = "file" if best_token == f_token else "keychain"
        remaining_min = int((best_expires / 1000) - time.time()) // 60 if best_expires else 0
        if len(candidates) > 1:
            log(f"credential sources: file={int((f_expires/1000)-time.time())//60 if f_expires else 'N/A'}m, "
                f"keychain={int((k_expires/1000)-time.time())//60 if k_expires else 'N/A'}m → "
                f"using {src_label} ({remaining_min}m)")

        cred_mtime = CRED_FILE.stat().st_mtime if CRED_FILE.exists() else 0.0
        return best_token, best_expires, cred_mtime

    @classmethod
    def _load_locked(cls, force: bool = False) -> str:
        file_exists = CRED_FILE.exists()
        mtime = CRED_FILE.stat().st_mtime if file_exists else 0.0
        # Reload when forced, first load, file changed, OR file deleted
        # (mtime==0 and _cred_mtime!=0 means file was removed — Keychain-only now)
        file_vanished = (not file_exists) and (cls._cred_mtime != 0.0)
        if force or cls._token is None or mtime != cls._cred_mtime or file_vanished:
            cls._token, cls._expires_at_ms, cls._cred_mtime = cls._read_credentials()
            log(f"loaded OAuth token sha12={cls.token_sha12()} {cls.expiry_summary()}")
        return cls._token

    @classmethod
    def token_sha12(cls) -> str:
        import hashlib
        if not cls._token:
            return "none"
        return hashlib.sha256(cls._token.encode()).hexdigest()[:12]

    @classmethod
    def expiry_summary(cls) -> str:
        if not cls._expires_at_ms:
            return "expires=unknown"
        seconds_left = int((cls._expires_at_ms / 1000) - time.time())
        return f"expires_in={seconds_left // 60}m"

    @classmethod
    def _expires_soon_locked(cls) -> bool:
        if not cls._expires_at_ms:
            return False
        return (cls._expires_at_ms / 1000) - time.time() < EXPIRY_REFRESH_MARGIN_SEC

    @classmethod
    def get_token(cls) -> str:
        with cls._state_lock:
            token = cls._load_locked(force=False)
            if cls._expires_soon_locked():
                log(f"OAuth token {cls.expiry_summary()}; refreshing before request")
                cls.refresh("expires soon")
                token = cls._load_locked(force=True)
            return token

    @classmethod
    def refresh(cls, reason: str) -> bool:
        with cls._refresh_lock:
            now = time.time()
            # Avoid hammering Claude Code if several requests fail together.
            if now - cls._last_refresh_attempt < 5:
                with cls._state_lock:
                    try:
                        cls._load_locked(force=True)
                    except Exception:
                        pass
                return bool(cls._token)

            cls._last_refresh_attempt = now
            cmd = [
                cls._claude_bin(),
                "-p",
                "--model",
                "opus",
                "--output-format",
                "json",
                "--no-session-persistence",
                "Reply with exactly OK.",
            ]
            log(f"refreshing OAuth via Claude Code ({reason})")
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(Path.home()),
                    capture_output=True,
                    text=True,
                    timeout=REFRESH_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                log(f"OAuth refresh timed out after {REFRESH_TIMEOUT}s")
                return False
            except FileNotFoundError:
                log(f"Claude binary not found: {cmd[0]}")
                return False
            except Exception as exc:
                log(f"OAuth refresh failed before execution: {type(exc).__name__}: {exc}")
                return False

            if result.returncode != 0:
                err = (result.stderr or result.stdout or "").strip().replace("\n", " ")[:500]
                log(f"OAuth refresh command failed rc={result.returncode}: {err}")
                return False

            with cls._state_lock:
                try:
                    cls._load_locked(force=True)
                except Exception as exc:
                    log(f"OAuth refresh succeeded but token reload failed: {type(exc).__name__}: {exc}")
                    return False
            log(f"OAuth refresh OK sha12={cls.token_sha12()} {cls.expiry_summary()}")
            return True


def flatten(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                t = b.get("type", "")
                if t == "text":
                    parts.append(b.get("text", ""))
                elif t == "tool_result":
                    parts.append(str(b.get("content", "")))
                elif t == "tool_use":
                    parts.append("[called " + str(b.get("name", "?")) + "]")
        return " ".join(parts)
    return str(content)


def fix_message(msg):
    role = msg.get("role", "user")
    content = msg.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = flatten(content)
    else:
        text = str(content)
    if role == "assistant" and not text.strip():
        return None
    if not text.strip():
        text = "(empty)"
    has_tools = isinstance(content, list) and any(
        b.get("type") in ("tool_use", "tool_result") for b in content
        if isinstance(b, dict))
    if has_tools:
        return {"role": role, "content": content}
    return {"role": role, "content": text}


class UpstreamHTTPError(Exception):
    def __init__(self, code: int, body: bytes, headers=None):
        super().__init__(f"upstream HTTP {code}")
        self.code = code
        self.body = body or b'{"type":"error","error":{"message":"upstream error"}}'
        self.headers = headers or {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send_json(self, code: int, payload: dict | bytes, content_type: str = "application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code if code < 600 else 502)
        self.send_header("Content-Type", content_type or "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, OSError):
            pass

    def _send_json_error(self, code: int, message: str, error_type: str = "proxy_error", detail: str | None = None):
        error = {"type": error_type, "message": message}
        if detail:
            error["detail"] = detail[:1000]
        self._send_json(code, {"type": "error", "error": error})

    def _make_upstream_request(self, body: dict, token: str) -> urllib.request.Request:
        return urllib.request.Request(
            f"{API}/v1/messages",
            data=json.dumps(body).encode(),
            headers={
                **CC_H,
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

    def _open_upstream_once(self, body: dict, token: str):
        req = self._make_upstream_request(body, token)
        try:
            return urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)
        except urllib.error.HTTPError as e:
            raise UpstreamHTTPError(e.code, e.read(), dict(e.headers)) from e

    def _open_upstream_with_auth_retry(self, body: dict):
        token = TokenManager.get_token()
        try:
            return self._open_upstream_once(body, token)
        except UpstreamHTTPError as e:
            if e.code != 401:
                raise

            # 401 handling: figure out whether this is a real expiry or
            # something else. Refreshing on EVERY 401 produces a token-rotation
            # storm — Anthropic invalidates each refresh's predecessor, so
            # parallel threads using the older token cascade into more 401s,
            # and Anthropic eventually flags the whole grant as compromised.
            #
            # Strategy:
            #   1. If the in-memory token is genuinely past `expiresAt`, do
            #      one refresh + retry (the historic happy path).
            #   2. If the token still looks valid, the 401 is most likely a
            #      transient edge / rate-limit hiccup. Reload the credentials
            #      file (in case another process refreshed it) and retry the
            #      same request once — but DO NOT trigger a Claude Code
            #      refresh, which would mint yet another token and worsen
            #      the storm.
            with TokenManager._state_lock:
                exp_ms = TokenManager._expires_at_ms
            now_ms = time.time() * 1000
            token_actually_expired = exp_ms and now_ms >= exp_ms

            if token_actually_expired:
                log("upstream returned 401 and token is past expiresAt; refreshing OAuth and retrying once")
                if not TokenManager.refresh("upstream 401, token expired"):
                    raise
                token = TokenManager.get_token()
                return self._open_upstream_once(body, token)

            # Token still nominally valid — treat 401 as transient.
            # Reload the credentials file in case a sibling process refreshed
            # it, then retry once with whatever token is now on disk.
            log("upstream returned 401 but token is not past expiresAt; reloading file + retrying once (no refresh)")
            with TokenManager._state_lock:
                try:
                    TokenManager._load_locked(force=True)
                except Exception:
                    pass
                token = TokenManager._token
            try:
                return self._open_upstream_once(body, token)
            except UpstreamHTTPError as e2:
                if e2.code != 401:
                    raise
                # Second 401 with a still-valid token — at this point a
                # refresh is the only remaining lever. One refresh max per
                # request, then surface the error if it persists.
                log("second 401 after file reload; refreshing OAuth as last resort")
                if not TokenManager.refresh("upstream 401 persisted after reload"):
                    raise
                token = TokenManager.get_token()
                return self._open_upstream_once(body, token)

    def _prepare_body(self, body: dict) -> tuple[dict, dict]:
        # ---- Cloak ----
        body["system"] = CC_SYS
        banned = ("thinking", "temperature", "top_p", "top_k",
                  "output_config", "metadata", "stop_sequences", "tool_choice")
        for f in banned:
            body.pop(f, None)
        body["max_tokens"] = min(body.get("max_tokens", 64000), 64000)

        msgs = []
        for m in body.get("messages", []):
            fixed = fix_message(m)
            if fixed:
                msgs.append(fixed)
        body["messages"] = msgs

        tool_map = {}
        tools = body.get("tools", [])
        if tools:
            mapped = []
            for t in tools:
                hs_name = t.get("name", "")
                if not hs_name:
                    continue
                cloaked = _cloaked_tool_name(hs_name)
                tool_map[cloaked] = hs_name
                schema = t.get("input_schema", {"type": "object"})
                if not isinstance(schema.get("properties"), dict):
                    schema["properties"] = {}
                schema.setdefault("type", "object")
                cc = cloaked.split(_NAMESPACE_SEP, 1)[0]
                mapped.append({
                    "name": cloaked,
                    # Keep the original Hermes description so the model knows
                    # what each tool does. The cloak surface is in the *prefix*
                    # of the cloaked name, not in the description.
                    "description": t.get("description") or CC_DESC.get(cc, ""),
                    "input_schema": schema,
                })
            body["tools"] = mapped
            body["tool_choice"] = {"type": "auto"}
        else:
            body.pop("tools", None)
        return body, tool_map

    def do_POST(self):
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json_error(400, "invalid JSON request", "invalid_request", str(exc))
            return

        want_stream = bool(body.get("stream", False))
        body, tool_map = self._prepare_body(body)

        try:
            with self._open_upstream_with_auth_retry(body) as resp:
                if want_stream:
                    self._send_stream_response(resp, tool_map)
                else:
                    self._send_message_response(resp, tool_map)
        except UpstreamHTTPError as e:
            self._send_json(e.code, e.body, e.headers.get("Content-Type", "application/json"))
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            self._send_json_error(502, "Anthropic upstream unreachable or timed out", "upstream_error", str(exc))
        except RuntimeError as exc:
            self._send_json_error(503, "Claude OAuth unavailable", "auth_error", str(exc))
        except Exception as exc:
            log(f"unexpected request failure: {type(exc).__name__}: {exc}")
            self._send_json_error(500, "cloaked proxy internal error", "proxy_error", f"{type(exc).__name__}: {exc}")

    def _send_stream_response(self, resp, tool_map: dict):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                chunk = resp.read(16384)
                if not chunk:
                    break
                decoded = chunk.decode(errors="replace")
                # Replace the longest cloaked name first so we don't match a
                # shorter prefix that's a substring of a longer name.
                for cn in sorted(tool_map, key=len, reverse=True):
                    hn = tool_map[cn]
                    decoded = decoded.replace('"name":"' + cn + '"', '"name":"' + hn + '"')
                self.wfile.write(decoded.encode())
                self.wfile.flush()
        except (BrokenPipeError, OSError):
            pass

    def _send_message_response(self, resp, tool_map: dict):
        data = resp.read()
        if not data:
            self._send_json_error(502, "empty upstream response", "upstream_error")
            return
        try:
            result = json.loads(data)
        except json.JSONDecodeError as exc:
            self._send_json_error(502, "invalid upstream JSON", "upstream_error", str(exc))
            return

        # Reverse-map tool names
        for block in result.get("content", []):
            if block.get("type") == "tool_use":
                cloaked = block.get("name", "")
                if cloaked in tool_map:
                    block["name"] = tool_map[cloaked]
                else:
                    # Structural fallback for any cloaked name we didn't
                    # explicitly emit (e.g. model invented one).
                    block["name"] = _uncloak_tool_name(cloaked)

        # Inject model context info into response
        model = result.get("model", "")
        ctx = MODEL_CONTEXT.get(model, 200000)
        result["_model_context"] = ctx
        result["_hermes_note"] = f"Model: {model} ({ctx//1000}k context)"

        self._send_json(200, result)


if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else 8318
    try:
        TokenManager.get_token()
    except Exception as exc:
        log(f"FATAL: cannot load/refresh Claude OAuth token: {type(exc).__name__}: {exc}")
        sys.exit(1)

    log(f":{port}")
    log(f"token sha12={TokenManager.token_sha12()} {TokenManager.expiry_summary()}")
    log("pure cloak — auto-refresh enabled, model-aware, threaded")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
