#!/usr/bin/env python3
"""Hermes → Anthropic OAuth Cloaking Proxy (v21 — namespaced tool cloak, no collisions)."""

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
    "User-Agent": "claude-cli/2.1.77 (external, cli)",
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "oauth-2025-04-20,interleaved-thinking-2025-05-14,token-counting-2024-11-01",
    "x-app": "cli",
}
UPSTREAM_TIMEOUT = int(os.environ.get("CLOAKED_PROXY_UPSTREAM_TIMEOUT", "180"))
REFRESH_TIMEOUT = int(os.environ.get("CLOAKED_PROXY_REFRESH_TIMEOUT", "120"))
EXPIRY_REFRESH_MARGIN_SEC = int(os.environ.get("CLOAKED_PROXY_EXPIRY_MARGIN_SEC", "600"))
# Model the OAuth-refresh probe asks Claude Code to use. This is a hidden pin:
# if Anthropic ever retires the pinned model, token refresh silently breaks.
# Override with CLOAKED_REFRESH_MODEL (must be a valid Claude Code model id).
REFRESH_MODEL = os.environ.get("CLOAKED_REFRESH_MODEL", "opus-4-8")

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
    "claude-opus-5": 1000000,
    "claude-opus-4-8": 1000000,
    "claude-fable-5": 1000000,
    "claude-opus-4-7": 1000000,
    "claude-opus-4-6": 1000000,
    "claude-opus-4-5": 200000,
    "claude-sonnet-4-6": 1000000,
    "claude-sonnet-4-20250514": 200000,
    "claude-sonnet-4-5-20250929": 200000,
    "claude-haiku-4-5": 200000,
    "claude-haiku-4-5-20251001": 200000,
}

# Models advertised via GET /v1/models (drives the Hermes model picker).
# Curated subset of MODEL_CONTEXT — the ids we actually want selectable, in
# display order. Both proxies' do_GET iterate this, so it's the single source.
LISTED_MODELS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5",
)

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
    """Sanity-check a Hermes tool name.

    Historically this rejected any name containing ``__`` because the cloak uses
    ``__`` as the namespace separator (``<CC>__<tool>``). However, Claude Code
    namespaces MCP servers as ``mcp__<server>__<tool>`` and Hermes occasionally
    flattens that to ``mcp__<server>_<tool>`` (a single ``__`` at the start).
    Both forms are valid on the wire — the cloaked-name builder passes them
    through ``_cc_for`` (which strip-prefixes ``mcp_`` and tries each regex) and
    the resulting ``<CC>__<canonical>`` form is what reaches Anthropic, so the
    internal ``__`` never appears downstream. The failsafe that remains: ban a
    LEADING ``__`` (would produce an empty CC name) and control bytes.
    """
    if not hs_name:
        raise ValueError("hs_name must be a non-empty string")
    if hs_name.startswith(_NAMESPACE_SEP):
        raise ValueError(
            f"hs_name {hs_name!r} starts with reserved separator {_NAMESPACE_SEP!r}"
        )
    for ch in hs_name:
        if ord(ch) < 0x20 or ch == "\x7f":
            raise ValueError(f"hs_name {hs_name!r} contains control bytes")
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


# --- Tool-arg type coercion -------------------------------------------------
# Claude (esp. Opus 4 via the OAuth/CC path) sometimes emits tool_use.input
# values as JSON strings even when input_schema declares a non-string type
# (e.g. offset="480" instead of 480, or a stringified array/object). The real
# Claude Code CLI coerces these silently; strict downstream validators (opencode)
# reject them. We coerce on the response path using each tool's input_schema.
_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _schema_types(spec: dict) -> list:
    """Collect the JSON-Schema scalar types declared for a property.

    Handles the plain ``type`` (str or list) plus the union wrappers that
    Zod/TypeBox-style generators (e.g. OpenCode's tool schemas) emit for
    optional params — ``anyOf`` / ``oneOf`` / ``allOf`` — including the common
    ``{"anyOf": [{"type": "number"}, {"type": "null"}]}`` nullable shape. Without
    this, a stringified ``offset="660"`` against an ``anyOf`` numeric schema was
    left uncoerced and rejected by the strict client-side validator.
    """
    if not isinstance(spec, dict):
        return []
    out = []
    t = spec.get("type")
    if isinstance(t, str):
        out.append(t)
    elif isinstance(t, list):
        out.extend(x for x in t if isinstance(x, str))
    for key in ("anyOf", "oneOf", "allOf"):
        sub = spec.get(key)
        if isinstance(sub, list):
            for member in sub:
                out.extend(_schema_types(member))
    # Dedup, preserve order.
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def coerce_tool_args(input_dict, schema) -> None:
    """In-place coerce string args to the type declared in schema.properties.

    Only acts when the value is a str and the schema is unambiguous. Numeric
    strings -> int/float (no scientific notation, bounded length to avoid
    surprising bigint coercion). Stringified JSON -> object/array via json.loads.
    Everything else is left untouched.
    """
    if not isinstance(input_dict, dict) or not isinstance(schema, dict):
        return
    props = schema.get("properties")
    if not isinstance(props, dict):
        return
    for k, v in list(input_dict.items()):
        if not isinstance(v, str) or k not in props:
            continue
        types = _schema_types(props[k])
        # When the schema declares no parseable type for this property (e.g. a
        # generator emitted a shape _schema_types can't read, or omitted the
        # type entirely), fall back to value-shape inference: a string that is
        # cleanly numeric/bool/JSON is coerced UNLESS the schema explicitly says
        # the field is a string. This is what makes stringified offset="1144"
        # against an opaque OpenCode read-tool schema coerce correctly without
        # the proxy having to recognize every JSON-Schema dialect.
        infer = not types
        if "string" in types and len(types) == 1:
            # Genuinely a string field — never coerce.
            continue
        # In infer mode (no schema type), be conservative: skip numeric strings
        # with a leading zero (e.g. "02134" zip codes, ids) since dropping the
        # zero would corrupt an identifier. Schema-typed numerics still coerce.
        infer_numeric_ok = not (len(v) > 1 and v.lstrip("-").startswith("0") and "." not in v)
        if ("integer" in types or (infer and infer_numeric_ok and _NUM_RE.match(v) and "." not in v)) \
                and _NUM_RE.match(v) and "." not in v and len(v) <= 18:
            try:
                input_dict[k] = int(v)
                continue
            except ValueError:
                pass
        if ("number" in types or (infer and infer_numeric_ok and _NUM_RE.match(v))) \
                and _NUM_RE.match(v) and len(v) <= 18:
            try:
                input_dict[k] = float(v)
                continue
            except ValueError:
                pass
        if ("boolean" in types or infer) and v in ("true", "false"):
            input_dict[k] = (v == "true")
            continue
        if ("object" in types and v.startswith("{")) or ("array" in types and v.startswith("[")) \
                or (infer and (v.startswith("{") or v.startswith("["))):
            try:
                input_dict[k] = json.loads(v)
            except (ValueError, json.JSONDecodeError):
                pass


def log(msg: str) -> None:
    print(f"[cloaked-proxy v21] {msg}", file=sys.stderr, flush=True)


def _cc_for(hs_name: str) -> str:
    if hs_name in _CC_FOR_HS:
        return _CC_FOR_HS[hs_name]
    # Strip ONE well-formed CC namespace prefix: ``mcp__`` (two underscores
    # after ``mcp``). Claude Code namespaces MCP servers as
    # ``mcp__<server>__<tool>`` and Hermes occasionally flattens that to
    # ``mcp__<server>_<tool>`` (a single ``__`` at the start). Both reduce to
    # the same canonical tool tail, which is what the regexes match against.
    # If the prefix is missing (bare ``browser_back``) the strip is a no-op.
    clean = hs_name.removeprefix("mcp__")
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
                REFRESH_MODEL,
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

    # OpenAI format: {"role": "tool", "content": "...", "tool_call_id": "..."}
    # Claude expects tool results inside a user message with content blocks.
    if role == "tool":
        if isinstance(content, str):
            result_content = content
        elif isinstance(content, list):
            result_content = flatten(content)
        else:
            result_content = str(content)
        tool_id = msg.get("tool_call_id", "")
        block = {"type": "tool_result", "content": result_content or "(empty)"}
        if tool_id:
            block["tool_use_id"] = tool_id
        return {"role": "user", "content": [block]}

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
            log("upstream returned 401; refreshing OAuth and retrying once")
            if not TokenManager.refresh("upstream 401"):
                raise
            token = TokenManager.get_token()
            return self._open_upstream_once(body, token)

    def _prepare_body(self, body: dict) -> tuple[dict, dict, dict]:
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
        tool_schemas = {}
        seen_cloaked = set()
        tools = body.get("tools", [])
        if tools:
            mapped = []
            for t in tools:
                hs_name = t.get("name", "")
                if not hs_name:
                    continue
                cloaked = _cloaked_tool_name(hs_name)
                if cloaked in seen_cloaked:
                    # Anthropic returns HTTP 400 if two tools share a name.
                    # Hermes' toolset registration can emit the same logical
                    # tool under multiple names that collapse to one cloaked
                    # name (e.g. once via a flat `browser_back` and again via
                    # `mcp__browser_back` if both prefixes survive the tool
                    # filter — or any future ``mcp__<server>_<tool>`` alias
                    # whose canonical collides with an existing entry).
                    # Keep the FIRST occurrence; still record the alias so
                    # response uncloaking always finds a hermes-side name.
                    tool_map.setdefault(cloaked, hs_name)
                    continue
                seen_cloaked.add(cloaked)
                tool_map[cloaked] = hs_name
                schema = t.get("input_schema", {"type": "object"})
                if not isinstance(schema.get("properties"), dict):
                    schema["properties"] = {}
                schema.setdefault("type", "object")
                tool_schemas[cloaked] = schema
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
        return body, tool_map, tool_schemas

    def do_GET(self):
        """Stub /v1/models for provider model-list checks."""
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send_json(200, {
                "object": "list",
                "data": [
                    {"id": mid, "object": "model", "owned_by": "anthropic"}
                    for mid in LISTED_MODELS
                ],
            })
        else:
            self._send_json_error(404, f"not found: {self.path}")

    def do_POST(self):
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json_error(400, "invalid JSON request", "invalid_request", str(exc))
            return

        want_stream = bool(body.get("stream", False))
        body, tool_map, tool_schemas = self._prepare_body(body)
        # Force non-streaming upstream so tool-arg coercion runs on a complete
        # JSON body; re-synthesize SSE downstream if the client asked to stream.
        body["stream"] = False

        try:
            with self._open_upstream_with_auth_retry(body) as resp:
                self._send_message_response(resp, tool_map, tool_schemas, as_sse=want_stream)
        except UpstreamHTTPError as e:
            self._send_json(e.code, e.body, e.headers.get("Content-Type", "application/json"))
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            self._send_json_error(502, "Anthropic upstream unreachable or timed out", "upstream_error", str(exc))
        except RuntimeError as exc:
            self._send_json_error(503, "Claude OAuth unavailable", "auth_error", str(exc))
        except Exception as exc:
            log(f"unexpected request failure: {type(exc).__name__}: {exc}")
            self._send_json_error(500, "cloaked proxy internal error", "proxy_error", f"{type(exc).__name__}: {exc}")

    def _send_message_response(self, resp, tool_map: dict, tool_schemas: dict | None = None, as_sse: bool = False):
        tool_schemas = tool_schemas or {}
        # The upstream model does not always echo the cloaked tool name we
        # sent (e.g. it returns bare ``read`` when we declared ``Skill__read``).
        # Build a hermes-side-name -> schema index so schema lookup succeeds
        # regardless of whether the model replied with the cloaked or the
        # uncloaked name. Without this, coercion is silently skipped and
        # stringified numerics (offset="505") reach strict clients (OpenCode).
        schemas_by_hs = {}
        for _ck, _hs in tool_map.items():
            _sc = tool_schemas.get(_ck)
            if _sc is not None:
                schemas_by_hs.setdefault(_hs, _sc)
        data = resp.read()
        if not data:
            self._send_json_error(502, "empty upstream response", "upstream_error")
            return
        try:
            result = json.loads(data)
        except json.JSONDecodeError as exc:
            self._send_json_error(502, "invalid upstream JSON", "upstream_error", str(exc))
            return

        for block in result.get("content", []):
            if block.get("type") == "tool_use":
                cloaked = block.get("name", "")
                if cloaked in tool_map:
                    block["name"] = tool_map[cloaked]
                else:
                    # Structural fallback for any cloaked name we didn't
                    # explicitly emit (e.g. model invented one).
                    block["name"] = _uncloak_tool_name(cloaked)
                _before = json.dumps(block.get("input")) if os.environ.get("CLOAK_TRACE") else None
                # Resolve schema by cloaked key first; if the model replied with
                # the bare hermes-side name (cloaked not in tool_schemas), fall
                # back to the hs-name index so coercion still runs.
                _schema = tool_schemas.get(cloaked)
                if _schema is None:
                    _schema = schemas_by_hs.get(block.get("name"))
                coerce_tool_args(block.get("input"), _schema)
                if os.environ.get("CLOAK_TRACE"):
                    _props = (_schema or {}).get("properties", {})
                    log("TRACE tool=%s cloaked=%s schema_found=%s props=%s before=%s after=%s as_sse=%s"
                        % (block.get("name"), cloaked, _schema is not None,
                           json.dumps(_props), _before, json.dumps(block.get("input")), as_sse))

        model = result.get("model", "")
        ctx = MODEL_CONTEXT.get(model, 200000)
        result["_model_context"] = ctx
        result["_hermes_note"] = f"Model: {model} ({ctx//1000}k context)"

        if as_sse:
            self._emit_message_as_sse(result)
        else:
            self._send_json(200, result)

    def _emit_message_as_sse(self, result: dict):
        # Synthesize an Anthropic-style SSE stream from a complete message so a
        # streaming client gets the coerced (non-stream) body. Whole content
        # blocks are sent at once rather than token-by-token.
        def sse(event: str, payload: dict) -> bytes:
            return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            content = result.get("content", [])
            start_msg = {k: v for k, v in result.items() if k != "content"}
            start_msg["content"] = []
            self.wfile.write(sse("message_start", {"type": "message_start", "message": start_msg}))

            for idx, block in enumerate(content):
                btype = block.get("type")
                if btype == "text":
                    self.wfile.write(sse("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}}))
                    self.wfile.write(sse("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": block.get("text", "")}}))
                elif btype == "tool_use":
                    self.wfile.write(sse("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "tool_use", "id": block.get("id", ""), "name": block.get("name", ""), "input": {}}}))
                    self.wfile.write(sse("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}))}}))
                else:
                    self.wfile.write(sse("content_block_start", {"type": "content_block_start", "index": idx, "content_block": block}))
                self.wfile.write(sse("content_block_stop", {"type": "content_block_stop", "index": idx}))

            self.wfile.write(sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": result.get("stop_reason"), "stop_sequence": result.get("stop_sequence")}, "usage": result.get("usage", {})}))
            self.wfile.write(sse("message_stop", {"type": "message_stop"}))
            self.wfile.flush()
        except (BrokenPipeError, OSError):
            pass


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
