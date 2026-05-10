#!/usr/bin/env python3
"""Hermes Soul Proxy v3 — captures gateway system prompt + SOUL.md, injects into messages.
   Routes: Hermes Gateway → this (:8319) → cloaked proxy (:8318) → Anthropic"""

import json
import re
import socket
import sys
import datetime
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DOWNSTREAM = "http://127.0.0.1:8318/v1/messages"
SOUL_FILE = Path.home() / ".hermes" / "SOUL.md"
DOWNSTREAM_TIMEOUT = 190

# Read SOUL.md from disk (stays fresh)
if SOUL_FILE.exists():
    HERMES_SOUL = SOUL_FILE.read_text().strip()
else:
    HERMES_SOUL = "# Hermes Agent Persona\nYou are Hermes. Caveman brain. Nick's general agent."

# Skills list
SKILLS_DIR = Path.home() / ".hermes" / "skills"
_SKILLS_NAMES = []
if SKILLS_DIR.exists():
    for sf in sorted(SKILLS_DIR.glob("**/SKILL.md")):
        try:
            m = re.search(r'^name:\s*(.+)$', sf.read_text(), re.MULTILINE)
            if m:
                _SKILLS_NAMES.append(m.group(1).strip())
        except Exception:
            pass

_SKILLS_BLOCK = ""
if _SKILLS_NAMES:
    lines = ["- " + n for n in _SKILLS_NAMES]
    _SKILLS_BLOCK = "<available_skills>\n" + "\n".join(lines) + "\n</available_skills>"


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} [soul-proxy v3] {msg}", file=sys.stderr, flush=True)


def _build_inject(gateway_system):
    """Build injection block: SOUL.md + gateway's dynamic context + skills.
    Find where SOUL.md ends in gateway system → take everything after it.
    That gives us: MEMORY + USER PROFILE + platform context + skills (all dynamic).
    Strip the verbose <available_skills> block, use our compact one.
    """
    # Find transition: SOUL.md ends with "You live by this soul." or similar
    # Then everything after is dynamic context (memory, user profile, etc.)
    transition = re.search(r'(You live by this soul\.)', gateway_system)

    dynamic_text = ""
    if transition:
        # Everything after "You live by this soul."
        after_soul = gateway_system[transition.end():]
        # Strip the verbose <available_skills> block (with descriptions)
        after_soul = re.sub(r'<available_skills>.*?</available_skills>', '', after_soul, flags=re.DOTALL)
        after_soul = after_soul.strip()
        if after_soul:
            dynamic_text = "## Current Context (from gateway)\n\n" + after_soul

    # Build final injection
    inject = (
        "<hermes_persona>\n" + HERMES_SOUL + "\n</hermes_persona>\n\n"
    )
    if dynamic_text:
        inject += dynamic_text + "\n\n"
    inject += _SKILLS_BLOCK
    inject += "\n\n**You are Hermes. Follow the persona above. Be caveman. Talk like Hermes.**\n"
    inject += "**Load skills with skill_view(name) when task matches.**\n"

    return inject


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

    def _send_json_error(self, code: int, message: str, error_type: str = "soul_proxy_error", detail: str | None = None):
        error = {"type": error_type, "message": message}
        if detail:
            error["detail"] = detail[:1000]
        self._send_json(code, {"type": "error", "error": error})

    def _parse_body(self) -> dict | None:
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            return json.loads(raw)
        except Exception as exc:
            self._send_json_error(400, "invalid JSON request", "invalid_request", str(exc))
            return None

    def _gateway_system(self, raw_system) -> str:
        if isinstance(raw_system, list) and raw_system:
            # Modern Hermes sends system as an array of text blocks
            parts = []
            for block in raw_system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        if isinstance(raw_system, str):
            return raw_system
        return ""

    def _inject_persona(self, body: dict) -> dict:
        gateway_system = self._gateway_system(body.get("system", ""))
        inject_block = _build_inject(gateway_system)

        # Inject into first user message
        msgs = body.get("messages", [])
        if msgs and msgs[0].get("role") == "user":
            first = msgs[0]
            content = first.get("content", "")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        if "<hermes_persona>" not in b.get("text", ""):
                            b["text"] = inject_block + "\n\n" + b.get("text", "")
                        break
            elif isinstance(content, str):
                if "<hermes_persona>" not in content:
                    first["content"] = inject_block + "\n\n" + content
            msgs[0] = first
            body["messages"] = msgs

        # Strip system prompt — cloaked proxy will set CC one-liner
        body.pop("system", None)
        return body

    def do_POST(self):
        body = self._parse_body()
        if body is None:
            return
        want_stream = bool(body.get("stream", False))
        body = self._inject_persona(body)

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            DOWNSTREAM,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": self.headers.get("User-Agent", "hermes"),
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=DOWNSTREAM_TIMEOUT) as resp:
                content_type = resp.headers.get("Content-Type", "application/json")
                if want_stream or content_type.startswith("text/event-stream"):
                    self._proxy_stream(resp, content_type)
                else:
                    result = resp.read()
                    self._send_json(200, result, content_type)
        except urllib.error.HTTPError as e:
            err = e.read() or b'{"type":"error","error":{"message":"downstream error"}}'
            self._send_json(e.code if e.code < 600 else 502, err, e.headers.get("Content-Type", "application/json"))
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            self._send_json_error(502, "cloaked proxy unreachable or timed out", "downstream_error", str(exc))
        except Exception as exc:
            log(f"unexpected request failure: {type(exc).__name__}: {exc}")
            self._send_json_error(500, "soul proxy internal error", "soul_proxy_error", f"{type(exc).__name__}: {exc}")

    def _proxy_stream(self, resp, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type or "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                chunk = resp.read(16384)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, OSError):
            pass


if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else 8319
    log(f":{port} → {DOWNSTREAM}")
    log(f"{len(_SKILLS_NAMES)} skills, captures gateway dynamic context")
    log("threaded, bounded downstream errors, stream passthrough enabled")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
