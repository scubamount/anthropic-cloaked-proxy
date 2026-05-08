#!/usr/bin/env python3
"""Hermes Soul Proxy v2 — captures gateway system prompt + SOUL.md, injects into messages.
   Routes: Hermes Gateway → this (:8319) → cloaked proxy (:8318) → Anthropic"""

import json, sys, re, urllib.request, urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

DOWNSTREAM = "http://127.0.0.1:8318/v1/messages"
SOUL_FILE = Path.home() / ".hermes" / "SOUL.md"

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
    def log_message(self, *a): pass

    def do_POST(self):
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.loads(raw)
        except:
            self.send_error(400)
            return

        # Capture gateway's system prompt (has dynamic memory/user context)
        raw_system = body.get("system", "")
        if isinstance(raw_system, list) and raw_system:
            # Modern Hermes sends system as an array of text blocks
            parts = []
            for block in raw_system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            gateway_system = "\n".join(parts)
        elif isinstance(raw_system, str):
            gateway_system = raw_system
        else:
            gateway_system = ""
        
        # Build injection from gateway context + disk SOUL.md
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

        # Forward to cloaked proxy
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
            with urllib.request.urlopen(req, timeout=180) as resp:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                result = resp.read()
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                self.wfile.write(result)
        except urllib.error.HTTPError as e:
            err = e.read() or b'{"type":"error","error":{"message":"error"}}'
            self.send_response(e.code if e.code < 600 else 502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                self.wfile.write(err)
            except:
                pass


if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else 8319
    print(f"[soul-proxy v2] :{port} → {DOWNSTREAM}", file=sys.stderr)
    print(f"[soul-proxy v2] {len(_SKILLS_NAMES)} skills, captures gateway dynamic context", file=sys.stderr)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
