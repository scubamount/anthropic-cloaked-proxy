#!/usr/bin/env python3
"""Hermes → Anthropic OAuth Cloaking Proxy (v18 — pure cloak, no personality injection)."""

import json, sys, re, urllib.request, urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

API = "https://api.anthropic.com"
CC_SYS = "You are Claude Code, Anthropic's official CLI for Claude."
CC_H = {
    "User-Agent": "claude-cli/2.1.77 (external, cli)",
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "oauth-2025-04-20,interleaved-thinking-2025-05-14,token-counting-2024-11-01",
    "x-app": "cli",
}

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

TOKEN = json.loads(CRED_FILE.read_text())["claudeAiOauth"]["accessToken"]
TOOL_MAP = {}

def flatten(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                t = b.get("type","")
                if t == "text":
                    parts.append(b.get("text",""))
                elif t == "tool_result":
                    parts.append(str(b.get("content","")))
                elif t == "tool_use":
                    parts.append("[called " + str(b.get("name","?")) + "]")
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
        b.get("type") in ("tool_use","tool_result") for b in content
        if isinstance(b, dict))
    if has_tools:
        return {"role": role, "content": content}
    return {"role": role, "content": text}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        global TOOL_MAP
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            self.send_error(400)
            return

        want_stream = body.get("stream", False)

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

        tools = body.get("tools", [])
        TOOL_MAP.clear()
        if tools:
            seen = set()
            mapped = []
            for t in tools:
                cc_name = _cc_for(t.get("name", ""))
                if cc_name not in seen:
                    seen.add(cc_name)
                    TOOL_MAP[cc_name] = t.get("name", cc_name)
                    schema = t.get("input_schema", {"type": "object"})
                    if not isinstance(schema.get("properties"), dict):
                        schema["properties"] = {}
                    schema.setdefault("type", "object")
                    mapped.append({
                        "name": cc_name,
                        "description": CC_DESC.get(cc_name, t.get("description", "")),
                        "input_schema": schema,
                    })
            body["tools"] = mapped
            body["tool_choice"] = {"type": "auto"}
        else:
            body.pop("tools", None)

        req = urllib.request.Request(
            f"{API}/v1/messages",
            data=json.dumps(body).encode(),
            headers={**CC_H, "Authorization": f"Bearer {TOKEN}",
                     "Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                if want_stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    while True:
                        chunk = resp.read(16384)
                        if not chunk:
                            break
                        decoded = chunk.decode(errors="replace")
                        # Reverse-map tool names back to Hermes
                        for cn, hn in TOOL_MAP.items():
                            decoded = decoded.replace(
                                '"name":"' + cn + '"', '"name":"' + hn + '"')
                        try:
                            self.wfile.write(decoded.encode())
                            self.wfile.flush()
                        except (BrokenPipeError, OSError):
                            break
                else:
                    data = resp.read()
                    if not data:
                        self.send_error(502, "empty upstream response")
                        return
                    result = json.loads(data)
                    # Reverse-map tool names
                    for block in result.get("content", []):
                        if block.get("type") == "tool_use" and \
                           block.get("name") in TOOL_MAP:
                            block["name"] = TOOL_MAP[block["name"]]
                    
                    # Inject model context info into response
                    model = result.get("model", "")
                    ctx = MODEL_CONTEXT.get(model, 200000)
                    usage = result.get("usage", {})
                    result["_model_context"] = ctx
                    result["_hermes_note"] = f"Model: {model} ({ctx//1000}k context)"
                    
                    out = json.dumps(result).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
        except urllib.error.HTTPError as e:
            err = e.read() or b'{"type":"error","error":{"message":"error"}}'
            self.send_response(e.code if e.code < 600 else 502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                self.wfile.write(err)
            except (BrokenPipeError, OSError):
                pass


if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else 8318
    print(f"[cloaked-proxy v18] :{port}", file=sys.stderr)
    print(f"[cloaked-proxy v18] token: {TOKEN[:20]}...", file=sys.stderr)
    print(f"[cloaked-proxy v18] pure cloak — no skills injection, model-aware", file=sys.stderr)

    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
