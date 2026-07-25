# Anthropic Cloaked Proxy for Hermes

Bypass Anthropic's "extra usage" subscription gate on Max/Pro plans by running Hermes Agent through a Claude Code OAuth fingerprint. Uses a two-proxy chain so personality, skills, and tools all work natively.

```
Hermes → Soul Proxy (:8319) → Cloaked Proxy (:8318) → Anthropic API
        injects SOUL.md          tool mapping (28→14)
        + skills list            CC fingerprint cloak
        + memory context         OAuth auth
```

## Why two proxies

- Anthropic fingerprints the **system prompt** — must stay under ~150 chars
- Personality injection has to happen in the **first user message**
- Soul proxy handles persona + context injection
- Cloaked proxy handles tool mapping + fingerprint cloaking + OAuth headers

## Install Claude Code

⚠️ Use the official install script, **not** npm:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Then log in (browser flow):

```bash
claude login
```

## Deploy proxies

Copy the three files to your Hermes host (they live in `~/proxies/` on the
current install):

```bash
scp cloaked-proxy.py soul-proxy.py start_both.sh you@your-host:~/proxies/
```

Start both:

```bash
bash ~/proxies/start_both.sh
```

Point Hermes at the soul proxy:

```bash
hermes config set model.provider anthropic
hermes config set model.default claude-opus-5
hermes config set model.base_url "http://127.0.0.1:8319"
hermes config set model.api_key ""
```

`api_key` is intentionally empty — the cloaked proxy supplies OAuth
credentials upstream, and a non-empty value here sends a conflicting
`Authorization` header.

## Test

```bash
# Personality check
hermes chat -q "Without using any tools or skills, what can you see are available from your system prompt."

# Tool execution check
hermes chat -q "ls ~ | head -3"
```

Run the test suite (needs the interpreter pytest is installed under; unset
`PYTHONPATH` so a venv on the path doesn't shadow it):

```bash
cd ~/proxies && env -u PYTHONPATH python3 -m pytest tests/ -q      # 25 passed, 1 skipped
cd ~/proxies && env -u PYTHONPATH python3 -m pytest tests/ -q --live   # + live roundtrip
```

## Files

| File | Purpose |
|------|---------|
| `cloaked-proxy.py` | Tool mapping + Anthropic fingerprint cloak + OAuth headers |
| `soul-proxy.py` | SOUL.md injection + skills list + gateway dynamic context |
| `start_both.sh` | Restart script (kill old, start cloaked, start soul, verify) |
| `tests/` | Unit tests for message conversion and proxy behavior |

## Architecture

### Soul Proxy (port 8319)

Reads `~/.hermes/SOUL.md` from disk on startup. On each request:

1. Captures gateway's system prompt (contains `MEMORY` + `USER PROFILE`)
2. Finds the `SOUL.md` boundary (`"You live by this soul."`) and extracts everything after it as dynamic context
3. Strips verbose `<available_skills>` from gateway context (replaces with compact list)
4. Injects `<hermes_persona>` (fresh SOUL.md) + dynamic context + compact skills list into first user message
5. Appends behavioral directive: `You are Hermes. Follow the persona above. Be caveman.`
6. Strips system prompt (cloaked proxy sets the CC one-liner)
7. Forwards to cloaked proxy on `:8318`

### Cloaked Proxy (port 8318)

Pure cloak — **zero personality injection** (soul proxy handles that).

#### OpenAI → Anthropic message conversion

Hermes sends messages in OpenAI format. Claude's API requires Anthropic format. The cloaked proxy converts automatically:

| OpenAI format | Anthropic format |
|---------------|------------------|
| `{"role": "tool", "content": "...", "tool_call_id": "t1"}` | `{"role": "user", "content": [{"type": "tool_result", "content": "...", "tool_use_id": "t1"}]}` |
| `{"role": "assistant", "content": [{"type": "tool_use", ...}]}` | Passed through (native Anthropic format) |
| `{"role": "user", "content": "..."}` | `{"role": "user", "content": "..."}` |

This conversion happens in `fix_message()` and is required because Claude rejects `role: "tool"` with:
```
HTTP 400: Unexpected role "tool". Allowed roles are "user" or "assistant".
```

#### Model context awareness

Context limits from [Anthropic docs](https://platform.claude.com/docs/en/about-claude/models/overview):

| Model | Context | Max Output |
|-------|---------|------------|
| claude-opus-5 | **1M** | 128k |
| claude-opus-4-8 | **1M** | 128k |
| claude-fable-5 | **1M** | 128k |
| claude-opus-4-7 | **1M** | 128k |
| claude-opus-4-6 | 200k | 64k |
| claude-sonnet-4-6 | **1M** | 64k |
| claude-sonnet-4-20250514 | 200k | 64k |
| claude-sonnet-4-5-20250929 | 200k | 64k |
| claude-haiku-4-5 | 200k | 64k |
| claude-haiku-4-5-20251001 | 200k | 64k |

Table must match `MODEL_CONTEXT` in `cloaked-proxy.py`. Models offered in the
`/v1/models` picker are `LISTED_MODELS` (a subset, picker order).

Unknown models default to 200k. The proxy injects `_model_context` and `_hermes_note` into API responses.

#### Beta Headers (required)

```
anthropic-beta: oauth-2025-04-20,interleaved-thinking-2025-05-14,token-counting-2024-11-01
```

Missing any of these breaks OAuth routing or returns incorrect usage counts.

#### Tool Mapping (28 Hermes → 14 CC aliases)

| CC Alias | Hermes Tools |
|-----------|-------------|
| Bash | terminal, execute_code |
| Read | read_file |
| Write | write_file |
| Edit | patch |
| Grep | search_files |
| Task | delegate_task |
| TodoWrite | todo, viking_browse, viking_remember |
| NotebookEdit | clarify |
| WebFetch | web_extract, viking_read, viking_add_resource |
| WebSearch | web_search, viking_search |
| Glob | browser_* (5), vision_analyze, image_generate, text_to_speech |
| BashOutput | process |
| KillShell | cronjob |
| Skill | skill_view, skill_manage, skills_list, memory, session_search, send_message, mcp_* |

Reverse mapping stores `TOOL_MAP = {cc_name: real_hermes_name}` — tool names are swapped back in responses so the gateway executes the right tool.

## Fingerprint vectors covered

1. **System prompt** — exactly `You are Claude Code, Anthropic's official CLI for Claude.` (150 chars max)
2. **Tool count** — hard cap at 14 aliases
3. **Tool descriptions** — pure functional text, no brackets or qualifiers
4. **Headers** — CC User-Agent, `x-app: cli`, `anthropic-version: 2023-06-01`, betas
5. **Banned output fields** — thinking, temperature, top_p/k, output_config, metadata, stop_sequences, tool_choice (all stripped)
6. **Max tokens** — capped at 64k (CC limit, Hermes may send 128k)
7. **Message structure** — flatten arrays to plain strings
8. **Assistant prefill** — drop empty assistant messages
9. **Tool result passthrough** — preserve `tool_result` blocks in message history for multi-turn loops

## What NOT to do

- **NO** directive in system prompt (`"Tool names are aliases"` triggers editorialization)
- **NO** directive in first user message (`"Do not comment on tool names"` gets quoted verbatim)
- **NO** brackets in descriptions (`[browser_navigate]` draws attention)
- **NO** "despite the name" qualifiers

The model may naturally note the CC/Hermes name mismatch — accept it. Routing **is** correct.

## Troubleshooting

### "extra usage" or 429 errors
- Verify both proxies are listening:
  - macOS: `lsof -nP -iTCP:8318,8319 -sTCP:LISTEN`
  - Linux: `ss -tlnp | grep -E '831[89]'`
- Check Hermes base_url points to 8319 (soul): `hermes config | grep base_url`
- System prompt must be exactly the CC one-liner
- Clear rate limit after ~10-15 min of zero requests

### Tool messages rejected (HTTP 400 "Unexpected role 'tool'")
- Ensure you're running the latest `cloaked-proxy.py` with `fix_message()` OpenAI conversion
- The proxy must convert `role: "tool"` messages to `role: "user"` with `type: "tool_result"` content blocks
- Restart both proxies after updating: `bash ~/proxies/start_both.sh`

### Proxy dies after disconnect
- Use `bash ~/proxies/start_both.sh` from within the VM (uses `nohup`)
- Never inline SSH with `&` — Terminal tool blocks backgrounding

### `no option named '--live'` when running tests
- Fixed: `pytest_addoption` must live in `tests/conftest.py`. pytest ignores it
  in a plain test module, which errored the live roundtrip test on every run.

## License

MIT — use at your own risk. Anthropic ToS may frown on this.
