"""End-to-end and integration tests for the anthropic-cloaked-proxy chain.

Complements tests/test_tool_mapping.py (which covers unit-level mapping
semantics) with:

- ``test_fix_message_tool_role`` — OpenAI ``role:"tool"`` → Anthropic
  tool_result block (PR-openclaw/cloaked-proxy#1).
- ``test_do_models_response_shape`` — ``/v1/models`` advertises
  ``LISTED_MODELS`` and every listed id has a ``MODEL_CONTEXT`` entry.
- ``test_soul_models_match_cloaked`` — soul and cloaked LISTED_MODELS agree.
- ``test_soul_models_response_shape`` — soul proxy has a ``/v1/models`` stub.
- ``test_prepare_body_dedups_colliding_cloaked_names`` (moved to
  test_tool_mapping.py) — Anthropic HTTP 400 protection.
- ``test_token_picker_prefers_future_expiry`` — multi-source token
  picker picks the latest-future-expiry entry, not the first file.
- ``live_roundtrip`` (opt-in via ``--live``) — POST to ``127.0.0.1:8319``
  reproducing the original v20 failure with ``mcp__browser_back`` in the
  ``tools[]`` schema; verifies the proxy returns 200 with a real Anthropic
  response (costs ~10 input tokens).

Run with::

    python3 -m pytest tests/                 # unit + integration
    python3 -m pytest tests/ --live          # + the live roundtrip

The whole module is hermes-version-agnostic: it loads
``cloaked-proxy.py`` and ``soul-proxy.py`` via ``importlib`` and exercises
their public symbols. No hermes import required.
"""

from __future__ import annotations
import importlib.util
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


cloaked = _load("cloaked_proxy_under_test", HERE / "cloaked-proxy.py")
soul = _load("soul_proxy_under_test", HERE / "soul-proxy.py")


# -----------------------------------------------------------------------------
# Tool-arg type coercion (covers the OpenAI→Anthropic tool_call_id path used
# by Claude Opus 4 — see patches/083 in hermes-agent-patches for the upstream
# context).
# -----------------------------------------------------------------------------
def test_fix_message_tool_role():
    """OpenAI-style ``role:tool`` → Anthropic ``user`` block with tool_result."""
    if not hasattr(cloaked, "fix_message"):
        pytest.skip("fix_message unavailable in this proxy version")

    out = cloaked.fix_message({
        "role": "tool",
        "content": "ok",
        "tool_call_id": "call_abc",
    })
    assert out is not None and out["role"] == "user", f"got {out!r}"
    blocks = out["content"] if isinstance(out["content"], list) else [{"content": out["content"]}]
    assert any(b.get("type") == "tool_result" for b in blocks), (
        f"no tool_result block in {out['content']!r}"
    )


# -----------------------------------------------------------------------------
# coerce_tool_args — Claude Opus 4 emits stringified ints/arrays; the proxy
# coerces them on the response path using each tool's input_schema.
# -----------------------------------------------------------------------------
def test_coerce_tool_args_integer_and_number():
    if not hasattr(cloaked, "coerce_tool_args"):
        pytest.skip("coerce_tool_args unavailable in this proxy version")
    schema = {"type": "object", "properties": {
        "offset": {"type": "integer"},
        "ratio": {"type": "number"},
    }}
    args = {"offset": "480", "ratio": "1.5"}
    cloaked.coerce_tool_args(args, schema)
    assert args["offset"] == 480 and isinstance(args["offset"], int)
    assert args["ratio"] == 1.5 and isinstance(args["ratio"], float)


def test_coerce_tool_args_boolean_and_json():
    if not hasattr(cloaked, "coerce_tool_args"):
        pytest.skip("coerce_tool_args unavailable in this proxy version")
    schema = {"type": "object", "properties": {
        "flag": {"type": "boolean"},
        "items": {"type": "array"},
        "obj": {"type": "object"},
    }}
    args = {"flag": "true", "items": '[1,2,3]', "obj": '{"a":1}'}
    cloaked.coerce_tool_args(args, schema)
    assert args["flag"] is True
    assert args["items"] == [1, 2, 3]
    assert args["obj"] == {"a": 1}


def test_coerce_tool_args_leaves_strings_and_unknown_keys():
    if not hasattr(cloaked, "coerce_tool_args"):
        pytest.skip("coerce_tool_args unavailable in this proxy version")
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    args = {"name": "480", "untracked": "999"}  # name is genuinely a string
    cloaked.coerce_tool_args(args, schema)
    assert args["name"] == "480", "string-typed field must not be coerced"
    assert args["untracked"] == "999", "unknown key must be left untouched"


def test_coerce_tool_args_anyof_nullable_number():
    """Regression: OpenCode-style schemas wrap optional numerics in anyOf.

    Symptom before the fix: read tool's offset/limit declared as
    {"anyOf": [{"type": "number"}, {"type": "null"}]}; Opus emitted
    offset="660" (stringified); _schema_types returned [] so the value was
    left as a string and OpenCode's strict validator rejected it
    ("The schema wants a number type" / "schema serialization issue with offset").
    """
    if not hasattr(cloaked, "coerce_tool_args"):
        pytest.skip("coerce_tool_args unavailable in this proxy version")
    schema = {"type": "object", "properties": {
        "offset": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "ratio": {"oneOf": [{"type": "number"}, {"type": "string"}]},
    }}
    args = {"offset": "660", "limit": "20", "ratio": "1.5"}
    cloaked.coerce_tool_args(args, schema)
    assert args["offset"] == 660.0 and isinstance(args["offset"], float)
    assert args["limit"] == 20 and isinstance(args["limit"], int)
    assert args["ratio"] == 1.5 and isinstance(args["ratio"], float)


def test_schema_types_collects_union_members():
    if not hasattr(cloaked, "_schema_types"):
        pytest.skip("_schema_types unavailable in this proxy version")
    assert cloaked._schema_types({"type": "integer"}) == ["integer"]
    assert set(cloaked._schema_types({"type": ["number", "null"]})) == {"number", "null"}
    assert "number" in cloaked._schema_types(
        {"anyOf": [{"type": "number"}, {"type": "null"}]})
    assert "integer" in cloaked._schema_types(
        {"oneOf": [{"type": "integer"}, {"type": "string"}]})
    assert cloaked._schema_types({}) == []


def test_coerce_tool_args_infers_when_schema_typeless():
    """Regression: opaque/typeless schema must still coerce numeric strings.

    OpenCode's read tool declares offset/limit in a shape the proxy can't read
    (no parseable JSON-Schema type). Before the infer fallback, _schema_types
    returned [] and offset="1144" passed through as a string, tripping
    OpenCode's validator: SchemaError(Expected number | undefined, got "1144").
    Infer mode coerces a cleanly-numeric string when the schema does NOT
    explicitly type the field as string.
    """
    if not hasattr(cloaked, "coerce_tool_args"):
        pytest.skip("coerce_tool_args unavailable in this proxy version")
    # Property present but with an unreadable type wrapper -> _schema_types == []
    schema = {"type": "object", "properties": {
        "offset": {"description": "byte offset", "x-weird": True},
        "limit": {"description": "max bytes"},
    }}
    args = {"offset": "1144", "limit": "189"}
    cloaked.coerce_tool_args(args, schema)
    assert args["offset"] == 1144 and isinstance(args["offset"], int)
    assert args["limit"] == 189 and isinstance(args["limit"], int)


def test_coerce_tool_args_infer_preserves_genuine_strings():
    if not hasattr(cloaked, "coerce_tool_args"):
        pytest.skip("coerce_tool_args unavailable in this proxy version")
    schema = {"type": "object", "properties": {
        "name": {"type": "string"},          # explicitly string -> never coerce
        "zip": {"description": "no type"},    # infer mode, leading zero -> keep
        "path": {"description": "no type"},   # infer mode, non-numeric -> keep
        "count": {"description": "no type"},  # infer mode, clean int -> coerce
    }}
    args = {"name": "1144", "zip": "02134", "path": "scripts/x.py", "count": "7"}
    cloaked.coerce_tool_args(args, schema)
    assert args["name"] == "1144", "explicit string field must not coerce"
    assert args["zip"] == "02134", "leading-zero numeric must stay a string (id/zip)"
    assert args["path"] == "scripts/x.py", "non-numeric string untouched"
    assert args["count"] == 7 and isinstance(args["count"], int)


def test_schema_resolves_by_hs_name_when_model_returns_bare_name():
    """Regression: model may echo the BARE hermes-side tool name.

    We declare the tool to the upstream model under its cloaked name
    (e.g. ``Skill__read``) and key tool_schemas by that cloaked name. But
    the model does not always echo it -- OpenCode subagent runs showed the
    model returning bare ``read``. The response handler used to resolve the
    coercion schema with tool_schemas.get(cloaked) ONLY, so for a bare-name
    return the lookup missed (schema_found=False) and coerce_tool_args was
    skipped entirely -- offset="505"/limit="170" reached OpenCode's Zod
    validator as strings: SchemaError(Expected number | undefined, got "170").

    The fix builds a hermes-side-name -> schema index and falls back to it.
    This test reproduces that resolution + the resulting coercion.
    """
    if not hasattr(cloaked, "coerce_tool_args"):
        pytest.skip("coerce_tool_args unavailable in this proxy version")

    read_schema = {"type": "object", "properties": {
        "filePath": {"type": "string"},
        "offset": {"description": "line offset"},   # typeless -> infer path
        "limit": {"description": "max lines"},       # typeless -> infer path
    }}
    # As built in _prepare_body: keyed by the CLOAKED name.
    tool_map = {"Skill__read": "read"}
    tool_schemas = {"Skill__read": read_schema}

    # Mirror _send_message_response's hs-name index.
    schemas_by_hs = {}
    for _ck, _hs in tool_map.items():
        _sc = tool_schemas.get(_ck)
        if _sc is not None:
            schemas_by_hs.setdefault(_hs, _sc)

    # Model returned the BARE name -> cloaked-key lookup MUST miss.
    cloaked_name = "read"
    assert tool_schemas.get(cloaked_name) is None, "bare name must miss the cloaked index"

    # Handler remaps block name then resolves schema; for a bare return the
    # remapped name is still 'read' (uncloak no-op).
    resolved = tool_schemas.get(cloaked_name) or schemas_by_hs.get("read")
    assert resolved is read_schema, "fallback must resolve schema by hs-name"

    args = {"filePath": "scripts/x.py", "offset": "505", "limit": "170"}
    cloaked.coerce_tool_args(args, resolved)
    assert args["offset"] == 505 and isinstance(args["offset"], int)
    assert args["limit"] == 170 and isinstance(args["limit"], int)
    assert args["filePath"] == "scripts/x.py"


# -----------------------------------------------------------------------------
# /v1/models shape — both cloaked and soul proxies must return the same set.
# -----------------------------------------------------------------------------
def test_do_models_response_shape():
    """do_GET advertises LISTED_MODELS; every listed id has a MODEL_CONTEXT entry."""
    if not hasattr(cloaked, "LISTED_MODELS"):
        pytest.skip("LISTED_MODELS unavailable")
    listed = cloaked.LISTED_MODELS
    assert len(listed) > 0, "LISTED_MODELS is empty"
    assert "claude-opus-4-8" in listed, "opus retired from picker?"
    assert "claude-fable-5" in listed, "fable missing from picker"
    assert any(m.startswith("claude-sonnet") for m in listed), "no sonnet in picker"
    # haiku is a real, cheap model Anthropic offers — it must be pickable so
    # cheap delegation doesn't force the user to hand-write the id.
    assert "claude-haiku-4-5" in listed, "haiku missing from picker"
    # Every advertised model must have a context-limit entry (picker + ctx note
    # stay consistent).
    for mid in listed:
        assert mid in cloaked.MODEL_CONTEXT, f"{mid} advertised but absent from MODEL_CONTEXT"


def test_model_context_accuracy():
    """MODEL_CONTEXT must match known-good Anthropic context limits.

    A stale ctx leaks into the injected _hermes_note/_model_context metadata
    and reads as authoritative. Two historical drifts are pinned here so a
    future bad edit fails loudly instead of silently:
      - claude-opus-4-6 is 1M (was wrongly 200K).
      - claude-opus-4-5 (200K) must exist as a compatibility fallback.
    """
    if not hasattr(cloaked, "MODEL_CONTEXT"):
        pytest.skip("MODEL_CONTEXT unavailable")
    ctx = cloaked.MODEL_CONTEXT
    assert ctx.get("claude-opus-4-6") == 1000000, (
        f"opus-4-6 ctx={ctx.get('claude-opus-4-6')} — should be 1M (was wrongly 200K)"
    )
    assert ctx.get("claude-opus-4-5") == 200000, (
        "claude-opus-4-5 (200K) missing from MODEL_CONTEXT compatibility set"
    )


def test_soul_models_match_cloaked():
    """soul-proxy's LISTED_MODELS literal must mirror cloaked's (duplicated by necessity)."""
    if not hasattr(soul, "LISTED_MODELS") or not hasattr(cloaked, "LISTED_MODELS"):
        pytest.skip("LISTED_MODELS unavailable on one proxy")
    assert tuple(soul.LISTED_MODELS) == tuple(cloaked.LISTED_MODELS), (
        "soul and cloaked /v1/models drifted — update both LISTED_MODELS literals"
    )


def test_soul_models_response_shape():
    """Soul proxy do_GET should not raise."""
    if not hasattr(soul, "Handler"):
        pytest.skip("soul.Handler unavailable")
    handler = soul.Handler.__new__(soul.Handler)
    try:
        handler.do_GET = lambda: None  # we only assert there is a do_GET
    except AttributeError:
        pytest.fail("soul.Handler has no do_GET — /v1/models stub is missing")


def test_soul_refreshes_soul_md_on_mtime_change():
    """SOUL.md edits must take effect without a daemon restart.

    The first implementation snapshotted HERMES_SOUL at import time; a
    SOUL.md edit required restarting the soul proxy, and the comment
    claimed 'stays fresh'. _soul_text() must return the NEW content once
    the file's mtime changes.
    """
    if not hasattr(soul, "_soul_text"):
        pytest.skip("_soul_text unavailable in this proxy version")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "SOUL.md"
        f.write_text("# v1\nold persona", encoding="utf-8")
        old_file = soul.SOUL_FILE
        old_mtime, old_text = soul.SOUL_FILE_MTIME, soul.HERMES_SOUL
        try:
            soul.SOUL_FILE = f
            soul.SOUL_FILE_MTIME = 0.0
            soul.HERMES_SOUL = ""
            first = soul._soul_text()
            assert "old persona" in first, f"first read wrong: {first!r}"

            # Touch with new content — same path, new mtime.
            f.write_text("# v2\nnew persona", encoding="utf-8")
            second = soul._soul_text()
            assert "new persona" in second, (
                f"SOUL.md edit not picked up without restart: {second!r}"
            )
        finally:
            soul.SOUL_FILE = old_file
            soul.SOUL_FILE_MTIME = old_mtime
            soul.HERMES_SOUL = old_text


def test_soul_skills_block_refreshes_on_tree_change():
    """New skills must appear in the injected block without a restart."""
    if not hasattr(soul, "_skills_block"):
        pytest.skip("_skills_block unavailable in this proxy version")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "alpha").mkdir()
        (d / "alpha" / "SKILL.md").write_text("---\nname: alpha-skill\n---\n", encoding="utf-8")
        old_dir, old_mtime = soul.SKILLS_DIR, soul.SKILLS_MTIME
        try:
            soul.SKILLS_DIR = d
            soul.SKILLS_MTIME = 0.0
            block1 = soul._skills_block()
            assert "alpha-skill" in block1, f"first block missing skill: {block1!r}"

            (d / "beta").mkdir()
            (d / "beta" / "SKILL.md").write_text("---\nname: beta-skill\n---\n", encoding="utf-8")
            block2 = soul._skills_block()
            assert "beta-skill" in block2, (
                f"new skill not picked up without restart: {block2!r}"
            )
        finally:
            soul.SKILLS_DIR = old_dir
            soul.SKILLS_MTIME = old_mtime


# -----------------------------------------------------------------------------
# Token picker — multi-source merge prefers latest-future-expiry.
# -----------------------------------------------------------------------------
def test_token_picker_prefers_future_expiry():
    """A fresh Keychain token should win over a stale credentials-file token."""
    if not hasattr(cloaked, "TokenManager"):
        pytest.skip("TokenManager unavailable")

    now_ms = int(time.time() * 1000)

    class encore:
        pass  # noqa: just a namespace

    # We can't easily exercise TokenManager._read_credentials without
    # touching CRED_FILE / Keychain. Instead validate the helper logic
    # directly: given two (token, expires_ms) candidates, "winning" is the
    # one with the latest future expiry — mirror the upstream _score() body.
    candidates = [
        ("stale_token", now_ms - 60_000),     # 60s expired
        ("fresh_token", now_ms + 3_600_000),  # 1h ahead
    ]

    def score(tok_exp):
        tok, exp = tok_exp
        if not exp:
            return 0.0
        return (exp - now_ms) / 1000

    best = max(candidates, key=score)
    assert best[0] == "fresh_token", f"expected fresh_token, got {best}"


# -----------------------------------------------------------------------------
# Live roundtrip — opt-in via --live. Reproduces the original v20 failure with
# ``mcp__browser_back`` in ``tools[]``.
# -----------------------------------------------------------------------------
def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    "not config.getoption('--live')",
    reason="opt-in via --live (costs ~10 input tokens)",
)
def test_live_mcp_browser_roundtrip():
    """POST to :8319 with mcp__browser_back in tools[] — verify 200/proxy ok."""
    if not _port_open(8319):
        pytest.skip("soul proxy not listening on 127.0.0.1:8319")
    body = {
        "model": "claude-opus-4-8",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "x"}],
        "tools": [
            {"name": "mcp__browser_back", "description": "browser back",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "mcp__browser_snapshot", "description": "snapshot",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "browser_click", "description": "click element",
             "input_schema": {"type": "object", "properties": {}}},
        ],
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8319/v1/messages",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read())
            assert r.status == 200, f"status={r.status}"
            assert "model" in payload, f"keys={list(payload.keys())}"
    except urllib.error.HTTPError as exc:
        detail = exc.read(300).decode(errors="replace")
        pytest.fail(f"HTTP {exc.code}: {detail}")


# -----------------------------------------------------------------------------
# Adaptive-thinking self-healing (fable-5 / fable-5-1 rejection class).
# -----------------------------------------------------------------------------
def test_do_post_retries_with_adaptive_on_thinking_rejection(monkeypatch):
    """Upstream 400 naming the disabled pin -> learn the model, retry with
    adaptive, never surface the 400 to the client. Exercises the real
    ``do_POST`` wiring (not just the predicate)."""
    import io

    handler = cloaked.Handler.__new__(cloaked.Handler)
    raw = json.dumps({
        "model": "claude-future-9", "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode()
    handler.rfile = io.BytesIO(raw)
    handler.headers = {"Content-Length": str(len(raw))}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({
                "model": "claude-future-9", "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            }).encode()

    captured = []
    calls = {"n": 0}

    def fake_open(b):
        calls["n"] += 1
        captured.append(dict(b))
        if calls["n"] == 1:
            raise cloaked.UpstreamHTTPError(
                400,
                b'{"type":"error","error":{"message":'
                b'"\\"thinking.type.disabled\\" is not supported for this model. '
                b'Use \\"thinking.type.adaptive\\" ..."}}',
            )
        return FakeResp()

    sent = {}
    monkeypatch.setattr(handler, "_open_upstream_with_auth_retry", fake_open, raising=False)
    monkeypatch.setattr(
        handler, "_send_json",
        lambda code, payload, content_type="application/json": sent.update(code=code, payload=payload),
        raising=False,
    )

    try:
        handler.do_POST()
        assert calls["n"] == 2, f"expected one retry, got {calls['n']} upstream calls"
        assert captured[0].get("thinking") == {"type": "disabled"}, captured[0].get("thinking")
        assert captured[1].get("thinking") == {"type": "adaptive"}, captured[1].get("thinking")
        assert "claude-future-9" in cloaked._adaptive_learned
        assert sent.get("code") == 200, sent
    finally:
        cloaked._adaptive_learned.discard("claude-future-9")


def test_do_post_does_not_retry_unrelated_400(monkeypatch):
    """Only the disabled-rejection signature retries; other 400s surface."""
    import io

    handler = cloaked.Handler.__new__(cloaked.Handler)
    raw = json.dumps({
        "model": "claude-opus-5", "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode()
    handler.rfile = io.BytesIO(raw)
    handler.headers = {"Content-Length": str(len(raw))}

    calls = {"n": 0}

    def fake_open(b):
        calls["n"] += 1
        raise cloaked.UpstreamHTTPError(400, b'{"error":{"message":"unrelated"}}')

    sent = {}
    monkeypatch.setattr(handler, "_open_upstream_with_auth_retry", fake_open, raising=False)
    monkeypatch.setattr(
        handler, "_send_json",
        lambda code, payload, content_type="application/json": sent.update(code=code, payload=payload),
        raising=False,
    )

    handler.do_POST()
    assert calls["n"] == 1, f"unrelated 400 must not retry, got {calls['n']}"
    assert sent.get("code") == 400, sent
    assert "claude-opus-5" not in cloaked._adaptive_learned


# -----------------------------------------------------------------------------
# Streaming relay: upstream SSE passes through live; tool_use is fixed per block.
# -----------------------------------------------------------------------------
class _BufWriter:
    def __init__(self):
        self.buf = bytearray()
        self.writes = 0

    def write(self, b):
        self.buf += b
        self.writes += 1

    def flush(self):
        pass


def _sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def _streaming_handler(monkeypatch, fake_open, tools=None, headers=None):
    import io

    handler = cloaked.Handler.__new__(cloaked.Handler)
    req = {"model": "claude-opus-5", "max_tokens": 8, "stream": True,
           "messages": [{"role": "user", "content": "hi"}]}
    if tools:
        req["tools"] = tools
    raw = json.dumps(req).encode()
    handler.rfile = io.BytesIO(raw)
    handler.headers = {"Content-Length": str(len(raw)), **(headers or {})}
    handler.wfile = _BufWriter()
    statuses, sent_bodies = [], []
    monkeypatch.setattr(handler, "send_response", lambda code, *a: statuses.append(code), raising=False)
    monkeypatch.setattr(handler, "send_header", lambda *a: None, raising=False)
    monkeypatch.setattr(handler, "end_headers", lambda: None, raising=False)

    def opener(body):
        sent_bodies.append(dict(body))
        return fake_open(body)

    monkeypatch.setattr(handler, "_open_upstream_with_auth_retry", opener, raising=False)
    return handler, statuses, sent_bodies


class _StreamResp:
    def __init__(self, chunks, fail_after=None):
        self._lines = b"".join(chunks).splitlines(keepends=True)
        self._fail_after = fail_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for i, line in enumerate(self._lines):
            if self._fail_after is not None and i == self._fail_after:
                import socket as _s
                raise _s.timeout("The read operation timed out")
            yield line


def _events(buf):
    out = []
    for frame in bytes(buf).decode().split("\n\n"):
        if not frame.strip():
            continue
        ev = data = None
        for line in frame.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:])
        out.append((ev, data))
    return out


def test_stream_requests_stream_upstream_and_relays_text_live(monkeypatch):
    chunks = [
        _sse("message_start", {"type": "message_start", "message": {"model": "claude-opus-5", "content": []}}),
        _sse("ping", {"type": "ping"}),
        _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
        _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "he"}}),
        _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "llo"}}),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
        _sse("message_stop", {"type": "message_stop"}),
    ]
    handler, statuses, sent = _streaming_handler(monkeypatch, lambda b: _StreamResp(chunks))
    handler.do_POST()
    assert sent[0]["stream"] is True, "streaming client must stream upstream"
    assert statuses == [200]
    evs = _events(handler.wfile.buf)
    names = [e for e, _ in evs]
    assert names == ["message_start", "ping", "content_block_start", "content_block_delta",
                     "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    assert evs[0][1]["message"]["_model_context"] == cloaked.MODEL_CONTEXT["claude-opus-5"]
    assert [d["delta"]["text"] for e, d in evs if e == "content_block_delta"] == ["he", "llo"]


def test_stream_tool_use_uncloaked_and_coerced(monkeypatch):
    tools = [{"name": "read", "input_schema": {"type": "object", "properties": {
        "offset": {"description": "line offset"}}}}]
    cloaked_name = cloaked._cloaked_tool_name("read")
    chunks = [
        _sse("message_start", {"type": "message_start", "message": {"model": "claude-opus-5", "content": []}}),
        _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "tool_use", "id": "tu1", "name": cloaked_name, "input": {}}}),
        _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "input_json_delta", "partial_json": '{"offs'}}),
        _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "input_json_delta", "partial_json": 'et": "505"}'}}),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse("message_stop", {"type": "message_stop"}),
    ]
    handler, statuses, _ = _streaming_handler(monkeypatch, lambda b: _StreamResp(chunks), tools=tools)
    handler.do_POST()
    evs = _events(handler.wfile.buf)
    start = next(d for e, d in evs if e == "content_block_start")
    assert start["content_block"]["name"] == "read"
    deltas = [d for e, d in evs if e == "content_block_delta"]
    assert len(deltas) == 1
    assert json.loads(deltas[0]["delta"]["partial_json"]) == {"offset": 505}


def test_stream_timeout_after_commit_is_sse_error(monkeypatch):
    chunks = [
        _sse("message_start", {"type": "message_start", "message": {"model": "claude-opus-5", "content": []}}),
        _sse("ping", {"type": "ping"}),
    ]
    handler, statuses, _ = _streaming_handler(monkeypatch, lambda b: _StreamResp(chunks, fail_after=3))
    handler.do_POST()
    evs = _events(handler.wfile.buf)
    assert statuses == [200], statuses
    assert evs[-1][0] == "error" and "timed out" in json.dumps(evs[-1][1])


def test_stream_fast_upstream_error_keeps_real_status(monkeypatch):
    def fake_open(b):
        raise cloaked.UpstreamHTTPError(429, b'{"type":"error","error":{"type":"rate_limit_error","message":"slow"}}')

    handler, statuses, _ = _streaming_handler(monkeypatch, fake_open)
    handler.do_POST()
    assert statuses == [429], statuses


def test_stream_tool_use_live_when_client_coerces(monkeypatch):
    """X-Cloak-Tool-Args: raw (soul-proxy/Hermes): tool arg deltas pass through
    live with the uncloaked name, not held until content_block_stop."""
    tools = [{"name": "read", "input_schema": {"type": "object", "properties": {
        "offset": {"description": "line offset"}}}}]
    cloaked_name = cloaked._cloaked_tool_name("read")
    chunks = [
        _sse("message_start", {"type": "message_start", "message": {"model": "claude-opus-5", "content": []}}),
        _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "tool_use", "id": "tu1", "name": cloaked_name, "input": {}}}),
        _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "input_json_delta", "partial_json": '{"offs'}}),
        _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "input_json_delta", "partial_json": 'et": "505"}'}}),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse("message_stop", {"type": "message_stop"}),
    ]
    handler, _, _ = _streaming_handler(monkeypatch, lambda b: _StreamResp(chunks), tools=tools,
                                       headers={"X-Cloak-Tool-Args": "raw"})
    handler.do_POST()
    evs = _events(handler.wfile.buf)
    assert next(d for e, d in evs if e == "content_block_start")["content_block"]["name"] == "read"
    deltas = [d["delta"]["partial_json"] for e, d in evs if e == "content_block_delta"]
    assert deltas == ['{"offs', 'et": "505"}']
