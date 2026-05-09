"""Unit tests for cloaked-proxy tool name mapping (no-collision fix)."""
import importlib.util
import sys
from pathlib import Path

# Load cloaked-proxy.py as a module (file is hyphenated so direct import is awkward)
spec = importlib.util.spec_from_file_location(
    "cloaked_proxy",
    Path(__file__).resolve().parent.parent / "cloaked-proxy.py",
)
cp = importlib.util.module_from_spec(spec)
sys.modules["cloaked_proxy"] = cp
spec.loader.exec_module(cp)


HS_TOOLS = [
    "browser_back", "browser_click", "browser_console", "browser_get_images",
    "browser_navigate", "browser_press", "browser_scroll", "browser_snapshot",
    "browser_type", "browser_vision", "clarify", "cronjob", "delegate_task",
    "execute_code", "image_generate", "memory", "patch", "process", "read_file",
    "search_files", "send_message", "session_search", "skill_manage", "skill_view",
    "skills_list", "terminal", "text_to_speech", "todo", "vision_analyze",
    "web_extract", "web_search", "write_file", "viking_search", "viking_read",
    "viking_browse", "viking_remember", "viking_add_resource",
]

CC_NAMES = {
    "Bash", "Read", "Write", "Edit", "Grep", "Glob", "Task", "TodoWrite",
    "NotebookEdit", "WebFetch", "WebSearch", "BashOutput", "KillShell", "Skill",
}


def test_every_tool_gets_unique_cloaked_name():
    cloaked = [cp._cloaked_tool_name(t) for t in HS_TOOLS]
    assert len(set(cloaked)) == len(HS_TOOLS), (
        f"collisions: {len(HS_TOOLS) - len(set(cloaked))}, names={cloaked}"
    )


def test_cloaked_name_starts_with_cc_name():
    for t in HS_TOOLS:
        cn = cp._cloaked_tool_name(t)
        prefix = cn.split(cp._NAMESPACE_SEP, 1)[0]
        assert prefix in CC_NAMES, (
            f"{t} cloaked as {cn}, prefix {prefix!r} not a CC name"
        )


def test_cloaked_name_is_reversible():
    for t in HS_TOOLS:
        cn = cp._cloaked_tool_name(t)
        recovered = cp._uncloak_tool_name(cn)
        assert recovered == t, f"{t} -> {cn} -> {recovered}"


def test_uncloak_passthrough_for_unknown_names():
    # If the model invents a name with no separator, return it unchanged so the
    # error surfaces clearly downstream
    assert cp._uncloak_tool_name("RandomGarbage") == "RandomGarbage"


def test_uncloak_unknown_with_separator_extracts_suffix():
    # If the model emits a structurally-valid cloaked name we haven't seen,
    # split on the separator and trust the suffix
    assert cp._uncloak_tool_name("Skill__some_new_tool") == "some_new_tool"


def test_validate_hs_name_rejects_separator():
    raised = False
    try:
        cp._validate_hs_name("bad__name")
    except ValueError:
        raised = True
    assert raised, "validator must reject names containing the separator"


def test_prepare_body_keeps_all_tools():
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": t, "description": f"desc {t}", "input_schema": {"type": "object", "properties": {}}}
            for t in HS_TOOLS
        ],
    }
    handler = cp.Handler.__new__(cp.Handler)  # bypass HTTP init
    out, tool_map = handler._prepare_body(dict(body))
    assert len(out["tools"]) == len(HS_TOOLS), (
        f"expected {len(HS_TOOLS)}, got {len(out['tools'])}"
    )
    out_names = {t["name"] for t in out["tools"]}
    assert len(out_names) == len(HS_TOOLS), "duplicate cloaked names emitted"
    # Every cloaked name reverses to a real Hermes name
    for cn, hn in tool_map.items():
        assert cp._uncloak_tool_name(cn) == hn
        assert hn in HS_TOOLS
    # Every original Hermes tool is represented
    represented = set(tool_map.values())
    assert represented == set(HS_TOOLS), (
        f"missing: {set(HS_TOOLS) - represented}"
    )


def test_prepare_body_preserves_input_schema():
    """The actual schema (image_url+question for vision_analyze) must reach the model unchanged."""
    schema = {
        "type": "object",
        "properties": {
            "image_url": {"type": "string"},
            "question": {"type": "string"},
        },
        "required": ["image_url", "question"],
    }
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "vision_analyze", "description": "vision", "input_schema": schema}],
    }
    handler = cp.Handler.__new__(cp.Handler)
    out, tool_map = handler._prepare_body(body)
    assert len(out["tools"]) == 1
    out_schema = out["tools"][0]["input_schema"]
    assert "image_url" in out_schema["properties"]
    assert "question" in out_schema["properties"]
    assert out_schema["required"] == ["image_url", "question"]


if __name__ == "__main__":
    import traceback
    failures = 0
    total = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            total += 1
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{total - failures}/{total} passed")
    raise SystemExit(failures)
