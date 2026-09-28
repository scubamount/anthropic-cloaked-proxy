#!/usr/bin/env python3
"""Verify the model-catalog invariants that keep breaking this proxy.

Runs offline against the real modules — no network, no inference quota. The
canonical gate on this repo is still `pytest tests/`; this is the fast
standalone check for "did a catalog edit keep the wiring honest", usable in a
pre-commit hook or over SSH where the pytest deps aren't installed.

    ./scripts/verify_models.py        # exit 0 = all invariants hold

Checks, each of which corresponds to a gate that has actually fired here:
  1. LISTED_MODELS parity — soul-proxy duplicates the literal because it cannot
     import the hyphenated cloaked-proxy module; a silent divergence makes the
     Hermes model picker show models that 400 at runtime.
  2. Every advertised model has a MODEL_CONTEXT entry (the picker needs a
     context length to budget).
  3. THINKING_ADAPTIVE_MODELS is a subset of MODEL_CONTEXT — a model that
     rejects thinking=disabled must be one the proxy will actually route.
  4. Every advertised model gets the thinking shape its set membership
     dictates. This is per-model, not global: Anthropic's adaptive-only line
     400s on {"type":"disabled"}, and the remaining models require the pin
     (opus-5 defaults thinking ON upstream, which is what 25c4239 fixed).
  5. Replayed `thinking` blocks are stripped from tool passthrough. Replaying
     one alongside thinking=disabled is an HTTP 400 "Invalid signature in
     thinking block" that kills a whole turn.
"""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The proxy refuses to boot a real refresh without a setup token; blank it so
# importing the module has no side effects.
os.environ.setdefault("CLOAKED_SETUP_TOKEN", "")


def load(name: str, path: Path):
    """Import a hyphenated proxy file as a module.

    ``spec_from_file_location`` returns None for an unknown extension, so fail
    loudly here rather than raising an opaque AttributeError on `.loader`.
    """
    if not path.is_file():
        raise SystemExit(f"missing proxy source: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path} as a python module")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    cp = load("cp_verify", ROOT / "cloaked-proxy.py")
    sp = load("sp_verify", ROOT / "soul-proxy.py")
    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    # 1. parity between the two proxies
    listed = list(cp.LISTED_MODELS)
    check(listed == list(sp.LISTED_MODELS), "LISTED_MODELS parity",
          f"{len(listed)} ids" if listed == list(sp.LISTED_MODELS)
          else f"cloaked={listed} soul={list(sp.LISTED_MODELS)}")

    # 2. every advertised model has a context limit
    missing = [m for m in listed if m not in cp.MODEL_CONTEXT]
    check(not missing, "every advertised model has a context limit",
          f"missing={missing}" if missing else f"{len(listed)} models")

    # 3. adaptive models are all routable (present in the context map)
    orphan = sorted(cp.THINKING_ADAPTIVE_MODELS - set(cp.MODEL_CONTEXT))
    check(not orphan, "adaptive models are all in MODEL_CONTEXT",
          f"orphans={orphan}" if orphan else f"{len(cp.THINKING_ADAPTIVE_MODELS)} adaptive")

    # 4. per-model thinking shape
    handler = cp.Handler.__new__(cp.Handler)
    wrong: list[str] = []
    for model in listed:
        body, _, _ = handler._prepare_body({
            "model": model, "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        })
        expected = ({"type": "adaptive"}
                    if model in cp.THINKING_ADAPTIVE_MODELS
                    else {"type": "disabled"})
        if body["thinking"] != expected:
            wrong.append(f"{model}: {body['thinking']} != {expected}")
    n_adaptive = sum(1 for m in listed if m in cp.THINKING_ADAPTIVE_MODELS)
    check(not wrong, "per-model thinking shape",
          "; ".join(wrong) if wrong
          else f"{n_adaptive} adaptive, {len(listed) - n_adaptive} disabled")

    # 5. replayed thinking blocks stripped from tool passthrough
    fixed = cp.fix_message({"role": "assistant", "content": [
        {"type": "thinking", "thinking": "", "signature": "CAISx"},
        {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]})
    kinds = [b["type"] for b in fixed["content"]]
    check(kinds == ["tool_use"], "replayed thinking block stripped",
          f"blocks={kinds}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} invariant(s) — {', '.join(failures)}")
        return 1
    print("OK: all model-catalog invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
