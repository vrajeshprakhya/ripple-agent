"""Offline tests for the tool tracer. No network, no key."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import agent as A  # noqa: E402


def _assistant(*tool_uses):
    return {
        "role": "assistant",
        "content": [{"toolUse": tu} for tu in tool_uses],
    }


def test_records_tool_calls_in_order():
    t = A.ToolTracer(echo=False)
    t(message=_assistant(
        {"toolUseId": "a", "name": "list_directory", "input": {"path": "."}},
        {"toolUseId": "b", "name": "calculate", "input": {"expression": "6*7"}},
    ))
    assert [name for name, _ in t.calls] == ["list_directory", "calculate"]
    assert t.calls[1][1] == {"expression": "6*7"}


def test_deduplicates_repeated_messages():
    # Strands emits the same assistant message more than once per turn.
    t = A.ToolTracer(echo=False)
    msg = _assistant({"toolUseId": "a", "name": "calculate", "input": {}})
    t(message=msg)
    t(message=msg)
    t(message=msg)
    assert len(t.calls) == 1


def test_same_tool_twice_with_different_ids_counts_twice():
    t = A.ToolTracer(echo=False)
    t(message=_assistant({"toolUseId": "a", "name": "calculate", "input": {"expression": "1"}}))
    t(message=_assistant({"toolUseId": "b", "name": "calculate", "input": {"expression": "2"}}))
    assert len(t.calls) == 2


def test_ignores_user_messages():
    t = A.ToolTracer(echo=False)
    t(message={"role": "user", "content": [{"text": "hello"}]})
    assert t.calls == []


def test_tolerates_text_only_and_malformed_content():
    t = A.ToolTracer(echo=False)
    t(message={"role": "assistant", "content": [{"text": "thinking"}]})
    t(message={"role": "assistant", "content": ["not a dict", None]})
    t(message={"role": "assistant"})  # no content key at all
    t(message=None)
    t(data="")  # falsy delta
    assert t.calls == []


def test_ignores_tool_use_without_a_name():
    t = A.ToolTracer(echo=False)
    t(message=_assistant({"toolUseId": "a", "input": {}}))
    assert t.calls == []


# --- Manifest accounting --------------------------------------------------
def test_captures_manifest_from_tools_key():
    t = A.ToolTracer(echo=False)
    t(tool_config={"tools": [{"name": "one"}, {"name": "two"}]})
    assert len(t.manifest) == 2


def test_captures_manifest_from_tool_specs_key():
    t = A.ToolTracer(echo=False)
    t(tool_config={"toolSpecs": [{"name": "one"}]})
    assert len(t.manifest) == 1


def test_captures_manifest_from_a_bare_list():
    t = A.ToolTracer(echo=False)
    t(tool_config=[{"name": "one"}, {"name": "two"}, {"name": "three"}])
    assert len(t.manifest) == 3


def test_manifest_captured_only_once():
    t = A.ToolTracer(echo=False)
    t(tool_config={"tools": [{"name": "one"}]})
    t(tool_config={"tools": [{"name": "a"}, {"name": "b"}, {"name": "c"}]})
    assert len(t.manifest) == 1  # the first one wins, not the last


def test_summary_reports_manifest_and_calls():
    t = A.ToolTracer(echo=False)
    t(tool_config={"tools": [{"name": "x", "schema": "y" * 100}]})
    t(message=_assistant({"toolUseId": "a", "name": "x", "input": {"k": "v"}}))
    out = t.summary()
    assert "1 tools" in out
    assert "tokens re-sent per turn" in out
    assert "1. x" in out


def test_summary_says_none_when_no_tools_used():
    t = A.ToolTracer(echo=False)
    assert "tool calls: none" in t.summary()


def test_summary_truncates_enormous_arguments():
    t = A.ToolTracer(echo=False)
    t(message=_assistant({"toolUseId": "a", "name": "x", "input": {"blob": "z" * 5000}}))
    out = t.summary()
    assert "..." in out
    assert len(out) < 400


def test_summary_works_with_no_manifest_seen():
    t = A.ToolTracer(echo=False)
    t(message=_assistant({"toolUseId": "a", "name": "x", "input": {}}))
    out = t.summary()
    assert "manifest" not in out
    assert "1. x" in out


# --- Echo behaviour -------------------------------------------------------
def test_echoes_streamed_text_when_enabled(capsys):
    t = A.ToolTracer(echo=True)
    t(data="hello ")
    t(data="world")
    assert capsys.readouterr().out == "hello world"


def test_stays_silent_when_echo_disabled(capsys):
    t = A.ToolTracer(echo=False)
    t(data="should not appear")
    assert capsys.readouterr().out == ""


# --- Manifest extraction helper ------------------------------------------
def test_manifest_tools_handles_unexpected_shapes():
    assert A._manifest_tools(None) == []
    assert A._manifest_tools("nonsense") == []
    assert A._manifest_tools({"unexpected": 1}) == [{"unexpected": 1}]
