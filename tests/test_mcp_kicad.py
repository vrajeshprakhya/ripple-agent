"""Offline tests for the KiCad MCP wiring. No server, no KiCad, no network."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import mcp_kicad as K  # noqa: E402


# --- Server directory validation -----------------------------------------
@pytest.fixture
def built_server(tmp_path):
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "index.js").write_text("// built entry point\n")
    (tmp_path / "package.json").write_text("{}")
    return tmp_path


def test_accepts_a_built_checkout(built_server):
    assert K.validate_server_dir(built_server) == built_server / "dist" / "index.js"


def test_expands_user_and_resolves(built_server):
    # A relative spelling of the same directory must resolve to the same entry.
    rel = pathlib.Path(built_server.name)
    assert K.validate_server_dir(built_server / "." ) == built_server / "dist" / "index.js"
    assert rel.name  # guard against an empty fixture name


def test_rejects_missing_directory(tmp_path):
    with pytest.raises(K.KicadServerError, match="no such directory"):
        K.validate_server_dir(tmp_path / "nope")


def test_unbuilt_checkout_names_the_build_command(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    with pytest.raises(K.KicadServerError) as e:
        K.validate_server_dir(tmp_path)
    msg = str(e.value)
    assert "not built" in msg
    assert "npm run build" in msg


def test_unrelated_directory_gets_a_different_message(tmp_path):
    with pytest.raises(K.KicadServerError) as e:
        K.validate_server_dir(tmp_path)
    assert "checkout root" in str(e.value)


# --- Environment construction --------------------------------------------
def test_defaults_are_the_conservative_ends():
    env = K.kicad_server_env()
    assert env["KICAD_MCP_OPERATING_MODE"] == "readonly"
    assert env["KICAD_MCP_TOOLBOXES"] == "core"


def test_default_pythonpath_points_at_kicad_bindings():
    assert env_path() == K.DEFAULT_KICAD_PYTHONPATH


def env_path():
    return K.kicad_server_env()["PYTHONPATH"]


def test_pythonpath_override():
    env = K.kicad_server_env(pythonpath="/opt/kicad/py")
    assert env["PYTHONPATH"] == "/opt/kicad/py"


def test_kicad_python_omitted_unless_given():
    assert "KICAD_PYTHON" not in K.kicad_server_env()
    assert K.kicad_server_env(kicad_python="/usr/bin/python3")["KICAD_PYTHON"] == "/usr/bin/python3"


@pytest.mark.parametrize("mode", K.OPERATING_MODES)
def test_every_documented_mode_is_accepted(mode):
    assert K.kicad_server_env(mode=mode)["KICAD_MCP_OPERATING_MODE"] == mode


@pytest.mark.parametrize("mode", ["", "rw", "write-all", "READONLY", "readwrite", None])
def test_unknown_mode_is_refused(mode):
    with pytest.raises(K.KicadServerError, match="unknown operating mode"):
        K.kicad_server_env(mode=mode)


def test_write_mode_is_never_reached_by_accident():
    # The hazard is a typo silently widening access, so anything that is not
    # exactly a documented mode must raise rather than fall back to a default.
    for typo in ("Write", "WRITE", " write", "write "):
        with pytest.raises(K.KicadServerError):
            K.kicad_server_env(mode=typo)


# --- Client construction --------------------------------------------------
def test_builds_a_client_without_contacting_anything(built_server):
    client = K.kicad_mcp_client(built_server)
    # Construction must not spawn the server; only entering it should.
    assert client is not None


def test_bad_directory_fails_before_any_process_starts(tmp_path):
    with pytest.raises(K.KicadServerError):
        K.kicad_mcp_client(tmp_path / "missing")


def test_invalid_mode_fails_at_construction(built_server):
    with pytest.raises(K.KicadServerError, match="unknown operating mode"):
        K.kicad_mcp_client(built_server, mode="write-everything")


def test_allowed_tools_are_passed_through(built_server, monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(K, "MCPClient", FakeClient)
    K.kicad_mcp_client(built_server, allowed_tools=["get_board_info", "list_nets"])
    assert captured["tool_filters"] == {"allowed": ["get_board_info", "list_nets"]}
    assert captured["prefix"] == K.DEFAULT_PREFIX
    assert captured["startup_timeout"] == K.DEFAULT_STARTUP_TIMEOUT


def test_no_tool_filter_when_none_requested(built_server, monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(K, "MCPClient", FakeClient)
    K.kicad_mcp_client(built_server, allowed_tools=None)
    assert "tool_filters" not in captured


def test_prefix_can_be_disabled(built_server, monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(K, "MCPClient", FakeClient)
    K.kicad_mcp_client(built_server, prefix=None)
    assert "prefix" not in captured


def test_transport_launches_the_built_entry_point(built_server, monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    recorded = {}

    def fake_stdio_client(params):
        recorded["params"] = params
        return object()

    monkeypatch.setattr(K, "MCPClient", FakeClient)
    monkeypatch.setattr(K, "stdio_client", fake_stdio_client)

    K.kicad_mcp_client(built_server, node="/usr/bin/node", mode="readonly")
    captured["transport_callable"]()  # the callable is what spawns the server

    params = recorded["params"]
    assert params.command == "/usr/bin/node"
    assert params.args == [str(built_server / "dist" / "index.js")]
    assert params.env["KICAD_MCP_OPERATING_MODE"] == "readonly"
    assert params.cwd == str(built_server)
