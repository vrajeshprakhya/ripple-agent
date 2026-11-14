"""Tests for the mcp-pcb-emcopilot launcher and, where it is installed, the server itself.

The live tests skip where the server is not installed (CI). The security regression test matters
most: the server as published runs a hostile Gerber macro as code, and this checks the local
patch is still in place.
"""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import mcp_pcb as M  # noqa: E402

SERVER_DIR = pathlib.Path.home() / "mcp-pcb-emcopilot"
EXE = SERVER_DIR / ".venv" / "bin" / "mcp-pcb-emcopilot"
installed = EXE.is_file()
needs_server = pytest.mark.skipif(not installed, reason="mcp-pcb-emcopilot is not installed")


def hammerstad(w, h, er):
    u = w / h
    a = 1 + (1 / 49) * math.log((u**4 + (u / 52) ** 2) / (u**4 + 0.432)) + (1 / 18.7) * math.log(1 + (u / 18.1) ** 3)
    b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
    eeff = (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / u) ** (-a * b)
    f = 6 + (2 * math.pi - 6) * math.exp(-((30.666 / u) ** 0.7528))
    return 60 * math.log(f / u + math.sqrt(1 + (2 / u) ** 2)) / math.sqrt(eeff)


# --- Launcher ---------------------------------------------------------------------
def _fake_install(tmp_path):
    exe = tmp_path / ".venv" / "bin" / "mcp-pcb-emcopilot"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    return tmp_path


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(M.PcbServerError, match="no such directory"):
        M.pcb_executable(tmp_path / "nope")


def test_an_uninstalled_checkout_is_told_the_audited_commit(tmp_path):
    with pytest.raises(M.PcbServerError) as e:
        M.pcb_executable(tmp_path)
    assert M.AUDITED_COMMIT in str(e.value) and "eval patch" in str(e.value)


def test_unknown_profile_is_refused(tmp_path):
    with pytest.raises(M.PcbServerError, match="unknown profile"):
        M.pcb_mcp_client(_fake_install(tmp_path), profile="everything")


def test_core_profile_filters_to_the_curated_set(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(M, "MCPClient", lambda **kw: seen.update(kw))
    M.pcb_mcp_client(_fake_install(tmp_path))
    assert seen["tool_filters"] == {"allowed": list(M.CORE_TOOLS)}


def test_all_profile_has_no_filter(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(M, "MCPClient", lambda **kw: seen.update(kw))
    M.pcb_mcp_client(_fake_install(tmp_path), profile="all")
    assert "tool_filters" not in seen


def test_explicit_tools_override_the_profile(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(M, "MCPClient", lambda **kw: seen.update(kw))
    M.pcb_mcp_client(_fake_install(tmp_path), allowed_tools=["pcb_get_nets"])
    assert seen["tool_filters"] == {"allowed": ["pcb_get_nets"]}


@pytest.mark.parametrize("excluded", ["pcb_lookup_limit_live", "pcb_suggest_next_actions",
                                      "pcb_attach_external_result", "pcb_generate_docx_report",
                                      "pcb_generate_html_report", "pcb_export_all_renders",
                                      "pcb_validate_with_openems"])
def test_the_core_profile_leaves_out_tools_that_write_files_or_instruct_the_model(excluded):
    assert excluded not in M.CORE_TOOLS


# --- The real server ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def live():
    client = M.pcb_mcp_client(SERVER_DIR)
    with client:
        yield client


def call(client, tool, **args):
    res = client.call_tool_sync("t", tool, args)
    text = "".join(c.get("text", "") for c in res.get("content", []))
    return res.get("status"), (json.loads(text) if text.startswith("{") else text)


@needs_server
def test_every_curated_name_exists_on_the_server_and_the_manifest_is_small(live):
    tools = live.list_tools_sync()
    assert {t.tool_name for t in tools} == set(M.CORE_TOOLS)  # an upstream rename fails here
    size = len(json.dumps([t.tool_spec for t in tools], default=str))
    assert size < 15_000  # about 2k tokens, against ~20k for all 131 tools


@needs_server
@pytest.mark.parametrize("w,h,er", [(3.0, 1.6, 4.2), (1.2, 0.8, 3.66), (2.0, 1.0, 10.2), (6.0, 1.6, 4.2)])
def test_microstrip_impedance_agrees_with_hammerstad(live, w, h, er):
    status, d = call(live, "pcb_calc_microstrip_impedance", trace_width_mm=w, dielectric_height_mm=h,
                     trace_thickness_mm=0.035, dielectric_constant=er)
    assert status == "success", d
    assert d["impedance_ohms"] == pytest.approx(hammerstad(w, h, er), rel=0.03)


@needs_server
def test_a_hostile_gerber_macro_does_not_execute_code(tmp_path):
    """Regression for the local patch: the published eval() ran this payload and created the marker."""
    marker = tmp_path / "pwned"
    payload = ("[c for c in ().__class__.__base__.__subclasses__() if c.__name__ == 'catch_warnings'][0]()"
               f"._module.__builtins__['__import__']('os').system('touch {marker}')")
    code = ("from mcp_pcb_emcopilot.parsers.gerber_parser import ApertureMacro\n"
            f"ApertureMacro(name='EVIL', primitives=['1,' + {payload!r}]).evaluate([])\n")
    subprocess.run([str(SERVER_DIR / ".venv" / "bin" / "python"), "-c", code], capture_output=True, timeout=60)
    assert not marker.exists(), "the Gerber macro eval() patch is missing: parsing a file ran its code"


@needs_server
def test_legitimate_gerber_macro_arithmetic_still_evaluates():
    code = ("from mcp_pcb_emcopilot.parsers.gerber_parser import ApertureMacro\n"
            "r = ApertureMacro(name='OK', primitives=['1,1,0.5*2,(1+1)/4,-0.25']).evaluate([])\n"
            "print(r[0].parameters)\n")
    out = subprocess.run([str(SERVER_DIR / ".venv" / "bin" / "python"), "-c", code],
                         capture_output=True, text=True, timeout=60).stdout
    assert out.strip() == "[1.0, 0.5, -0.25]"


@needs_server
def test_the_board_counts_match_an_independent_parse(live):
    fixture = SERVER_DIR / "tests" / "fixtures" / "mixed_signal_4layer.kicad_pcb"
    status, d = call(live, "pcb_parse_layout", file_path=str(fixture))
    assert status == "success", d
    assert len(d["vias"]) == 10 if isinstance(d["vias"], list) else d["vias"] == 10
    # the server counts KiCad's unconnected pseudo-net (net 0, empty name); 24 named + that one
    assert (len(d["nets"]) if isinstance(d["nets"], list) else d["nets"]) == 25
    # The stackup is right: four physical copper layers. Its list holds the board's layer table
    # (typed signal/power/user) and then the physical stack (typed copper/prepreg/core), so filter
    # on the type, not the name, which appears in both.
    _, s = call(live, "pcb_get_stackup", session_id=d["session_id"])
    copper = [layer for layer in s["layers"] if layer["type"] == "copper"]
    assert [layer["name"] for layer in copper] == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    # Known quirk of the third-party tool, pinned so it is noticed if it changes: the headline
    # layer_count is 5 on this 4-copper-layer board, so use the stackup for the real number.
    assert d["layer_count"] == 5
