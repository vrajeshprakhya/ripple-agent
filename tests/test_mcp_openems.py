"""Tests for the mcp-openems launcher and, where it is installed, the server's numbers.

The live tests compare the third-party server's analytical output with formulas computed
here from first principles, so they check that the numbers are RIGHT rather than merely
present. They skip when the server is not installed (CI does not install it).
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import mcp_openems as M  # noqa: E402

SERVER_DIR = pathlib.Path.home() / "mcp-openems"
installed = (SERVER_DIR / ".venv" / "bin" / "mcp-openems").is_file()
needs_server = pytest.mark.skipif(not installed, reason="mcp-openems is not installed")
C_MM_PER_S = 299_792_458_000.0


def hammerstad_z0(w_mm: float, h_mm: float, er: float) -> tuple[float, float]:
    """Hammerstad-Jensen quasi-static microstrip: (effective permittivity, Z0 in ohms)."""
    u = w_mm / h_mm
    a = 1 + (1 / 49) * math.log((u**4 + (u / 52) ** 2) / (u**4 + 0.432)) + (1 / 18.7) * math.log(1 + (u / 18.1) ** 3)
    b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
    eeff = (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / u) ** (-a * b)
    f = 6 + (2 * math.pi - 6) * math.exp(-((30.666 / u) ** 0.7528))
    return eeff, 60 * math.log(f / u + math.sqrt(1 + (2 / u) ** 2)) / math.sqrt(eeff)


# --- Launcher ---------------------------------------------------------------------
def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(M.OpenemsServerError, match="no such directory"):
        M.openems_executable(tmp_path / "nope")


def test_an_uninstalled_checkout_is_told_how_to_install_it(tmp_path):
    with pytest.raises(M.OpenemsServerError) as e:
        M.openems_executable(tmp_path)
    msg = str(e.value)
    assert M.AUDITED_COMMIT in msg and "uv venv" in msg and "mcp-openems" in msg


def test_the_console_script_is_found(tmp_path):
    exe = tmp_path / ".venv" / "bin" / "mcp-openems"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    assert M.openems_executable(tmp_path) == exe.resolve()


def test_client_is_built_without_starting_anything(tmp_path):
    exe = tmp_path / ".venv" / "bin" / "mcp-openems"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    assert M.openems_mcp_client(tmp_path) is not None


def test_filter_is_passed_through_and_no_prefix_is_added(tmp_path, monkeypatch):
    exe = tmp_path / ".venv" / "bin" / "mcp-openems"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    seen = {}
    monkeypatch.setattr(M, "MCPClient", lambda **kw: seen.update(kw))
    M.openems_mcp_client(tmp_path, allowed_tools=["openems_create_microstrip"])
    assert seen["tool_filters"] == {"allowed": ["openems_create_microstrip"]}
    assert "prefix" not in seen  # the server's names already begin with openems_


def test_no_filter_key_when_none_requested(tmp_path, monkeypatch):
    exe = tmp_path / ".venv" / "bin" / "mcp-openems"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    seen = {}
    monkeypatch.setattr(M, "MCPClient", lambda **kw: seen.update(kw))
    M.openems_mcp_client(tmp_path)
    assert "tool_filters" not in seen


# --- The real server, checked against independent physics ---------------------------
@pytest.fixture(scope="module")
def live():
    client = M.openems_mcp_client(SERVER_DIR)
    with client:
        yield client


def call(client, name, **args):
    res = client.call_tool_sync("t", name, args)
    text = "".join(c.get("text", "") for c in res.get("content", []))
    return res.get("status"), (json.loads(text) if text.startswith("{") else text)


@needs_server
def test_server_exposes_its_sixteen_tools(live):
    names = {t.tool_name for t in live.list_tools_sync()}
    assert len(names) == 16
    assert {"openems_create_microstrip", "openems_create_dipole", "openems_create_patch",
            "openems_check_installation", "openems_generate_script"} <= names


@needs_server
def test_it_reports_honestly_whether_openems_itself_is_available(live):
    status, d = call(live, "openems_check_installation")
    assert status == "success" and isinstance(d["openems_available"], bool)
    if not d["openems_available"]:
        assert "import_error" in d and "simulation does not" in d["message"]


@needs_server
@pytest.mark.parametrize("w,h,er", [(3.0, 1.6, 4.2), (1.2, 0.8, 3.66), (0.3, 0.2, 3.38), (5.0, 1.0, 10.2)])
def test_microstrip_impedance_matches_hammerstad(live, w, h, er):
    status, d = call(live, "openems_create_microstrip", frequency_ghz=2.4, trace_width_mm=w,
                     trace_length_mm=20.0, dielectric_height_mm=h, substrate_er=er)
    assert status == "success", d
    calc = d["design"]["calculated"]
    eeff, z0 = hammerstad_z0(w, h, er)
    assert calc["z0_ohms"] == pytest.approx(z0, rel=0.01)
    assert calc["effective_er"] == pytest.approx(eeff, rel=0.01)


@needs_server
def test_microstrip_derived_quantities_are_consistent_with_each_other(live):
    _, d = call(live, "openems_create_microstrip", frequency_ghz=2.4, trace_width_mm=3.0,
                trace_length_mm=20.0, dielectric_height_mm=1.6)
    c = d["design"]["calculated"]
    lam = C_MM_PER_S / (2.4e9 * math.sqrt(c["effective_er"]))
    assert c["wavelength_mm"] == pytest.approx(lam, rel=0.001)
    assert c["electrical_length_deg"] == pytest.approx(20.0 / lam * 360, rel=0.001)
    assert c["delay_ps_per_mm"] == pytest.approx(math.sqrt(c["effective_er"]) / C_MM_PER_S * 1e12, rel=0.001)


@needs_server
def test_a_wider_trace_has_lower_impedance(live):
    z = []
    for w in (1.0, 2.0, 3.0, 4.0):
        _, d = call(live, "openems_create_microstrip", frequency_ghz=2.4, trace_width_mm=w,
                    trace_length_mm=20.0, dielectric_height_mm=1.6)
        z.append(d["design"]["calculated"]["z0_ohms"])
    assert z == sorted(z, reverse=True)


@needs_server
def test_dipole_wavelength_and_length_are_physical(live):
    _, d = call(live, "openems_create_dipole", frequency_ghz=2.45)
    dim = d["design"]["dimensions"]
    assert dim["wavelength_mm"] == pytest.approx(C_MM_PER_S / 2.45e9, rel=0.001)
    assert 0.44 < dim["total_length_mm"] / dim["wavelength_mm"] < 0.50  # a shortened half-wave dipole


@needs_server
def test_an_unknown_design_id_is_an_error_not_a_crash(live):
    status, text = call(live, "openems_get_design", design_id="does-not-exist")
    assert status == "error" or (isinstance(text, dict) and text.get("success") is False)
