"""Tests for the validated full-wave server.

The gate and validation tests are offline. The live tests run a real openEMS simulation (about a
minute each) and skip where the solver is not installed, which includes CI.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import mcp_emverify as M  # noqa: E402
from emverify_mcp import core  # noqa: E402
from emverify_mcp.core import EmError  # noqa: E402

installed = core.solver_status().get("available", False)
needs_solver = pytest.mark.skipif(not installed, reason="openEMS solver not installed")

GOOD = {"width_mm": 3.0, "height_mm": 1.6, "er": 4.2, "f_max_ghz": 6.0, "length_mm": 60.0}


def raw(power=(0.998, 1.0, 1.001), z0=50.66, eeff=3.236):
    return {"power_balance": {"min": power[0], "median": power[1], "max": power[2]},
            "z0_ohm": {"low_band_median": z0, "at_f_max": z0}, "eeff": {"low_band_median": eeff, "at_f_max": eeff},
            "s11_worst_db": -48.6, "s21_worst_db": -0.01, "f_ghz": [1.0], "s11_db": [-48.0], "s21_db": [-0.01],
            "grid": [10, 10, 10]}


# --- Validation ---------------------------------------------------------------------
def test_a_valid_request_passes_and_is_made_float():
    v = core.validate({**GOOD, "width_mm": 3})
    assert v["width_mm"] == 3.0 and isinstance(v["width_mm"], float)


@pytest.mark.parametrize("key,bad", [
    ("width_mm", 0.0), ("width_mm", 25.0), ("height_mm", 0.01), ("er", 0.5), ("er", 25.0),
    ("f_max_ghz", 0.1), ("f_max_ghz", 50.0), ("length_mm", 1.0), ("length_mm", 500.0),
])
def test_out_of_range_values_are_refused(key, bad):
    with pytest.raises(EmError, match="outside the validated range"):
        core.validate({**GOOD, key: bad})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "3", None, True, [3.0]])
def test_non_numbers_and_non_finite_values_are_refused(bad):
    with pytest.raises(EmError):
        core.validate({**GOOD, "width_mm": bad})


def test_a_missing_field_is_refused():
    p = dict(GOOD)
    del p["er"]
    with pytest.raises(EmError, match="missing er"):
        core.validate(p)


def test_a_line_too_short_for_the_measurement_plane_is_refused():
    with pytest.raises(EmError, match="12 times"):
        core.validate({**GOOD, "height_mm": 5.0, "length_mm": 40.0})


def test_there_is_no_way_to_pass_code():
    # The model supplies numbers only; extra keys never reach the simulation script.
    v = core.validate({**GOOD, "script": "__import__('os').system('id')"})
    assert set(v) == set(core.LIMITS)


# --- Hammerstad, against independent numbers ---------------------------------------
@pytest.mark.parametrize("w,h,er,z0", [(3.0, 1.6, 4.2, 51.72), (2.0, 1.0, 10.2, 32.96), (1.2, 0.8, 3.66, 62.48)])
def test_hammerstad_matches_values_the_mcp_openems_server_agreed_with(w, h, er, z0):
    assert core.hammerstad(w, h, er)[0] == pytest.approx(z0, abs=0.05)


# --- The gate ----------------------------------------------------------------------
def test_a_physical_result_passes_with_the_comparison_attached():
    r = core.judge(GOOD, raw())
    assert r["energy_check"]["passed"] and r["agrees_with_theory"]
    assert r["analytical_hammerstad"]["z0_ohm"] == pytest.approx(51.72, abs=0.05)
    assert r["difference_percent"]["z0"] == pytest.approx(-2.1, abs=0.2)


@pytest.mark.parametrize("power", [(1.7, 1.8, 1.9), (0.99, 1.0, 1.04), (0.75, 0.77, 0.81), (0.5, 0.6, 0.7)])
def test_non_physical_results_are_refused_not_reported(power):
    # 1.7 to 1.9 is what the generated mcp-openems script produced for a matched line.
    with pytest.raises(EmError, match="refusing to report"):
        core.judge(GOOD, raw(power=power))


def test_a_wrong_impedance_is_reported_but_flagged_not_hidden():
    r = core.judge(GOOD, raw(z0=80.0))
    assert r["energy_check"]["passed"] and not r["agrees_with_theory"]
    assert r["difference_percent"]["z0"] > 40


def test_the_agreement_threshold_is_five_percent():
    z_h = core.hammerstad(3.0, 1.6, 4.2)[0]
    assert core.judge(GOOD, raw(z0=z_h * 1.049))["agrees_with_theory"]
    assert not core.judge(GOOD, raw(z0=z_h * 1.06))["agrees_with_theory"]


def test_delay_follows_from_effective_permittivity():
    r = core.judge(GOOD, raw(eeff=4.0))
    assert r["full_wave"]["delay_ps_per_mm"] == pytest.approx(2.0 / 299.792458 * 1000, rel=1e-3)


# --- Solver discovery -----------------------------------------------------------------
def test_a_missing_solver_python_is_explained(monkeypatch, tmp_path):
    monkeypatch.setenv("EMVERIFY_PYTHON", str(tmp_path / "nope"))
    s = core.solver_status()
    assert s["available"] is False and "EMVERIFY_PYTHON" in s["reason"]


def test_a_python_without_openems_is_reported_unavailable(monkeypatch):
    monkeypatch.setenv("EMVERIFY_PYTHON", sys.executable)  # the agent venv has no openEMS
    s = core.solver_status()
    assert s["available"] is False
    assert "openEMS" in s["reason"] or "CSXCAD" in s["reason"]


# --- Launcher -----------------------------------------------------------------------
def test_launcher_passes_the_solver_python_and_prefix(tmp_path, monkeypatch):
    seen, params = {}, {}
    monkeypatch.setattr(M, "MCPClient", lambda **kw: seen.update(kw))
    monkeypatch.setattr(M, "stdio_client", lambda p: params.setdefault("p", p) or object())
    M.emverify_mcp_client(solver_python="/opt/py/bin/python", python="/usr/bin/python3")
    seen["transport_callable"]()
    p = params["p"]
    assert p.args == ["-m", "emverify_mcp"] and p.env["EMVERIFY_PYTHON"] == "/opt/py/bin/python"
    assert seen["prefix"] == "emverify"


def test_launcher_filter_passthrough(monkeypatch):
    seen = {}
    monkeypatch.setattr(M, "MCPClient", lambda **kw: seen.update(kw))
    M.emverify_mcp_client(allowed_tools=["simulate_microstrip"])
    assert seen["tool_filters"] == {"allowed": ["simulate_microstrip"]}


# --- Live: the real solver, end to end ----------------------------------------------------
def call(client, tool, **args):
    res = client.call_tool_sync("t", tool, args)
    text = "".join(c.get("text", "") for c in res.get("content", []))
    return res.get("status"), (json.loads(text) if text.startswith("{") else text)


@pytest.fixture(scope="module")
def live():
    client = M.emverify_mcp_client()
    with client:
        yield client


@needs_solver
def test_server_lists_its_tools(live):
    names = {t.tool_name for t in live.list_tools_sync()}
    assert names == {"emverify_check_solver", "emverify_simulate_microstrip"}


@needs_solver
def test_check_solver_reports_versions(live):
    status, d = call(live, "check_solver")
    assert status == "success" and d["available"] and d["openEMS"]


@needs_solver
def test_full_wave_microstrip_agrees_with_theory_and_conserves_energy(live):
    status, d = call(live, "simulate_microstrip", trace_width_mm=3.0, dielectric_height_mm=1.6,
                     substrate_er=4.2, f_max_ghz=6.0, line_length_mm=60.0)
    assert status == "success", d
    assert d["agrees_with_theory"]
    assert d["energy_check"]["power_balance_max"] <= 1.03 and d["energy_check"]["power_balance_min"] >= 0.9
    assert abs(d["difference_percent"]["z0"]) < 5 and abs(d["difference_percent"]["effective_er"]) < 5
    assert d["full_wave"]["s11_worst_db"] < -30  # a straight line is matched to its own impedance


@needs_solver
def test_a_wider_trace_has_lower_simulated_impedance(live):
    z = []
    for w in (1.5, 3.0):
        _, d = call(live, "simulate_microstrip", trace_width_mm=w, dielectric_height_mm=1.6,
                    substrate_er=4.2, f_max_ghz=4.0, line_length_mm=40.0)
        z.append(d["full_wave"]["z0_ohm"])
    assert z[0] > z[1]


@needs_solver
def test_an_out_of_range_request_is_refused_before_any_simulation(live):
    status, text = call(live, "simulate_microstrip", trace_width_mm=500.0, dielectric_height_mm=1.6,
                        substrate_er=4.2)
    assert status == "error" and "outside the validated range" in text
