"""Tests for the ngspice MCP launcher and, where ngspice exists, the server end to end."""
from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import mcp_spice as M  # noqa: E402
from ngspice_mcp import core  # noqa: E402

needs_ngspice = pytest.mark.skipif(not core.have_ngspice(), reason="ngspice not installed")

RC = "RC lowpass\nV1 in 0 DC 0 AC 1 PULSE(0 1 0 1n 1n 5m 10m)\nR1 in out 1k\nC1 out 0 159.15n\n"
TAU = 1000 * 159.15e-9


# --- Launcher ---------------------------------------------------------------
def test_env_carries_resolved_roots(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    env = M.spice_server_env([a, b])
    assert env["NGSPICE_MCP_ROOTS"].split(":") == [str(a.resolve()), str(b.resolve())]
    assert env["PYTHONPATH"] == str(M.REPO)


def test_a_missing_root_is_refused(tmp_path):
    with pytest.raises(M.SpiceServerError, match="not a directory"):
        M.spice_server_env([tmp_path / "nope"])


def test_no_roots_is_refused():
    with pytest.raises(M.SpiceServerError, match="at least one"):
        M.spice_server_env([])


def test_a_missing_ngspice_binary_is_refused(tmp_path):
    with pytest.raises(M.SpiceServerError, match="no such ngspice"):
        M.spice_server_env([tmp_path], ngspice=str(tmp_path / "ngspice"))


def test_ngspice_binary_is_passed_through_when_given(tmp_path):
    fake = tmp_path / "ngspice"
    fake.write_text("#!/bin/sh\n")
    assert M.spice_server_env([tmp_path], ngspice=str(fake))["NGSPICE_BIN"] == str(fake)


def test_client_is_built_without_starting_anything(tmp_path):
    assert M.spice_mcp_client([tmp_path]) is not None


def test_filter_and_prefix_are_passed_through(tmp_path, monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(M, "MCPClient", Fake)
    M.spice_mcp_client([tmp_path], allowed_tools=["run_ac"])
    assert seen["tool_filters"] == {"allowed": ["run_ac"]}
    assert seen["prefix"] == "spice"


def test_no_filter_key_when_none_requested(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(M, "MCPClient", lambda **kw: seen.update(kw))
    M.spice_mcp_client([tmp_path], prefix=None)
    assert "tool_filters" not in seen and "prefix" not in seen


def test_transport_runs_the_module_from_the_repo(tmp_path, monkeypatch):
    seen, params = {}, {}
    monkeypatch.setattr(M, "MCPClient", lambda **kw: seen.update(kw))
    monkeypatch.setattr(M, "stdio_client", lambda p: params.setdefault("p", p) or object())
    M.spice_mcp_client([tmp_path], python="/usr/bin/python3")
    seen["transport_callable"]()
    p = params["p"]
    assert p.command == "/usr/bin/python3" and p.args == ["-m", "ngspice_mcp"]
    assert p.cwd == str(M.REPO)
    assert p.env["NGSPICE_MCP_ROOTS"] == str(tmp_path.resolve())


# --- The real server over stdio ----------------------------------------------------
def _call(client, name, **args):
    res = client.call_tool_sync("t", name, args)
    text = "".join(c.get("text", "") for c in res.get("content", []))
    return res.get("status"), text


def _json(text):
    return json.loads(text)


@pytest.fixture
def live(tmp_path):
    (tmp_path / "rc.cir").write_text(RC)
    client = M.spice_mcp_client([tmp_path])
    with client:
        yield client, tmp_path


@needs_ngspice
def test_server_lists_its_tools(live):
    client, _ = live
    names = {t.tool_name for t in client.list_tools_sync()}
    assert {"spice_run_ac", "spice_run_transient", "spice_run_op", "spice_run_dc_sweep",
            "spice_measure", "spice_get_vector", "spice_check_netlist",
            "spice_export_kicad_netlist"} <= names


@needs_ngspice
def test_ac_through_the_server_finds_the_three_db_point(live):
    client, root = live
    status, text = _call(client, "run_ac", start_hz="1", stop_hz="1meg", points_per_decade=40,
                         probes=["v(out)"], netlist_path=str(root / "rc.cir"))
    assert status == "success", text
    bw = _json(text)["bandwidth_3db_hz"]["v(out)"]
    assert bw == pytest.approx(1 / (2 * math.pi * TAU), rel=0.01)


@needs_ngspice
def test_transient_then_measure_rise_time_by_result_id(live):
    client, root = live
    status, text = _call(client, "run_transient", step="1u", stop="2m", max_step="1u",
                         probes=["v(out)"], netlist_path=str(root / "rc.cir"))
    assert status == "success", text
    rid = _json(text)["result_id"]
    status, text = _call(client, "measure", result_id=rid, kind="rise_time_10_90", vector="v(out)")
    assert status == "success", text
    assert _json(text)["seconds"] == pytest.approx(2.197 * TAU, rel=0.02)


@needs_ngspice
def test_op_returns_node_voltages(live):
    client, _ = live
    status, text = _call(client, "run_op", netlist="d\nV1 in 0 DC 5\nR1 in mid 10k\nR2 mid 0 10k\n")
    assert status == "success", text
    assert _json(text)["values"]["v(mid)"] == pytest.approx(2.5)


@needs_ngspice
def test_a_netlist_outside_the_roots_is_refused(live, tmp_path_factory):
    client, _ = live
    outside = tmp_path_factory.mktemp("elsewhere") / "x.cir"
    outside.write_text(RC)
    status, text = _call(client, "run_op", netlist_path=str(outside))
    assert status == "error" and "outside the allowed roots" in text


@needs_ngspice
def test_a_hostile_netlist_is_refused_by_the_server_and_nothing_runs(live):
    client, root = live
    marker = root / "pwned"
    status, text = _call(client, "run_op", netlist=RC + f".control\nshell touch {marker}\n.endc\n")
    assert status == "error" and "not allowed" in text
    assert not marker.exists()


@needs_ngspice
def test_a_probe_that_is_a_command_is_refused(live):
    client, _ = live
    status, text = _call(client, "run_ac", start_hz="1", stop_hz="1k", probes=["v(out); shell id"], netlist=RC)
    assert status == "error"


@needs_ngspice
def test_giving_both_or_neither_netlist_form_is_an_error(live):
    client, root = live
    assert _call(client, "run_op")[0] == "error"
    assert _call(client, "run_op", netlist=RC, netlist_path=str(root / "rc.cir"))[0] == "error"


@needs_ngspice
def test_compute_is_exposed_and_gets_the_shunt_current_right(live):
    client, _ = live
    assert "spice_compute" in {t.tool_name for t in client.list_tools_sync()}
    status, text = _call(client, "compute", expression="1.00002 mV / 1 mohm")
    assert status == "success", text
    r = _json(text)
    assert r["unit"] == "A" and r["display"] == "1.00002 A"


@needs_ngspice
@pytest.mark.parametrize("name,args", [
    ("run_ac", {"start_hz": "1", "stop_hz": "1k", "netlist": RC}),
    ("run_transient", {"step": "1u", "stop": "100u", "netlist": RC}),
    ("run_dc_sweep", {"source": "V1", "start": "0", "stop": "1", "step": "0.5", "netlist": RC}),
])
def test_sweeps_without_probes_are_refused_not_dumped(live, name, args):
    # On a real board this used to return every one of 99 vectors: 144 KB from one call.
    client, _ = live
    status, text = _call(client, name, **args)
    assert status == "error" and "name the vectors" in text and len(text) < 400


@needs_ngspice
def test_too_many_probes_are_refused(live):
    client, _ = live
    status, text = _call(client, "run_ac", start_hz="1", stop_hz="1k", netlist=RC,
                         probes=[f"v(n{i})" for i in range(13)])
    assert status == "error" and "at most 12" in text


@needs_ngspice
def test_compute_reads_values_from_a_run_op_result_with_units_attached(live):
    client, _ = live
    net = "board\nVS in 0 DC 5\nR1 in sense 100\nRSH sense 0 0.5\n"  # 10 mA... 5/100.5 A through 0.5 ohm
    _, text = _call(client, "run_op", netlist=net)
    rid = _json(text)["result_id"]
    status, text = _call(client, "compute", expression="v(sense) / 0.5 ohm", result_id=rid)
    assert status == "success", text
    r = _json(text)
    assert r["unit"] == "A" and r["value_si"] == pytest.approx(5 / 100.5, rel=1e-4)
    assert r["display"].endswith("mA")  # 49.75 mA, stated with its unit rather than a bare number


@needs_ngspice
def test_compute_resolves_a_differential_reference(live):
    client, _ = live
    net = "d\nVS in 0 DC 5\nR1 in mid 1k\nR2 mid 0 1k\n"
    rid = _json(_call(client, "run_op", netlist=net)[1])["result_id"]
    status, text = _call(client, "compute", expression="v(in,mid) / 1 kohm", result_id=rid)
    assert status == "success", text
    assert _json(text)["value_si"] == pytest.approx(2.5e-3)


@needs_ngspice
def test_compute_refuses_a_vector_without_a_result(live):
    client, _ = live
    status, text = _call(client, "compute", expression="v(out) / 1 kohm")
    assert status == "error" and "needs a result_id" in text


@needs_ngspice
def test_compute_refuses_references_into_a_non_operating_point_result(live):
    client, _ = live
    rid = _json(_call(client, "run_transient", step="1u", stop="10u", probes=["v(out)"], netlist=RC)[1])["result_id"]
    status, text = _call(client, "compute", expression="v(out) / 1 kohm", result_id=rid)
    assert status == "error" and "operating-point" in text


@needs_ngspice
def test_compute_refuses_an_unknown_vector(live):
    client, _ = live
    rid = _json(_call(client, "run_op", netlist="d\nV1 a 0 DC 1\nR1 a 0 1k\n")[1])["result_id"]
    status, text = _call(client, "compute", expression="v(nope) / 1 kohm", result_id=rid)
    assert status == "error" and "no vector" in text


@needs_ngspice
def test_compute_refuses_unlike_quantities_with_a_readable_reason(live):
    client, _ = live
    status, text = _call(client, "compute", expression="1 V + 1 A")
    assert status == "error" and "cannot combine" in text


@needs_ngspice
def test_every_analysis_result_states_its_units(live):
    client, root = live
    for name, args in [
        ("run_op", {"netlist": "d\nV1 in 0 DC 5\nR1 in 0 1k\n"}),
        ("run_ac", {"start_hz": "1", "stop_hz": "1k", "probes": ["v(out)"], "netlist": RC}),
        ("run_transient", {"step": "1u", "stop": "100u", "probes": ["v(out)"], "netlist": RC}),
        ("run_dc_sweep", {"source": "V1", "start": "0", "stop": "1", "step": "0.5", "probes": ["v(in)"],
                          "netlist": "d\nV1 in 0 DC 0\nR1 in 0 1k\n"}),
    ]:
        status, text = _call(client, name, **args)
        assert status == "success", (name, text)
        assert "SI base units" in _json(text)["units"], name


SHUNT = "shunt\nI1 0 a DC 1\nR1 a 0 0.001\nV9 b 0 DC 3.688\nRL b 0 1k\n"  # 1 A through 1 mohm


@needs_ngspice
def test_derived_shunt_current_comes_back_in_amperes_ready_made(live):
    # The exact failure: 1 mV across a 1 mohm shunt, reported as milliamperes.
    client, _ = live
    status, text = _call(client, "run_op", netlist=SHUNT,
                         derived={"shunt_current": "v(a) / 1 mohm", "cell": "v(b)"})
    assert status == "success", text
    d = _json(text)["derived"]
    assert d["shunt_current"]["unit"] == "A"
    assert d["shunt_current"]["display"] == "1 A"
    assert d["cell"]["display"] == "3.688 V"


@needs_ngspice
def test_derived_works_with_a_differential_reference(live):
    client, _ = live
    status, text = _call(client, "run_op", netlist="d\nVS in 0 DC 5\nR1 in mid 1k\nR2 mid 0 1k\n",
                         derived={"i": "v(in,mid) / 1 kohm"})
    assert status == "success", text
    assert _json(text)["derived"]["i"]["display"] == "2.5 mA"


@needs_ngspice
def test_op_without_derived_has_no_derived_key(live):
    client, _ = live
    status, text = _call(client, "run_op", netlist=SHUNT)
    assert status == "success" and "derived" not in _json(text)


@needs_ngspice
def test_a_bad_derived_expression_fails_the_whole_call_with_its_reason(live):
    client, _ = live
    status, text = _call(client, "run_op", netlist=SHUNT, derived={"x": "v(a) + 1 A"})
    assert status == "error" and "cannot combine" in text


@needs_ngspice
def test_a_derived_reference_to_an_unknown_node_is_refused(live):
    client, _ = live
    status, text = _call(client, "run_op", netlist=SHUNT, derived={"x": "v(nope) / 1 ohm"})
    assert status == "error" and "no vector" in text


@needs_ngspice
@pytest.mark.parametrize("label", ["", "1x", "a b", "x;y", "a" * 50, "x-y"])
def test_derived_names_must_be_plain_identifiers(live, label):
    client, _ = live
    status, text = _call(client, "run_op", netlist=SHUNT, derived={label: "v(a) / 1 ohm"})
    assert status == "error" and "plain identifier" in text


@needs_ngspice
def test_too_many_derived_quantities_are_refused(live):
    client, _ = live
    status, text = _call(client, "run_op", netlist=SHUNT,
                         derived={f"q{i}": "v(a) / 1 ohm" for i in range(13)})
    assert status == "error" and "at most 12" in text


@needs_ngspice
def test_unknown_result_id_is_a_clear_error(live):
    client, _ = live
    status, text = _call(client, "measure", result_id="nope", kind="rms", vector="v(out)")
    assert status == "error" and "unknown result_id" in text
