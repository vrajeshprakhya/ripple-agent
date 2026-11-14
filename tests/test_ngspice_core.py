"""Tests for the ngspice MCP core: the sanitizer, the parser and, where ngspice is
installed, real simulations checked against closed-form answers.

The sanitizer tests matter most. Everything a model sends reaches ngspice through
it, and ngspice's control language can run shell commands.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ngspice_mcp import core  # noqa: E402
from ngspice_mcp.core import SpiceError  # noqa: E402

needs_ngspice = pytest.mark.skipif(not core.have_ngspice(), reason="ngspice not installed")

RC = """RC lowpass
V1 in 0 DC 0 AC 1 PULSE(0 1 0 1n 1n 5m 10m)
R1 in out 1k
C1 out 0 159.15n
"""
TAU = 1000 * 159.15e-9  # 159.15 us


# --- Sanitizer: accepts ------------------------------------------------------
def test_accepts_a_plain_circuit():
    clean, notes = core.sanitize_netlist(RC)
    assert "R1 in out 1k" in clean
    assert notes == []


def test_first_line_text_becomes_a_comment_title():
    clean, _ = core.sanitize_netlist(RC)
    assert clean.splitlines()[0] == "* RC lowpass"


def test_first_line_that_is_a_component_is_not_swallowed_as_a_title():
    # ngspice treats line 1 as a title, which would silently delete this resistor.
    clean, notes = core.sanitize_netlist("R1 a b 1k\nR2 b 0 1k\n")
    assert "R1 a b 1k" in clean.splitlines()
    assert any("looked like a component" in n for n in notes)


@pytest.mark.parametrize("title", [
    "Two-stage RC low-pass",           # T is a transmission line: once read as a component
    "Series RLC band-pass",            # S is a switch
    "Low pass filter design",          # L would be an inductor
    "Voltage divider example circuit",
    "Current source with load",
    "Resistor ladder for testing",
])
def test_ordinary_titles_starting_with_element_letters_stay_titles(title):
    clean, notes = core.sanitize_netlist(f"{title}\nV1 a 0 DC 1\nR1 a 0 1k\n")
    assert clean.splitlines()[0] == f"* {title}"
    assert notes == []


@pytest.mark.parametrize("first", ["R1 a b 1k", "C1 out 0 100n", "V1 in 0 DC 5", "V1 in 0 5", "I1 0 a 1m"])
def test_a_real_first_component_is_kept(first):
    clean, notes = core.sanitize_netlist(f"{first}\nR9 a 0 1k\n")
    assert first in clean.splitlines()
    assert any("looked like a component" in n for n in notes)


def test_analysis_cards_and_end_are_stripped_not_rejected():
    clean, notes = core.sanitize_netlist(RC + ".op\n.tran 1u 1m\n.ac dec 10 1 1meg\n.dc V1 0 1 0.1\n.end\n")
    for card in (".op", ".tran", ".ac", ".dc", ".end"):
        assert not any(l.strip().lower().startswith(card) for l in clean.splitlines())
    assert len(notes) == 4


def test_model_and_subckt_and_param_pass():
    clean, _ = core.sanitize_netlist(
        "* t\n.param rv=1k\n.model dmod D(Is=1e-14)\n.subckt buf a b\nR1 a b {rv}\n.ends\nX1 in out buf\n")
    assert ".model dmod" in clean and ".subckt buf" in clean


def test_continuation_lines_are_joined():
    clean, _ = core.sanitize_netlist("* t\nR1 a b\n+ 1k\n")
    assert "R1 a b 1k" in clean


# --- Sanitizer: refuses (the part that matters) -------------------------------
@pytest.mark.parametrize("directive", [
    ".control", " .control", "\t.control", ".CONTROL", ".Control", ".endc", ".ENDC",
    ".write /tmp/x.raw", ".print tran v(out)", ".plot tran v(out)", ".measure tran x max v(out)",
    ".csparam x=1", ".cm foo", ".spice shell", ".shell id",
])
def test_refuses_dangerous_or_unlisted_directives(directive):
    with pytest.raises(SpiceError, match="not allowed"):
        core.sanitize_netlist(f"* t\nR1 a b 1k\n{directive}\n")


def test_refuses_a_directive_split_across_continuation_lines():
    # joined it reads ".cont rol", which is not on the allowlist either
    with pytest.raises(SpiceError):
        core.sanitize_netlist("* t\n.cont\n+rol\nshell id\n.endc\n")


@pytest.mark.parametrize("sep", ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", " ", " ", "\x00", "\x1b"])
def test_refuses_line_separators_python_and_ngspice_disagree_on(sep):
    # Python's splitlines() would see a hidden directive as its own line; a checker
    # built on a different split than the consumer is exactly how one gets past it.
    with pytest.raises(SpiceError, match="control characters"):
        core.sanitize_netlist(f"* t\nR1 a b 1k{sep}.control\nshell id\n.endc\n")


def test_refuses_backticks():
    with pytest.raises(SpiceError, match="backtick"):
        core.sanitize_netlist("* t\nR1 a b `id`\n")


def test_refuses_oversized_netlists():
    with pytest.raises(SpiceError, match="larger than"):
        core.sanitize_netlist("* t\n" + "R1 a b 1k\n" * 200_000)


def test_carriage_returns_normalise_to_newlines_before_checking():
    with pytest.raises(SpiceError, match="not allowed"):
        core.sanitize_netlist("* t\rR1 a b 1k\r.control\rshell id\r.endc\r")


# --- Includes -----------------------------------------------------------------
@pytest.fixture
def root(tmp_path):
    r = tmp_path / "models"
    r.mkdir()
    (r / "diode.lib").write_text(".model d1 D\n")
    (r / "notes.py").write_text("print('x')\n")
    return r


def test_include_inside_root_is_allowed_and_made_absolute(root):
    clean, _ = core.sanitize_netlist(f"* t\n.include {root}/diode.lib\nR1 a b 1k\n", roots=[root])
    assert f'.include "{(root / "diode.lib").resolve()}"' in clean


def test_relative_include_resolves_against_base_dir(root):
    clean, _ = core.sanitize_netlist("* t\n.lib diode.lib\n", roots=[root], base_dir=root)
    assert str((root / "diode.lib").resolve()) in clean


def test_include_with_no_roots_configured_is_refused():
    with pytest.raises(SpiceError, match="no include roots"):
        core.sanitize_netlist("* t\n.include /etc/passwd\n")


@pytest.mark.parametrize("target", ["/etc/passwd", "/etc/hosts", "../../etc/passwd"])
def test_include_outside_root_is_refused(root, target):
    with pytest.raises(SpiceError):
        core.sanitize_netlist(f"* t\n.include {target}\n", roots=[root], base_dir=root)


def test_include_of_a_non_model_suffix_is_refused(root):
    with pytest.raises(SpiceError, match="not allowed"):
        core.sanitize_netlist(f"* t\n.include {root}/notes.py\n", roots=[root])


def test_include_through_a_symlink_to_outside_is_refused(root, tmp_path):
    secret = tmp_path / "secret.lib"
    secret.write_text(".model x D\n")
    link = root / "innocent.lib"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(SpiceError, match="outside the allowed roots"):
        core.sanitize_netlist(f"* t\n.include {link}\n", roots=[root])


def test_include_of_a_missing_file_is_refused(root):
    with pytest.raises(SpiceError, match="does not exist"):
        core.sanitize_netlist(f"* t\n.include {root}/nope.lib\n", roots=[root])


def test_include_with_trailing_junk_is_refused(root):
    with pytest.raises(SpiceError, match="unexpected text"):
        core.sanitize_netlist(f"* t\n.lib {root}/diode.lib ; shell id\n", roots=[root])


@pytest.mark.parametrize("body", [
    ".control\nshell touch /tmp/x\n.endc\n",
    "* innocent looking\n.CONTROL\nshell id\n.ENDC\n",
    ".model d1 D\n.write /tmp/steal.raw\n",
    ".model d1 D\n .control\n",
    ".model d1 D\n.cont\n+rol\n",
    ".model d1 D\x0c.control\n",
    ".model d1 D `id`\n",
])
def test_a_hostile_library_inside_the_root_is_refused(root, body):
    # Being under an allowed root must not make a file's CONTENT trusted: ngspice
    # splices an included file into the deck, so its .control block would run.
    (root / "evil.lib").write_text(body)
    with pytest.raises(SpiceError):
        core.sanitize_netlist(f"* t\n.include {root}/evil.lib\n", roots=[root])


def test_a_clean_library_with_sections_and_subckts_passes(root):
    (root / "vendor.lib").write_text(
        ".lib TYPICAL\n.param kp=1\n.subckt amp a b\nR1 a b 1k\n.ends\n.model q1 NPN\n.endl TYPICAL\n")
    clean, _ = core.sanitize_netlist(f"* t\n.lib {root}/vendor.lib TYPICAL\n", roots=[root])
    assert "vendor.lib" in clean


def test_a_nested_include_to_a_hostile_file_is_refused(root):
    (root / "evil.lib").write_text(".control\nshell id\n.endc\n")
    (root / "outer.lib").write_text(f".include {root}/evil.lib\n.model d1 D\n")
    with pytest.raises(SpiceError):
        core.sanitize_netlist(f"* t\n.include {root}/outer.lib\n", roots=[root])


def test_a_nested_include_outside_the_root_is_refused(root, tmp_path):
    outside = tmp_path / "elsewhere.lib"
    outside.write_text(".model x D\n")
    (root / "outer.lib").write_text(f".include {outside}\n")
    with pytest.raises(SpiceError, match="outside the allowed roots"):
        core.sanitize_netlist(f"* t\n.include {root}/outer.lib\n", roots=[root])


def test_an_include_cycle_does_not_recurse_forever(root):
    (root / "a.lib").write_text(f".include {root}/b.lib\n")
    (root / "b.lib").write_text(f".include {root}/a.lib\n")
    with pytest.raises(SpiceError, match="nested more than"):
        core.sanitize_netlist(f"* t\n.include {root}/a.lib\n", roots=[root])


def test_read_netlist_file_refuses_paths_outside_roots(root, tmp_path):
    outside = tmp_path / "out.cir"
    outside.write_text("* x\n")
    with pytest.raises(SpiceError, match="outside the allowed roots"):
        core.read_netlist_file(str(outside), [root])


# --- Validators -----------------------------------------------------------------
@pytest.mark.parametrize("v", ["10k", "159.15n", "1e-3", "2meg", "10uF", "-5", "+1.5", ".5", "100"])
def test_numbers_accepted(v):
    assert core.spice_number(v, "x") == v


@pytest.mark.parametrize("v", ["", "k", "1;shell", "1 2", "$(id)", "1k\nshell id", "1,2", "1)", "abc", "1k`id`"])
def test_numbers_rejected(v):
    with pytest.raises(SpiceError):
        core.spice_number(v, "x")


@pytest.mark.parametrize("v", ["v(out)", "i(v1)", "V(OUT)", "v(/vbat)", "v(a,b)", "frequency", "time", "v(out#branch)"])
def test_vectors_accepted(v):
    assert core.spice_vector(v) == v.lower()


@pytest.mark.parametrize("v", ["", "v(out); shell", "v(out) x", "v(out", "(out)", "v(out)\nshell", "$x"])
def test_vectors_rejected(v):
    with pytest.raises(SpiceError):
        core.spice_vector(v)


def test_source_name_cannot_carry_a_command():
    with pytest.raises(SpiceError):
        core.control_block("dc", {"source": "V1; shell id", "start": "0", "stop": "1", "step": "0.1"}, [])


def test_control_block_is_built_only_from_validated_pieces():
    c = core.control_block("ac", {"start": "1", "stop": "1meg", "points_per_decade": 10}, ["V(Out)"])
    assert "ac dec 10 1 1meg" in c and "write out.raw v(out)" in c
    assert c.startswith(".control") and c.strip().endswith(".endc")


@pytest.mark.parametrize("ppd", [0, -1, 1001])
def test_points_per_decade_is_bounded(ppd):
    with pytest.raises(SpiceError):
        core.control_block("ac", {"start": "1", "stop": "10", "points_per_decade": ppd}, [])


def test_unknown_analysis_is_refused():
    with pytest.raises(SpiceError):
        core.control_block("shell", {}, [])


# --- Raw parsing ------------------------------------------------------------------
REAL_RAW = """Title: t
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 3
Variables:
\t0\ttime\ttime
\t1\tv(out)\tvoltage
Values:
 0\t0.0
\t0.0
 1\t1.0e-3
\t0.5
 2\t2.0e-3
\t0.75
"""
CPLX_RAW = REAL_RAW.replace("Flags: real", "Flags: complex").replace("Transient", "AC").replace(
    "\t0.0\n 1", "\t0.0,0.0\n 1").replace("0.0\n\t0.0", "0.0,0.0\n\t1.0,0.0")


def test_parses_a_real_rawfile():
    plot, cplx, v = core.parse_ascii_raw(REAL_RAW)
    assert plot == "Transient Analysis" and not cplx
    assert v["time"] == [0.0, 1e-3, 2e-3]
    assert v["v(out)"] == [0.0, 0.5, 0.75]


def test_parses_complex_values():
    text = ("Title: t\nPlotname: AC\nFlags: complex\nNo. Variables: 2\nNo. Points: 2\nVariables:\n"
            "\t0\tfrequency\tfrequency\n\t1\tv(out)\tvoltage\nValues:\n"
            " 0\t1.0,0.0\n\t0.5,-0.5\n 1\t2.0,0.0\n\t0.25,-0.25\n")
    plot, cplx, v = core.parse_ascii_raw(text)
    assert cplx and v["v(out)"] == [complex(0.5, -0.5), complex(0.25, -0.25)]


def test_truncated_rawfile_is_refused():
    with pytest.raises(SpiceError, match="truncated"):
        core.parse_ascii_raw(REAL_RAW.rsplit("\t0.75", 1)[0])


def test_rawfile_without_values_is_refused():
    with pytest.raises(SpiceError, match="no Values"):
        core.parse_ascii_raw("Title: t\nNo. Variables: 1\n")


# --- Measurements ------------------------------------------------------------------
def test_bandwidth_of_a_first_order_response():
    f = [10 ** (i / 20) for i in range(0, 121)]  # 1 Hz to 1 MHz
    fc = 1000.0
    mag = [core.db(1 / complex(1, x / fc)) for x in f]
    assert core.bandwidth_3db(f, mag) == pytest.approx(fc, rel=0.01)


def test_bandwidth_is_none_when_it_never_rolls_off():
    assert core.bandwidth_3db([1, 10, 100], [0.0, 0.0, -0.1]) is None


def test_rise_time_of_an_exponential_step():
    t = [i * 1e-6 for i in range(0, 2001)]
    y = [1 - math.exp(-x / TAU) for x in t]
    assert core.rise_time(t, y) == pytest.approx(2.197 * TAU, rel=0.01)


def test_rise_time_of_a_flat_signal_is_none():
    assert core.rise_time([0, 1, 2], [1, 1, 1]) is None


def test_downsample_keeps_the_endpoints():
    pts = core.downsample(list(range(1000)), list(range(1000)), 10)
    assert pts[0] == (0, 0) and pts[-1] == (999, 999) and len(pts) <= 10


def test_stats():
    s = core.stats([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["peak_to_peak"] == 2.0 and s["final"] == 3.0


# --- Live simulations against closed-form answers -----------------------------------
def _run(analysis, params, probes=(), netlist=RC):
    clean, _ = core.sanitize_netlist(netlist)
    return core.run_ngspice(clean, analysis, params, list(probes))


@needs_ngspice
def test_ac_bandwidth_matches_one_over_two_pi_rc():
    r = _run("ac", {"start": "1", "stop": "1meg", "points_per_decade": 50}, ["v(out)"])
    freqs = [c.real for c in r.vectors["frequency"]]
    bw = core.bandwidth_3db(freqs, [core.db(c) for c in r.vectors["v(out)"]])
    assert bw == pytest.approx(1 / (2 * math.pi * TAU), rel=0.01)


@needs_ngspice
def test_transient_rise_time_matches_2_197_tau():
    r = _run("tran", {"step": "1u", "stop": "2m", "max_step": "1u"}, ["v(out)"])
    t = [x.real if isinstance(x, complex) else x for x in r.vectors["time"]]
    assert core.rise_time(t, r.vectors["v(out)"]) == pytest.approx(2.197 * TAU, rel=0.02)


@needs_ngspice
def test_operating_point_of_a_divider():
    net = "divider\nV1 in 0 DC 5\nR1 in mid 10k\nR2 mid 0 10k\n"
    r = _run("op", {}, [], net)
    assert r.vectors["v(mid)"][0] == pytest.approx(2.5)
    assert r.vectors["i(v1)"][0] == pytest.approx(-0.25e-3)


@needs_ngspice
def test_dc_sweep_is_linear_through_a_divider():
    net = "divider\nV1 in 0 DC 0\nR1 in mid 10k\nR2 mid 0 30k\n"
    r = _run("dc", {"source": "V1", "start": "0", "stop": "4", "step": "1"}, ["v(mid)"], net)
    assert r.vectors["v(mid)"] == pytest.approx([0, 0.75, 1.5, 2.25, 3.0])


@needs_ngspice
def test_a_malformed_component_is_a_failure_not_a_silent_result():
    # ngspice calls this a warning, drops the part and simulates what is left. The
    # operating point it then returns is its table of constants (pi, boltz, ...).
    with pytest.raises(SpiceError, match="rejected part of the netlist"):
        _run("op", {}, [], "bad\nR1 a\n")


@needs_ngspice
def test_a_good_circuit_with_one_bad_part_still_fails():
    with pytest.raises(SpiceError, match="rejected"):
        _run("op", {}, [], "t\nV1 a 0 DC 1\nR1 a 0 1k\nR2 a\n")


@needs_ngspice
def test_an_ill_posed_circuit_is_refused_not_given_numbers():
    # An inductor straight across an ideal source is a DC short. ngspice prints
    # "singular matrix" as a WARNING and still returns values for it.
    with pytest.raises(SpiceError, match="rejected"):
        _run("op", {}, [], "t\nV1 a 0 DC 3.3\nL1 a 0 1m\nR1 a 0 1k\n")


@needs_ngspice
def test_an_empty_circuit_does_not_return_constants_as_an_answer():
    with pytest.raises(SpiceError):
        _run("op", {}, [], "just a title\n")


@needs_ngspice
def test_a_runaway_simulation_is_killed_at_the_timeout():
    clean, _ = core.sanitize_netlist(RC)
    import time
    t0 = time.time()
    with pytest.raises(SpiceError, match="did not finish"):
        core.run_ngspice(clean, "tran", {"step": "1p", "stop": "100"}, ["v(out)"], timeout=2)
    assert time.time() - t0 < 15


# --- Hostile netlists end to end: nothing may execute ----------------------------------
@needs_ngspice
@pytest.mark.parametrize("payload", [
    ".control\nshell touch {m}\n.endc",
    ".CONTROL\nshell touch {m}\n.ENDC",
    " .control\n shell touch {m}\n .endc",
    ".cont\n+rol\nshell touch {m}\n.endc",
    ".include /etc/passwd\nshell touch {m}",
])
def test_hostile_netlists_are_refused_and_never_run(tmp_path, payload):
    marker = tmp_path / "pwned"
    net = RC + payload.format(m=marker) + "\n"
    with pytest.raises(SpiceError):
        _run("op", {}, [], net)
    assert not marker.exists()


@needs_ngspice
def test_a_bare_shell_line_in_a_netlist_does_not_execute(tmp_path):
    # Outside a control block ngspice reads this as a component card, not a command.
    marker = tmp_path / "pwned"
    net = RC + f"shell touch {marker}\n"
    try:
        _run("op", {}, [], net)
    except SpiceError:
        pass
    assert not marker.exists()


@needs_ngspice
def test_ngspice_runs_in_a_scrubbed_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    r = _run("op", {}, [], "env\nV1 a 0 DC 1\nR1 a 0 1k\n")
    assert "hunter2" not in r.log
