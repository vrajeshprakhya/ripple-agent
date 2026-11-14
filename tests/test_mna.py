"""The independent solver, checked against closed forms and against ngspice."""
from __future__ import annotations

import math
import pathlib
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from datagen import mna  # noqa: E402
from ngspice_mcp import core  # noqa: E402

needs_ngspice = pytest.mark.skipif(not core.have_ngspice(), reason="ngspice not installed")


def els(text):
    return mna.parse("t\n" + text)


# --- Value parsing ---------------------------------------------------------------------
@pytest.mark.parametrize("text,value", [("10k", 1e4), ("4.7k", 4700), ("100n", 1e-7), ("2meg", 2e6), ("1u", 1e-6),
                                        ("22p", 22e-12), ("1.5", 1.5), ("1e-3", 1e-3), ("10uF", 1e-5), ("5V", 5.0),
                                        ("1M", 1e-3), ("3.3m", 3.3e-3), ("1G", 1e9)])
def test_spice_values(text, value):
    assert mna.spice_value(text) == pytest.approx(value)


def test_m_is_milli_not_mega_as_in_spice():
    assert mna.spice_value("1M") == pytest.approx(1e-3)  # SPICE is case-insensitive: meg is mega


@pytest.mark.parametrize("bad", ["", "abc", "1x", "--1", "k"])
def test_unreadable_values_are_refused(bad):
    with pytest.raises(mna.MnaError):
        mna.spice_value(bad)


# --- Closed forms -------------------------------------------------------------------------
def test_resistive_divider():
    e = els("V1 in 0 5\nR1 in out 1k\nR2 out 0 3k\n")
    v, _ = mna.op(e)
    assert v["out"] == pytest.approx(3.75)


def test_source_current_is_delivered_positive():
    e = els("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n")
    assert mna.delivered_current(e, "v1") == pytest.approx(2.5e-3)


def test_current_source_polarity():
    # I n1 n2 pushes current from n1 through the source to n2, so the node n2 rises.
    e = els("I1 0 a 1m\nR1 a 0 2k\n")
    assert mna.op(e)[0]["a"] == pytest.approx(2.0)


def test_capacitor_is_open_and_inductor_short_at_dc():
    e = els("V1 in 0 6\nL1 in a 1m\nR1 a b 1k\nC1 b 0 1u\nR2 b 0 1k\n")
    assert mna.op(e)[0]["b"] == pytest.approx(3.0)


def test_rc_lowpass_at_the_corner():
    r, c = 1000.0, 159.155e-9
    e = els(f"V1 in 0 DC 0 AC 1\nR1 in out {r}\nC1 out 0 {c}\n")
    f0 = 1 / (2 * math.pi * r * c)
    v = mna.ac_node(e, "out", np.array([f0]))[0]
    assert abs(v) == pytest.approx(1 / math.sqrt(2), rel=1e-4)
    assert math.degrees(np.angle(v)) == pytest.approx(-45.0, abs=0.01)


def test_series_rlc_peaks_at_the_resonant_frequency():
    l, c, r = 10e-3, 100e-9, 50.0
    e = els(f"V1 in 0 DC 0 AC 1\nL1 in a {l}\nC1 a b {c}\nR1 b 0 {r}\n")
    f = np.geomspace(100, 1e5, 4000)
    mag = np.abs(mna.ac_node(e, "b", f))
    assert f[mag.argmax()] == pytest.approx(1 / (2 * math.pi * math.sqrt(l * c)), rel=0.01)
    assert mag.max() == pytest.approx(1.0, rel=1e-3)  # all the voltage across R at resonance


# --- Refusals ----------------------------------------------------------------------------------
@pytest.mark.parametrize("text,msg", [
    ("D1 a 0 dmod\n", "not supported"), ("Q1 a b c npn\n", "not supported"), ("M1 a b c d nmos\n", "not supported"),
    ("R1 a 0 1k\nR1 b 0 1k\n", "duplicate"), ("R1 a a 1k\n", "both terminals"), ("R1 a 0 -5\n", "positive"),
    ("R1 a 0\n", "needs two nodes"), ("R1 a 0 1k 2k\n", "extra fields"),
    ("R1 a b 1k\n", "ground"),
])
def test_unsupported_or_malformed_netlists_are_refused(text, msg):
    with pytest.raises(mna.MnaError, match=msg):
        e = els(text)
        mna.op(e)


def test_a_floating_node_is_refused_as_singular():
    with pytest.raises(mna.MnaError, match="singular"):
        mna.op(els("V1 a 0 1\nR1 a 0 1k\nR2 b c 1k\n"))


def test_two_sources_in_a_loop_are_refused():
    with pytest.raises(mna.MnaError, match="singular"):
        mna.op(els("V1 a 0 1\nV2 a 0 2\nR1 a 0 1k\n"))


# --- Against ngspice on random circuits ----------------------------------------------------------
def random_netlist(rng: random.Random) -> str:
    """A random connected R/C/L network driven by one DC+AC source, with a resistive path to ground."""
    n = rng.randint(3, 6)
    lines = ["rand", "V1 n1 0 DC 3.3 AC 1"]
    values = [100, 220, 470, 1e3, 2.2e3, 4.7e3, 10e3, 47e3]
    for k in range(2, n + 1):  # a spanning chain guarantees every node connects to the source
        lines.append(f"R{k} n{k - 1} n{k} {rng.choice(values)}")
    lines.append(f"R{n + 1} n{n} 0 {rng.choice(values)}")
    for k in range(rng.randint(2, 4)):
        a, b = rng.sample(["0"] + [f"n{i}" for i in range(1, n + 1)], 2)
        kind = rng.choice("rcl")
        val = {"r": rng.choice(values), "c": rng.choice([1e-9, 10e-9, 100e-9, 1e-6]),
               "l": rng.choice([1e-4, 1e-3, 10e-3])}[kind]
        lines.append(f"{kind.upper()}x{k} {a} {b} {val}")
    return "\n".join(lines) + "\n"


@needs_ngspice
@pytest.mark.parametrize("seed", range(25))
def test_matches_ngspice_on_random_circuits(seed):
    rng = random.Random(seed)
    text = random_netlist(rng)
    e = mna.parse(text)
    clean, _ = core.sanitize_netlist(text)

    # Some random circuits are ill-posed (an inductor straight across the source is a DC short).
    # ngspice warns "singular matrix" and used to return numbers anyway; both solvers must refuse.
    try:
        v, branch = mna.op(e)
        mna_ok = True
    except mna.MnaError:
        mna_ok = False
    try:
        r = core.run_ngspice(clean, "op", {}, [])
        ng_ok = True
    except core.SpiceError:
        ng_ok = False
    assert mna_ok == ng_ok, f"solvers disagree on whether the circuit is solvable:\n{text}"
    if not mna_ok:
        return

    # operating point
    for node, value in v.items():
        if node != "0":
            assert r.vectors[f"v({node})"][0] == pytest.approx(value, rel=1e-6, abs=1e-9), node
    assert r.vectors["i(v1)"][0] == pytest.approx(branch["v1"], rel=1e-6, abs=1e-12)

    # AC at three frequencies, magnitude and phase at the last node
    probe = f"v({mna.nodes_of(e)[-1]})"
    ra = core.run_ngspice(clean, "ac", {"start": "100", "stop": "1meg", "points_per_decade": 10}, [probe])
    freqs = np.array([c.real for c in ra.vectors["frequency"]])
    node = mna.nodes_of(e)[-1]
    ours = mna.ac_node(e, node, freqs)
    theirs = np.array(ra.vectors[probe])
    assert np.abs(theirs) == pytest.approx(np.abs(ours), rel=1e-6, abs=1e-12)
    assert np.angle(theirs) == pytest.approx(np.angle(ours), abs=1e-6)
