"""Tests for the unit-aware calculator. The first test is the failure that motivated it."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ngspice_mcp import units as U  # noqa: E402
from ngspice_mcp.core import SpiceError  # noqa: E402


def c(expr):
    return U.compute(expr)


def test_millivolts_over_milliohms_is_amperes_not_milliamperes():
    # A model reported "1.00002 mA" for exactly this. 1 mV / 1 mOhm is 1 A.
    r = c("1.00002 mV / 1 mohm")
    assert r["unit"] == "A"
    assert r["value_si"] == pytest.approx(1.00002)
    assert r["display"] == "1.00002 A"


def test_the_same_division_written_with_si_base_values():
    r = c("0.001000019913 V / 0.001 ohm")
    assert r["unit"] == "A" and r["value_si"] == pytest.approx(1.000019913)


@pytest.mark.parametrize("expr,value,unit", [
    ("3.688 V * 1 A", 3.688, "W"),
    ("5 V / 10 kohm", 0.0005, "A"),
    ("1 kohm * 100 nF", 1e-4, "s"),
    ("1 / (2 * 3.14159265 * 1 kohm * 159.15 nF)", 1000.031, "Hz"),
    ("10 V + 5 mV", 10.005, "V"),
    ("2 A * 3 ohm", 6.0, "V"),
    ("1 kohm + 500 ohm", 1500.0, "ohm"),
    ("(1 V + 2 V) / (1 A + 2 A)", 1.0, "ohm"),
    ("-5 V + 2 V", -3.0, "V"),
])
def test_arithmetic_with_units(expr, value, unit):
    r = c(expr)
    assert r["value_si"] == pytest.approx(value, rel=1e-6)
    assert r["unit"] == unit


def test_one_over_a_time_is_a_frequency():
    assert c("1 / 1 ms")["unit"] == "Hz"
    assert c("1 / 1 ms")["value_si"] == pytest.approx(1000.0)


@pytest.mark.parametrize("text,value", [
    ("1 pF", 1e-12), ("1 nF", 1e-9), ("1 uF", 1e-6), ("1 µF", 1e-6), ("1 mF", 1e-3),
    ("1 F", 1.0), ("1 kohm", 1e3), ("1 Mohm", 1e6), ("1 megohm", 1e6), ("1 Gohm", 1e9),
    ("2 MHz", 2e6), ("2 mHz", 2e-3), ("5 ms", 5e-3), ("5 us", 5e-6), ("1 mV", 1e-3),
    ("1 Ω", 1.0), ("1kΩ", 1e3), ("4.7uF", 4.7e-6), ("1e3 ohm", 1e3), (".5 A", 0.5),
])
def test_prefixes_and_spellings(text, value):
    assert c(text)["value_si"] == pytest.approx(value)


def test_milli_and_mega_are_not_confused():
    assert c("1 mohm")["value_si"] == pytest.approx(1e-3)
    assert c("1 Mohm")["value_si"] == pytest.approx(1e6)


# --- Refusals: the point of a unit check ---------------------------------------
@pytest.mark.parametrize("expr", ["1 V + 1 A", "5 A - 2 ohm", "1 s + 1 Hz", "3 V + 4"])
def test_adding_unlike_quantities_is_refused(expr):
    with pytest.raises(U.UnitError, match="cannot combine"):
        c(expr)


@pytest.mark.parametrize("expr", ["1 V / 0 ohm", "1 / 0"])
def test_division_by_zero_is_refused(expr):
    with pytest.raises(U.UnitError, match="division by zero"):
        c(expr)


@pytest.mark.parametrize("expr", [
    "", "   ", "1 +", "(1 V", "1 V)", "* 2", "1 V 2 V", "1 parsec", "abc", "1m",
    "__import__('os')", "1; shell", "1 V ** 2 V",
])
def test_malformed_input_is_refused(expr):
    with pytest.raises(SpiceError):
        c(expr)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('id')", "open('/etc/passwd')", "1 V; rm -rf /", "`id`", "$(id)",
    "().__class__", "lambda: 1",
])
def test_it_is_not_an_eval(expr):
    with pytest.raises(SpiceError):
        c(expr)


def test_a_bare_prefix_without_a_unit_is_not_silently_accepted():
    # "1m" in SPICE means milli, but with no unit there is nothing to attach it to.
    with pytest.raises(SpiceError):
        c("1m")
    with pytest.raises(SpiceError):
        c("10k * 2 V")


def test_a_fractional_unit_is_refused():
    with pytest.raises(U.UnitError, match="fractional unit"):
        c("2 V ** 0.5")


def test_squaring_a_quantity_gives_a_composite_unit():
    r = c("2 V ** 2")
    assert r["value_si"] == pytest.approx(4.0) and "^" in r["unit"]


# --- Display ---------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    (0.001, "1 mV"), (1.00002, "1.00002 V"), (4.7e-6, "4.7 uV"), (1500.0, "1.5 kV"),
    (2e-12, "2 pV"), (0.0, "0 V"),
])
def test_si_format_picks_a_readable_prefix(value, expected):
    assert U.si_format(value, U.UNITS["V"]) == expected


def test_dimensionless_results_have_no_unit_text():
    r = c("10 V / 5 V")
    assert r["unit"] == "dimensionless" and r["value_si"] == pytest.approx(2.0)
    assert r["display"] == "2"


def test_vector_references_are_read_through_a_resolver_with_units_attached():
    values = {"v(/srp)": 0.00100002, "v(/srn)": 0.0, "i(v1)": -0.5}
    r = U.compute("(v(/srp) - v(/srn)) / 1 mohm", resolve=values.__getitem__)
    assert r["unit"] == "A" and r["display"] == "1.00002 A"
    assert U.compute("i(v1) * 4 V", resolve=values.__getitem__)["unit"] == "W"


def test_a_vector_reference_without_a_resolver_is_refused():
    with pytest.raises(U.UnitError, match="needs a result_id"):
        c("v(out) / 1 kohm")


def test_vector_references_are_case_insensitive_and_unit_checked():
    values = {"v(out)": 2.0}
    assert U.compute("V(OUT) / 1 kohm", resolve=values.__getitem__)["unit"] == "A"
    with pytest.raises(U.UnitError, match="cannot combine"):
        U.compute("v(out) + 1 A", resolve=values.__getitem__)


def test_result_echoes_the_expression():
    assert c("  1 V  ")["expression"] == "1 V"
