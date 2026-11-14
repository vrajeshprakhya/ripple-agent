"""The grader is evidence, so it is tested against answers a real model actually gave."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "demo"))

import grade_board_answer as G  # noqa: E402

# Verbatim lines from the first batch of runs (see the README).
RUN1_CONTRADICTS = (
    "- Shunt current: (v(/srp) - v(/srn)) / 1 mOhm = (0.001000019913116972 V - 0 V) / 0.001 Ohm = **1.000 mA**\n"
    "2. **Sense-shunt current** = **1.000 mA** (1.00002 A, i.e. the 1 A commanded load current)\n")
RUN2_RIGHT = "2. **Sense shunt current** = (0.001000019913116972 V - 0.0 V) / 1 mohm = **1.00002 A**\n"
RUN4_WRONG = "2. **Sense shunt current**: (0.001000019913116972 V - 0.0 V) / 0.001 Ohm = **1.00002 mA**.\n"


def test_a_milliampere_claim_is_wrong_even_beside_a_correct_ampere_claim():
    assert G.grade_current(RUN1_CONTRADICTS) == "wrong"


def test_one_ampere_is_right():
    assert G.grade_current(RUN2_RIGHT) == "ok"
    assert G.grade_current("The shunt carries 1.000 A.") == "ok"


def test_one_milliampere_is_wrong():
    assert G.grade_current(RUN4_WRONG) == "wrong"
    assert G.grade_current("current = 1.000 mA") == "wrong"


def test_one_thousand_milliamperes_is_right():
    # The reference itself is logged as i_mA = 1000.02; that is the same current.
    assert G.grade_current("i = 1000.02 mA") == "ok"


def test_no_current_stated_is_none():
    assert G.grade_current("the cell is at 3.688 V") == "none"


@pytest.mark.parametrize("text", ["7.957 kHz", "7,956.95 Hz", "7956.95 Hz", "7.9569 kHz", "7.96 kHz"])
def test_the_reference_corner_in_any_spelling_is_right(text):
    assert G.grade_corner(f"the -3 dB frequency is {text}") == "ok"


def test_a_corner_just_outside_one_percent_is_wrong():
    assert G.grade_corner("the -3 dB frequency is 8.1 kHz") == "wrong"


@pytest.mark.parametrize("text", ["11.25 kHz", "11,253.98 Hz", "1.1 kHz", "11.254 kHz"])
def test_the_wrong_corners_seen_in_practice_are_wrong(text):
    assert G.grade_corner(f"the -3 dB frequency is {text}") == "wrong"


def test_sweep_limits_alone_do_not_count_as_a_claim():
    assert G.grade_corner("swept 10 Hz to 1 MHz") == "none"


def test_cell_voltage():
    assert G.grade_cell("cell 1 = 3.688 V") == "ok"
    assert G.grade_cell("cell 1 = 3.7 V") == "none"


def test_full_grade_of_a_good_answer():
    g = G.grade("3.688 V, 1.00002 A and a corner of 7.957 kHz")
    assert g == {"cell": "ok", "current": "ok", "corner": "ok"}
