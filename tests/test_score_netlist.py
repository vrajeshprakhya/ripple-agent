"""Offline tests for the netlist scorer: it must be hard to fool, since the whole
point is judging the circuit rather than the model's account of it."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "demo"))

import score_netlist as S  # noqa: E402

GOOD = {
    "VBAT": {"BAT1/1", "Q1/D", "C1/1", "T1/1"},
    "BATN": {"BAT1/2", "C1/2", "T1/2", "R1/1"},
    "GND": {"R1/2"},
    "LOAD": {"Q1/S"},
    "GATE": {"Q1/G"},
}


def perfect(nets):
    st, nm = S.score(nets)
    return all(st.values()) and all(nm.values())


def test_the_intended_circuit_scores_fully():
    assert perfect(GOOD)


def test_pin_choice_on_symmetric_parts_does_not_matter():
    flipped = {
        "VBAT": {"BAT1/1", "Q1/D", "C1/2", "T1/2"},
        "BATN": {"BAT1/2", "C1/1", "T1/1", "R1/2"},
        "GND": {"R1/1"},
        "LOAD": {"Q1/S"},
        "GATE": {"Q1/G"},
    }
    assert perfect(flipped)


def test_net_names_are_scored_separately_from_structure():
    renamed = {
        "A": GOOD["VBAT"], "B": GOOD["BATN"], "C": GOOD["GND"],
        "D": GOOD["LOAD"], "E": GOOD["GATE"],
    }
    st, nm = S.score(renamed)
    assert all(st.values())  # right circuit
    assert not any(nm.values())  # wrong names


def test_swapped_cell_polarity_fails():
    bad = dict(GOOD)
    bad["VBAT"] = {"BAT1/2", "Q1/D", "C1/1", "T1/1"}
    bad["BATN"] = {"BAT1/1", "C1/2", "T1/2", "R1/1"}
    st, _ = S.score(bad)
    assert not all(st.values())


def test_shorting_the_cell_fails():
    shorted = {"VBAT": GOOD["VBAT"] | GOOD["BATN"], "GND": {"R1/2"},
               "LOAD": {"Q1/S"}, "GATE": {"Q1/G"}}
    st, _ = S.score(shorted)
    assert not st["cell terminals on different nets"]


def test_a_cap_across_nothing_fails():
    bad = dict(GOOD)
    bad["BATN"] = {"BAT1/2", "T1/2", "R1/1"}
    bad["VBAT"] = {"BAT1/1", "Q1/D", "T1/1"}
    bad["X"] = {"C1/1", "C1/2"}
    st, _ = S.score(bad)
    assert not st["C1 straddles the cell"]


def test_an_unwired_schematic_scores_nothing_structural():
    st, nm = S.score({})
    assert not any(st.values()) and not any(nm.values())


def test_fet_source_shorted_to_the_cell_fails():
    bad = dict(GOOD)
    bad["VBAT"] = GOOD["VBAT"] | {"Q1/S"}
    bad["LOAD"] = set()
    st, _ = S.score(bad)
    assert not st["Q1 source on its own net"]


# --- Parser ---------------------------------------------------------------
def test_sexp_parser_handles_quoted_strings_with_parens_and_spaces():
    tree = S.parse_sexp('(a "b (c) d" (e "f"))')
    assert tree == ["a", "b (c) d", ["e", "f"]]


def test_sexp_parser_handles_escaped_quotes():
    assert S.parse_sexp(r'(a "x \"y\" z")') == ["a", r"x \"y\" z"]


def test_unconnected_pin_nets_do_not_count_as_wired():
    # KiCad reports an unwired pin as its own net named unconnected-(...).
    bad = {
        "VBAT": {"BAT1/1", "Q1/D", "C1/1", "T1/1"},
        "BATN": {"BAT1/2", "C1/2", "T1/2", "R1/1"},
        "unconnected-(R1-Pad2)": {"R1/2"},
        "unconnected-(Q1-S-PadS)": {"Q1/S"},
        "unconnected-(Q1-G-PadG)": {"Q1/G"},
    }
    st, _ = S.score(bad)
    assert not st["Q1 source on its own net"]
    assert not st["Q1 gate on its own net"]
    assert not st["R1 other pin on its own net"]


def test_unwired_cell_cannot_pass_by_both_nets_being_none():
    st, _ = S.score({"unconnected-(BAT1)": {"BAT1/1"}, "unconnected-(BAT1b)": {"BAT1/2"}})
    assert not any(st.values())


def test_merging_the_load_terminals_into_one_net_fails():
    # Found in a live run: the FET source and the free pin of the shunt both on "LOAD".
    merged = {
        "CELL+": {"BAT1/1", "C1/1", "Q1/D", "T1/1"},
        "CELL-": {"BAT1/2", "C1/2", "R1/1", "T1/2"},
        "LOAD": {"Q1/S", "R1/2"},
    }
    st, _ = S.score(merged)
    assert not st["load terminals on different nets"]
    assert st["Q1 source on its own net"]  # the older checks alone did not catch it


def test_fet_across_the_cell_fails():
    # Found in a live run: Q1 source on the cell negative net.
    across = {
        "CELL_POS": {"BAT1/1", "C1/1", "Q1/D", "T1/1"},
        "CELL_NEG": {"BAT1/2", "C1/2", "Q1/S", "R1/1", "T1/2"},
        "LOAD_NEG": {"R1/2"},
    }
    st, _ = S.score(across)
    assert not st["Q1 source on its own net"]


def test_distinct_load_terminals_pass():
    assert S.score(GOOD)[0]["load terminals on different nets"]
