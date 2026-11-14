"""A deterministic 'engineering calculator' baseline: textbook formulas for the three training families.

It recognises a resistive divider, an RC low-pass and a series RLC band-pass by their structure and
applies the closed-form answer. Anything else it declines. If a fine-tuned model only matches this
on the training families and does no better elsewhere, fine-tuning has taught it to reproduce
textbook formulas and nothing more.
"""
from __future__ import annotations

import math

from . import design, mna
from .families import E12, E24


def _family(elements: list[mna.Element]) -> tuple[str, dict] | None:
    kinds = sorted(e.kind for e in elements)
    by = {e.kind: [x for x in elements if x.kind == e.kind] for e in elements}
    v = by.get("v", [None])[0]
    if kinds == ["r", "r", "v"]:
        r_top = next((r for r in by["r"] if v.n1 in (r.n1, r.n2)), None)
        r_bot = next((r for r in by["r"] if "0" in (r.n1, r.n2) and r is not r_top), None)
        if r_top and r_bot and v.n2 == "0":
            out = ({r_top.n1, r_top.n2} - {v.n1}).pop()
            if out in (r_bot.n1, r_bot.n2):
                return "divider", {"vs": v.value, "r1": r_top.value, "r2": r_bot.value, "out": out}
    if kinds == ["c", "r", "v"]:
        r, c = by["r"][0], by["c"][0]
        if v.n1 in (r.n1, r.n2) and "0" in (c.n1, c.n2):
            out = ({r.n1, r.n2} - {v.n1}).pop()
            if out in (c.n1, c.n2):
                return "rc_lowpass", {"r": r.value, "c": c.value, "out": out}
    if kinds == ["c", "l", "r", "v"]:
        r = by["r"][0]
        if "0" in (r.n1, r.n2):
            return "series_rlc", {"l": by["l"][0].value, "c": by["c"][0].value, "r": r.value,
                                  "out": ({r.n1, r.n2} - {"0"}).pop()}
    return None


def predict(problem: dict) -> str:
    """The calculator's reply for a problem, in the same one-line form a model is asked for."""
    if problem["task"] == "design":
        return _design(problem)
    try:
        elements = mna.parse(problem["netlist"])
    except mna.MnaError:
        return "unsupported"
    fam = _family(elements)
    spec = problem["measurement"]
    if fam is None or spec.get("node", fam[1].get("out")) != fam[1].get("out"):
        return "unsupported"
    name, p = fam
    k = spec["kind"]
    if name == "divider":
        if k == "op_voltage":
            return f"{p['vs'] * p['r2'] / (p['r1'] + p['r2']):.6g} V"
        if k == "op_current":
            return f"{p['vs'] / (p['r1'] + p['r2']):.6g} A"
    if name == "rc_lowpass":
        fc = 1 / (2 * math.pi * p["r"] * p["c"])
        if k == "f3db_hz":
            return f"{fc:.6g} Hz"
        x = spec["freq_hz"] / fc
        if k == "ac_gain_db":
            return f"{-10 * math.log10(1 + x * x):.6g} dB"
        if k == "ac_phase_deg":
            return f"{-math.degrees(math.atan(x)):.6g} deg"
    if name == "series_rlc":
        f0 = 1 / (2 * math.pi * math.sqrt(p["l"] * p["c"]))
        if k == "peak_freq_hz":
            return f"{f0:.6g} Hz"
        q = math.sqrt(p["l"] / p["c"]) / p["r"]
        x = q * (spec["freq_hz"] / f0 - f0 / spec["freq_hz"])
        if k == "ac_gain_db":
            return f"{-10 * math.log10(1 + x * x):.6g} dB"
        if k == "ac_phase_deg":
            return f"{-math.degrees(math.atan(x)):.6g} deg"
    return "unsupported"


def _nearest(value: float, series: list[float]) -> float:
    d = math.floor(math.log10(value))
    cands = [m * 10.0 ** e for e in (d - 1, d, d + 1) for m in series]
    return min(cands, key=lambda c: abs(math.log(c / value)))


def _design(problem: dict) -> str:
    """Textbook design: the ideal value, snapped to the required series."""
    series = {"E24": E24, "E12": E12}[problem["series"]]
    value = _nearest(problem["reference_value"], series)
    unit = {"ohm": "ohm", "F": "F", "H": "H"}[problem["unit"]]
    return f"{problem['unknown']} = {value:.4g} {unit}"
