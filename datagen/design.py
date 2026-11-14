"""Design and fault-correction problems, judged by simulating what the model chose.

The model is asked to choose one component value. Nothing in the answer is compared with a stored
number: the chosen value is dropped into the circuit and the resulting behaviour is measured with
ngspice and the independent solver, then compared with the target. So a design succeeds only if it
actually works, and a value that differs from our reference but also works is accepted.
"""
from __future__ import annotations

import hashlib
import math
import random
import re

from . import measure, mna
from .families import E12, E24, SUPPLIES, TRAIN, human, pick, spice_str
from .measure import DatasetError

TOLERANCE = 0.05
_UNIT_NAMES = {"ohm": r"(?:ohms?|Ω)", "F": r"(?:F|farads?)", "H": r"(?:H|henr(?:y|ies))"}
_PREFIX = {"": 1.0, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9}
SERIES = {"E24": E24, "E12": E12}


def is_standard(value: float, series: str) -> bool:
    """Whether the value is a standard series value (to 1%), in any decade."""
    if not value > 0 or not math.isfinite(value):
        return False
    mant = value / 10 ** math.floor(math.log10(value) + 1e-12)
    return any(abs(mant - m) / m < 0.01 for m in SERIES[series] + [10.0])


def parse_value(text: str, unit: str) -> float | None:
    """Pull one component value out of a reply such as 'R1 = 1.6 kohm'. Returns None if there is none."""
    unit_re = _UNIT_NAMES[unit]
    pat = re.compile(rf"(?P<num>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<pre>[pnuµmkMG]?)\s*(?P<unit>{unit_re})?(?![A-Za-z])")
    after_equals = text.split("=")[-1] if "=" in text else text
    for chunk in (after_equals, text):
        found = [m for m in pat.finditer(chunk) if m.group("unit") or m.group("pre") or "=" in text]
        if found:
            m = found[0]
            return float(m.group("num")) * _PREFIX[m.group("pre") or ""]
    return None


def evaluate(problem: dict, value: float) -> dict:
    """Simulate the chosen value and report whether it meets the target."""
    out = {"value": value, "constraint_ok": is_standard(value, problem["series"]), "success": False}
    lo, hi = problem.get("limits", (0, math.inf))
    if not (lo <= value <= hi and value > 0):
        out["error"] = "value outside the allowed range"
        return out
    netlist = problem["template"].format(X=spice_str(value))
    try:
        elements = mna.parse(netlist)
        spec = measure.validate_spec(problem["measurement"], elements)
        a, b = measure.ngspice_value(netlist, spec), measure.mna_value(elements, spec)
    except DatasetError as e:
        out["error"] = str(e)
        return out
    target = problem["target"]["value"]
    out.update(actual=a, solvers_agree=measure.agree(spec["kind"], a, b), error_pct=100 * abs(a - target) / abs(target))
    out["success"] = out["error_pct"] <= 100 * TOLERANCE and out["solvers_agree"]
    return out


def _prompt(body: str, name: str, unit_word: str, series: str | None) -> str:
    rule = f" {name} must be a standard {series} value." if series else ""
    return f"{body}{rule}\nReply with exactly one line: {name} = <value> <{unit_word}>."


def _make(kind: str, body: str, template: str, unknown: str, unit: str, series: str, target: dict, measurement: dict,
          limits: tuple, answer_hint: float) -> dict:
    pid = hashlib.sha1((kind + template + body).encode()).hexdigest()[:12]
    return {"id": pid, "task": "design", "category": "design", "family": kind, "prompt": _prompt(
        body, unknown, {"ohm": "ohm value, e.g. 4.7 kohm", "F": "farad value, e.g. 22 nF", "H": "henry value"}[unit], series),
        "template": template, "unknown": unknown, "unit": unit, "series": series, "target": target,
        "measurement": measurement, "limits": list(limits), "tolerance": TOLERANCE,
        "reference_value": answer_hint}


def lpf_resistor(rng: random.Random, narrative: str = "design") -> dict:
    cap = pick(rng, E12, 1e-9, 1e-6)
    fc = rng.choice([100, 159, 250, 400, 600, 1000, 1600, 2500, 4000, 6300, 10000])
    ideal = 1 / (2 * math.pi * fc * cap)
    template = f"RC low-pass\nV1 in 0 DC 0 AC 1\nR1 in out {{X}}\nC1 out 0 {spice_str(cap)}\n"
    if narrative == "correction":
        wrong = pick(rng, E24, ideal * 5, ideal * 20)  # a decade-class assembly mistake
        wrong_fc = 1 / (2 * math.pi * wrong * cap)
        body = (f"An RC low-pass filter was built with R1 = {human(wrong, 'ohm')} and C1 = {human(cap, 'F')}, but a "
                f"measurement shows its -3 dB cutoff is {wrong_fc:.4g} Hz instead of the required {fc} Hz. "
                "Keep C1 and give the corrected R1.")
        family = "fault_correction"
    else:
        body = (f"Design an RC low-pass filter (V1, then R1 in series, then C1 to ground) with C1 = {human(cap, 'F')} "
                f"so that its -3 dB cutoff frequency is {fc} Hz. Choose R1.")
        family = "design_lpf"
    return _make(family, body, template, "R1", "ohm", "E24", {"kind": "f3db_hz", "value": float(fc)},
                 {"kind": "f3db_hz", "node": "out"}, (1.0, 1e8), ideal)


def divider_resistor(rng: random.Random) -> dict:
    vs, r1 = rng.choice(SUPPLIES), pick(rng, E24, 1e3, 1e5)
    vout = round(vs * rng.choice([0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8]), 3)
    template = f"Divider\nV1 in 0 DC {vs:g}\nR1 in out {spice_str(r1)}\nR2 out 0 {{X}}\n"
    body = (f"A resistive divider (V1 = {vs:g} V, then R1 = {human(r1, 'ohm')} in series, then R2 to ground) must give "
            f"{vout:g} V at the node between R1 and R2 with no load. Choose R2.")
    return _make("design_divider", body, template, "R2", "ohm", "E24", {"kind": "op_voltage", "value": vout},
                 {"kind": "op_voltage", "node": "out"}, (1.0, 1e8), vout * r1 / (vs - vout))


def rlc_capacitor(rng: random.Random) -> dict:
    ind, res = pick(rng, E12, 1e-4, 1e-1), pick(rng, E24, 5, 200)
    f0 = rng.choice([500, 1000, 2000, 5000, 10000, 20000, 50000])
    ideal = 1 / ((2 * math.pi * f0) ** 2 * ind)
    template = f"Series RLC\nV1 in 0 DC 0 AC 1\nL1 in a {spice_str(ind)}\nC1 a out {{X}}\nR1 out 0 {spice_str(res)}\n"
    body = (f"A series RLC band-pass circuit (V1, then L1 = {human(ind, 'H')}, then C1, then R1 = {human(res, 'ohm')} to "
            f"ground, output across R1) must peak at {f0} Hz. Choose C1.")
    return _make("design_rlc", body, template, "C1", "F", "E12", {"kind": "peak_freq_hz", "value": float(f0)},
                 {"kind": "peak_freq_hz", "node": "out"}, (1e-12, 1e-2), ideal)


def design_problems(rng: random.Random, n: int = 10) -> list[dict]:
    """3 low-pass, 3 divider, 2 RLC and 2 fault-correction problems for n = 10, rotating beyond that."""
    plan = [lambda: lpf_resistor(rng), lambda: divider_resistor(rng), lambda: rlc_capacitor(rng),
            lambda: lpf_resistor(rng, "correction")]
    weights = [3, 3, 2, 2]
    seq = [i for i, w in enumerate(weights) for _ in range(w)]
    out, seen = [], set()
    k = 0
    while len(out) < n:
        p = plan[seq[k % len(seq)]]()
        k += 1
        if p["id"] not in seen:
            seen.add(p["id"])
            out.append(p)
    return out
