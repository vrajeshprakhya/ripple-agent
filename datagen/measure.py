"""Measurements: what a dataset question can ask, answered by two independent solvers.

A measurement spec is a small dict, for example {"kind": "ac_gain_db", "node": "out", "freq_hz": 1000}.
Each kind has a fixed unit, a question template, and a computation in ngspice and in the
independent MNA solver. Question text is always rendered from the spec, never taken from a model,
so what is asked and what is measured cannot drift apart.
"""
from __future__ import annotations

import math

import numpy as np

from ngspice_mcp import core

from . import mna


class DatasetError(ValueError):
    """An example that is refused, with the reason (counted in the manifest)."""


UNITS = {"op_voltage": "V", "op_current": "A", "ac_gain_db": "dB", "ac_phase_deg": "deg",
         "f3db_hz": "Hz", "peak_freq_hz": "Hz"}
# The sweep starts two decades below the lowest answer allowed (1 Hz) so that its first point is a
# true low-frequency reference. Starting at 1 Hz made "-3 dB" mean "3 dB below the 1 Hz value",
# which for a 10 Hz filter is 0.9% away from the textbook cutoff.
SWEEP = (0.01, 1e9, 100)  # Hz, Hz, points per decade


def render_question(spec: dict, ac_source: str = "v1") -> str:
    k, node = spec["kind"], spec.get("node")
    if k == "op_voltage":
        return f"What is the DC voltage at node {node} with respect to ground (node 0), in volts?"
    if k == "op_current":
        return (f"What DC current does voltage source {spec['source'].upper()} deliver to the circuit "
                "(positive when it powers the load), in amperes?")
    if k == "ac_gain_db":
        return (f"The AC source has magnitude 1 V. What is the voltage gain at node {node} at "
                f"{_hz(spec['freq_hz'])}, in dB (20*log10 of the magnitude)?")
    if k == "ac_phase_deg":
        return (f"The AC source has magnitude 1 V and zero phase. What is the phase of the voltage at node "
                f"{node} at {_hz(spec['freq_hz'])}, in degrees?")
    if k == "f3db_hz":
        return (f"The AC source has magnitude 1 V. What is the -3 dB frequency of the response at node {node}, "
                "taken relative to its low-frequency (DC) value, in hertz?")
    if k == "peak_freq_hz":
        return (f"The AC source has magnitude 1 V. At what frequency is the voltage magnitude at node {node} "
                "greatest, in hertz?")
    raise DatasetError(f"unknown measurement kind {k!r}")


def _hz(f: float) -> str:
    for scale, name in ((1e6, "MHz"), (1e3, "kHz"), (1.0, "Hz")):
        if f >= scale:
            return f"{f / scale:.6g} {name}"
    return f"{f:.6g} Hz"


def validate_spec(spec: object, elements: list[mna.Element]) -> dict:
    """Return a normalised copy of a spec that this circuit can answer, or raise DatasetError."""
    if not isinstance(spec, dict) or spec.get("kind") not in UNITS:
        raise DatasetError("measurement kind is missing or unknown")
    kind = spec["kind"]
    out = {"kind": kind}
    nodes = set(mna.nodes_of(elements))
    if kind == "op_current":
        src = str(spec.get("source", "")).lower()
        if not any(e.kind == "v" and e.name == src for e in elements):
            raise DatasetError(f"no voltage source {src!r} for op_current")
        out["source"] = src
    else:
        node = str(spec.get("node", "")).lower()
        if node not in nodes:
            raise DatasetError(f"measured node {node!r} is not in the circuit")
        out["node"] = node
    if kind in ("ac_gain_db", "ac_phase_deg"):
        f = spec.get("freq_hz")
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not 1.0 <= f <= 1e9:
            raise DatasetError("freq_hz must be a number from 1 Hz to 1 GHz")
        out["freq_hz"] = float(f)
    if UNITS[kind] in ("dB", "deg", "Hz"):
        ac = [e for e in elements if e.kind in "vi" and e.ac != 0]
        if len(ac) != 1 or ac[0].kind != "v" or abs(ac[0].ac - 1.0) > 1e-12:
            raise DatasetError("AC measurements need exactly one voltage source with AC magnitude 1")
    return out


# --- ngspice ----------------------------------------------------------------------------------
def _sweep(clean: str, node: str) -> tuple[np.ndarray, np.ndarray]:
    lo, hi, ppd = SWEEP
    r = core.run_ngspice(clean, "ac", {"start": str(lo), "stop": str(hi), "points_per_decade": ppd}, [f"v({node})"])
    return (np.array([c.real for c in r.vectors["frequency"]]), np.array(r.vectors[f"v({node})"]))


def _peak(freqs: np.ndarray, mag: np.ndarray) -> float:
    """Frequency of the maximum, refined by a parabola through the log-magnitude around it."""
    i = int(np.argmax(mag))
    if i in (0, len(mag) - 1):
        raise DatasetError("the response has no interior peak in 1 Hz to 1 GHz")
    x, y = np.log10(freqs[i - 1:i + 2]), np.log10(np.maximum(mag[i - 1:i + 2], 1e-300))
    a, b, _ = np.polyfit(x, y, 2)
    if a >= 0:
        raise DatasetError("the response peak is not a maximum")
    return float(10 ** (-b / (2 * a)))


def _f3db(freqs: np.ndarray, mag: np.ndarray) -> float:
    ref = mag[0]
    if ref < 1e-3:
        raise DatasetError("low-frequency response is too small to define a -3 dB point (not a low-pass shape)")
    value = core.bandwidth_3db(list(freqs), [core.db(m / ref) for m in mag])
    if value is None:
        raise DatasetError("the response never falls 3 dB below its low-frequency value by 1 GHz")
    return float(value)


def ngspice_value(netlist: str, spec: dict) -> float:
    try:
        clean, _ = core.sanitize_netlist(netlist)
        k = spec["kind"]
        if k in ("op_voltage", "op_current"):
            r = core.run_ngspice(clean, "op", {}, [])
            if k == "op_voltage":
                return float(r.vectors[f"v({spec['node']})"][0].real)
            return -float(r.vectors[f"i({spec['source']})"][0].real)  # delivered current
        if k in ("ac_gain_db", "ac_phase_deg"):
            f = spec["freq_hz"]
            r = core.run_ngspice(clean, "ac", {"start": f"{f:.12g}", "stop": f"{2 * f:.12g}", "points_per_decade": 1},
                                 [f"v({spec['node']})"])
            v = r.vectors[f"v({spec['node']})"][0]
            return core.db(v) if k == "ac_gain_db" else math.degrees(np.angle(v))
        freqs, v = _sweep(clean, spec["node"])
        return _peak(freqs, np.abs(v)) if k == "peak_freq_hz" else _f3db(freqs, np.abs(v))
    except core.SpiceError as e:
        raise DatasetError(f"ngspice refused the circuit: {str(e).splitlines()[0][:100]}") from e


# --- the independent solver --------------------------------------------------------------------
def mna_value(elements: list[mna.Element], spec: dict) -> float:
    try:
        k = spec["kind"]
        if k == "op_voltage":
            return mna.op(elements)[0][spec["node"]]
        if k == "op_current":
            return mna.delivered_current(elements, spec["source"])
        if k in ("ac_gain_db", "ac_phase_deg"):
            v = mna.ac_node(elements, spec["node"], np.array([spec["freq_hz"]]))[0]
            return core.db(v) if k == "ac_gain_db" else math.degrees(np.angle(v))
        lo, hi, ppd = SWEEP
        freqs = np.geomspace(lo, hi, int(math.log10(hi / lo) * 4 * ppd) + 1)  # four times finer than ngspice
        mag = np.abs(mna.ac_node(elements, spec["node"], freqs))
        return _peak(freqs, mag) if k == "peak_freq_hz" else _f3db(freqs, mag)
    except mna.MnaError as e:
        raise DatasetError(f"independent solver refused the circuit: {e}") from e


def agree(kind: str, a: float, b: float) -> bool:
    """Whether two solvers' answers match to the tolerance of this kind of quantity."""
    if kind == "ac_gain_db":
        return abs(a - b) <= 0.01
    if kind == "ac_phase_deg":
        d = abs(a - b) % 360
        return min(d, 360 - d) <= 0.05
    if kind in ("f3db_hz", "peak_freq_hz"):
        return abs(a - b) <= 0.005 * max(abs(a), abs(b))
    return abs(a - b) <= 1e-5 * max(abs(a), abs(b)) + 1e-12


# Answers outside these ranges are numerical noise or a degenerate question, not engineering.
def sane(kind: str, v: float) -> bool:
    if not math.isfinite(v):
        return False
    return {"op_voltage": 1e-6 <= abs(v) <= 1e4, "op_current": 1e-9 <= abs(v) <= 1e2,
            "ac_gain_db": -120 <= v <= 60, "ac_phase_deg": -180 <= v <= 180,
            "f3db_hz": 1 <= v <= 1e9, "peak_freq_hz": 1 <= v <= 1e9}[kind]


def format_quantity(value: float, unit: str) -> str:
    """The answer as text: SI prefix and four significant digits (plain 3 decimals for dB and degrees)."""
    if unit in ("dB", "deg"):
        return f"{value:.3f} {unit}"
    prefixes = {-12: "p", -9: "n", -6: "u", -3: "m", 0: "", 3: "k", 6: "M", 9: "G"}
    exp = 0 if value == 0 else max(-12, min(9, 3 * math.floor(math.log10(abs(value)) / 3)))
    mant = float(f"{value / 10 ** exp:.4g}")
    if abs(mant) >= 1000 and exp < 9:  # rounding carried into the next prefix
        exp, mant = exp + 3, float(f"{value / 10 ** (exp + 3):.4g}")
    return f"{mant:g} {prefixes[exp]}{unit}"
