"""A small independent circuit solver (modified nodal analysis) for linear R, L, C, V, I networks.

Why it exists: the dataset's answers come from ngspice, but when a language model proposes the
circuits, trusting one solver is not enough. Every example must also agree with this second
solver, which shares no code or conventions with ngspice. A bug or a quirk in either shows up as
a disagreement and the example is rejected.

Conventions (SPICE's, so the two can be compared):
  * node "0" is ground; a voltage source V n+ n- has its branch current positive from n+ through
    the source to n-, so a source powering a load carries a NEGATIVE branch current;
  * DC: capacitors are open and inductors are short circuits;
  * AC: s = j*2*pi*f, and the AC source magnitude is whatever the netlist says.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

_SUFFIX = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}
# SPICE ignores any trailing letters after a value, which would accept "1x" or "5 volts"; for a
# dataset of model-proposed circuits only a unit after the scale suffix is allowed.
_VALUE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)(meg|mil|[fpnumkgt])?(?:v|a|f|h|hz|s|ohms?)?$", re.I)


class MnaError(ValueError):
    """A netlist outside what this solver supports, or one that cannot be solved."""


def spice_value(text: str) -> float:
    m = _VALUE.match(text.strip())
    if not m:
        raise MnaError(f"cannot read value {text!r}")
    v = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    if suffix == "mil":
        raise MnaError("mil suffix is not supported")
    return v * _SUFFIX.get(suffix, 1.0)


@dataclass
class Element:
    kind: str   # r l c v i
    name: str
    n1: str
    n2: str
    value: float = 0.0   # R ohms, L henries, C farads, source DC value
    ac: float = 0.0      # source AC magnitude


def parse(netlist: str) -> list[Element]:
    """Parse the elements of a netlist. Anything that is not R, L, C, V or I is refused."""
    out: list[Element] = []
    seen: set[str] = set()
    for raw in netlist.replace("\r", "\n").split("\n")[1:]:  # line 1 is the title
        line = raw.split(";")[0].strip()
        if not line or line.startswith("*") or line.startswith("."):
            continue
        tok = line.split()
        name, kind = tok[0].lower(), tok[0][0].lower()
        if kind not in "rlcvi":
            raise MnaError(f"element {tok[0]} is not supported (only R L C V I)")
        if name in seen:
            raise MnaError(f"duplicate element name {tok[0]}")
        seen.add(name)
        if len(tok) < 4:
            raise MnaError(f"element {tok[0]} needs two nodes and a value")
        n1, n2 = tok[1].lower(), tok[2].lower()
        if n1 == n2:
            raise MnaError(f"element {tok[0]} has both terminals on node {n1}")
        if kind in "rlc":
            if len(tok) != 4:
                raise MnaError(f"element {tok[0]} has unexpected extra fields")
            val = spice_value(tok[3])
            if not (val > 0 and math.isfinite(val)):
                raise MnaError(f"element {tok[0]} must have a positive value")
            out.append(Element(kind, name, n1, n2, val))
            continue
        # sources: "V1 a b 5", "V1 a b DC 5", "V1 a b AC 1", "V1 a b DC 0 AC 1"
        dc, ac, rest, i = 0.0, 0.0, [t.lower() for t in tok[3:]], 0
        if rest and rest[0] not in ("dc", "ac"):
            dc, i = spice_value(rest[0]), 1
        while i < len(rest):
            key = rest[i]
            if key not in ("dc", "ac") or i + 1 >= len(rest):
                raise MnaError(f"cannot read source specification of {tok[0]}")
            val = spice_value(rest[i + 1])
            if key == "dc":
                dc = val
            else:
                ac = val
            i += 2
        out.append(Element(kind, name, n1, n2, dc, ac))
    if not out:
        raise MnaError("no elements")
    return out


def nodes_of(elements: list[Element]) -> list[str]:
    seen: list[str] = []
    for e in elements:
        for n in (e.n1, e.n2):
            if n != "0" and n not in seen:
                seen.append(n)
    if not any("0" in (e.n1, e.n2) for e in elements):
        raise MnaError("no connection to ground (node 0)")
    return seen


def _solve(elements: list[Element], omega: float | None) -> tuple[dict[str, complex], dict[str, complex]]:
    """Solve one operating point (omega None) or one AC frequency. Returns node voltages and branch currents."""
    nodes = nodes_of(elements)
    idx = {n: i for i, n in enumerate(nodes)}
    branches = [e for e in elements if e.kind == "v" or (e.kind == "l" and omega is None)]
    size = len(nodes) + len(branches)
    a = np.zeros((size, size), dtype=complex)
    z = np.zeros(size, dtype=complex)
    bidx = {e.name: len(nodes) + k for k, e in enumerate(branches)}

    def stamp(row, col, val):
        a[row, col] += val

    for e in elements:
        i, j = idx.get(e.n1), idx.get(e.n2)
        if e.kind in "rc" or (e.kind == "l" and omega is not None):
            if e.kind == "r":
                y = 1.0 / e.value
            elif e.kind == "c":
                y = 0.0 if omega is None else 1j * omega * e.value
            else:
                y = 1.0 / (1j * omega * e.value)
            for p, q, s in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
                if p is not None and q is not None:
                    stamp(p, q, s * y)
        elif e.kind == "i":
            amp = e.value if omega is None else e.ac
            if i is not None:
                z[i] -= amp   # current leaves n1 through the source toward n2
            if j is not None:
                z[j] += amp
        else:  # v, or l at DC (a 0 V branch)
            k = bidx[e.name]
            if i is not None:
                stamp(i, k, 1)
                stamp(k, i, 1)
            if j is not None:
                stamp(j, k, -1)
                stamp(k, j, -1)
            if e.kind == "v":
                z[k] = e.value if omega is None else e.ac
    try:
        if np.linalg.cond(a) > 1e14:
            raise MnaError("the circuit is singular (floating node, shorted source or loop of sources)")
        x = np.linalg.solve(a, z)
    except np.linalg.LinAlgError as err:
        raise MnaError(f"the circuit is singular: {err}") from err
    volts = {n: x[idx[n]] for n in nodes}
    volts["0"] = 0j
    return volts, {e.name: x[bidx[e.name]] for e in branches}


def op(elements: list[Element]) -> tuple[dict[str, float], dict[str, float]]:
    """DC operating point: real node voltages and branch currents (SPICE sign)."""
    v, i = _solve(elements, None)
    return {k: x.real for k, x in v.items()}, {k: x.real for k, x in i.items()}


def ac_node(elements: list[Element], node: str, freqs: np.ndarray) -> np.ndarray:
    """Complex voltage at one node over a frequency array."""
    node = node.lower()
    if node not in nodes_of(elements) and node != "0":
        raise MnaError(f"no node {node!r}")
    return np.array([_solve(elements, 2 * math.pi * f)[0][node] for f in freqs])


def delivered_current(elements: list[Element], source: str) -> float:
    """Current a voltage source delivers to the circuit (positive when it powers a load)."""
    _, branch = op(elements)
    key = source.lower()
    if key not in branch:
        raise MnaError(f"no voltage source {source!r}")
    return -branch[key]
