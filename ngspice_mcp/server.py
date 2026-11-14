"""MCP server exposing ngspice board-level analysis over stdio.

Netlists come in two ways, either inline (`netlist`) or as a file (`netlist_path`),
and the file must sit under the allowed roots (NGSPICE_MCP_ROOTS, os.pathsep
separated, defaulting to the working directory). Every netlist goes through
core.sanitize_netlist, and the control block ngspice runs is always built here.

Results are kept in memory under a short id so a long waveform never has to be
pushed through the model's context: the run tools return a summary, and
`get_vector` or `measure` fetch more on demand.
"""
from __future__ import annotations

import functools
import os
import re
import subprocess
import uuid
from collections import OrderedDict
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import core
from .core import SpiceError

mcp = MCPServer(
    "ngspice",
    instructions=(
        "Board-level SPICE analysis with ngspice. Supply a circuit as `netlist` text or a "
        "`netlist_path` under the allowed roots; analysis cards and .control are not needed "
        "and are not allowed, the tools choose the analysis. Run op, dc, ac or transient, "
        "then use measure for numbers such as the -3 dB bandwidth or rise time."
    ),
)

def _tool(fn):
    """Register fn as an MCP tool, turning SpiceError into a ToolError the model can read.

    The SDK hides the text of any exception it does not recognise (it treats it as a
    crash, so server internals do not leak), which would reduce every refusal to
    "Error executing tool run_op". ToolError is the one type whose message reaches
    the client, so the deliberate failures are converted and real crashes stay masked.
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except SpiceError as e:
            raise ToolError(str(e)) from e
        except subprocess.TimeoutExpired as e:
            raise ToolError(f"{e.cmd[0] if e.cmd else 'a subprocess'} timed out after {e.timeout}s") from e

    return mcp.tool()(wrapped)


_RESULTS: "OrderedDict[str, core.RunResult]" = OrderedDict()
_MAX_RESULTS = 50


def roots() -> list[Path]:
    raw = os.environ.get("NGSPICE_MCP_ROOTS", "")
    found = [Path(p) for p in raw.split(os.pathsep) if p]
    return found or [Path.cwd()]


def _circuit(netlist: str | None, netlist_path: str | None) -> tuple[str, list[str]]:
    if bool(netlist) == bool(netlist_path):
        raise SpiceError("give exactly one of netlist or netlist_path")
    base = None
    if netlist_path:
        netlist = core.read_netlist_file(netlist_path, roots())
        base = Path(netlist_path).expanduser().resolve().parent
    return core.sanitize_netlist(netlist or "", roots=roots(), base_dir=base)


MAX_PROBES = 12


def _need_probes(probes: list[str] | None) -> list[str]:
    """Sweeps must name what they return.

    With no probes ngspice writes every vector, and on a real board that is 99 of them: a
    144 KB response, around 36k tokens, from a single call. An operating point is the one
    analysis where the whole picture is wanted, so it alone is exempt.
    """
    if not probes:
        raise SpiceError("name the vectors to return with probes, for example ['v(out)'] or a "
                         "differential ['v(a,b)']. Run run_op first to see every node name.")
    if len(probes) > MAX_PROBES:
        raise SpiceError(f"at most {MAX_PROBES} probes per run, got {len(probes)}")
    return probes


def _store(result: core.RunResult) -> str:
    rid = uuid.uuid4().hex[:8]
    _RESULTS[rid] = result
    while len(_RESULTS) > _MAX_RESULTS:
        _RESULTS.popitem(last=False)
    return rid


def _get(result_id: str) -> core.RunResult:
    if result_id not in _RESULTS:
        raise SpiceError(f"unknown result_id {result_id!r}; run an analysis first")
    return _RESULTS[result_id]


def _real(v) -> float:
    return v.real if isinstance(v, complex) else float(v)


def _scale(r: core.RunResult) -> tuple[str, list[float]]:
    name = next(iter(r.vectors))
    return name, [_real(x) for x in r.vectors[name]]


def _vector_for(r: core.RunResult, vector: str) -> list:
    key = core.spice_vector(vector)
    if key not in r.vectors:
        raise SpiceError(f"no vector {vector!r}; available: {', '.join(sorted(r.vectors))}")
    return r.vectors[key]


UNIT_NOTE = ("every value is in SI base units: v(...) in volts, i(...) in amperes, time in seconds, "
             "frequency in hertz. 0.001 in a v() vector is one millivolt, not one volt.")


def _resolver(r: core.RunResult):
    """A function that reads a v(...) or i(...) value out of an operating-point result."""

    def resolve(name: str) -> float:
        if name in r.vectors:
            return _real(r.vectors[name][0])
        if name.startswith("v(") and "," in name:  # v(a,b) is v(a) - v(b)
            a, b = name[2:-1].split(",", 1)
            if f"v({a})" in r.vectors and f"v({b})" in r.vectors:
                return _real(r.vectors[f"v({a})"][0]) - _real(r.vectors[f"v({b})"][0])
        raise SpiceError(f"no vector {name!r} in that result")

    return resolve


MAX_DERIVED = 12
_LABEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")


@_tool
def compute(expression: str, result_id: str | None = None) -> dict:
    """Unit-aware arithmetic. Use this for ANY derived quantity instead of calculating in your head.

    Write each quantity as a number plus a unit: "1.00002 mV / 1 mohm" gives 1.00002 A (volts divided
    by ohms is amperes whatever the prefixes). Units: V A ohm W F H Hz s S, prefixes p n u m k M G.
    Adding unlike quantities (V + A) is refused, which catches mistakes.

    Pass the result_id of a run_op to read node values directly, so you never retype digits or guess
    a unit: compute("(v(/srp) - v(/srn)) / 1 mohm", result_id=...) returns the shunt current with its
    unit already attached. v(...) is in volts and i(...) in amperes. Use the "display" field.
    """
    from . import units

    resolve = None
    if result_id:
        r = _get(result_id)
        if r.analysis != "op":
            raise SpiceError("vector references need an operating-point (run_op) result")
        resolve = _resolver(r)
    return units.compute(expression, resolve)


@_tool
def check_netlist(netlist: str | None = None, netlist_path: str | None = None) -> dict:
    """Validate a netlist without simulating it. Reports what would be stripped and why a netlist is refused."""
    clean, notes = _circuit(netlist, netlist_path)
    return {"ok": True, "lines": len(clean.splitlines()), "notes": notes}


@_tool
def run_op(netlist: str | None = None, netlist_path: str | None = None,
           derived: dict[str, str] | None = None) -> dict:
    """DC operating point. Returns every node voltage and source current, and any derived quantities.

    Ask for the numbers you actually need with `derived`, a mapping from a name you choose to a
    unit-aware expression over the result, and read them back ready-made with their unit attached:
        derived={"cell1": "v(/vc1f,/vc0f)", "shunt_current": "(v(/srp) - v(/srn)) / 1 mohm"}
    returns derived.cell1.display = "3.688 V" and derived.shunt_current.display = "1.00002 A".
    Use those "display" strings as they are rather than dividing or relabelling the raw values:
    volts over ohms is amperes, and a bare 0.001 is a millivolt, not a milliampere.
    """
    from . import units

    labels = derived or {}
    if len(labels) > MAX_DERIVED:
        raise SpiceError(f"at most {MAX_DERIVED} derived quantities per run")
    for label in labels:
        if not _LABEL.match(label):
            raise SpiceError(f"derived name {label!r} must be a plain identifier such as shunt_current")
    clean, notes = _circuit(netlist, netlist_path)
    r = core.run_ngspice(clean, "op", {}, [])
    values = {k: _real(v[0]) for k, v in r.vectors.items()}
    out = {"result_id": _store(r), "values": values, "units": UNIT_NOTE, "notes": notes + r.notes}
    if labels:
        resolve = _resolver(r)
        out["derived"] = {label: units.compute(expr, resolve) for label, expr in labels.items()}
    return out


@_tool
def run_dc_sweep(source: str, start: str, stop: str, step: str, probes: list[str] | None = None,
                 netlist: str | None = None, netlist_path: str | None = None,
                 max_points: int = 25) -> dict:
    """Sweep an independent source (for example V1) from start to stop. Numbers use SPICE suffixes (10k, 1m)."""
    clean, notes = _circuit(netlist, netlist_path)
    r = core.run_ngspice(clean, "dc", {"source": source, "start": start, "stop": stop, "step": step},
                         _need_probes(probes))
    sname, xs = _scale(r)
    table = {name: [(x, _real(y)) for x, y in core.downsample(xs, r.vectors[name], max_points)]
             for name in r.vectors if name != sname}
    return {"result_id": _store(r), "sweep": sname, "points": len(xs), "table": table,
            "units": UNIT_NOTE, "notes": notes + r.notes}


@_tool
def run_ac(start_hz: str, stop_hz: str, points_per_decade: int = 20, probes: list[str] | None = None,
           netlist: str | None = None, netlist_path: str | None = None, max_points: int = 15) -> dict:
    """AC small-signal sweep. Returns magnitude in dB and phase in degrees, plus the -3 dB frequency of each probe."""
    import cmath
    import math

    clean, notes = _circuit(netlist, netlist_path)
    r = core.run_ngspice(clean, "ac", {"start": start_hz, "stop": stop_hz,
                                       "points_per_decade": points_per_decade}, _need_probes(probes))
    _, freqs = _scale(r)
    out, bw = {}, {}
    for name, series in r.vectors.items():
        if name == "frequency":
            continue
        mags = [core.db(c) for c in series]
        phases = [math.degrees(cmath.phase(c)) for c in series]
        rows = core.downsample(freqs, list(zip(mags, phases)), max_points)
        out[name] = [{"hz": f, "db": round(m, 3), "deg": round(p, 2)} for f, (m, p) in rows]
        bw[name] = core.bandwidth_3db(freqs, mags)
    return {"result_id": _store(r), "points": len(freqs), "table": out, "bandwidth_3db_hz": bw,
            "units": UNIT_NOTE, "notes": notes + r.notes}


@_tool
def run_transient(step: str, stop: str, probes: list[str] | None = None, max_step: str | None = None,
                  netlist: str | None = None, netlist_path: str | None = None,
                  max_points: int = 25) -> dict:
    """Transient simulation. Returns per-probe statistics and a downsampled waveform."""
    clean, notes = _circuit(netlist, netlist_path)
    r = core.run_ngspice(clean, "tran", {"step": step, "stop": stop, "max_step": max_step or ""},
                         _need_probes(probes))
    sname, ts = _scale(r)
    out, st = {}, {}
    for name, series in r.vectors.items():
        if name == sname:
            continue
        ys = [_real(v) for v in series]
        st[name] = core.stats(ys)
        out[name] = [(t, y) for t, y in core.downsample(ts, ys, max_points)]
    return {"result_id": _store(r), "points": len(ts), "end_time": ts[-1], "stats": st, "waveform": out,
            "units": UNIT_NOTE, "notes": notes + r.notes}


@_tool
def get_vector(result_id: str, vector: str, max_points: int = 100) -> dict:
    """Fetch a downsampled vector from an earlier run, for example v(out) from a transient."""
    r = _get(result_id)
    sname, xs = _scale(r)
    ys = [_real(v) for v in _vector_for(r, vector)]
    return {"scale": sname, "points": len(xs), "data": core.downsample(xs, ys, max_points)}


@_tool
def measure(result_id: str, kind: str, vector: str) -> dict:
    """Derive a number from an earlier run. kind is one of: bandwidth_3db (AC), rise_time_10_90 (transient step),
    peak_to_peak, average, rms, final_value, minimum, maximum."""
    r = _get(result_id)
    series = _vector_for(r, vector)
    _, xs = _scale(r)
    if kind == "bandwidth_3db":
        if not r.is_complex:
            raise SpiceError("bandwidth_3db needs an AC result")
        value = core.bandwidth_3db(xs, [core.db(c) for c in series])
        return {"kind": kind, "vector": vector, "hz": value,
                "note": None if value is not None else "magnitude never fell 3 dB within the sweep"}
    ys = [_real(v) for v in series]
    s = core.stats(ys)
    table = {"peak_to_peak": s["peak_to_peak"], "average": s["mean"], "rms": s["rms"],
             "final_value": s["final"], "minimum": s["min"], "maximum": s["max"]}
    if kind == "rise_time_10_90":
        value = core.rise_time(xs, ys)
        return {"kind": kind, "vector": vector, "seconds": value,
                "note": None if value is not None else "no 10 to 90 percent transition found"}
    if kind not in table:
        raise SpiceError(f"unknown kind {kind!r}; choose bandwidth_3db, rise_time_10_90, " + ", ".join(table))
    return {"kind": kind, "vector": vector, "value": table[kind]}


@_tool
def export_kicad_netlist(schematic_path: str, output_path: str | None = None) -> dict:
    """Export a KiCad schematic to a SPICE netlist with kicad-cli, ready for the run tools. Both paths must be under the allowed roots."""
    sch = Path(schematic_path).expanduser()
    if not core._resolved_within(sch, [r.resolve() for r in roots()]):
        raise SpiceError(f"{schematic_path} is outside the allowed roots")
    if not sch.is_file() or sch.suffix != ".kicad_sch":
        raise SpiceError(f"not a .kicad_sch file: {schematic_path}")
    out = Path(output_path).expanduser() if output_path else sch.with_suffix(".cir")
    if not core._resolved_within(out.parent / out.name, [r.resolve() for r in roots()]):
        raise SpiceError(f"{out} is outside the allowed roots")
    proc = subprocess.run(["kicad-cli", "sch", "export", "netlist", "--format", "spice",
                           "--output", str(out), str(sch)],
                          capture_output=True, text=True, timeout=60)
    if not out.is_file():
        raise SpiceError("kicad-cli produced no netlist: " + (proc.stderr or proc.stdout).strip()[:300])
    text = out.read_text(errors="replace")
    elements = [ln for ln in text.splitlines() if ln[:1].isalpha()]
    return {"path": str(out), "elements": len(elements),
            "preview": text.splitlines()[:25],
            "note": "KiCad only emits simulation-ready lines for symbols that carry a SPICE model; "
                    "run check_netlist next and expect parts without one to be missing or to fail."}


def main() -> None:
    mcp.run(transport="stdio")
