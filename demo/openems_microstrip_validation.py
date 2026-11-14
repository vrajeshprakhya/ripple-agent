#!/usr/bin/env python3
"""Validate a full-wave openEMS simulation against transmission-line theory.

Takes the simulation script that mcp-openems generates for a microstrip, mismatches its two
lumped ports to ZREF ohms (default 100) so the reflection is large and depends sensitively on
the line's impedance and electrical length, runs it, and compares S11 and S21 with the exact
response of a lossless uniform line between the same reference impedances (ABCD matrices),
using the design's own analytical Z0 and effective permittivity.

Run with the Python that has the openEMS bindings, e.g.
    ~/mcp-openems/.venv/bin/python demo/openems_microstrip_validation.py [generated_script.py]

Without an argument it asks the server for a 3 mm x 20 mm microstrip on 1.6 mm FR-4 (er 4.2).
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np

C0 = 299_792_458.0
ZREF = float(os.environ.get("ZREF", "100"))
HOME = pathlib.Path.home()


def generate() -> tuple[str, dict]:
    """Ask mcp-openems for the design and its script (needs the strands MCP client)."""
    from mcp import StdioServerParameters, stdio_client
    from strands.tools.mcp import MCPClient

    def transport():
        return stdio_client(StdioServerParameters(
            command=str(HOME / "mcp-openems/.venv/bin/mcp-openems"), args=[], cwd=str(HOME / "mcp-openems")))

    def call(c, tool, **kw):
        r = c.call_tool_sync("t", tool, kw)
        return json.loads("".join(x.get("text", "") for x in r["content"]))

    with MCPClient(transport, startup_timeout=60) as c:
        d = call(c, "openems_create_microstrip", frequency_ghz=2.4, trace_width_mm=3.0, trace_length_mm=20.0,
                 dielectric_height_mm=1.6, substrate_er=4.2, name="validation")
        s = call(c, "openems_generate_script", design_id=d["design_id"])
    return s["script"], d["design"]


def theory(f: np.ndarray, z0: float, eeff: float, length_m: float, zref: float):
    """S11 and S21 of a lossless line of impedance z0 between two zref ports."""
    th = 2 * np.pi * f * math.sqrt(eeff) * length_m / C0
    a = d = np.cos(th)
    b = 1j * z0 * np.sin(th)
    c = 1j * np.sin(th) / z0
    den = a + b / zref + c * zref + d
    return (a + b / zref - c * zref - d) / den, 2 / den


def main() -> int:
    if len(sys.argv) > 1:
        script, calc = pathlib.Path(sys.argv[1]).read_text(), None
    else:
        script, calc = generate()

    z0 = float(re.search(r"z0_calc\s*=\s*([\d.]+)", script).group(1))
    length_mm = float(re.search(r"trace_l\s*=\s*([\d.]+)", script).group(1))
    eeff = calc["calculated"]["effective_er"] if calc else 3.1896

    # Mismatch the ports, and replace the plotting tail with a save of the raw S-parameters.
    n1 = script.count("1, z0_calc,")
    n2 = script.count("2, z0_calc,")
    assert n1 == 1 and n2 == 1, f"expected one definition of each port, found {n1} and {n2}"
    script = script.replace("1, z0_calc,", f"1, {ZREF},").replace("2, z0_calc,", f"2, {ZREF},")
    script = script.split("# Plot results")[0]
    out = pathlib.Path(tempfile.mkdtemp(prefix="oems_val_")) / "s.npz"
    script += f"\nnp.savez({str(out)!r}, f=f, s11=s11, s21=s21)\n"
    run = out.parent / "run.py"
    run.write_text(script)

    print(f"line: z0={z0} ohm  eeff={eeff}  length={length_mm} mm  ports={ZREF} ohm", flush=True)
    p = subprocess.run([sys.executable, str(run)], capture_output=True, text=True)
    if not out.exists():
        print("simulation produced no results:\n", (p.stdout + p.stderr)[-1500:])
        return 2

    r = np.load(out)
    f, s11, s21 = r["f"], r["s11"], r["s21"]
    t11, t21 = theory(f, z0, eeff, length_mm * 1e-3, ZREF)

    def db(x):
        return 20 * np.log10(np.maximum(np.abs(x), 1e-12))

    print("\n GHz   |S11| sim  theory   |S21| sim  theory   S21 phase sim  theory")
    for i in range(0, len(f), 50):
        ph = lambda v: math.degrees(np.angle(v))
        print(f"{f[i] / 1e9:5.2f}  {db(s11[i]):8.2f} {db(t11[i]):8.2f}  {db(s21[i]):9.2f} {db(t21[i]):8.2f}"
              f"   {ph(s21[i]):10.1f} {ph(t21[i]):9.1f}")

    band = (f > 1.5e9) & (f < 3.3e9)  # inside the excitation band, away from its edges
    d11, d21 = np.abs(db(s11) - db(t11))[band], np.abs(db(s21) - db(t21))[band]
    dph = np.abs(np.angle(s21 / t21, deg=True))[band]
    print(f"\n1.5-3.3 GHz:  |S11| differs by {np.median(d11):.2f} dB (median), {d11.max():.2f} dB (max)")
    print(f"              |S21| differs by {np.median(d21):.2f} dB (median), {d21.max():.2f} dB (max)")
    print(f"              S21 phase differs by {np.median(dph):.1f} deg (median), {dph.max():.1f} deg (max)")
    # Energy conservation: a passive two-port cannot return more power than it was given.
    pw = (np.abs(s11) ** 2 + np.abs(s21) ** 2)[band]
    print(f"power balance |S11|^2+|S21|^2 (must be <= 1; ~1 for a lossless line): "
          f"median {np.median(pw):.3f}, max {pw.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
