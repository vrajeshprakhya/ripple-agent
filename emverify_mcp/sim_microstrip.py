"""A straight microstrip line, full wave, run by the Python that has the openEMS bindings.

This file is executed as a subprocess by the emverify server, never imported by it, because
the bindings live in a different interpreter from the agent's. It reads one JSON object from
argv[1] and prints one JSON object as the last line of stdout.

Original code written against the openEMS Python API (openEMS is GPL-3.0; this file calls it
as a library and contains none of its tutorial code). The design choices that matter, and
that the script emitted by mcp-openems lacked, are:

  * the ground plane is the solver's PEC lower boundary, not a thin sheet that has to land
    on a mesh line;
  * the ports are microstrip ports, which launch the quasi-TEM mode and measure the line's
    own impedance and propagation constant from the simulated fields, rather than lumped
    resistors across the end faces;
  * mesh lines are placed around each strip edge, where the fields are singular, and the
    mesh is smoothed so neighbouring cells never differ by much;
  * the domain is wide and tall enough for the fringing fields to decay before the boundary.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

UNIT = 1e-3  # lengths are in millimetres


def build_and_run(p: dict, workdir: str) -> dict:
    w, h, er = p["width_mm"], p["height_mm"], p["er"]
    fmax = p["f_max_ghz"] * 1e9
    half = p["length_mm"] / 2.0

    # Cell sizes: 40 per wavelength in the dielectric at the top frequency, four times finer
    # near the strip, matching the field gradient there.
    coarse = C0 / (fmax * np.sqrt(er)) / UNIT / 40.0
    # Wavelength alone under-resolves a narrow strip (a first version put too few cells across
    # a 1.2 mm trace and was 4% off in impedance), so the fine size also resolves the strip
    # width and the substrate height.
    fine = min(coarse / 4.0, w / 12.0, h / 6.0)
    # Field singularity at a conductor edge: one line a third of a fine cell inside the edge
    # and one two thirds outside it.
    edge = fine * np.array([2.0 / 3.0, -1.0 / 3.0])

    sim = openEMS(EndCriteria=1e-4)
    sim.SetGaussExcite(fmax / 2, fmax / 2)
    #                 x-min   x-max   y-min  y-max  z-min  z-max
    sim.SetBoundaryCond(["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"])

    csx = ContinuousStructure()
    sim.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    y_margin = max(10 * h, 3 * w)
    z_top = 11 * h

    mesh.AddLine("x", [-half, 0, half])
    mesh.SmoothMeshLines("x", coarse)

    mesh.AddLine("y", 0)
    mesh.AddLine("y", w / 2 + edge)
    mesh.AddLine("y", -w / 2 - edge)
    mesh.SmoothMeshLines("y", fine)
    mesh.AddLine("y", [-(w / 2 + y_margin), w / 2 + y_margin])
    mesh.SmoothMeshLines("y", coarse)

    mesh.AddLine("z", np.linspace(0, h, 5))
    mesh.AddLine("z", z_top)
    mesh.SmoothMeshLines("z", coarse)

    substrate = csx.AddMaterial("substrate", epsilon=er)
    substrate.AddBox([mesh.GetLine("x", 0), mesh.GetLine("y", 0), 0],
                     [mesh.GetLine("x", -1), mesh.GetLine("y", -1), h])

    metal = csx.AddMetal("strip")
    # The source is fed `feed` in from the port end (clear of the absorbing layer), and the
    # impedance and propagation constant are measured further along at `meas`. The two must be
    # well apart: a probe within a couple of substrate heights of the source reads its near
    # field, not the guided mode (first attempt: 0.6 mm apart, effective permittivity 5.7 on a
    # substrate of 4.2, which is impossible).
    feed = 10 * coarse
    meas = max(0.7 * half, feed + 6 * h)
    # Each port builds the half of the strip on its own side of x = 0.
    port1 = sim.AddMSLPort(1, metal, [-half, -w / 2, h], [0, w / 2, 0], "x", "z", excite=-1,
                           FeedShift=feed, MeasPlaneShift=meas, priority=10)
    port2 = sim.AddMSLPort(2, metal, [half, -w / 2, h], [0, w / 2, 0], "x", "z",
                           MeasPlaneShift=meas, priority=10)

    sim.Run(workdir, cleanup=True)

    f = np.linspace(fmax / 20, fmax, 160)
    for port in (port1, port2):
        port.CalcPort(workdir, f)  # reference impedance defaults to the line's own

    s11 = port1.uf_ref / port1.uf_inc
    s21 = port2.uf_ref / port1.uf_inc
    z0 = np.real(port1.Z_ref)
    beta = np.real(port1.beta)
    eeff = (beta * C0 / (2 * np.pi * f)) ** 2
    power = np.abs(s11) ** 2 + np.abs(s21) ** 2
    db = lambda x: 20 * np.log10(np.maximum(np.abs(x), 1e-12))
    low = f <= 0.4 * fmax  # the quasi-static end of the band, where Hammerstad applies best

    pick = np.linspace(0, len(f) - 1, 8).astype(int)
    return {
        "f_ghz": [round(float(f[i]) / 1e9, 4) for i in pick],
        "s11_db": [round(float(db(s11[i])), 2) for i in pick],
        "s21_db": [round(float(db(s21[i])), 3) for i in pick],
        "s11_worst_db": round(float(db(s11).max()), 2),
        "s21_worst_db": round(float(db(s21).min()), 3),
        "power_balance": {"min": float(power.min()), "median": float(np.median(power)),
                          "max": float(power.max())},
        "z0_ohm": {"low_band_median": float(np.median(z0[low])), "at_f_max": float(z0[-1])},
        "eeff": {"low_band_median": float(np.median(eeff[low])), "at_f_max": float(eeff[-1])},
        "grid": [len(mesh.GetLines(a)) for a in ("x", "y", "z")],
    }


def main() -> int:
    params = json.loads(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="emverify_") as work:
        result = build_and_run(params, os.path.join(work, "sim"))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
