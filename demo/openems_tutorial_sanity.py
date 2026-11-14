#!/usr/bin/env python3
"""Does this openEMS build conserve energy on an official tutorial?

Runs the project's own MSL_NotchFilter tutorial headless and checks |S11|^2 + |S21|^2 <= 1
(a lossless passive two-port cannot return more power than it was given). If the build and
solver are sound, this passes whatever is wrong with any generated script.

Run with the Python that has the openEMS bindings:
    ~/mcp-openems/.venv/bin/python demo/openems_tutorial_sanity.py
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

TUTORIAL = pathlib.Path.home() / "opt/src/openEMS-Project/openEMS/python/Tutorials/MSL_NotchFilter.py"


def main() -> int:
    src = TUTORIAL.read_text()
    cut = src.index("fig, axis")  # drop the plotting, keep the simulation and the S-parameters
    out = pathlib.Path(tempfile.mkdtemp(prefix="oems_tut_"))
    body = src[:cut] + f"""
import numpy as _np
_pw = _np.abs(s11) ** 2 + _np.abs(s21) ** 2
_np.savez({str(out / 's.npz')!r}, f=f, s11=s11, s21=s21)
"""
    (out / "run.py").write_text(body)
    p = subprocess.run([sys.executable, str(out / "run.py")], capture_output=True, text=True, cwd=out,
                       env={**os.environ, "MPLBACKEND": "Agg"})
    if not (out / "s.npz").exists():
        print("tutorial produced no results:\n", (p.stdout + p.stderr)[-1500:])
        return 2
    import numpy as np

    r = np.load(out / "s.npz")
    f, s11, s21 = r["f"], r["s11"], r["s21"]
    pw = np.abs(s11) ** 2 + np.abs(s21) ** 2
    db = lambda x: 20 * np.log10(np.maximum(np.abs(x), 1e-12))
    print(f"{len(f)} frequency points, {f[0] / 1e9:.2f} to {f[-1] / 1e9:.2f} GHz")
    print(f"|S11| {db(s11).min():.1f} to {db(s11).max():.1f} dB   |S21| {db(s21).min():.1f} to {db(s21).max():.1f} dB")
    print(f"power balance |S11|^2+|S21|^2: min {pw.min():.3f}, median {np.median(pw):.3f}, max {pw.max():.3f}")
    ok = pw.max() < 1.05
    print("VERDICT:", "energy conserved (max <= 1.05)" if ok else "ENERGY NOT CONSERVED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
