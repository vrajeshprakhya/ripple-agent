"""Validated full-wave microstrip simulation: parameter checking, a safe runner, and a gate.

The model supplies four numbers. It never supplies a script: the simulation code is the fixed
file next to this one, so there is nothing for a prompt to inject into. Results are refused,
not reported, when they break energy conservation, because a lossless passive line cannot
return more power than it receives and a model will happily quote a number it was handed.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import resource
import signal
import subprocess
import sys

SIM_SCRIPT = pathlib.Path(__file__).resolve().parent / "sim_microstrip.py"
DEFAULT_SOLVER_PYTHON = pathlib.Path.home() / "mcp-openems" / ".venv" / "bin" / "python"
RUN_TIMEOUT_S = 600

# Bounds keep a request inside what the mesh rules were validated on (0.2 to 3 mm strips,
# er 3.4 to 10.2) with room either side, and stop a request that would need a huge grid.
LIMITS = {
    "width_mm": (0.1, 20.0),
    "height_mm": (0.05, 10.0),
    "er": (1.0, 20.0),
    "f_max_ghz": (0.5, 20.0),
    "length_mm": (5.0, 200.0),
}
# Observed on four geometries: power balance 0.987 to 1.001. A passive lossless line cannot
# exceed 1, so anything above 1.03 is the extraction failing; far below 1 means energy is
# going somewhere it should not (a leaky boundary, a probe in the near field).
POWER_MAX, POWER_MIN = 1.03, 0.90
# Worst observed error against Hammerstad was 2.7 percent (Z0) and 2.0 percent (eeff).
AGREE_TOL = 0.05


class EmError(RuntimeError):
    """A refused request or a refused result, with a message the caller can act on."""


def validate(params: dict) -> dict:
    out = {}
    for key, (lo, hi) in LIMITS.items():
        if key not in params:
            raise EmError(f"missing {key}")
        v = params[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise EmError(f"{key} must be a finite number, got {v!r}")
        if not lo <= v <= hi:
            raise EmError(f"{key} = {v} is outside the validated range {lo} to {hi}")
        out[key] = float(v)
    # The measurement plane has to clear the source and the absorbing layer.
    if out["length_mm"] < 12 * out["height_mm"]:
        raise EmError("line_length_mm must be at least 12 times dielectric_height_mm so the "
                      "measurement plane clears the source's near field")
    return out


def hammerstad(w_mm: float, h_mm: float, er: float) -> tuple[float, float]:
    """Quasi-static Hammerstad-Jensen microstrip: (Z0 in ohms, effective permittivity)."""
    u = w_mm / h_mm
    a = 1 + (1 / 49) * math.log((u**4 + (u / 52) ** 2) / (u**4 + 0.432)) + (1 / 18.7) * math.log(1 + (u / 18.1) ** 3)
    b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
    eeff = (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / u) ** (-a * b)
    f = 6 + (2 * math.pi - 6) * math.exp(-((30.666 / u) ** 0.7528))
    return 60 * math.log(f / u + math.sqrt(1 + (2 / u) ** 2)) / math.sqrt(eeff), eeff


def solver_python() -> pathlib.Path:
    p = pathlib.Path(os.environ.get("EMVERIFY_PYTHON") or DEFAULT_SOLVER_PYTHON).expanduser()
    if not p.is_file():
        raise EmError(f"solver Python not found: {p}. Set EMVERIFY_PYTHON to a Python that can "
                      "import openEMS and CSXCAD (see the README for the source build).")
    return p


def solver_status() -> dict:
    """Whether the openEMS bindings import under the solver Python, and their versions."""
    try:
        py = solver_python()
    except EmError as e:
        return {"available": False, "reason": str(e)}
    r = subprocess.run([str(py), "-c", "import CSXCAD, openEMS; print(CSXCAD.__version__, openEMS.__version__)"],
                       capture_output=True, text=True, timeout=60, env={"PATH": "/usr/bin:/bin"})
    if r.returncode != 0:
        return {"available": False, "python": str(py), "reason": r.stderr.strip().splitlines()[-1][:200]}
    csxcad, oems = r.stdout.split()
    return {"available": True, "python": str(py), "CSXCAD": csxcad, "openEMS": oems}


def _limits(cpu_s: int):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (12 << 30, 12 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2 << 30, 2 << 30))
    return apply


def run_microstrip(params: dict, timeout: int = RUN_TIMEOUT_S) -> dict:
    p = validate(params)
    py = solver_python()
    # Threads: the solver is OpenMP; cap it so a run leaves the machine usable.
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C", "OMP_NUM_THREADS": "4"}
    proc = subprocess.Popen([str(py), str(SIM_SCRIPT), json.dumps(p)], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                            start_new_session=True, preexec_fn=_limits(timeout * 4))
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise EmError(f"the simulation did not finish within {timeout}s and was killed")
    lines = out.strip().splitlines()
    try:
        raw = json.loads(lines[-1])
    except (IndexError, ValueError):
        raise EmError("the simulation produced no result. Its last output:\n" + "\n".join(lines[-8:]))
    return judge(p, raw)


def judge(p: dict, raw: dict) -> dict:
    """Gate the raw result on energy conservation, then set it beside the analytical answer."""
    pw = raw["power_balance"]
    if pw["max"] > POWER_MAX or pw["min"] < POWER_MIN:
        raise EmError(
            f"refusing to report: the simulated S-parameters are not physical (|S11|^2+|S21|^2 "
            f"ranged {pw['min']:.3f} to {pw['max']:.3f}; a lossless passive line must stay in "
            f"{POWER_MIN} to {POWER_MAX}). The numbers would look plausible and be wrong.")
    z0_h, eeff_h = hammerstad(p["width_mm"], p["height_mm"], p["er"])
    z0, eeff = raw["z0_ohm"]["low_band_median"], raw["eeff"]["low_band_median"]
    ez, ee = (z0 - z0_h) / z0_h, (eeff - eeff_h) / eeff_h
    return {
        "inputs": p,
        "full_wave": {"z0_ohm": round(z0, 2), "effective_er": round(eeff, 3),
                      "delay_ps_per_mm": round(math.sqrt(eeff) / 299.792458 * 1000, 3),
                      "s11_worst_db": raw["s11_worst_db"], "s21_worst_db": raw["s21_worst_db"]},
        "analytical_hammerstad": {"z0_ohm": round(z0_h, 2), "effective_er": round(eeff_h, 3)},
        "difference_percent": {"z0": round(100 * ez, 1), "effective_er": round(100 * ee, 1)},
        "agrees_with_theory": abs(ez) <= AGREE_TOL and abs(ee) <= AGREE_TOL,
        "energy_check": {"power_balance_min": round(pw["min"], 3), "power_balance_max": round(pw["max"], 3),
                         "passed": True},
        "sweep": {"f_ghz": raw["f_ghz"], "s11_db": raw["s11_db"], "s21_db": raw["s21_db"]},
        "grid_cells": raw["grid"],
        "note": ("Z0 and effective permittivity are the median over the low 40% of the band, where the "
                 "quasi-static formula applies best. Validated on four geometries (worst 2.7% Z0, "
                 "2.0% effective permittivity against Hammerstad); outside that range treat the "
                 "difference as information, not an error."),
    }
