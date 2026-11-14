"""Circuit families with closed-form answers, and the parameter ranges that define 'seen' and 'unseen'.

Three families make up the training distribution: resistive dividers, RC low-pass filters and
series RLC band-pass circuits. Three more (CR high-pass, loaded divider, two-stage RC) never appear
in training and exist only to test structural generalisation. Every generator returns the analytic
answer so the pipeline can require formula, ngspice and the independent solver to agree.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8,
       7.5, 8.2, 9.1]
E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
SUPPLIES = [1.8, 3.3, 5.0, 9.0, 12.0, 15.0, 24.0]


@dataclass(frozen=True)
class Ranges:
    """Where component values and the quantities they set are allowed to fall."""
    R: tuple = (100.0, 1e5)
    C: tuple = (100e-12, 10e-6)
    L: tuple = (10e-6, 100e-3)
    R_rlc: tuple = (1.0, 1e3)
    fc: tuple = (10.0, 1e6)        # RC cutoff
    f0: tuple = (200.0, 5e6)       # RLC resonance
    Q: tuple = (0.5, 30.0)
    # when set, at least one of these must hold for a sample to count as out of range
    outside_only: bool = False


TRAIN = Ranges()
# Out of range: components or the resulting frequency beyond anything in training.
OOD = Ranges(R=(1.0, 5e6), C=(1e-12, 470e-6), L=(1e-6, 1.0), R_rlc=(0.1, 5e3), fc=(1.0, 3e7), f0=(20.0, 5e7),
             Q=(0.3, 60.0), outside_only=True)


def _inside(v: float, rng_: tuple) -> bool:
    return rng_[0] <= v <= rng_[1]


def _series_values(series: list[float], lo: float, hi: float) -> list[float]:
    out = []
    for d in range(-13, 10):
        for m in series:
            v = float(f"{m * 10.0 ** d:.6g}")
            if lo * 0.999 <= v <= hi * 1.001:
                out.append(v)
    return sorted(set(out))


def pick(rng: random.Random, series: list[float], lo: float, hi: float) -> float:
    return rng.choice(_series_values(series, lo, hi))


def spice_str(v: float) -> str:
    for scale, suffix in ((1e9, "g"), (1e6, "meg"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "u"),
                          (1e-9, "n"), (1e-12, "p"), (1e-15, "f")):
        if v >= scale * 0.9999:
            return f"{v / scale:.4g}{suffix}"
    return f"{v:.4g}"


def human(v: float, unit: str) -> str:
    """Component values as a person would write them: 4.7 kohm, 100 nF."""
    table = {"ohm": [(1e6, "Mohm"), (1e3, "kohm"), (1.0, "ohm")],
             "F": [(1.0, "F"), (1e-3, "mF"), (1e-6, "uF"), (1e-9, "nF"), (1e-12, "pF")],
             "H": [(1.0, "H"), (1e-3, "mH"), (1e-6, "uH")]}[unit]
    for scale, name in table:
        if v >= scale * 0.9999:
            return f"{v / scale:.4g} {name}"
    return f"{v:.4g} {unit}"


def _round_sig(x: float, n: int = 3) -> float:
    return float(f"{x:.{n}g}")


@dataclass
class Gen:
    family: str
    netlist: str
    spec: dict
    analytic: float
    rationale: str
    params: dict = field(default_factory=dict)


# --- Training families ---------------------------------------------------------------------------
def divider(rng: random.Random, r: Ranges = TRAIN) -> Gen:
    vs, r1, r2 = rng.choice(SUPPLIES), pick(rng, E24, *r.R), pick(rng, E24, *r.R)
    net = f"Resistive divider\nV1 in 0 DC {vs:g}\nR1 in out {spice_str(r1)}\nR2 out 0 {spice_str(r2)}\n"
    if rng.random() < 0.5:
        return Gen("divider", net, {"kind": "op_voltage", "node": "out"}, vs * r2 / (r1 + r2),
                   f"Vout = Vs*R2/(R1+R2) = {vs:g}*{r2:g}/({r1:g}+{r2:g}).", dict(vs=vs, r1=r1, r2=r2))
    return Gen("divider", net, {"kind": "op_current", "source": "v1"}, vs / (r1 + r2),
               f"I = Vs/(R1+R2) = {vs:g}/({r1:g}+{r2:g}).", dict(vs=vs, r1=r1, r2=r2))


def _fits(r: Ranges, values: dict[str, float], derived: tuple[str, float] | None) -> bool:
    """For ordinary ranges everything must lie inside; for out-of-range, at least one thing must lie outside TRAIN."""
    inside_train = all(_inside(v, getattr(TRAIN, k)) for k, v in values.items()) and (
        derived is None or _inside(derived[1], getattr(TRAIN, derived[0])))
    ok_range = all(_inside(v, getattr(r, k)) for k, v in values.items()) and (
        derived is None or _inside(derived[1], getattr(r, derived[0])))
    return ok_range and (not inside_train if r.outside_only else True)


def rc_lowpass(rng: random.Random, r: Ranges = TRAIN) -> Gen:
    for _ in range(500):
        res, cap = pick(rng, E24, *r.R), pick(rng, E12, *r.C)
        fc = 1 / (2 * math.pi * res * cap)
        if _fits(r, {"R": res, "C": cap}, ("fc", fc)):
            break
    else:
        raise RuntimeError("could not sample an RC low-pass in range")
    net = f"RC low-pass\nV1 in 0 DC 0 AC 1\nR1 in out {spice_str(res)}\nC1 out 0 {spice_str(cap)}\n"
    kind = rng.choice(["f3db_hz", "ac_gain_db", "ac_phase_deg"])
    p = dict(r=res, c=cap, fc=fc)
    if kind == "f3db_hz":
        return Gen("rc_lowpass", net, {"kind": kind, "node": "out"}, fc, "fc = 1/(2*pi*R*C).", p)
    f = _round_sig(fc * rng.choice([0.1, 0.2, 0.5, 1, 2, 3, 5, 10]))
    x = f / fc
    if kind == "ac_gain_db":
        return Gen("rc_lowpass", net, {"kind": kind, "node": "out", "freq_hz": f}, -10 * math.log10(1 + x * x),
                   "|H| = 1/sqrt(1+(f/fc)^2) with fc = 1/(2*pi*R*C).", p)
    return Gen("rc_lowpass", net, {"kind": kind, "node": "out", "freq_hz": f}, -math.degrees(math.atan(x)),
               "phase = -atan(f/fc) with fc = 1/(2*pi*R*C).", p)


def series_rlc(rng: random.Random, r: Ranges = TRAIN) -> Gen:
    for _ in range(2000):
        ind, cap, res = pick(rng, E12, *r.L), pick(rng, E12, *r.C), pick(rng, E24, *r.R_rlc)
        f0 = 1 / (2 * math.pi * math.sqrt(ind * cap))
        q = math.sqrt(ind / cap) / res
        if _fits(r, {"L": ind, "C": cap, "R_rlc": res}, ("f0", f0)) and r.Q[0] <= q <= r.Q[1]:
            break
    else:
        raise RuntimeError("could not sample an RLC in range")
    net = (f"Series RLC band-pass\nV1 in 0 DC 0 AC 1\nL1 in a {spice_str(ind)}\nC1 a out {spice_str(cap)}\n"
           f"R1 out 0 {spice_str(res)}\n")
    p = dict(l=ind, c=cap, r=res, f0=f0, q=q)
    kind = rng.choice(["peak_freq_hz", "ac_gain_db", "ac_phase_deg"])
    if kind == "peak_freq_hz":
        return Gen("series_rlc", net, {"kind": kind, "node": "out"}, f0, "f0 = 1/(2*pi*sqrt(L*C)).", p)
    m = rng.choice([0.25, 0.5, 0.8, 1.25, 2.0, 4.0])
    f = _round_sig(f0 * m)
    x = q * (f / f0 - f0 / f)
    if kind == "ac_gain_db":
        return Gen("series_rlc", net, {"kind": kind, "node": "out", "freq_hz": f}, -10 * math.log10(1 + x * x),
                   "|H| = 1/sqrt(1+Q^2*(f/f0-f0/f)^2), Q = sqrt(L/C)/R.", p)
    return Gen("series_rlc", net, {"kind": kind, "node": "out", "freq_hz": f}, -math.degrees(math.atan(x)),
               "phase = -atan(Q*(f/f0-f0/f)), Q = sqrt(L/C)/R.", p)


# --- Families that never appear in training ---------------------------------------------------------
def cr_highpass(rng: random.Random, r: Ranges = TRAIN) -> Gen:
    for _ in range(500):
        res, cap = pick(rng, E24, *r.R), pick(rng, E12, *r.C)
        fc = 1 / (2 * math.pi * res * cap)
        if _fits(r, {"R": res, "C": cap}, ("fc", fc)):
            break
    net = f"CR high-pass\nV1 in 0 DC 0 AC 1\nC1 in out {spice_str(cap)}\nR1 out 0 {spice_str(res)}\n"
    f = _round_sig(fc * rng.choice([0.1, 0.2, 0.5, 1, 2, 5, 10]))
    x = f / fc
    if rng.random() < 0.5:
        return Gen("cr_highpass", net, {"kind": "ac_gain_db", "node": "out", "freq_hz": f},
                   20 * math.log10(x / math.sqrt(1 + x * x)), "|H| = (f/fc)/sqrt(1+(f/fc)^2).", dict(fc=fc))
    return Gen("cr_highpass", net, {"kind": "ac_phase_deg", "node": "out", "freq_hz": f},
               math.degrees(math.atan(1 / x)), "phase = atan(fc/f).", dict(fc=fc))


def loaded_divider(rng: random.Random, r: Ranges = TRAIN) -> Gen:
    vs, r1, r2, rl = rng.choice(SUPPLIES), pick(rng, E24, *r.R), pick(rng, E24, *r.R), pick(rng, E24, *r.R)
    rp = r2 * rl / (r2 + rl)
    net = (f"Loaded divider\nV1 in 0 DC {vs:g}\nR1 in out {spice_str(r1)}\nR2 out 0 {spice_str(r2)}\n"
           f"R3 out 0 {spice_str(rl)}\n")
    if rng.random() < 0.5:
        return Gen("loaded_divider", net, {"kind": "op_voltage", "node": "out"}, vs * rp / (r1 + rp),
                   "R2 and the load are in parallel, then divide.", {})
    return Gen("loaded_divider", net, {"kind": "op_current", "source": "v1"}, vs / (r1 + rp),
               "The source sees R1 in series with R2||R3.", {})


def two_stage_rc(rng: random.Random, r: Ranges = TRAIN) -> Gen:
    for _ in range(500):
        r1, r2 = pick(rng, E24, *r.R), pick(rng, E24, *r.R)
        c1, c2 = pick(rng, E12, *r.C), pick(rng, E12, *r.C)
        fc = 1 / (2 * math.pi * math.sqrt(r1 * c1 * r2 * c2))
        if _fits(r, {"R": r1, "C": c1}, ("fc", fc)) and _inside(r2, r.R) and _inside(c2, r.C):
            break
    net = (f"Two-stage RC low-pass\nV1 in 0 DC 0 AC 1\nR1 in a {spice_str(r1)}\nC1 a 0 {spice_str(c1)}\n"
           f"R2 a out {spice_str(r2)}\nC2 out 0 {spice_str(c2)}\n")
    a1, a2 = r1 * c1 + r2 * c2 + r1 * c2, r1 * c1 * r2 * c2

    def mag(f: float) -> float:
        w = 2 * math.pi * f
        return 1 / math.hypot(1 - a2 * w * w, a1 * w)

    if rng.random() < 0.5:
        # |H|^2 = 1/2 is a quadratic in w^2: a2^2 w^4 + (a1^2 - 2 a2) w^2 - 1 = 0. A bisection
        # bracketed around fc missed the corner when the two poles were far apart.
        b = a1 * a1 - 2 * a2
        w2 = (-b + math.sqrt(b * b + 4 * a2 * a2)) / (2 * a2 * a2)
        f3 = math.sqrt(w2) / (2 * math.pi)
        return Gen("two_stage_rc", net, {"kind": "f3db_hz", "node": "out"}, f3,
                   "H(s) = 1/(1+s(R1C1+R2C2+R1C2)+s^2 R1C1R2C2); solve |H| = 1/sqrt(2).", {})
    f = _round_sig(fc * rng.choice([0.1, 0.3, 1, 3]))
    return Gen("two_stage_rc", net, {"kind": "ac_gain_db", "node": "out", "freq_hz": f}, 20 * math.log10(mag(f)),
               "|H| from the second-order transfer function of the loaded ladder.", {})


def _bisect(fn, lo: float, hi: float) -> float:
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if fn(mid) > 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


TRAIN_FAMILIES = {"divider": divider, "rc_lowpass": rc_lowpass, "series_rlc": series_rlc}
CONFIG_FAMILIES = {"cr_highpass": cr_highpass, "loaded_divider": loaded_divider, "two_stage_rc": two_stage_rc}
