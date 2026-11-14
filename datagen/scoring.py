"""Automatic scoring, summaries and paired comparison for the benchmark.

Numerical answers are scored by relative error against the verified value. dB and degree answers
use absolute error instead, because relative error is meaningless near 0 dB or 0 degrees. Format
compliance is reported separately from correctness: a model can learn to answer in the requested
shape without getting any better at physics, and these must not be confused.
"""
from __future__ import annotations

import math
import re
import statistics

from . import design

_PREFIX = {"": 1.0, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9}
_UNIT_RE = {"V": "V", "A": "A", "Hz": "Hz", "dB": "dB", "deg": "(?:deg(?:rees)?|°)"}
_PREFIXABLE = {"V": "pnuµmkMG", "A": "pnuµmk", "Hz": "mkMG", "dB": "", "deg": ""}
EPS = {"V": 1e-6, "A": 1e-9, "Hz": 1e-3}
ABS_TOL = {"dB": (0.5, 0.1), "deg": (5.0, 1.0)}   # (within "5%"-class, within "1%"-class)
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_answer(text: str, unit: str) -> dict:
    """Extract the model's numeric answer in base units.

    Takes the last number that carries the expected unit. `strict` means the whole reply is just a
    number and its unit. A bare number with no unit is accepted but flagged `unit_missing`.
    """
    text = text.strip()
    pre = _PREFIXABLE[unit]
    pat = re.compile(rf"(?P<num>{NUM})\s*(?P<pre>[{pre}]?)\s*(?P<unit>{_UNIT_RE[unit]})(?![A-Za-z])" if pre else
                     rf"(?P<num>{NUM})\s*(?P<pre>)(?P<unit>{_UNIT_RE[unit]})(?![A-Za-z])")
    matches = list(pat.finditer(text))
    if matches:
        m = matches[-1]
        value = float(m.group("num")) * _PREFIX[m.group("pre") or ""]
        prefix_class = f"[{pre}]?" if pre else ""  # an empty [] is not a valid character class
        strict = bool(re.fullmatch(rf"\s*{NUM}\s*{prefix_class}\s*{_UNIT_RE[unit]}\s*\.?\s*", text))
        return {"value": value, "unit_missing": False, "strict": strict}
    # A bare number counts only when it is the whole reply. Taking the last number from free text
    # once scored "-7" out of a truncated "10^{-7}" as a valid answer.
    if re.fullmatch(rf"\s*{NUM}\s*\.?\s*", text):
        return {"value": float(text.strip().rstrip(".")), "unit_missing": True, "strict": False}
    return {"value": None, "unit_missing": False, "strict": False}


def score_predict(problem: dict, response: str) -> dict:
    unit, kind = problem["answer"]["unit"], problem["measurement"]["kind"]
    truth = problem["answer"]["value"]
    p = parse_answer(response, unit)
    rec = {"id": problem["id"], "category": problem["category"], "family": problem["family"], "kind": kind,
           "task": "predict", "valid": p["value"] is not None, "strict": p["strict"], "unit_missing": p["unit_missing"],
           "pred": p["value"], "truth": truth, "ok5": False, "ok1": False, "err": None, "err_type": None}
    if p["value"] is None:
        return rec
    if unit in ABS_TOL:
        err = abs(p["value"] - truth)
        if unit == "deg":
            err = min(err % 360, 360 - err % 360)
        rec.update(err=err, err_type="abs", ok5=err <= ABS_TOL[unit][0], ok1=err <= ABS_TOL[unit][1])
    else:
        err = abs(p["value"] - truth) / max(abs(truth), EPS[unit])
        rec.update(err=err, err_type="rel", ok5=err <= 0.05, ok1=err <= 0.01)
    return rec


def score_design(problem: dict, response: str) -> dict:
    value = design.parse_value(response, problem["unit"])
    rec = {"id": problem["id"], "category": "design", "family": problem["family"], "task": "design",
           "valid": value is not None, "strict": bool(re.fullmatch(r"\s*\w+\s*=\s*\S+(?:\s+\S+)?\s*\.?\s*", response)),
           "ok5": False, "constraint_ok": False, "err": None, "err_type": "design_pct", "pred": value}
    if value is None:
        return rec
    ev = design.evaluate(problem, value)
    rec.update(ok5=bool(ev["success"]), constraint_ok=bool(ev["constraint_ok"]), err=ev.get("error_pct"),
               actual=ev.get("actual"))
    if "error" in ev:
        rec["note"] = ev["error"]
    return rec


def score(problem: dict, response: str, *, truncated: bool = False, transport_error: bool = False) -> dict:
    """Score one reply. A truncated reply has no final answer; a transport error is not the model's fault.

    Both are recorded as invalid and flagged, so the summary can show them apart from wrong answers.
    """
    if truncated or transport_error:
        response = ""
    rec = score_design(problem, response) if problem["task"] == "design" else score_predict(problem, response)
    rec["truncated"], rec["transport_error"] = truncated, transport_error
    return rec


def _rate(num: int, den: int) -> str:
    return f"{num}/{den}"


def summarise(records: list[dict]) -> dict:
    """Overall and per-category metrics, always with raw counts beside rates."""
    def block(rs: list[dict]) -> dict:
        n = len(rs)
        rel = [r["err"] for r in rs if r["err_type"] == "rel" and r["err"] is not None]
        absd = [r["err"] for r in rs if r["err_type"] == "abs" and r["err"] is not None]
        des = [r for r in rs if r["task"] == "design"]
        out = {"n": n, "truncated": sum(r.get("truncated", False) for r in rs),
               "transport_errors": sum(r.get("transport_error", False) for r in rs),
               "valid": sum(r["valid"] for r in rs), "strict_format": sum(r["strict"] for r in rs),
               "correct_within_tolerance": sum(r["ok5"] for r in rs),
               "correct_tight": sum(r.get("ok1", False) for r in rs),
               "median_rel_error": statistics.median(rel) if rel else None,
               "median_abs_error_db_deg": statistics.median(absd) if absd else None}
        toks = [r["completion_tokens"] for r in rs if r.get("completion_tokens")]
        # tokens spent per answer: a tuned model that answers briefly is a format change, not physics
        out["median_completion_tokens"] = statistics.median(toks) if toks else None
        if des:
            out["design_success"] = sum(r["ok5"] for r in des)
            out["design_constraint_ok"] = sum(r["constraint_ok"] for r in des)
            out["design_n"] = len(des)
        return out

    cats = sorted({r["category"] for r in records})
    return {"overall": block(records), "by_category": {c: block([r for r in records if r["category"] == c]) for c in cats}}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value on the discordant pairs (b: A only, c: B only). A guide, not a verdict."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(a: list[dict], b: list[dict]) -> dict:
    """Paired comparison on the same problems: counts of who got each right, per category."""
    ia, ib = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    ids = sorted(set(ia) & set(ib))
    missing = sorted(set(ia) ^ set(ib))

    def tally(subset: list[str]) -> dict:
        both = sum(ia[i]["ok5"] and ib[i]["ok5"] for i in subset)
        only_a = sum(ia[i]["ok5"] and not ib[i]["ok5"] for i in subset)
        only_b = sum(ib[i]["ok5"] and not ia[i]["ok5"] for i in subset)
        neither = len(subset) - both - only_a - only_b
        # the failure the plan warns about: B answers in the requested form where A did not, yet B is wrong where A was right
        fmt_up_num_down = sum(ib[i]["strict"] and not ia[i]["strict"] and ia[i]["ok5"] and not ib[i]["ok5"]
                              for i in subset)
        return {"n": len(subset), "both_correct": both, "only_A_correct": only_a, "only_B_correct": only_b,
                "neither": neither, "mcnemar_exact_p": round(mcnemar_exact(only_a, only_b), 4),
                "A_strict_format": sum(ia[i]["strict"] for i in subset),
                "B_strict_format": sum(ib[i]["strict"] for i in subset),
                "B_better_format_but_wrong_where_A_right": fmt_up_num_down}

    cats = sorted({ia[i]["category"] for i in ids})
    return {"unpaired_ids": missing, "overall": tally(ids),
            "by_category": {c: tally([i for i in ids if ia[i]["category"] == c]) for c in cats}}
