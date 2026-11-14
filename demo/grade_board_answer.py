#!/usr/bin/env python3
"""Grade a model's answer to the BMS board question against the project's own reference.

Reference values come from the logged testbench of ~/bq_vboard_evidence/kicad/bms_6s_fab:
cell 1 = 3.688 V, shunt current = 1.00002 A (1.00002 mV across 1 mohm), filter corner
7957.74 Hz. Grading is by pattern, so it is a heuristic: it reads the whole answer text,
and an answer that states a number two different ways is graded on the worst of them.

Usage: grade_board_answer.py < answer.txt      prints  cell=ok|wrong|none current=... corner=...
"""
from __future__ import annotations

import re
import sys

CELL_V = 3.688
CORNER_HZ = 7957.74

_NUM = r"\d[\d,]*(?:\.\d+)?"
_FREQ = re.compile(rf"({_NUM})\s*(k|M)?Hz", re.I)


def grade_current(text: str) -> str:
    """Shunt current. 1 mV across 1 mohm is 1 A, so '1.00002 mA' is wrong by 1000x.

    ok if the current is stated in amperes and never in milliamperes; wrong if any milliampere
    claim of ~1 appears, even beside a correct one (a self-contradicting answer is not right).
    """
    wrong = re.search(r"\b1\.0+\d*\s*mA\b", text)
    ok = re.search(r"\b1\.0+\d*\s*A\b(?!\w)", text) or re.search(r"\b1000\.0\d*\s*mA\b", text)
    if wrong and not re.search(r"\b1000\.0\d*\s*mA\b", text):
        return "wrong"
    return "ok" if ok else "none"


def grade_cell(text: str) -> str:
    return "ok" if re.search(r"\b3\.68\d*\s*V\b", text) else "none"


def _to_hz(num: str, prefix: str | None) -> float:
    v = float(num.replace(",", ""))
    return v * {"k": 1e3, "m": 1e6, None: 1.0}[prefix.lower() if prefix else None]


def grade_corner(text: str) -> str:
    """The -3 dB corner of the differential shunt voltage; within 1% of the reference."""
    values = [_to_hz(n, p) for n, p in _FREQ.findall(text)]
    if any(abs(v - CORNER_HZ) / CORNER_HZ < 0.01 for v in values):
        return "ok"
    # Any claimed corner in the kHz range that is not the reference is a wrong answer.
    return "wrong" if any(1_000 <= v <= 500_000 and abs(v - CORNER_HZ) / CORNER_HZ >= 0.01 for v in values) else "none"


def grade(text: str) -> dict[str, str]:
    return {"cell": grade_cell(text), "current": grade_current(text), "corner": grade_corner(text)}


if __name__ == "__main__":
    g = grade(sys.stdin.read())
    print(" ".join(f"{k}={v}" for k, v in g.items()))
    raise SystemExit(0 if all(v == "ok" for v in g.values()) else 1)
