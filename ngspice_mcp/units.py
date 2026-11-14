"""Unit-aware arithmetic, so a model does not have to track SI prefixes in its head.

Motivating failure: handed a shunt voltage of 1.00002 mV, a model divided it by 1 mOhm
and reported "1.00002 mA". The simulator's number was right and the model's label was
wrong by a factor of 1000. Dimensional analysis makes that class of error impossible:
volts divided by ohms is amperes whatever prefixes were typed, and adding volts to
amperes is refused rather than computed.

Not eval: expressions are parsed by a small recursive-descent parser that only knows
numbers, quantities, + - * / **, unary signs and parentheses.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .core import SpiceError


class UnitError(SpiceError):
    """A malformed expression or a dimensionally inconsistent one."""


# Exponents of (kg, m, s, A), enough for every electrical quantity used here.
Dims = tuple[int, int, int, int]
_NONE: Dims = (0, 0, 0, 0)

UNITS: dict[str, Dims] = {
    "V": (1, 2, -3, -1), "A": (0, 0, 0, 1), "ohm": (1, 2, -3, -2), "W": (1, 2, -3, 0),
    "F": (-1, -2, 4, 2), "H": (1, 2, -2, -2), "Hz": (0, 0, -1, 0), "s": (0, 0, 1, 0),
    "S": (-1, -2, 3, 2),
}
# Names tried, in order, when turning a result's dimensions back into a unit.
_DISPLAY_ORDER = ["V", "A", "ohm", "W", "F", "H", "Hz", "s", "S"]

PREFIXES = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "": 1.0,
            "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
_NUMBER = r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
# prefix + unit, longest unit names first so "ohm" is not read as "o" "hm"
_UNIT_WORDS = sorted(list(UNITS) + ["Ω", "Ohm", "OHM", "ohms"], key=len, reverse=True)
_QUANTITY = re.compile(
    rf"(?P<num>{_NUMBER})\s*(?P<prefix>meg|[pnuµmkMGT])?(?P<unit>{'|'.join(map(re.escape, _UNIT_WORDS))})(?![A-Za-z0-9_])"
)
_PLAIN = re.compile(_NUMBER)


@dataclass(frozen=True)
class Quantity:
    value: float
    dims: Dims = _NONE

    def __add__(self, o): return Quantity(self.value + self._same(o).value, self.dims)
    def __sub__(self, o): return Quantity(self.value - self._same(o).value, self.dims)
    def __mul__(self, o): return Quantity(self.value * o.value, tuple(a + b for a, b in zip(self.dims, o.dims)))

    def __truediv__(self, o):
        if o.value == 0:
            raise UnitError("division by zero")
        return Quantity(self.value / o.value, tuple(a - b for a, b in zip(self.dims, o.dims)))

    def __neg__(self): return Quantity(-self.value, self.dims)

    def power(self, p: float) -> "Quantity":
        dims = [d * p for d in self.dims]
        if any(abs(d - round(d)) > 1e-9 for d in dims):
            raise UnitError(f"exponent {p} would give a fractional unit")
        return Quantity(self.value ** p, tuple(int(round(d)) for d in dims))

    def _same(self, o: "Quantity") -> "Quantity":
        if self.dims != o.dims:
            raise UnitError(f"cannot combine {unit_name(self.dims)} with {unit_name(o.dims)}")
        return o


def unit_name(dims: Dims) -> str:
    if dims == _NONE:
        return "dimensionless"
    for name in _DISPLAY_ORDER:
        if UNITS[name] == dims:
            return name
    parts = [f"{b}^{e}" if e != 1 else b for b, e in zip(("kg", "m", "s", "A"), dims) if e]
    return "*".join(parts)


_VECTOR = r"[vViI]\([A-Za-z0-9_.:#$/+\-,]+\)"
_TOKEN = re.compile(
    rf"\s*(?:(?P<vec>{_VECTOR})|(?P<q>{_QUANTITY.pattern})|(?P<n>{_NUMBER})|(?P<op>\*\*|[-+*/()]))")


def _tokens(expr: str, resolve=None) -> list[tuple[str, object]]:
    out, pos = [], 0
    expr = expr.strip()
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m:
            raise UnitError(f"cannot read {expr[pos:pos + 12]!r}: use numbers with units such as "
                            "1.5 mV, 10 kohm, 4.7 uF, or plain numbers")
        if m.group("vec"):
            if resolve is None:
                raise UnitError(f"{m.group('vec')} needs a result_id to read its value from")
            name = m.group("vec").lower()
            out.append(("q", Quantity(float(resolve(name)), UNITS["V"] if name[0] == "v" else UNITS["A"])))
        elif m.group("q"):
            q = _QUANTITY.fullmatch(m.group("q").strip())
            prefix = {"meg": "M"}.get(q.group("prefix") or "", q.group("prefix") or "")
            unit = "ohm" if q.group("unit") in ("Ω", "Ohm", "OHM", "ohms") else q.group("unit")
            out.append(("q", Quantity(float(q.group("num")) * PREFIXES[prefix], UNITS[unit])))
        elif m.group("n"):
            out.append(("q", Quantity(float(m.group("n")))))
        else:
            out.append(("op", m.group("op")))
        pos = m.end()
    if not out:
        raise UnitError("empty expression")
    return out


class _Parser:
    def __init__(self, toks): self.t, self.i = toks, 0

    def peek(self): return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def parse(self) -> Quantity:
        v = self.expr()
        if self.i != len(self.t):
            raise UnitError("unexpected text after the expression")
        return v

    def expr(self):
        v = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            r = self.term()
            v = v + r if op == "+" else v - r
        return v

    def term(self):
        v = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            r = self.unary()
            v = v * r if op == "*" else v / r
        return v

    def unary(self):
        if self.peek() in (("op", "-"), ("op", "+")):
            op = self.take()[1]
            v = self.unary()
            return -v if op == "-" else v
        return self.power()

    def power(self):
        base = self.atom()
        if self.peek() == ("op", "**"):
            self.take()
            sign = -1 if self.peek() == ("op", "-") and self.take() else 1
            kind, e = self.take()
            if kind != "q" or e.dims != _NONE:
                raise UnitError("an exponent must be a plain number")
            return base.power(sign * e.value)
        return base

    def atom(self):
        kind, val = self.take()
        if kind == "q":
            return val
        if (kind, val) == ("op", "("):
            v = self.expr()
            if self.take() != ("op", ")"):
                raise UnitError("missing closing parenthesis")
            return v
        raise UnitError("expected a number or a quantity with a unit")


def si_format(value: float, dims: Dims) -> str:
    """6 significant figures with the SI prefix that keeps the mantissa in [1, 1000)."""
    unit = "" if dims == _NONE else unit_name(dims)
    if value == 0 or not math.isfinite(value):
        return f"{value:g} {unit}".strip()
    if "^" in unit or "*" in unit:  # a composite unit has no SI prefix of its own
        return f"{value:.6g} {unit}"
    exp = max(-12, min(12, 3 * math.floor(math.log10(abs(value)) / 3)))
    prefix = {-12: "p", -9: "n", -6: "u", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T"}[exp]
    return f"{value / 10 ** exp:.6g} {prefix}{unit}".strip()


def compute(expression: str, resolve=None) -> dict:
    """Evaluate a unit-aware expression. Quantities are a number plus a unit, e.g. 1.00002 mV."""
    q = _Parser(_tokens(expression, resolve)).parse()
    return {"expression": expression.strip(), "value_si": q.value, "unit": unit_name(q.dims),
            "display": si_format(q.value, q.dims)}
