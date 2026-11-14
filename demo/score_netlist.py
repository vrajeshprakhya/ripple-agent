#!/usr/bin/env python3
"""Score a wired schematic against the intended 1S protection topology.

Reads KiCad's own exported netlist, never the model's report of what it did, so
a model claiming a connection it did not make cannot score.

The intended circuit has five parts: BAT1 (cell), Q1 (N-FET), C1, T1 (thermistor),
R1 (shunt). R1, C1 and T1 are two-terminal and symmetric, so the checks are
phrased in terms of which nets their two pins straddle, not which pin is which.

    VBAT  = BAT1/+  Q1/D  one pin of C1  one pin of T1
    BATN  = BAT1/-  other pin of C1  other pin of T1  one pin of R1
    GND   = other pin of R1
    LOAD  = Q1/S
    GATE  = Q1/G

Usage: score_netlist.py SCHEMATIC.kicad_sch   (exit 0 only if every check passes)
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

TOKEN = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+')


def parse_sexp(text: str):
    """Minimal s-expression reader; atoms and quoted strings both come back as str."""
    stack: list[list] = [[]]
    for tok in TOKEN.findall(text):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok[1:-1] if tok.startswith('"') else tok)
    return stack[0][0]


def netlist(schematic: Path) -> dict[str, set[str]]:
    """Map net name -> set of 'REF/PIN', from KiCad's exported netlist."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "n.net"
        r = subprocess.run(
            ["kicad-cli", "sch", "export", "netlist", "--format", "kicadsexpr",
             "--output", str(out), str(schematic)],
            capture_output=True, text=True,
        )
        if not out.exists():
            raise RuntimeError(f"kicad-cli failed: {(r.stderr or r.stdout).strip()[:200]}")
        tree = parse_sexp(out.read_text())

    nets: dict[str, set[str]] = {}
    for section in tree:
        if isinstance(section, list) and section and section[0] == "nets":
            for net in section[1:]:
                if not (isinstance(net, list) and net and net[0] == "net"):
                    continue
                name, pins = "", set()
                for part in net[1:]:
                    if isinstance(part, list) and part and part[0] == "name":
                        name = part[1].lstrip("/")
                    elif isinstance(part, list) and part and part[0] == "node":
                        ref = pin = ""
                        for f in part[1:]:
                            if isinstance(f, list) and f and f[0] == "ref":
                                ref = f[1]
                            elif isinstance(f, list) and f and f[0] == "pin":
                                pin = f[1]
                        pins.add(f"{ref}/{pin}")
                nets[name] = pins
    return nets


def score(nets: dict[str, set[str]]) -> tuple[dict[str, bool], dict[str, bool]]:
    # KiCad lists every unconnected pin as its own net called "unconnected-(...)".
    # Those are the ABSENCE of a connection, so they must not count as a net a pin
    # "is on", or an unwired FET source would pass "on its own net".
    where = {
        pin: name
        for name, pins in nets.items()
        if not name.startswith("unconnected-")
        for pin in pins
    }

    def n(pin: str) -> str | None:
        return where.get(pin)

    vbat, batn = n("BAT1/1"), n("BAT1/2")  # Battery_Cell pin 1 is +, pin 2 is -
    cell_ok = vbat is not None and batn is not None and vbat != batn
    used = {vbat, batn}

    def own_net(pin: str, *also_not: str | None) -> bool:
        net = n(pin)
        return net is not None and net not in used and net not in also_not

    # Every check that mentions the cell nets is gated on cell_ok: with the cell
    # unwired both nets are None, and None == None would otherwise pass.
    structural = {
        "cell terminals on different nets": cell_ok,
        "Q1 drain joins cell +": cell_ok and n("Q1/D") == vbat,
        "C1 straddles the cell": cell_ok and {n("C1/1"), n("C1/2")} == {vbat, batn},
        "T1 straddles the cell": cell_ok and {n("T1/1"), n("T1/2")} == {vbat, batn},
        "R1 one pin on cell -": cell_ok and batn in {n("R1/1"), n("R1/2")},
        "R1 other pin on its own net": cell_ok
        and sum(1 for p in ("R1/1", "R1/2") if own_net(p)) == 1,
        "Q1 source on its own net": cell_ok and own_net("Q1/S"),
        "Q1 gate on its own net": cell_ok and own_net("Q1/G", n("Q1/S")),
        # The load sits between the FET source and the shunt's free pin, so those
        # must be different nets. Merging them leaves the load with one terminal.
        "load terminals on different nets": cell_ok and (
            len(free := [n(p) for p in ("R1/1", "R1/2") if own_net(p)]) == 1
            and n("Q1/S") is not None
            and n("Q1/S") != free[0]
        ),
    }

    named = {
        "net VBAT holds cell +": "BAT1/1" in nets.get("VBAT", set()),
        "net BATN holds cell -": "BAT1/2" in nets.get("BATN", set()),
        "net LOAD holds Q1/S": "Q1/S" in nets.get("LOAD", set()),
        "net GATE holds Q1/G": "Q1/G" in nets.get("GATE", set()),
        "net GND holds an R1 pin": bool(nets.get("GND", set()) & {"R1/1", "R1/2"}),
    }
    return structural, named


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        nets = netlist(Path(argv[1]))
    except RuntimeError as e:
        print(f"SCORE error: {e}")
        return 2
    structural, named = score(nets)
    s, nm = sum(structural.values()), sum(named.values())
    print(f"SCORE structural={s}/{len(structural)} named={nm}/{len(named)}")
    for label, group in (("structural", structural), ("named", named)):
        for k, v in group.items():
            if not v:
                print(f"  MISSING ({label}): {k}")
    return 0 if s == len(structural) and nm == len(named) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
