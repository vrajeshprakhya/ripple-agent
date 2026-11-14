"""ngspice runner, netlist sanitizer and result parser behind the ngspice MCP server.

A SPICE simulator driven by a language model has one hazard that matters more
than the rest: ngspice's `.control` language can run shell commands, read and
write arbitrary files and load shared objects, so a netlist written by a model
(or by text the model has read) is effectively code. The design here is that the
model never supplies a control block at all.

  * The model supplies only a CIRCUIT. Every dot-directive in it must be on a
    short allowlist; analysis cards are stripped, and `.control` is rejected.
  * This module writes the `.control` block itself, from validated numbers and
    names, so the only commands that ever reach ngspice are ones built here.
  * `.include` and `.lib` are allowed only for files under configured roots.
  * ngspice runs in a throwaway directory, with a scrubbed environment, CPU,
    memory and file-size limits and a wall-clock timeout.

This narrows the attack surface; it does not make a hostile netlist safe to run
as a different user. See the README for the residual risk.
"""
from __future__ import annotations

import math
import os
import re
import resource
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_NGSPICE = Path.home() / "ngspice-install" / "bin" / "ngspice"
MAX_NETLIST_BYTES = 1_000_000
RUN_TIMEOUT_S = 60
MAX_RAW_BYTES = 200 * 1024 * 1024

ALLOWED_DIRECTIVES = frozenset(
    {"model", "subckt", "ends", "param", "global", "ic", "nodeset",
     "options", "option", "temp", "save", "func", "include", "lib",
     "title"}  # KiCad's export opens with ".title ..."; it only names the deck
)
# Stripped rather than rejected: the server chooses the analysis, so a netlist
# exported from KiCad that carries its own `.op` or `.tran` still works.
ANALYSIS_DIRECTIVES = frozenset({"op", "ac", "tran", "dc", "noise", "tf", "pz", "sens", "disto"})
INCLUDE_SUFFIXES = frozenset({".lib", ".mod", ".cir", ".sub", ".inc", ".sp", ".sp3", ".lb", ".txt"})

# Characters ngspice and Python disagree about as line breaks, plus controls.
# Python splits on several that ngspice does not, which is how a directive can be
# hidden from a checker, so none of them are accepted at all.
_BAD_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x85  ]")
_DIRECTIVE = re.compile(r"^\.(\w+)")
_ELEMENT_LETTERS = set("RCLVIDQMJKBEFGHSTUWXYZ")
_ELEMENT_CARD = re.compile(r"^[A-Za-z][A-Za-z0-9_$#.:\\-]*\s+\S+\s+\S+")

# A SPICE number with an optional scale suffix and trailing unit letters
# ("10k", "159.15n", "1e-3", "2meg", "10uF"). Letters only, so nothing that could
# carry a separator or a command.
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?[A-Za-z]*$")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.]*$")
_VECTOR = re.compile(r"^[A-Za-z][A-Za-z0-9_.:#$]*(\([A-Za-z0-9_.:#$,+\-/]*\))?$")


_EXPECTED_PLOT = {"op": "operating point", "dc": "dc transfer", "ac": "ac analysis", "tran": "transient"}
_PROBLEM = re.compile(
    r"\b(error|fatal)\b|ignored!|not a valid|could not find|unknown (model|subckt|device)|no such|"
    r"too few|too many", re.I)


class SpiceError(RuntimeError):
    """Anything the caller should see as a failed request, with a usable message."""


# --- Validation -------------------------------------------------------------
def spice_number(value: str, what: str) -> str:
    v = str(value).strip()
    if not _NUMBER.match(v):
        raise SpiceError(f"{what} must be a SPICE number like 10k, 159.15n or 1e-3, got {value!r}")
    return v


def spice_name(value: str, what: str) -> str:
    v = str(value).strip()
    if not _NAME.match(v):
        raise SpiceError(f"{what} must be a plain name like V1, got {value!r}")
    return v


def spice_vector(value: str) -> str:
    v = str(value).strip().lower()
    if not _VECTOR.match(v):
        raise SpiceError(f"vector must look like v(out) or i(v1), got {value!r}")
    return v


def _resolved_within(path: Path, roots: list[Path]) -> bool:
    real = path.resolve()
    return any(real == r or r in real.parents for r in roots)


def read_netlist_file(path: str, roots: list[Path]) -> str:
    """Read a netlist, refusing anything outside the configured roots (symlinks included)."""
    p = Path(path).expanduser()
    if not _resolved_within(p, [r.resolve() for r in roots]):
        raise SpiceError(f"{path} is outside the allowed roots: {', '.join(str(r) for r in roots)}")
    if not p.is_file():
        raise SpiceError(f"not a file: {path}")
    if p.stat().st_size > MAX_NETLIST_BYTES:
        raise SpiceError(f"netlist larger than {MAX_NETLIST_BYTES} bytes")
    return p.read_text(errors="replace")


# --- Sanitizer ---------------------------------------------------------------
def _looks_like_element(line: str) -> bool:
    s = line.strip()
    return bool(s) and s[0].upper() in _ELEMENT_LETTERS and bool(_ELEMENT_CARD.match(s))


def sanitize_netlist(
    text: str, *, roots: list[Path] | None = None, base_dir: Path | None = None
) -> tuple[str, list[str]]:
    """Return (circuit text safe to hand to ngspice, notes about what was changed).

    Raises SpiceError for anything dangerous. The returned text has no title
    line problem (the first line is always made a comment), no analysis cards
    and no `.end`; the caller adds its own control block.
    """
    roots = [r.resolve() for r in (roots or [])]
    if len(text.encode("utf-8", "replace")) > MAX_NETLIST_BYTES:
        raise SpiceError(f"netlist larger than {MAX_NETLIST_BYTES} bytes")
    if _BAD_CHARS.search(text):
        raise SpiceError("netlist contains control characters or unusual line separators")
    if "`" in text:
        raise SpiceError("backticks are not allowed in a netlist")

    raw = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # ngspice always treats line 1 as a title and ignores it. Make it explicit,
    # so a netlist that starts straight on a component does not lose that part.
    while raw and not raw[0].strip():
        raw.pop(0)
    notes: list[str] = []
    if raw:
        first = raw[0].strip()
        if first.startswith("*") or first.startswith("."):
            pass
        elif _looks_like_element(first):
            notes.append("first line looked like a component, not a title; kept as a component")
            raw.insert(0, "* circuit")
        else:
            raw[0] = "* " + first
    # Join continuation lines so a directive split across lines is still seen whole.
    lines: list[str] = []
    for ln in raw:
        if ln.lstrip().startswith("+") and lines:
            lines[-1] += " " + ln.lstrip()[1:].strip()
        else:
            lines.append(ln)

    out: list[str] = []
    for n, ln in enumerate(lines, 1):
        s = ln.strip()
        if not s or s.startswith("*"):
            out.append(ln)
            continue
        m = _DIRECTIVE.match(s)
        if not m:
            out.append(ln)
            continue
        name = m.group(1).lower()
        if name == "end":
            continue
        if name in ANALYSIS_DIRECTIVES:
            notes.append(f"line {n}: stripped analysis card .{name} (the tools choose the analysis)")
            continue
        if name not in ALLOWED_DIRECTIVES:
            raise SpiceError(f"line {n}: directive .{name} is not allowed")
        if name in ("include", "lib"):
            out.append(_checked_include(s, name, n, roots, base_dir))
            continue
        out.append(ln)
    return "\n".join(out), notes


def _checked_include(line: str, name: str, n: int, roots: list[Path], base_dir: Path | None) -> str:
    rest = line[len(name) + 1:].strip()
    if not rest:
        raise SpiceError(f"line {n}: .{name} needs a file")
    if rest[0] in "\"'":
        quote = rest[0]
        end = rest.find(quote, 1)
        if end < 0:
            raise SpiceError(f"line {n}: unterminated quote in .{name}")
        target, tail = rest[1:end], rest[end + 1:].strip()
    else:
        parts = rest.split(None, 1)
        target, tail = parts[0], (parts[1] if len(parts) > 1 else "")
    if not roots:
        raise SpiceError(f"line {n}: .{name} is not allowed because no include roots are configured")
    p = Path(target).expanduser()
    if not p.is_absolute():
        p = (base_dir or roots[0]) / p
    if p.suffix.lower() not in INCLUDE_SUFFIXES:
        raise SpiceError(f"line {n}: .{name} of a {p.suffix or 'suffixless'} file is not allowed")
    if not _resolved_within(p, roots):
        raise SpiceError(f"line {n}: .{name} {target} is outside the allowed roots")
    if not p.is_file():
        raise SpiceError(f"line {n}: .{name} file does not exist: {target}")
    if tail and not re.match(r"^[A-Za-z0-9_.]+$", tail):
        raise SpiceError(f"line {n}: unexpected text after .{name} file")
    _scan_library(p.resolve(), roots)
    return f'.{name} "{p.resolve()}"' + (f" {tail}" if tail else "")


# A model library may also close a section (.endl) and use conditionals, which the
# top-level circuit has no need for.
LIBRARY_DIRECTIVES = ALLOWED_DIRECTIVES | {"endl", "end", "if", "elseif", "else", "endif"}
MAX_LIBRARY_BYTES = 5_000_000
MAX_INCLUDE_DEPTH = 4


def _scan_library(path: Path, roots: list[Path], depth: int = 0) -> None:
    """Refuse an included file that would run anything ngspice could execute.

    ngspice splices an `.include`d file into the deck, so a `.control` block inside
    it runs exactly as one in the netlist would. Checking only WHERE the file lives
    would leave that open: any file under a root would be trusted with its contents.
    """
    if depth > MAX_INCLUDE_DEPTH:
        raise SpiceError(f"includes nested more than {MAX_INCLUDE_DEPTH} deep at {path.name}")
    if path.stat().st_size > MAX_LIBRARY_BYTES:
        raise SpiceError(f"{path.name} is larger than {MAX_LIBRARY_BYTES} bytes")
    text = path.read_text(errors="replace")
    if _BAD_CHARS.search(text) or "`" in text:
        raise SpiceError(f"{path.name} contains control characters or backticks")
    lines: list[str] = []
    for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if ln.lstrip().startswith("+") and lines:
            lines[-1] += " " + ln.lstrip()[1:].strip()
        else:
            lines.append(ln)
    for n, ln in enumerate(lines, 1):
        s = ln.strip()
        m = _DIRECTIVE.match(s) if s and not s.startswith("*") else None
        if not m:
            continue
        name = m.group(1).lower()
        if name not in LIBRARY_DIRECTIVES:
            raise SpiceError(f"{path.name} line {n}: directive .{name} is not allowed in an included file")
        if name in ("include", "lib"):
            rest = s[len(name) + 1:].strip().strip("\"'")
            first = rest.split()[0] if rest.split() else ""
            # `.lib NAME` inside a library opens a section and names no file.
            if "/" in first or Path(first).suffix.lower() in INCLUDE_SUFFIXES:
                nested = Path(first).expanduser()
                nested = nested if nested.is_absolute() else path.parent / nested
                if not _resolved_within(nested, roots) or not nested.is_file():
                    raise SpiceError(f"{path.name} line {n}: nested .{name} {first} is outside the allowed roots or missing")
                if nested.suffix.lower() not in INCLUDE_SUFFIXES:
                    raise SpiceError(f"{path.name} line {n}: nested .{name} of a {nested.suffix} file is not allowed")
                _scan_library(nested.resolve(), roots, depth + 1)


# --- Control block -------------------------------------------------------------
def control_block(analysis: str, params: dict[str, str], probes: list[str]) -> str:
    """Build the only control commands ngspice ever sees, from validated pieces."""
    if analysis == "op":
        cmd = "op"
    elif analysis == "dc":
        cmd = "dc {source} {start} {stop} {step}".format(
            source=spice_name(params["source"], "source"),
            start=spice_number(params["start"], "start"),
            stop=spice_number(params["stop"], "stop"),
            step=spice_number(params["step"], "step"),
        )
    elif analysis == "ac":
        ppd = int(params["points_per_decade"])
        if not 1 <= ppd <= 1000:
            raise SpiceError("points_per_decade must be 1 to 1000")
        cmd = "ac dec {ppd} {a} {b}".format(
            ppd=ppd,
            a=spice_number(params["start"], "start frequency"),
            b=spice_number(params["stop"], "stop frequency"),
        )
    elif analysis == "tran":
        cmd = "tran {step} {stop}".format(
            step=spice_number(params["step"], "time step"),
            stop=spice_number(params["stop"], "stop time"),
        )
        if params.get("max_step"):
            cmd += " 0 " + spice_number(params["max_step"], "max step")
    else:
        raise SpiceError(f"unknown analysis {analysis!r}")
    what = " ".join(spice_vector(p) for p in probes) if probes else "all"
    return "\n".join([".control", "set filetype=ascii", "set noaskquit", cmd,
                      f"write out.raw {what}", ".endc"])


# --- Running -------------------------------------------------------------------
def ngspice_binary() -> str:
    env = os.environ.get("NGSPICE_BIN")
    if env and Path(env).is_file():
        return env
    if DEFAULT_NGSPICE.is_file():
        return str(DEFAULT_NGSPICE)
    found = shutil.which("ngspice")
    if found:
        return found
    raise SpiceError("ngspice not found: set NGSPICE_BIN or put ngspice on PATH")


def have_ngspice() -> bool:
    try:
        ngspice_binary()
        return True
    except SpiceError:
        return False


def _limits(cpu_s: int):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_RAW_BYTES, MAX_RAW_BYTES))
        resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    return apply


@dataclass
class RunResult:
    analysis: str
    plot: str
    is_complex: bool
    vectors: dict[str, list]
    log: str = ""
    notes: list[str] = field(default_factory=list)


def run_ngspice(circuit: str, analysis: str, params: dict[str, str], probes: list[str],
                *, timeout: int = RUN_TIMEOUT_S) -> RunResult:
    """Run one analysis on an already-sanitized circuit."""
    binary = ngspice_binary()
    ctl = control_block(analysis, params, probes)
    with tempfile.TemporaryDirectory(prefix="ngspice_mcp_") as wd:
        work = Path(wd)
        (work / "circuit.cir").write_text(circuit.rstrip("\n") + "\n" + ctl + "\n.end\n")
        try:
            proc = subprocess.Popen(
                [binary, "-b", "circuit.cir"], cwd=work, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env={"PATH": "/usr/bin:/bin", "HOME": str(work), "LANG": "C"},
                start_new_session=True, preexec_fn=_limits(timeout + 5),
            )
            try:
                log, _ = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                raise SpiceError(f"ngspice did not finish within {timeout}s and was killed")
        finally:
            pass
        raw = work / "out.raw"
        if not raw.is_file():
            tail = "\n".join(log.strip().splitlines()[-12:])
            raise SpiceError("simulation produced no results. ngspice said:\n" + tail)
        plot, is_complex, vectors = parse_ascii_raw(raw.read_text())

    # ngspice reports a malformed component as a WARNING, drops it, and simulates
    # whatever is left. For a tool a model reads, a result for the wrong circuit
    # is worse than an error, so those warnings are failures.
    problems = [ln.strip() for ln in log.splitlines() if _PROBLEM.search(ln)]
    if problems:
        raise SpiceError("ngspice rejected part of the netlist:\n" + "\n".join(problems[:6]))
    # With no usable circuit ngspice can still write a file, holding its built-in
    # constants table (pi, boltz, kelvin ...). Check the plot is the one asked for.
    if not plot.lower().startswith(_EXPECTED_PLOT[analysis]):
        raise SpiceError(f"ngspice did not run the {analysis} analysis (it produced a {plot!r} plot); "
                         "the circuit is probably empty or has no path between its nodes")
    soft = [ln.strip() for ln in log.splitlines() if ln.strip().lower().startswith("warning")]
    return RunResult(analysis, plot, is_complex, vectors, log=log, notes=soft[:5])


# --- Raw file parsing -----------------------------------------------------------
def parse_ascii_raw(text: str) -> tuple[str, bool, dict[str, list]]:
    """Parse ngspice's ASCII rawfile into {vector name: [values]} (complex where flagged)."""
    head, _, body = text.partition("\nValues:")
    if not body:
        raise SpiceError("rawfile has no Values section")
    plot, is_complex, nvars, npoints, names = "", False, 0, 0, []
    in_vars = False
    for ln in head.splitlines():
        if in_vars:
            parts = ln.split()
            if len(parts) >= 2:
                names.append(parts[1].lower())
            continue
        if ln.startswith("Plotname:"):
            plot = ln.split(":", 1)[1].strip()
        elif ln.startswith("Flags:"):
            is_complex = "complex" in ln.lower()
        elif ln.startswith("No. Variables:"):
            nvars = int(ln.split(":")[1])
        elif ln.startswith("No. Points:"):
            npoints = int(ln.split(":")[1])
        elif ln.startswith("Variables:"):
            in_vars = True
    if not nvars or len(names) != nvars:
        raise SpiceError("rawfile header is inconsistent")
    toks = body.split()
    if len(toks) < npoints * (nvars + 1):
        raise SpiceError("rawfile is truncated")

    def conv(tok: str):
        if "," in tok:
            re_, im = tok.split(",")
            return complex(float(re_), float(im))
        return complex(float(tok), 0.0) if is_complex else float(tok)

    vectors: dict[str, list] = {nm: [] for nm in names}
    pos = 0
    for _ in range(npoints):
        pos += 1  # point index
        for nm in names:
            vectors[nm].append(conv(toks[pos]))
            pos += 1
    return plot, is_complex, vectors


# --- Measurements ----------------------------------------------------------------
def db(c: complex | float) -> float:
    return 20 * math.log10(max(abs(c), 1e-300))


def downsample(xs: list, ys: list, n: int) -> list[tuple]:
    if len(xs) <= n:
        return list(zip(xs, ys))
    idx = sorted({round(i * (len(xs) - 1) / (n - 1)) for i in range(n)})
    return [(xs[i], ys[i]) for i in idx]


def bandwidth_3db(freq: list[float], mag_db: list[float]) -> float | None:
    """Frequency where magnitude first falls 3.0103 dB below its starting value."""
    if not mag_db:
        return None
    target = mag_db[0] - 3.0103
    for i in range(1, len(mag_db)):
        if mag_db[i] <= target <= mag_db[i - 1]:
            f0, f1 = math.log10(freq[i - 1]), math.log10(freq[i])
            m0, m1 = mag_db[i - 1], mag_db[i]
            frac = 0.0 if m0 == m1 else (m0 - target) / (m0 - m1)
            return 10 ** (f0 + frac * (f1 - f0))
    return None


def _cross(t: list[float], y: list[float], level: float) -> float | None:
    for i in range(1, len(y)):
        a, b = y[i - 1], y[i]
        if (a - level) * (b - level) <= 0 and a != b:
            return t[i - 1] + (level - a) / (b - a) * (t[i] - t[i - 1])
    return None


def rise_time(t: list[float], y: list[float], lo: float = 0.1, hi: float = 0.9) -> float | None:
    """10 to 90 percent rise time between the first and last sample (for a step response)."""
    y0, y1 = y[0], y[-1]
    if y1 == y0:
        return None
    t_lo, t_hi = _cross(t, y, y0 + lo * (y1 - y0)), _cross(t, y, y0 + hi * (y1 - y0))
    return None if t_lo is None or t_hi is None else t_hi - t_lo


def stats(y: list[float]) -> dict[str, float]:
    n = len(y)
    mean = sum(y) / n
    return {"min": min(y), "max": max(y), "mean": mean, "final": y[-1],
            "rms": math.sqrt(sum(v * v for v in y) / n), "peak_to_peak": max(y) - min(y)}
