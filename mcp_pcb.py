"""Launcher for the third-party mcp-pcb-emcopilot server (AGPL-3.0), with a curated tool profile.

Provenance, stated plainly: the author's repository (RFingAdam/mcp-pcb-emcopilot) is not public.
The code used here is a public copy under a different account (Yverr0y/mcp-pcb-emcopilot) at
b3f1895, whose origin cannot be verified from git alone, plus ONE local commit that replaces an
exploitable eval() in the Gerber parser (see README). Install it outside this repo:

    git clone https://github.com/Yverr0y/mcp-pcb-emcopilot ~/mcp-pcb-emcopilot
    git -C ~/mcp-pcb-emcopilot checkout b3f1895      # then apply the eval patch described in the README
    cd ~/mcp-pcb-emcopilot && uv venv --python 3.14 .venv && \\
        uv pip install --python .venv/bin/python -e ".[sparams,ground,sexpdata]"

Python 3.14 matters: its default tar extraction filter blocks symlink and path-escape members in
the ODB++ archive reader, where 3.12 and 3.13 would extract them.

The server exposes 131 tools (about 20k tokens per turn), far too many for a 65B model, so only the
`core` profile is attached by default. It is read-only analysis plus closed-form calculators, and
it deliberately leaves out three groups:

  * tools that write files to a path the model chooses (report writers, PNG exporters);
  * tools whose output is addressed to the model as instructions to act on
    (pcb_lookup_limit_live returns a "deferred next-action Claude should execute", and
    pcb_suggest_next_actions and pcb_attach_external_result are the same orchestration pattern),
    which this agent treats as untrusted data;
  * the openEMS bridge, which this repo replaces with the validated emverify server.

Like the KiCad server, it reads any file path it is given, so point it only at boards you trust.
"""
from __future__ import annotations

import pathlib
from typing import Any, Iterable

from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

AUDITED_COMMIT = "b3f1895"
DEFAULT_STARTUP_TIMEOUT = 90

# Read-only analysis and calculators useful for reviewing a KiCad board. Verified against the
# server's real tool list by the tests, so an upstream rename fails loudly instead of silently
# shrinking the manifest.
CORE_TOOLS = (
    "pcb_parse_layout", "pcb_get_stackup", "pcb_get_nets", "pcb_classify_nets", "pcb_detect_interfaces",
    "pcb_calc_microstrip_impedance", "pcb_calc_stripline_impedance", "pcb_calc_differential_impedance",
    "pcb_calc_trace_width", "pcb_analyze_return_paths", "pcb_find_split_crossings",
    "pcb_analyze_decoupling", "pcb_analyze_pdn", "pcb_analyze_crosstalk", "pcb_analyze_emi_risk",
    "pcb_analyze_grounding", "pcb_analyze_current_loop", "pcb_analyze_differential_pair",
)
PROFILES = ("core", "all")


class PcbServerError(RuntimeError):
    """The server is not installed where it was said to be."""


def pcb_executable(server_dir: str | pathlib.Path) -> pathlib.Path:
    d = pathlib.Path(server_dir).expanduser().resolve()
    if not d.is_dir():
        raise PcbServerError(f"no such directory: {d}")
    exe = d / ".venv" / "bin" / "mcp-pcb-emcopilot"
    if not exe.is_file():
        raise PcbServerError(
            f"{exe} not found. Install the server as described at the top of mcp_pcb.py "
            f"(commit {AUDITED_COMMIT} plus the local eval patch) into {d}.")
    return exe


def pcb_mcp_client(
    server_dir: str | pathlib.Path,
    *,
    profile: str = "core",
    allowed_tools: Iterable[str] | None = None,
    startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
) -> MCPClient:
    """An MCPClient for the PCB server. `allowed_tools` overrides the profile when given."""
    if profile not in PROFILES:
        raise PcbServerError(f"unknown profile {profile!r}; expected one of {', '.join(PROFILES)}")
    exe = pcb_executable(server_dir)

    def transport():
        return stdio_client(StdioServerParameters(command=str(exe), args=[], cwd=str(exe.parent.parent.parent)))

    kwargs: dict[str, Any] = {"transport_callable": transport, "startup_timeout": startup_timeout}
    names = list(allowed_tools) if allowed_tools else (list(CORE_TOOLS) if profile == "core" else None)
    if names:
        kwargs["tool_filters"] = {"allowed": names}
    return MCPClient(**kwargs)
