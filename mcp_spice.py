"""Launcher for this repo's own ngspice MCP server (the `ngspice_mcp` package).

Unlike the KiCad server this one is not third-party: it lives in this repo, runs as
a separate stdio subprocess of the same Python, and takes its filesystem boundary
from the roots passed here. The roots matter. The server reads netlists and writes
KiCad exports only beneath them, and `.include` is honoured only for files beneath
them, so they play the role RIPPLE_AGENT_ROOT plays for the agent's own file tools.
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Any, Iterable

from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

REPO = pathlib.Path(__file__).resolve().parent
DEFAULT_PREFIX = "spice"
# The server starts in well under a second; this only has to outlast a cold import.
DEFAULT_STARTUP_TIMEOUT = 30


class SpiceServerError(RuntimeError):
    """The configured roots or binary are not usable."""


def spice_server_env(roots: Iterable[str | pathlib.Path], ngspice: str | None = None) -> dict[str, str]:
    """Environment for the server process; refuses missing roots rather than guessing."""
    resolved = []
    for r in roots:
        p = pathlib.Path(r).expanduser().resolve()
        if not p.is_dir():
            raise SpiceServerError(f"spice root is not a directory: {p}")
        resolved.append(str(p))
    if not resolved:
        raise SpiceServerError("at least one spice root is required")
    env = {"NGSPICE_MCP_ROOTS": os.pathsep.join(resolved), "PYTHONPATH": str(REPO)}
    if ngspice:
        if not pathlib.Path(ngspice).expanduser().is_file():
            raise SpiceServerError(f"no such ngspice binary: {ngspice}")
        env["NGSPICE_BIN"] = str(pathlib.Path(ngspice).expanduser())
    return env


def spice_mcp_client(
    roots: Iterable[str | pathlib.Path],
    *,
    ngspice: str | None = None,
    allowed_tools: Iterable[str] | None = None,
    startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
    prefix: str | None = DEFAULT_PREFIX,
    python: str | None = None,
) -> MCPClient:
    """An MCPClient that launches `python -m ngspice_mcp` over stdio."""
    env = spice_server_env(roots, ngspice)

    def transport():
        return stdio_client(
            StdioServerParameters(
                command=python or sys.executable,
                args=["-m", "ngspice_mcp"],
                env=env,
                cwd=str(REPO),
            )
        )

    kwargs: dict[str, Any] = {"transport_callable": transport, "startup_timeout": startup_timeout}
    if prefix:
        kwargs["prefix"] = prefix
    if allowed_tools:
        kwargs["tool_filters"] = {"allowed": list(allowed_tools)}
    return MCPClient(**kwargs)
