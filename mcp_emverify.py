"""Launcher for this repo's own validated full-wave server (the `emverify_mcp` package).

It runs under the agent's Python, and starts the simulation under a DIFFERENT Python: the one
with the openEMS bindings built against it (EMVERIFY_PYTHON, default ~/mcp-openems/.venv).
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any, Iterable

from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

REPO = pathlib.Path(__file__).resolve().parent
DEFAULT_PREFIX = "emverify"
# A full-wave run takes tens of seconds, so the startup timeout is not the constraint; the
# server itself bounds each run.
DEFAULT_STARTUP_TIMEOUT = 30


def emverify_mcp_client(
    *,
    solver_python: str | None = None,
    allowed_tools: Iterable[str] | None = None,
    startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
    prefix: str | None = DEFAULT_PREFIX,
    python: str | None = None,
) -> MCPClient:
    """An MCPClient that launches `python -m emverify_mcp` over stdio."""
    env = {"PYTHONPATH": str(REPO)}
    if solver_python:
        env["EMVERIFY_PYTHON"] = str(pathlib.Path(solver_python).expanduser())

    def transport():
        return stdio_client(StdioServerParameters(
            command=python or sys.executable, args=["-m", "emverify_mcp"], env=env, cwd=str(REPO)))

    kwargs: dict[str, Any] = {"transport_callable": transport, "startup_timeout": startup_timeout}
    if prefix:
        kwargs["prefix"] = prefix
    if allowed_tools:
        kwargs["tool_filters"] = {"allowed": list(allowed_tools)}
    return MCPClient(**kwargs)
