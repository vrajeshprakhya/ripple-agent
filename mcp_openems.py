"""Launcher for the third-party mcp-openems server (RFingAdam/mcp-openems, AGPL-3.0).

Install it OUTSIDE this repo, in its own venv, from the commit that was audited:

    git clone https://github.com/RFingAdam/mcp-openems ~/mcp-openems
    git -C ~/mcp-openems checkout 46ceb5d
    cd ~/mcp-openems && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .

Nothing from that project is copied into this MIT repo; it runs as a separate stdio
process. Audit of 46ceb5d: no subprocess calls, no network calls, no install-time hooks,
runtime dependencies `mcp` and `numpy` only. It writes no files and takes no paths, so
unlike the SPICE and KiCad servers there are no roots to configure. The script templates
it EMITS do create and delete a fixed directory under the system temp dir when someone
runs them, which only matters once openEMS is installed.

Without openEMS built from source the server still does analytical design and script
generation, and says so through openems_check_installation.
"""
from __future__ import annotations

import pathlib
from typing import Any, Iterable

from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

AUDITED_COMMIT = "46ceb5d"
# The server's tool names already begin with "openems_", so no extra prefix is added.
DEFAULT_STARTUP_TIMEOUT = 60


class OpenemsServerError(RuntimeError):
    """The server is not installed where it was said to be."""


def openems_executable(server_dir: str | pathlib.Path) -> pathlib.Path:
    """Return the server's console script, or explain how to install it."""
    d = pathlib.Path(server_dir).expanduser().resolve()
    if not d.is_dir():
        raise OpenemsServerError(f"no such directory: {d}")
    exe = d / ".venv" / "bin" / "mcp-openems"
    if not exe.is_file():
        raise OpenemsServerError(
            f"{exe} not found. Clone https://github.com/RFingAdam/mcp-openems into {d}, check out "
            f"{AUDITED_COMMIT}, then: uv venv --python 3.12 .venv && uv pip install --python "
            ".venv/bin/python -e ."
        )
    return exe


def openems_mcp_client(
    server_dir: str | pathlib.Path,
    *,
    allowed_tools: Iterable[str] | None = None,
    startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
) -> MCPClient:
    """An MCPClient that launches the mcp-openems console script over stdio."""
    exe = openems_executable(server_dir)

    def transport():
        return stdio_client(StdioServerParameters(command=str(exe), args=[], cwd=str(exe.parent.parent.parent)))

    kwargs: dict[str, Any] = {"transport_callable": transport, "startup_timeout": startup_timeout}
    if allowed_tools:
        kwargs["tool_filters"] = {"allowed": list(allowed_tools)}
    return MCPClient(**kwargs)
