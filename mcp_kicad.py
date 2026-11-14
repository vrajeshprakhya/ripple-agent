"""Integration with the mixelpixx/KiCAD-MCP-Server MCP server.

The server is a Node process speaking MCP over stdio, which shells out to
KiCad's Python bindings (pcbnew) and optionally its IPC API. It is not pip
installable: clone the `stable` branch, `npm install && npm run build`, and
point `--kicad` at the resulting directory.

Two of its environment variables matter more than the rest, and the defaults
here are deliberately the conservative ends of both:

`KICAD_MCP_OPERATING_MODE` selects `readonly`, `manufacturing` or `write`.
Anything other than readonly lets the model mutate real PCB and schematic
files, so readonly is the default and widening it is an explicit choice.

`KICAD_MCP_TOOLBOXES` selects which tool groups are exposed. The full server
advertises somewhere between 184 and 253 tools depending on which part of its
README you read. That is a poor fit for a 65B model: tool-selection accuracy
falls off well before that count, and every schema is re-sent as input on every
turn. `core` cuts it to roughly 37.
"""
from __future__ import annotations

import pathlib
from typing import Any, Iterable

from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

# KiCad ships pcbnew outside the system site-packages, so the server's Python
# side needs it on PYTHONPATH explicitly.
DEFAULT_KICAD_PYTHONPATH = "/usr/lib/kicad/lib/python3/dist-packages"

OPERATING_MODES = ("readonly", "manufacturing", "write")
DEFAULT_MODE = "readonly"
DEFAULT_TOOLBOXES = "core"
DEFAULT_PREFIX = "kicad"

# The server builds an index and probes KiCad on startup, which is slower than
# the 30s MCPClient default on a cold run.
DEFAULT_STARTUP_TIMEOUT = 90


class KicadServerError(RuntimeError):
    """The configured server directory is not a usable built server."""


def validate_server_dir(server_dir: str | pathlib.Path) -> pathlib.Path:
    """Return the server's entry point, or explain what is missing.

    Checked eagerly because the alternative is an opaque stdio timeout
    ninety seconds later.
    """
    d = pathlib.Path(server_dir).expanduser().resolve()
    if not d.is_dir():
        raise KicadServerError(f"no such directory: {d}")
    entry = d / "dist" / "index.js"
    if not entry.is_file():
        if (d / "package.json").is_file():
            raise KicadServerError(
                f"{d} looks like the repo but is not built: {entry} is missing.\n"
                "Run: npm install && npm run build"
            )
        raise KicadServerError(
            f"{entry} not found. --kicad wants the KiCAD-MCP-Server checkout root."
        )
    return entry


def kicad_server_env(
    *,
    toolboxes: str = DEFAULT_TOOLBOXES,
    mode: str = DEFAULT_MODE,
    pythonpath: str | None = None,
    log_level: str = "info",
    kicad_python: str | None = None,
) -> dict[str, str]:
    """Build the server's environment, refusing an unknown operating mode."""
    if mode not in OPERATING_MODES:
        raise KicadServerError(
            f"unknown operating mode {mode!r}; expected one of {', '.join(OPERATING_MODES)}"
        )
    env = {
        "KICAD_MCP_OPERATING_MODE": mode,
        "KICAD_MCP_TOOLBOXES": toolboxes,
        "KICAD_MCP_LOG_LEVEL": log_level,
        "PYTHONPATH": pythonpath or DEFAULT_KICAD_PYTHONPATH,
    }
    if kicad_python:
        env["KICAD_PYTHON"] = kicad_python
    return env


def kicad_mcp_client(
    server_dir: str | pathlib.Path,
    *,
    node: str = "node",
    toolboxes: str = DEFAULT_TOOLBOXES,
    mode: str = DEFAULT_MODE,
    pythonpath: str | None = None,
    kicad_python: str | None = None,
    allowed_tools: Iterable[str] | None = None,
    startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
    prefix: str | None = DEFAULT_PREFIX,
) -> MCPClient:
    """An MCPClient that launches the KiCad server over stdio.

    `allowed_tools` narrows the manifest a second time, client side, on top of
    whatever `toolboxes` already excluded. `prefix` namespaces the tool names so
    a generic one like `search` cannot collide with a local tool.
    """
    entry = validate_server_dir(server_dir)
    env = kicad_server_env(
        toolboxes=toolboxes,
        mode=mode,
        pythonpath=pythonpath,
        kicad_python=kicad_python,
    )

    def transport():
        return stdio_client(
            StdioServerParameters(
                command=node,
                args=[str(entry)],
                env=env,
                cwd=str(entry.parent.parent),
            )
        )

    kwargs: dict[str, Any] = {
        "transport_callable": transport,
        "startup_timeout": startup_timeout,
    }
    if prefix:
        kwargs["prefix"] = prefix
    if allowed_tools:
        kwargs["tool_filters"] = {"allowed": list(allowed_tools)}
    return MCPClient(**kwargs)
