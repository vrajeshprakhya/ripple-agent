"""MCP server for validated full-wave electromagnetic checks.

The point of this server, against the generator in mcp-openems, is that it can be trusted:
the simulation is a fixed validated script, the model supplies numbers rather than code, and
a result that breaks energy conservation is refused instead of returned.
"""
from __future__ import annotations

import functools

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import core
from .core import EmError

mcp = MCPServer(
    "emverify",
    instructions=(
        "Full-wave electromagnetic simulation with openEMS, validated against theory. Use "
        "simulate_microstrip to check a trace's impedance and propagation, and read the "
        "agrees_with_theory field and the difference_percent before relying on it."
    ),
)


def _tool(fn):
    """Register fn, turning EmError into a ToolError whose message reaches the model."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except EmError as e:
            raise ToolError(str(e)) from e

    return mcp.tool()(wrapped)


@_tool
def check_solver() -> dict:
    """Report whether the openEMS solver is installed and which versions."""
    return core.solver_status()


@_tool
def simulate_microstrip(trace_width_mm: float, dielectric_height_mm: float, substrate_er: float,
                        f_max_ghz: float = 6.0, line_length_mm: float = 60.0) -> dict:
    """Full-wave FDTD simulation of a straight microstrip line over a ground plane.

    Returns the line's characteristic impedance, effective permittivity and delay measured from the
    simulated fields, beside the analytical Hammerstad values and the percentage difference, plus
    S-parameters and an energy-conservation check. A simulation that is not physical is refused.

    Takes roughly 10 to 60 seconds. line_length_mm must be at least 12 times the dielectric height.
    Z0 and effective permittivity are medians over the lowest 40 percent of the band up to f_max_ghz.
    """
    return core.run_microstrip({"width_mm": trace_width_mm, "height_mm": dielectric_height_mm,
                                "er": substrate_er, "f_max_ghz": f_max_ghz, "length_mm": line_length_mm})


def main() -> None:
    mcp.run(transport="stdio")
