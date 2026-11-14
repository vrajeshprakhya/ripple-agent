#!/usr/bin/env bash
# Capability probe: can Ripple 65B drive the KiCad MCP server well enough to
# build a trivial BMS schematic from scratch?
#
# This is a measurement, not a deliverable. The design it produces is not
# expected to be fabricable or even to pass ERC. What is being measured is:
#   - how many tools the manifest carries, and what it costs per turn
#   - whether the model selects the right tool out of that manifest
#   - where it gives up, loops, or invents a tool that does not exist
#
# It deliberately works in an isolated directory, because it runs the server in
# write mode and so the agent can create and modify real design files.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${KICAD_MCP_DIR:?set KICAD_MCP_DIR to a built KiCAD-MCP-Server checkout}"
WORK="${RIPPLE_DEMO_DIR:-$HOME/ripple_agent_demo/bms_1s}"
TOOLBOXES="${KICAD_MCP_TOOLBOXES:-core}"

# Refuse to run anywhere near real design work.
case "$(readlink -m "$WORK")" in
  *bq_vboard*|*bms_vboard*|*_evidence*)
    echo "refusing to run in $WORK: that path holds real design work" >&2
    exit 1
    ;;
esac

mkdir -p "$WORK"
if [ -n "$(ls -A "$WORK" 2>/dev/null)" ]; then
  echo "note: $WORK is not empty; the agent may modify what is there" >&2
fi

echo "=== probe configuration ==="
echo "server:    $KICAD_MCP_DIR"
echo "workdir:   $WORK   (write mode)"
echo "toolboxes: $TOOLBOXES"
echo

run_stage() {
  local n="$1" prompt="$2"
  echo "################ stage $n ################"
  echo "prompt: $prompt"
  echo
  # Each stage is a fresh agent, so a stage that derails cannot poison the
  # next one; that also makes the per-stage tool traces comparable.
  RIPPLE_AGENT_ROOT="$WORK" ./run.sh \
    --kicad "$KICAD_MCP_DIR" \
    --kicad-mode write \
    --kicad-toolboxes "$TOOLBOXES" \
    --trace-tools \
    --max-tokens 2000 \
    --window 20 \
    "$prompt" || echo "[stage $n exited $?]"
  echo
}

run_stage 1 "What KiCad tools do you have available? Group them by what they operate on, and name the one you would use to create a new project. Do not create anything yet."

run_stage 2 "Create a new KiCad project called bms_1s in $WORK. Report the exact files you created and the tool you used for each."

run_stage 3 "In the bms_1s project at $WORK, add these parts from the official KiCad symbol libraries: a Device:Battery_Cell, a Device:R for a current shunt, a Device:Thermistor_NTC, a Device:Q_NMOS as the discharge FET, and a Device:C decoupling cap. Report each part you placed and its reference designator."

run_stage 4 "In bms_1s at $WORK, wire it as a single-cell protection circuit: cell positive to the load through the NMOS FET, the shunt in the low side return path, the thermistor across the cell for temperature, and the cap across the cell. List each connection you made and the tool call that made it."

run_stage 5 "Run an electrical rules check on bms_1s at $WORK and report the result verbatim. If you cannot, say which tool you looked for and what was missing."

echo "=== probe complete ==="
echo "workdir contents:"
ls -la "$WORK"
