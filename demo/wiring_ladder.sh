#!/usr/bin/env bash
# Wiring ladder: how much of a circuit can Ripple 65B work out for itself?
#
# Three prompts that name nets and parts but give NO pin identifiers, so the
# model has to discover pin names with the server's tools. Each run starts from
# a fresh copy of the same five-part, unwired template and is judged by
# score_netlist.py reading KiCad's own exported netlist, never the model's
# account of what it did.
#
#   A  per-connection: "BAT1 positive terminal -> VBAT", with net names
#   B  net-centric:    "VBAT joins the cell positive, the FET drain, ..."
#   C  functional:     net names supplied, topology described in prose
#   D  open-ended:     the original prompt: no net names, no pins, design it yourself
#
# Usage: KICAD_MCP_DIR=~/KiCAD-MCP-Server demo/wiring_ladder.sh [runs-per-level] ["A B C D"]
set -uo pipefail
cd "$(dirname "$0")/.."

: "${KICAD_MCP_DIR:?set KICAD_MCP_DIR to a built KiCAD-MCP-Server checkout}"
RUNS="${1:-3}"
LEVELS="${2:-A B C}"
LAB="${RIPPLE_LADDER_DIR:-$HOME/ripple_agent_demo/ladder}"
TEMPLATE="$LAB/template"

[ -f "$TEMPLATE/bms_1s.kicad_sch" ] || { echo "no template at $TEMPLATE (build it first)" >&2; exit 1; }
case "$(readlink -m "$LAB")" in
  *bq_vboard*|*bms_vboard*|*_evidence*) echo "refusing to run in $LAB" >&2; exit 1 ;;
esac

# The manifest is held constant across every level and run. It includes the
# server's pin-discovery tools, because a prompt with no pin names is unfair
# without them.
TOOLS=(--kicad-tool list_schematic_components --kicad-tool list_symbol_pins
       --kicad-tool get_schematic_pin_locations --kicad-tool connect_to_net
       --kicad-tool add_schematic_net_label --kicad-tool add_schematic_wire)

prompt_for() {  # $1 level, $2 schematic path
  case "$1" in
    A) cat <<EOF
In the KiCad schematic $2, make these connections using the net names given. Work out the pin names yourself.
BAT1 positive terminal -> net VBAT. BAT1 negative terminal -> net BATN.
Q1 drain -> VBAT. Q1 source -> LOAD. Q1 gate -> GATE.
One terminal of C1 -> VBAT and the other -> BATN.
One terminal of T1 -> VBAT and the other -> BATN.
One end of R1 -> BATN and the other end -> GND.
Report each tool call you made.
EOF
    ;;
    B) cat <<EOF
In the KiCad schematic $2, create these nets. Work out the pin names yourself.
VBAT joins the positive terminal of BAT1, the drain of Q1, one terminal of C1 and one terminal of T1.
BATN joins the negative terminal of BAT1, the other terminal of C1, the other terminal of T1 and one end of R1.
GND is the other end of R1. LOAD is the source of Q1. GATE is the gate of Q1.
Report each tool call you made.
EOF
    ;;
    C) cat <<EOF
In the KiCad schematic $2, wire a single-cell protection circuit using the net names VBAT (cell positive), BATN (cell negative), GND, LOAD and GATE.
Q1 is the high-side discharge FET between VBAT and LOAD, with its gate on GATE.
R1 is the current shunt between BATN and GND.
Thermistor T1 and capacitor C1 each sit across the cell, from VBAT to BATN.
Work out the pin names yourself. Report each tool call you made.
EOF
    ;;
    D) cat <<EOF
In the KiCad schematic $2, wire it as a single-cell protection circuit: cell positive to the load through the NMOS FET, the shunt in the low side return path, the thermistor across the cell for temperature, and the cap across the cell. You choose the net names and work out the pin names yourself. List each connection you made and the tool call that made it.
EOF
    ;;
  esac
}

SUMMARY="$LAB/summary_${LEVELS// /}.tsv"
printf 'level\trun\tcalls\tpintool\treadfile\tcost\tstructural\tnamed\n' > "$SUMMARY"

for level in $LEVELS; do
  for i in $(seq 1 "$RUNS"); do
    dir="$LAB/trial_${level}_${i}"
    rm -rf "$dir"; cp -r "$TEMPLATE" "$dir"
    sch="$dir/bms_1s.kicad_sch"
    log="$dir/run.log"
    echo "=== level $level run $i ==="
    RIPPLE_AGENT_ROOT="$dir" timeout 300 ./run.sh \
      --kicad "$KICAD_MCP_DIR" --kicad-mode write --kicad-toolboxes all "${TOOLS[@]}" \
      --trace-tools --max-tokens 3000 --window 40 \
      "$(prompt_for "$level" "$sch")" > "$log" 2>&1
    calls=$(grep -oE 'tool calls \(([0-9]+)\)' "$log" | grep -oE '[0-9]+' | tail -1); calls=${calls:-0}
    pintool=$(grep -qE '^ +[0-9]+\. kicad_(list_symbol_pins|get_schematic_pin_locations)' "$log" && echo yes || echo no)
    readfile=$(grep -qE '^ +[0-9]+\. read_text_file' "$log" && echo yes || echo no)
    cost=$(grep -oE '\$[0-9]+\.[0-9]+' "$log" | tail -1); cost=${cost:-?}
    score=$(python3 demo/score_netlist.py "$sch" | head -1)
    st=$(sed -E 's/.*structural=([0-9]+\/[0-9]+).*/\1/' <<<"$score")
    nm=$(sed -E 's/.*named=([0-9]+\/[0-9]+).*/\1/' <<<"$score")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$level" "$i" "$calls" "$pintool" "$readfile" "$cost" "$st" "$nm" | tee -a "$SUMMARY"
  done
done

echo; echo "=== summary ($SUMMARY) ==="; column -t -s $'\t' "$SUMMARY"
