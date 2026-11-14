#!/usr/bin/env bash
# Ask Ripple 65B the same board-level SPICE question N times and grade each answer.
#
# The board is a COPY of the 6S BMS fab netlist, so nothing under the evidence directory is
# ever a server root. The reference values come from that project's own logged testbench.
#
# Usage: demo/spice_board_probe.sh [runs]        (default 5)
set -uo pipefail
cd "$(dirname "$0")/.."

RUNS="${1:-5}"
SRC="${RIPPLE_BOARD_SRC:-$HOME/bq_vboard_evidence/kicad}"
LAB="${RIPPLE_SPICE_LAB:-$HOME/ripple_agent_demo/spice}"

mkdir -p "$LAB/logs"
cp "$SRC/bms_6s_fab.cir" "$SRC/bms_fab_models.lib" "$LAB/"
# KiCad writes an absolute UNC path for the models include; localise it to the copy.
sed -i 's|^\.include .*bms_fab_models.*|.include "bms_fab_models.lib"|' "$LAB/bms_6s_fab.cir"

PROMPT="The board netlist is $LAB/bms_6s_fab.cir. Using the SPICE tools: (1) at the nominal operating point, find the voltage of cell 1, which is v(/vc1f) minus v(/vc0f), and the current through the sense shunt, which is v(/srp) minus v(/srn) divided by 1 milliohm. (2) Run an AC sweep from 10 Hz to 1 MHz and find the -3 dB frequency of the differential voltage v(/srp,/srn). Report the three numbers with units."

printf 'run\tcalls\tcompute\tcell\tcurrent\tcorner\tcost\n'
for i in $(seq 1 "$RUNS"); do
  log="$LAB/logs/probe_$i.log"
  RIPPLE_AGENT_ROOT="$LAB" timeout 280 ./run.sh --spice --spice-root "$LAB" \
    --trace-tools --max-tokens 2500 "$PROMPT" > "$log" 2>&1
  answer=$(grep -vE '^ +[0-9]+\. |^manifest|^tool calls|^\[tokens|^spice mcp' "$log")
  grade=$(python3 demo/grade_board_answer.py <<<"$answer")
  calls=$(grep -oE 'tool calls \(([0-9]+)\)' "$log" | grep -oE '[0-9]+' | tail -1)
  usedcompute=$(grep -qE '^ +[0-9]+\. spice_compute' "$log" && echo yes || echo no)
  cost=$(grep -oE '\$[0-9]+\.[0-9]+' "$log" | tail -1)
  cell=$(sed -E 's/.*cell=([a-z]+).*/\1/' <<<"$grade")
  cur=$(sed -E 's/.*current=([a-z]+).*/\1/' <<<"$grade")
  cor=$(sed -E 's/.*corner=([a-z]+).*/\1/' <<<"$grade")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "${calls:-0}" "$usedcompute" "$cell" "$cur" "$cor" "${cost:-?}"
done
