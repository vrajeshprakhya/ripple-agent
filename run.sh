#!/usr/bin/env bash
# Run the Ripple 65B Strands agent inside its own venv.
#   ./run.sh "prompt"      single shot
#   ./run.sh               interactive REPL
set -euo pipefail
cd "$(dirname "$0")"
exec ./.venv/bin/python agent.py "$@"
