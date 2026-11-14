#!/usr/bin/env bash
# Run the Ripple agent against the fine-tuned (LoRA-merged) model on Modal instead of OpenRouter.
#   ./run_lora.sh "prompt"        single shot, with the ngspice tool
#   ./run_lora.sh                 interactive REPL
set -euo pipefail
cd "$(dirname "$0")"
export RIPPLE_BASE_URL="https://vrajeshprakhya--ripple-serve-lora-serve-lora-h200.us-east.modal.run/v1"
export RIPPLE_MODEL="ripple-lora"
export RIPPLE_KEY_FILE="$HOME/.ripple_modal_key"
unset OPENROUTER_API_KEY   # it takes precedence over the key file and would be sent to Modal
exec ./.venv/bin/python agent.py --any-provider --spice --ngspice "$HOME/ngspice-install/bin/ngspice" --max-tokens 4096 "$@"
