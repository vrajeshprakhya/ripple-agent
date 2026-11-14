# ripple-agent

A [Strands Agents](https://strandsagents.com) agent powered by **Ripple 65B Preview**
(`ricursiveai/ripple-65b-preview`) through the [OpenRouter](https://openrouter.ai) API,
with a small sandboxed toolset.

## Why the OpenAI provider

Strands ships no OpenRouter-specific model provider. OpenRouter speaks the OpenAI wire
protocol, so the built-in `OpenAIModel` is pointed at OpenRouter's base URL instead:

```python
from strands import Agent
from strands.models.openai import OpenAIModel

model = OpenAIModel(
    client_args={"api_key": key, "base_url": "https://openrouter.ai/api/v1"},
    model_id="ricursiveai/ripple-65b-preview",
    params={"max_tokens": 1024, "temperature": 0.3},
)
agent = Agent(model=model, tools=[...], system_prompt=SYSTEM_PROMPT)
```

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install 'strands-agents[openai]' strands-agents-tools
```

Supply an OpenRouter key either way round:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
# or, read automatically if the variable is unset:
umask 077 && printf '%s' 'sk-or-v1-...' > ~/.openrouter_key
```

Create a key at <https://openrouter.ai/keys>. Ripple 65B has no free endpoint, so the
account needs credit on it. A typical tool-using turn costs well under a cent.

## Usage

```bash
./run.sh "what is in this directory?"    # single shot
./run.sh                                 # interactive REPL, state persists across turns
./run.sh --no-tools "hello"              # plain completion, no tool loop
./run.sh --provider GMICloud "hello"     # pin one provider (repeatable, order matters)
./run.sh --any-provider "hello"          # let OpenRouter route freely
```

Token usage and an upper-bound cost print to stderr after each turn.

## Tools

| Tool | Does |
|---|---|
| `list_directory` | Lists a directory with sizes |
| `read_text_file` | Reads a UTF-8 file, truncating at `max_bytes` |
| `calculate` | Exact arithmetic, walked over an AST rather than `eval` |
| `current_time` | Local time, ISO 8601 |

The two filesystem tools are confined to a sandbox root, the working directory by
default, overridable with `RIPPLE_AGENT_ROOT`. Paths resolving outside it are refused.

## Provider routing

Ripple 65B is served by three OpenRouter endpoints with materially different specs:

| Provider | Context | Price per Mtok (in / out) | Quantization |
|---|---|---|---|
| PrimeIntellect | 1,048,576 | $0.10 / $0.30 | bf16 |
| GMICloud | 1,048,576 | $0.20 / $0.60 | bf16 |
| Together | 131,072 | $0.20 / $0.60 | unspecified |

Left unpinned, OpenRouter picks for itself, so your context window and numeric precision
vary per call. `PROVIDER_ORDER` in `agent.py` states a preference and keeps
`allow_fallbacks` on, so an unhealthy endpoint costs a retry rather than a failure.

**Known issue:** as of 2026-11-13, PrimeIntellect returns a provider error on every
request, so GMICloud serves these calls in practice. PrimeIntellect is kept first in the
order because it is both the cheapest and a 1M-context bf16 endpoint when it recovers.

The cost figure printed after each turn is an **upper bound**: Strands does not surface
which provider served a call, so it assumes the dearer rate. The raw OpenRouter response
carries an exact `cost` field, which Strands discards.

## Caveats

OpenRouter warns that OpenAI-compatible endpoints can deviate from the spec in places.
Tool calling is verified working here (Ripple 65B has native tool calling), but other
features may behave differently than they would against OpenAI directly.

## Licence

MIT
