# ripple-agent

[![tests](https://github.com/vrajeshprakhya/ripple-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/vrajeshprakhya/ripple-agent/actions/workflows/ci.yml)

A [Strands Agents](https://strandsagents.com) agent powered by **Ripple 65B Preview**
(`ricursiveai/ripple-65b-preview`) through the [OpenRouter](https://openrouter.ai) API,
with a sandboxed toolset and exact per-turn cost reporting.

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
./.venv/bin/python -m pip install -r requirements.txt
```

Supply an OpenRouter key either way round:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
# or, read automatically if the variable is unset:
umask 077 && printf '%s' 'sk-or-v1-...' > ~/.openrouter_key
```

Create a key at <https://openrouter.ai/keys>. Ripple 65B has no free endpoint, so the
account needs credit on it. A tool-using turn typically costs under a hundredth of a cent.

## Usage

```bash
./run.sh "what is in this directory?"    # single shot
./run.sh                                 # interactive REPL, state persists across turns
./run.sh --no-tools "hello"              # plain completion, no tool loop
./run.sh --provider GMICloud "hello"     # pin one provider (repeatable, order matters)
./run.sh --any-provider "hello"          # let OpenRouter route freely
./run.sh --window 10 "hello"             # keep only the last 10 turns in context
```

After each turn a line like this goes to stderr:

```
[tokens in=1339 out=163 total=1502  $0.000089 via GMICloud]
```

## Tools

| Tool | Does |
|---|---|
| `list_directory` | Lists a directory with sizes |
| `read_text_file` | Reads a UTF-8 file, truncating at `max_bytes` |
| `calculate` | Exact arithmetic, walked over an AST rather than `eval` |
| `current_time` | Local time, ISO 8601 |

The two filesystem tools are confined to a sandbox root, the working directory by
default, overridable with `RIPPLE_AGENT_ROOT`. Containment is checked against the
**resolved** path, so `..` traversal, absolute paths and symlinks pointing outside the
root are all refused rather than merely discouraged.

`calculate` walks an AST and permits numbers and arithmetic operators only: no names,
calls, subscripts or comprehensions. It also refuses exponents above 1024, since
`9**9**9` is cheap to write and would otherwise hang the agent inside a C-level loop.

## Cost reporting

Strands normalizes the model's usage payload down to token counts and discards
everything else, so neither `AgentResult` nor a `callback_handler` ever sees
OpenRouter's own `provider` and `cost` fields. Setting `usage: {"include": true}` in
`extra_body` makes OpenRouter report an exact cost, and a thin wrapper around the
OpenAI client tees the streamed chunks to read it back. `OpenAIModel` accepts a
pre-built client for exactly this purpose, which its docstring describes as
"centralizing observability".

This matters more than it sounds. Estimating cost from token counts and headline rates
overstated a real turn by **4x**, because prompt caching discounts cached input tokens
by roughly an order of magnitude and the token counts alone cannot show that. If the tee
ever stops working after an upstream upgrade, the line falls back to a token-based
estimate labelled `<=$... est` rather than failing.

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

**PrimeIntellect is intermittent.** On 2026-11-13 it returned a provider error on every
request, then served normally a few hours later. It is kept first in the order because it
is both the cheapest and a 1M-context bf16 endpoint, and `allow_fallbacks` turns an outage
into a retry rather than a failure. The `via <provider>` field in the per-turn report
tells you who actually served a given call, so do not assume.

## Context window

Ripple's window is large, but an unbounded REPL will still walk into it, and every
retained turn is re-billed as input on the next call. A
`SlidingWindowConversationManager` keeps the last `--window` turns (40 by default), so
long sessions stay bounded in both context and cost.

## Tests

```bash
./.venv/bin/python -m pip install -r requirements-dev.txt
./.venv/bin/python -m pytest tests -q
```

83 tests, entirely offline: no network, no API key, no KiCad, no Node, no MCP server, so
they cost nothing and run in CI on every push. They cover sandbox escape refusal
(including via symlink), calculator safety, key-loading precedence, the usage-reporting
fallback, the KiCad server validation and mode handling, and the tool tracer. The live
model path is deliberately not tested in CI, because a key in CI would mean every push
spends credit.

## KiCad MCP server

`mcp_kicad.py` wires in [mixelpixx/KiCAD-MCP-Server](https://github.com/mixelpixx/KiCAD-MCP-Server),
a Node process speaking MCP over stdio that drives KiCad through its `pcbnew`
Python bindings, with an experimental IPC backend.

It has prerequisites this repo cannot install for you:

| Needs | Why |
|---|---|
| KiCad 9.0+ with `pcbnew` | The server's whole backend. Verify with `python3 -c "import pcbnew; print(pcbnew.GetBuildVersion())"` |
| Node.js 18+ | The server is TypeScript, built with `npm run build` |
| Python 3.9+ and its `requirements.txt` | The server's Python side |

```bash
git clone --branch stable https://github.com/mixelpixx/KiCAD-MCP-Server.git
cd KiCAD-MCP-Server && npm install && pip3 install -r requirements.txt && npm run build
```

Then:

```bash
./run.sh --kicad ~/KiCAD-MCP-Server "what layers does this board use?"
./run.sh --kicad ~/KiCAD-MCP-Server --kicad-toolboxes all        # the full manifest
./run.sh --kicad ~/KiCAD-MCP-Server --kicad-mode write           # lets the model edit designs
./run.sh --kicad ~/KiCAD-MCP-Server --kicad-tool get_board_info  # narrow, client side
```

### Two defaults chosen deliberately

**`--kicad-mode readonly`.** The server's `KICAD_MCP_OPERATING_MODE` also accepts
`manufacturing` and `write`, either of which lets the model mutate real PCB and
schematic files. Readonly is the default here and widening it is an explicit flag. A
mode that is not exactly one of the three documented spellings raises rather than
falling back, so a typo cannot silently widen access.

**`--kicad-toolboxes core`,** roughly 37 tools. The full server advertises between 184
and 253 depending on which part of its README you read. That is a bad fit for a 65B
model: tool-selection accuracy degrades well before that count, and every schema is
re-sent as input on every turn, which the 131k Together endpoint would feel. Tools are
also namespaced with a `kicad` prefix so a generic name cannot collide with a local one.

The server directory is validated eagerly, before the agent or the API key are touched,
because the alternative is an opaque stdio timeout 90 seconds later. An unbuilt checkout
is told to run `npm run build`; a wrong directory is told what `--kicad` expects.

### Measuring it

`--trace-tools` reports the manifest size and every tool the model actually chose:

```
manifest: 4 tools, 1216 chars (~304 tokens re-sent per turn)
tool calls (3):
  1. list_directory {'path': '.'}
  2. read_text_file {'path': 'run.sh'}
  3. calculate {'expression': '2*21'}
```

That measurement is the point rather than a debugging aid. At roughly 300 characters per
simple tool, a 37-tool `core` manifest should cost a few thousand input tokens per turn,
and the full 184-plus manifest tens of thousands, on every single call. KiCad's schemas
are richer than these four, so treat those as floors.

`demo/kicad_bms_demo.sh` is a staged capability probe: it asks the agent to enumerate its
KiCad tools, create a project, place parts from the official symbol libraries, wire a
single-cell protection circuit, and run ERC. Each stage is a fresh agent so one derailing
cannot poison the next, and it refuses to run in any path matching `bq_vboard`,
`bms_vboard` or `_evidence`. It is a measurement of where the model succeeds and fails,
not a way to produce a design worth fabricating.

## Caveats

OpenRouter warns that OpenAI-compatible endpoints can deviate from the spec in places.
Tool calling is verified working here (Ripple 65B has native tool calling), but other
features may behave differently than they would against OpenAI directly.

The client wrapper that recovers exact cost reaches past the public API, since no public
hook exposes unmodelled response fields. It is written to degrade rather than break.

## Licence

MIT
