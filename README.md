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

**`--kicad-toolboxes core`.** The server's own README gives three different tool counts
(253, 244, 184). Measured against the running v2.9.0 server, the manifest is:

| Mode / toolbox | Tools | Manifest, per turn |
|---|---|---|
| readonly / core (default) | 14 | ~1.5k tokens |
| write / core | 37 | ~6.6k tokens |
| readonly / all | 86 | ~17k tokens |
| write / all | 250 | ~66.6k tokens |

The manifest is re-sent as input on every call, so the full write manifest would eat half
of the 131k Together endpoint's window before the conversation starts. `core` is the
default for that reason. The server can also unlock groups at runtime
(`kicad_open_toolbox`), and the tracer saw the count grow from 37 to 41 mid-session. Tools
are namespaced with a `kicad` prefix so a generic name cannot collide with a local one.

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

That measurement is the point rather than a debugging aid. The four local tools cost about
300 tokens. KiCad's richer schemas cost about 180 tokens per tool (see the table above).

### Probe results (2026-11-13, one run, so anecdote rather than benchmark)

Ripple 65B, `write` / `core`, KiCad 9.0.8, server v2.9.0. Whole probe cost about $0.002.

| Stage | Outcome |
|---|---|
| 1. Enumerate tools | Answered |
| 2. Create project | **Worked**: one `kicad_create_project` call wrote the four project files |
| 3. Place 5 library parts | **Worked**: 5 `kicad_add_schematic_component` calls, correct nested arguments, designators BAT1/R1/T1/Q1/C1, all five symbols present in the file |
| 4. Wire the circuit | **Refused 4 of 4 with no pin-discovery tool in the manifest; attempts it every time once one is present, though open-ended designs come out wrong** (see below) |
| 5. Run ERC | **Failed on `core`, worked on a narrow manifest** (see below) |

**The server writes KiCad 10 format** (`version 20260101`, `generator_version "10.0"`,
hardcoded in `python/commands/schematic.py`), so KiCad 9.0.8 reports `Failed to load
schematic`. Use KiCad 10 with this server. With 10.0.8 installed the same file loads.

### Follow-up experiments

Stage 4 and 5 were re-run with the system prompt corrected (it had claimed the agent
only had filesystem and arithmetic tools, which stopped being true once an MCP server
attached; `compose_system_prompt` now describes MCP tools and exempts them from the
sandbox rule) and then with a narrow manifest.

| Run | Stage 4 (wiring) | Stage 5 (ERC) |
|---|---|---|
| `core`, 37 tools, original prompt | refused | refused, no tool call |
| `core`, 37 tools, corrected prompt | refused | refused, no tool call |
| 5 and 2 tools (relevant only) | refused | **worked** |

The ERC run called `list_directory` then `kicad_run_erc` with correct arguments and
reported 17 violations. An independent `kicad-cli sch erc` run on the same file also
reports 17, so the result is real.

What this does and does not show:

- **Manifest size matters for ERC.** `kicad_run_erc` lives only in the `all` toolbox, so
  `core` genuinely lacks it. The model did not call `kicad_search_tools` or
  `kicad_open_toolbox` to find it, but used it as soon as it was in front of it.
- **The wiring refusal was not about the sandbox, and was only half wrong about tools.**
  "Outside the sandbox" was false (the model listed that very directory in the ERC run).
  But "I have no tool that shows pin names" was TRUE: `list_schematic_components` reports
  pin counts only, and the 5-tool manifest held no pin-discovery tool. An earlier version of
  this README called all of its stated reasons false; that was an overreach. The cause was
  found by experiment, below.

### Why wiring failed, and what Ripple can and cannot do (revised)

An earlier version of this section concluded "the task, not the tools": that Ripple can
execute a given design but not originate one. That was wrong about the cause, and the
data that overturned it follows.

The first explicit-pins run (23 calls) looked like a clean contrast with the open-ended
refusals, but its manifest differed from the refused runs in a way I had not controlled:
it let the model fall back on reading the schematic file for pin names. The ladder below
holds the manifest constant: 6 KiCad tools (`list_schematic_components`,
`list_symbol_pins`, `get_schematic_pin_locations`, `connect_to_net`,
`add_schematic_net_label`, `add_schematic_wire`) plus the 4 local ones, about 1.7k tokens
per turn. Twelve runs, each from a fresh copy of the same unwired five-part template,
scored by `demo/score_netlist.py` against KiCad's own exported netlist, never the model's
report. `demo/wiring_ladder.sh` reproduces it.

| Level | The prompt supplies | Perfect (9/9) | Notes |
|---|---|---|---|
| A | every connection, with net names and roles ("BAT1 positive -> VBAT") | 3 of 3 | 17 calls each |
| B | net membership ("VBAT joins the positive terminal of BAT1, the drain of Q1...") | 3 of 3 | 17 calls each |
| C | net names plus a prose topology ("Q1 is the high-side FET between VBAT and LOAD...") | 2 of 3 | C1 made 5 calls, wired nothing |
| D | only the original open-ended prompt | **0 of 3** | 16 calls each, all wrong |

Levels A to C cost about $0.0003 to $0.0005 per run: one `list_schematic_components`, five
`get_schematic_pin_locations`, eleven `connect_to_net`. No run read the schematic file.
The pin-discovery tool is what the model reached for every time.

**Level D is the finding.** With the same pin-discovery tools, the open-ended prompt that
was refused 4 of 4 times earlier is now attempted every time. But the designs are wrong:

- D1 and D3 put the FET source and the shunt's free pin on one net called `LOAD`, so the
  load's two terminals are the same node.
- D2 put the FET source on the cell's negative net, which puts the FET across the cell.

So: **given net names it executes reliably (8 of 9); asked to originate the topology it
completes the job and gets it wrong (0 of 3).** That resembles the earlier hypothesis but
for a different reason: the refusals were a missing tool, and the real limitation is design
correctness.

How the scorer was tightened, because it matters for trust: the first rubric had 8 checks
and gave D1 and D3 a 7/8, missing that merged load terminals are a design error. A ninth
check ("load terminals on different nets") was added, with regression tests built from
these two runs, and every trial was re-scored under it. A to C are unaffected. A floating
gate is not counted as a defect at level D, because that prompt never asked for gate drive.

Caveats: 12 runs, one model, one small circuit, one temperature. The scorer encodes one
reading of the prompt, though the three level D designs are wrong by inspection, not
merely different. The first explicit run's observation that Ripple guessed Q1's pin names
wrong twice before reading the file is explained by the missing discovery tool, not by any
reasoning failure.

`--kicad-tool` matches the server's own tool names **without** the `kicad_` prefix
(`run_erc`, not `kicad_run_erc`). A filter matching nothing now exits with an error
rather than running an agent that truthfully reports having no tools.

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
