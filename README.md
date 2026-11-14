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

**The default order is now GMICloud, then PrimeIntellect, and Together is left out.**
Two measurements (2026-11-14, small samples, worth re-checking) drove that:

- **PrimeIntellect intermittently ends a streamed tool call early.** The stream carries
  `finish_reason: "tool_calls"` while the arguments JSON is still open, so Strands logs
  `failed to parse tool input json, defaulting to empty dict` and the server receives `{}`.
  Against a 16-tool manifest it did this in 4 of 6 agent runs; GMICloud did it in 0 of 8. It
  did not reproduce with a one-tool request (32 of 32 valid on both providers), so the trigger
  is something in the larger request. None of the earlier 23 experiment logs shows it.
- **Together returned HTTP 404 to a request carrying tools.**

PrimeIntellect stays as the fallback: it is half the price, so an outage on the first choice
costs a retry rather than a failure. It was also down entirely for a stretch on 2026-11-13.
The `via <provider>` field in the per-turn report says who actually served a call.

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

About 250 tests, offline: no network, no API key, no KiCad, no Node (the ngspice ones need ngspice and skip without it), so
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

## ngspice MCP server

`ngspice_mcp/` is this repo's own MCP server for board-level SPICE analysis, built because
the open-source options were unlicensed (`gtnoble/ngspice-mcp`, `ghensley/circuit-sim-mcp`)
or GPL-3.0 with an `mcp<2` pin that conflicts with this environment (`clanker-lover/spicebridge`),
and `spice2rnm` contains no MCP server to reuse. It wraps whatever ngspice is installed
(`NGSPICE_BIN`, then `~/ngspice-install/bin/ngspice`, then `PATH`) and runs over stdio.

```bash
./run.sh --spice --spice-root ~/boards "run an AC sweep of /path/to/board.cir"
./run.sh --spice --spice-tool run_ac --spice-tool measure "..."      # narrow the manifest
./run.sh --kicad ~/KiCAD-MCP-Server --spice "wire it, then simulate it"   # both servers
```

| Tool | Does |
|---|---|
| `run_op` | Operating point: every node voltage and source current |
| `run_dc_sweep` | Sweep a source, with a downsampled table |
| `run_ac` | AC sweep: dB and phase, plus the -3 dB frequency of each probe |
| `run_transient` | Transient: per-probe statistics and a downsampled waveform |
| `get_vector`, `measure` | Fetch or derive numbers from an earlier run by `result_id` (bandwidth, 10-90 rise time, rms, ...) |
| `check_netlist` | Validate without simulating |
| `export_kicad_netlist` | `kicad-cli` schematic to SPICE netlist, inside the roots |

Probes accept ngspice's differential form (`v(/srp,/srn)`), which board-level transfer
functions usually need. Results stay in server memory under a short id, so a long waveform
never has to pass through the model's context. The manifest is 8 tools; with the 4 local
ones that is about 1.9k tokens per turn.

### Why the model never writes a control block

ngspice's `.control` language can run shell commands, read and write arbitrary files and load
shared objects, so a netlist written by a model, or by text the model has read, is effectively
code. The design is that the model supplies only a circuit:

- Every dot-directive must be on a short allowlist. Analysis cards are stripped (so a KiCad
  export carrying its own `.op` still works) and `.control` is refused in any case or spacing.
- The `.control` block ngspice runs is always built by the server, from numbers and names that
  match strict patterns, so a probe like `v(out); shell id` is refused.
- `.include` and `.lib` are honoured only for files under the configured roots, resolved
  through symlinks, with a suffix allowlist. **The included file's contents are scanned too**,
  recursively and with a depth limit, because ngspice splices an included file into the deck
  and a `.control` block inside it would run. Being under a root does not make content trusted.
- Python and ngspice disagree about what a line break is (`\x0c`, `\x85`, ` ` ...). A
  checker built on a different split than the consumer is how a directive gets hidden, so
  those characters are refused outright.
- ngspice runs in a throwaway directory with a scrubbed environment, CPU, memory and file-size
  limits, and a wall-clock timeout that kills the process group.

**This narrows the attack surface; it is not a sandbox.** There is no network isolation and
no separate user, and there is a small window between the scan of an included file and
ngspice reading it. Do not point `--spice-root` at a directory a hostile party can write to.
The KiCad server is *not* confined by these roots.

Two failure modes found while testing are worth knowing, because both would have misled a model:

- ngspice reports a malformed component as a **warning**, drops it, and simulates the rest.
  The "operating point" returned was its built-in constants table (`pi`, `boltz`, `kelvin`).
  Warnings that mean "ignored" or "not a valid" are now failures, and the result's plot type
  must match the analysis asked for.
- The MCP SDK hides the text of any exception it does not recognise, so every refusal arrived
  as "Error executing tool run_op". Deliberate failures are now raised as `ToolError`, the
  one type whose message reaches the model.

### Checked against a real board

`~/bq_vboard_evidence/kicad/bms_6s_fab.cir`, the KiCad export of a 6S BMS, copied to a
scratch directory and compared with that project's own logged testbench results.

| | Logged reference | These tools, no model involved |
|---|---|---|
| Cell 1 voltage | 3.688 V | 3.6880 V |
| Shunt voltage | 1.00002 mV | 1.00002 mV |
| Kelvin-filter corner | 7957.74 Hz | 7956.9 Hz (0.01% off: interpolation at 20 points per decade) |

Then Ripple 65B was asked for the same quantities, three times. It made 2 or 3 tool calls per
run at about $0.0004 to $0.0007, and **got the cell voltage (3.688 V) and the corner
(7.957 kHz) right every time. It got the shunt current wrong every time**: it reported
"1.00002 mA" where 1.00002 mV across 1 mOhm is 1.00002 **A**. The simulator's number was
right; the model divided it itself, never used its `calculate` tool, and attached the wrong
unit. Treat a model's unit labels on derived quantities as unverified even when every number
it was handed was correct.

## openEMS MCP server (third party)

`mcp_openems.py` launches [RFingAdam/mcp-openems](https://github.com/RFingAdam/mcp-openems)
(AGPL-3.0) as a separate stdio process. Nothing from it is copied into this MIT repo.

```bash
git clone https://github.com/RFingAdam/mcp-openems ~/mcp-openems
git -C ~/mcp-openems checkout 46ceb5d                     # the commit that was audited
cd ~/mcp-openems && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
./run.sh --openems ~/mcp-openems "design a 50 ohm microstrip for 2.4 GHz on 1.6 mm FR-4"
```

Audit of `46ceb5d`, by reading the source (not by running it): no subprocess calls, no network
calls, no install-time hooks, runtime dependencies `mcp` and `numpy` only. It writes no files and
takes no paths, so there are no roots to configure. The script templates it *emits* create and
delete a fixed directory under the system temp dir when someone runs them, which only matters
once openEMS is installed.

**openEMS itself is built from source here.** It is not on PyPI or apt. The C++ libraries
(`thliebig/openEMS-Project` at `e4cd015`, `update_openEMS.sh ~/opt/openEMS --disable-GUI`) took
2 min 28 s with 4 jobs and needed only `cmake libhdf5-dev libtinyxml-dev libboost-all-dev` from apt.
The Python bindings are built inside the `mcp-openems` venv (`build_python.sh --cpp-install-dir`,
3 min 10 s) and are pinned to that venv's Python 3.12; a different interpreter or an HDF5/boost
soname bump means a rebuild. They import with a scrubbed environment, because the libraries are
found through a baked-in RUNPATH, and `openems_check_installation` now reports
`openems_available: true`.

Its 16 tools cost about 2.3k tokens per turn. Checked against independent physics rather than
taken on trust: its microstrip impedance matches Hammerstad-Jensen to within 1% across four
substrates (epsilon_r 3.38 to 10.2), and its effective permittivity, wavelength, electrical
length and delay are mutually consistent. Five runs of "find the width that gives 50 ohms on
1.6 mm FR-4 at 2.4 GHz", through Ripple 65B on the default provider order, all converged:
3.17 mm at the tool's default epsilon_r of 4.2 (my reference: 3.18 mm) in three runs, and
3.065 mm at epsilon_r 4.4 in two, which is also correct for that permittivity. One of those
said "4.4, as specified" though no permittivity was specified.

### Full-wave validation: the build passes, the generated script does not

Two checks, both using energy conservation (a lossless passive two-port cannot return more
power than it was given, so `|S11|^2 + |S21|^2` must be at most 1):

| Simulation | Power balance | Verdict |
|---|---|---|
| Official `MSL_NotchFilter` tutorial (`demo/openems_tutorial_sanity.py`, 35 s) | 0.977 to 1.001 | **Sound**: 52.9 dB notch, S11 down to -33 dB |
| Script generated by `openems_generate_script` for a 3 mm x 20 mm microstrip, ports matched to the line (`demo/openems_microstrip_validation.py`, 77 s) | 1.7 to 1.9 | **Not physical**: a matched line shows |S11| of about -0.2 dB, nearly total reflection |

So the openEMS build and solver are fine, and **the simulation scripts `mcp-openems` generates should
not be trusted for verification as they stand.** With the ports mismatched to 100 ohm against the
analytical response, the S21 phase tracks theory (median error about 6 degrees) but the magnitudes
are off by 4 to 5 dB, and the power balance exceeds 1 there too. I did not find the root cause. The
generated script differs from the working tutorial in ways that are plausible suspects, not
established ones: plain lumped ports where the tutorial uses `AddMSLPort` with feed and
measurement-plane shifts, no `SmoothMeshLines`, no explicit `ref_impedance` in `CalcPort`, and a
10 mm absorbing margin that is small against the quarter wavelength (about 62 mm at 1.2 GHz).

The analytical tools (impedance, effective permittivity, delay) are unaffected and were checked
against Hammerstad above. What this means in practice: use the design tools, and treat any
S-parameters from the generated scripts as unverified until they pass a power-balance check.

### What is not here

**`mcp-pcb-emcopilot`** was the intended companion, and is not integrated. The author's
repository returns 404 (listings call it private), and the only public copy is under a different
account (`Yverr0y/mcp-pcb-emcopilot`) whose provenance cannot be verified from git alone. Reading
it turned up an `eval()` in its Gerber parser (`gerber_parser.py:127`) running on text from the
file with empty `__builtins__`, which is a known-escapable pattern and a real concern for files
that come from outside. Nothing from it was installed or run.

## Caveats

OpenRouter warns that OpenAI-compatible endpoints can deviate from the spec in places.
Tool calling is verified working here (Ripple 65B has native tool calling), but other
features may behave differently than they would against OpenAI directly.

The client wrapper that recovers exact cost reaches past the public API, since no public
hook exposes unmodelled response fields. It is written to degrade rather than break.

## Licence

MIT
