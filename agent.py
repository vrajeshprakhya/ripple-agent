#!/usr/bin/env python3
"""A Strands agent powered by Ripple 65B Preview, served over the OpenRouter API.

Strands ships no OpenRouter-specific provider; OpenRouter speaks the OpenAI wire
protocol, so the built-in OpenAI provider is pointed at its base URL instead.

Usage:
    ./run.sh "what is in this directory?"     # single shot
    ./run.sh                                  # interactive REPL
    ./run.sh --no-tools "hello"               # plain completion, no tool loop
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as _dt
import operator
import os
import pathlib
import sys

import mcp_kicad
import mcp_openems
import mcp_spice
from strands import Agent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models.openai import OpenAIModel

MODEL_ID = "ricursiveai/ripple-65b-preview"
BASE_URL = "https://openrouter.ai/api/v1"
KEY_FILE = pathlib.Path.home() / ".openrouter_key"

# OpenRouter load-balances across three endpoints with materially different specs:
#   PrimeIntellect  1,048,576 ctx  $0.10/$0.30 per Mtok  bf16
#   GMICloud        1,048,576 ctx  $0.20/$0.60 per Mtok  bf16
#   Together          131,072 ctx  $0.20/$0.60 per Mtok  quantization unspecified
# Left unpinned it picks for itself, so the context window and the precision you
# get vary per call. Pin the order and keep fallbacks for availability.
#
# GMICloud goes first, and Together is left out, for two reasons found by measurement
# on 2026-11-14 (small samples; re-check before relying on them):
#   * PrimeIntellect intermittently ends a streamed tool call early: the stream carries
#     finish_reason "tool_calls" while the arguments JSON is still open, so Strands logs
#     "failed to parse tool input json, defaulting to empty dict" and the server receives {}.
#     Against a 16-tool manifest it did this in 4 of 6 agent runs; GMICloud in 0 of 3.
#     It did not reproduce with a one-tool request (32 of 32 valid on both providers), so the
#     trigger is something in the larger request. Earlier it was also down for a stretch.
#   * Together returned HTTP 404 to a request carrying tools.
# PrimeIntellect stays as the fallback: it is half the price, so an outage on the first
# choice costs a retry rather than a failure.
PROVIDER_ORDER = ["GMICloud", "PrimeIntellect"]

# Per-Mtok rates for the fallback cost estimate, used only when OpenRouter does
# not report an exact cost. GMICloud and Together both charge this;
# PrimeIntellect is half, so the estimate is an upper bound.
RATE_IN, RATE_OUT = 0.20, 0.60

# Turns kept in context. Ripple's window is large, but an unbounded REPL will
# still walk into it, and every retained turn is re-billed as input on the next
# call. Sliding the window keeps long sessions bounded in cost and context.
DEFAULT_WINDOW = 40

SYSTEM_PROMPT = """You are a concise technical assistant running inside a Strands agent loop.

You have tools for reading the local filesystem and doing arithmetic. Use them
rather than guessing: if a question is about a file or directory, read it first.
State plainly when a tool returns an error or when something is outside the
sandbox root, and never invent file contents you did not read."""

MCP_ADDENDUM = """

You also have tools supplied by an attached MCP server. Their names carry a
prefix. They are NOT covered by the sandbox rule above, which restricts only your
local file tools, so a path that your file tools refuse may be perfectly valid
for them. When a task is about something those tools operate on, use them rather
than asking the user to paste files. Before concluding that a capability does not
exist, look for it: if a tool for searching or listing tools is available, call
it, and report what it returned.

State every physical quantity with its SI unit. Do not do arithmetic on quantities
in your head: if a unit-aware calculation tool is available, use it, because slips
in prefixes (milli versus base units) are easy to make and invisible. For example
volts divided by ohms is amperes."""


def compose_system_prompt(has_mcp_tools: bool) -> str:
    """The base prompt, plus a description of the MCP tools when there are any.

    The base prompt says the agent has filesystem and arithmetic tools. Once an
    MCP server is attached that is no longer true, and a model told it has only
    those tools will say so, including in front of a manifest that lists dozens
    more.
    """
    return SYSTEM_PROMPT + (MCP_ADDENDUM if has_mcp_tools else "")


# --- Sandbox --------------------------------------------------------------
# File tools are confined to SANDBOX_ROOT so the agent cannot wander the disk.
SANDBOX_ROOT = pathlib.Path(
    os.environ.get("RIPPLE_AGENT_ROOT", pathlib.Path.cwd())
).resolve()


def resolve_in_sandbox(path: str, root: pathlib.Path | None = None) -> pathlib.Path:
    """Resolve path under root, refusing anything that escapes it.

    Absolute paths, .. traversal and symlinks are all resolved first, so the
    containment check sees the real target rather than the spelling.
    """
    base = (root or SANDBOX_ROOT).resolve()
    raw = pathlib.Path(path)
    p = (raw if raw.is_absolute() else base / raw).resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"path escapes sandbox root {base}: {path}")
    return p


# --- Tool implementations -------------------------------------------------
# Kept as plain functions so they are testable without the Strands tool
# machinery; the @tool wrappers below are what the agent actually sees.
def list_directory_impl(path: str = ".", root: pathlib.Path | None = None) -> str:
    try:
        d = resolve_in_sandbox(path, root)
    except ValueError as e:
        return f"error: {e}"
    if not d.is_dir():
        return f"error: not a directory: {d}"
    rows = []
    for e in sorted(d.iterdir()):
        kind = "dir " if e.is_dir() else "file"
        size = "-" if e.is_dir() else str(e.stat().st_size)
        rows.append(f"{kind} {size:>10}  {e.name}")
    return "\n".join(rows) or "(empty)"


def read_text_file_impl(
    path: str, max_bytes: int = 20000, root: pathlib.Path | None = None
) -> str:
    try:
        f = resolve_in_sandbox(path, root)
    except ValueError as e:
        return f"error: {e}"
    if not f.is_file():
        return f"error: not a file: {f}"
    size = f.stat().st_size
    text = f.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
    if size > max_bytes:
        text += f"\n...[truncated at {max_bytes} of {size} bytes]"
    return text


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Guards against an expression that is cheap to write and ruinous to evaluate,
# such as 9**9**9, which would hang the agent inside a C-level loop.
_MAX_POW_EXP = 1024


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXP:
            raise ValueError(f"exponent above {_MAX_POW_EXP} refused")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculate_impl(expression: str) -> str:
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"error: {e}"


# --- Tools ----------------------------------------------------------------
@tool
def list_directory(path: str = ".") -> str:
    """List the entries in a directory, with sizes.

    Args:
        path: Directory to list, relative to the sandbox root.
    """
    return list_directory_impl(path)


@tool
def read_text_file(path: str, max_bytes: int = 20000) -> str:
    """Read a UTF-8 text file and return its contents.

    Args:
        path: File to read, relative to the sandbox root.
        max_bytes: Truncate the file after this many bytes.
    """
    return read_text_file_impl(path, max_bytes)


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Numbers and operators only, no names or calls.

    Args:
        expression: For example "(0.10 * 23 + 0.30 * 38) / 1000000".
    """
    return calculate_impl(expression)


@tool
def current_time() -> str:
    """Return the current local date and time in ISO 8601 form."""
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# --- Usage capture --------------------------------------------------------
class UsageTap:
    """Records the exact cost and serving provider OpenRouter reports.

    Strands normalizes the usage payload down to token counts and discards
    everything else, so neither the AgentResult nor the callback handler ever
    sees OpenRouter's own `provider` and `cost` fields. They are read off the
    raw chunks instead, at the one seam Strands offers for this: OpenAIModel
    accepts a pre-built client, documented for "centralizing observability".
    """

    def __init__(self) -> None:
        self.provider: str | None = None
        self.cost: float | None = None

    def observe(self, chunk) -> None:
        extra = getattr(chunk, "model_extra", None) or {}
        if extra.get("provider"):
            self.provider = extra["provider"]
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            u_extra = getattr(usage, "model_extra", None) or {}
            if u_extra.get("cost") is not None:
                self.cost = u_extra["cost"]


class ToolTracer:
    """Records which tools the model chose, and the manifest it chose from.

    The point of this is measurement rather than debugging. With a large MCP
    server attached, the two things worth knowing are whether the model picks
    the right tool out of a manifest that size, and what carrying the manifest
    costs per turn, since every schema is re-sent as input on every call.

    Installing a callback handler replaces the default one, which is what
    prints streamed text, so this echoes `data` deltas to keep that behaviour.
    """

    def __init__(self, echo: bool = True) -> None:
        self.calls: list[tuple[str, object]] = []
        self.manifest: list | None = None
        self.echo = echo
        self._seen: set[str] = set()

    def __call__(self, **kw) -> None:
        if self.echo:
            data = kw.get("data")
            if data:
                print(data, end="", flush=True)

        if self.manifest is None:
            config = kw.get("tool_config")
            if config:
                self.manifest = _manifest_tools(config)

        message = kw.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                use = block.get("toolUse")
                # The same message is emitted more than once per turn, so
                # dedupe on the id rather than counting a call twice.
                if use and use.get("name") and use.get("toolUseId") not in self._seen:
                    self._seen.add(use.get("toolUseId"))
                    self.calls.append((use["name"], use.get("input")))

    def summary(self) -> str:
        lines = []
        if self.manifest is not None:
            import json

            size = len(json.dumps(self.manifest, default=str))
            # Schema JSON tokenizes at roughly 4 characters per token; this is
            # an order-of-magnitude figure, not an accounting one.
            lines.append(
                f"manifest: {len(self.manifest)} tools, {size} chars "
                f"(~{size // 4} tokens re-sent per turn)"
            )
        if self.calls:
            lines.append(f"tool calls ({len(self.calls)}):")
            for i, (name, args) in enumerate(self.calls, 1):
                rendered = str(args)
                if len(rendered) > 120:
                    rendered = rendered[:117] + "..."
                lines.append(f"  {i}. {name} {rendered}")
        else:
            lines.append("tool calls: none")
        return "\n".join(lines)


def _manifest_tools(config) -> list:
    """Pull the tool list out of a tool_config of whichever shape."""
    if isinstance(config, dict):
        for key in ("tools", "toolSpecs", "tool_specs"):
            if isinstance(config.get(key), list):
                return config[key]
        return [config]
    if isinstance(config, list):
        return config
    return []


def instrumented_client(api_key: str, tap: UsageTap):
    """An AsyncOpenAI client that tees streamed chunks through `tap`.

    This wraps a method on the SDK client rather than using a public hook,
    because no public hook exposes unmodelled response fields. If an openai or
    strands upgrade changes how the stream is consumed, the symptom is a
    missing exact cost and a fallback to the estimate, not a broken agent.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)
    original = client.chat.completions.create

    async def create(*args, **kwargs):
        response = await original(*args, **kwargs)
        if not kwargs.get("stream"):
            tap.observe(response)
            return response

        async def teed():
            async for chunk in response:
                tap.observe(chunk)
                yield chunk

        return teed()

    client.chat.completions.create = create
    return client


# --- Wiring ---------------------------------------------------------------
def load_key(env: dict | None = None, key_file: pathlib.Path | None = None) -> str:
    env = os.environ if env is None else env
    key_file = KEY_FILE if key_file is None else key_file
    key = env.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    if key_file.is_file():
        key = key_file.read_text().strip()
        if key:
            return key
    raise SystemExit(
        f"No OpenRouter key. Set OPENROUTER_API_KEY or write one to {key_file}.\n"
        "Create a key at https://openrouter.ai/keys"
    )


def build_agent(
    *,
    use_tools: bool = True,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    provider_order=None,
    window: int = DEFAULT_WINDOW,
    extra_tools=None,
    tracer: ToolTracer | None = None,
) -> tuple[Agent, UsageTap]:
    # `usage.include` is what makes OpenRouter report an exact cost; both it and
    # provider routing are OpenRouter body fields rather than OpenAI inference
    # parameters, so they ride along in extra_body.
    extra_body: dict = {"usage": {"include": True}}
    order = provider_order if provider_order is not None else PROVIDER_ORDER
    if order:
        extra_body["provider"] = {"order": order, "allow_fallbacks": True}

    tap = UsageTap()
    model = OpenAIModel(
        client=instrumented_client(load_key(), tap),
        model_id=MODEL_ID,
        params={
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": extra_body,
        },
    )
    tools = [list_directory, read_text_file, calculate, current_time] if use_tools else []
    tools += list(extra_tools or [])
    kwargs = {}
    if tracer is not None:
        # Only override the default handler when tracing, since doing so is
        # what takes over printing the streamed output.
        kwargs["callback_handler"] = tracer
    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=compose_system_prompt(bool(extra_tools)),
        conversation_manager=SlidingWindowConversationManager(window_size=window),
        **kwargs,
    )
    return agent, tap


def format_report(usage: dict, tap: UsageTap) -> str:
    """One line of token accounting, exact where OpenRouter told us the cost."""
    tin = usage.get("inputTokens", 0)
    tout = usage.get("outputTokens", 0)
    total = usage.get("totalTokens", 0)
    where = f" via {tap.provider}" if tap.provider else ""
    if tap.cost is not None:
        money = f"${tap.cost:.6f}"
    else:
        est = tin / 1e6 * RATE_IN + tout / 1e6 * RATE_OUT
        money = f"<=${est:.6f} est"
    return f"[tokens in={tin} out={tout} total={total}  {money}{where}]"


def report(result, tap: UsageTap) -> None:
    try:
        print("\n" + format_report(result.metrics.accumulated_usage, tap), file=sys.stderr)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Strands agent on Ripple 65B via OpenRouter"
    )
    ap.add_argument("prompt", nargs="*", help="prompt; omit for an interactive REPL")
    ap.add_argument("--no-tools", action="store_true", help="disable the tool loop")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"turns kept in context (default {DEFAULT_WINDOW})",
    )
    ap.add_argument(
        "--provider",
        action="append",
        help="pin an OpenRouter provider; repeatable, order matters",
    )
    ap.add_argument(
        "--any-provider",
        action="store_true",
        help="let OpenRouter route freely instead of pinning",
    )
    g = ap.add_argument_group("KiCad MCP server")
    g.add_argument(
        "--kicad",
        metavar="DIR",
        help="path to a built mixelpixx/KiCAD-MCP-Server checkout",
    )
    g.add_argument(
        "--kicad-mode",
        default=mcp_kicad.DEFAULT_MODE,
        choices=mcp_kicad.OPERATING_MODES,
        help="anything but readonly lets the model modify real design files",
    )
    g.add_argument(
        "--kicad-toolboxes",
        default=mcp_kicad.DEFAULT_TOOLBOXES,
        help="server-side tool groups; 'core' is ~37 tools, 'all' is 184+",
    )
    g.add_argument(
        "--kicad-tool",
        action="append",
        metavar="NAME",
        help="narrow to these tool names, client side; repeatable",
    )
    g.add_argument(
        "--kicad-log-level",
        default=mcp_kicad.DEFAULT_LOG_LEVEL,
        choices=mcp_kicad.LOG_LEVELS,
        help="server log verbosity; info prints a line per tool group to stderr",
    )
    g.add_argument("--node", default="node", help="node binary to launch the server with")
    s = ap.add_argument_group("ngspice MCP server")
    s.add_argument(
        "--spice",
        action="store_true",
        help="attach this repo's ngspice MCP server (board-level SPICE analysis)",
    )
    s.add_argument(
        "--spice-root",
        action="append",
        metavar="DIR",
        help="directory the server may read netlists from and write under; repeatable "
        "(default: the sandbox root)",
    )
    s.add_argument(
        "--spice-tool",
        action="append",
        metavar="NAME",
        help="narrow to these tool names, without the spice_ prefix (run_ac, not spice_run_ac)",
    )
    s.add_argument("--ngspice", metavar="PATH", help="ngspice binary (default: auto-detect)")
    e = ap.add_argument_group("openEMS MCP server (third party, AGPL-3.0)")
    e.add_argument(
        "--openems",
        metavar="DIR",
        help="path to an installed RFingAdam/mcp-openems checkout (see mcp_openems.py)",
    )
    e.add_argument(
        "--openems-tool",
        action="append",
        metavar="NAME",
        help="narrow to these tool names as listed, e.g. openems_create_microstrip; repeatable",
    )
    ap.add_argument(
        "--trace-tools",
        action="store_true",
        help="report the tool manifest size and every tool the model chose",
    )
    args = ap.parse_args()

    order = [] if args.any_provider else args.provider

    with contextlib.ExitStack() as stack:
        extra_tools = []
        if args.kicad:
            try:
                client = mcp_kicad.kicad_mcp_client(
                    args.kicad,
                    node=args.node,
                    mode=args.kicad_mode,
                    toolboxes=args.kicad_toolboxes,
                    log_level=args.kicad_log_level,
                    allowed_tools=args.kicad_tool,
                )
            except mcp_kicad.KicadServerError as e:
                print(f"kicad mcp: {e}", file=sys.stderr)
                return 2
            stack.enter_context(client)
            kicad_tools = client.list_tools_sync()
            if not kicad_tools:
                # An empty manifest is never what was wanted, and the agent would
                # otherwise carry on and truthfully report that it has no tools.
                hint = (
                    " --kicad-tool matches the server's own tool names WITHOUT the "
                    "'kicad_' prefix (run_erc, not kicad_run_erc)."
                    if args.kicad_tool
                    else ""
                )
                print(f"kicad mcp: server exposed 0 tools.{hint}", file=sys.stderr)
                return 2
            extra_tools.extend(kicad_tools)
            print(
                f"kicad mcp: {len(kicad_tools)} tools "
                f"(mode={args.kicad_mode}, toolboxes={args.kicad_toolboxes})",
                file=sys.stderr,
            )

        if args.spice:
            try:
                sclient = mcp_spice.spice_mcp_client(
                    args.spice_root or [SANDBOX_ROOT],
                    ngspice=args.ngspice,
                    allowed_tools=args.spice_tool,
                )
            except mcp_spice.SpiceServerError as e:
                print(f"spice mcp: {e}", file=sys.stderr)
                return 2
            stack.enter_context(sclient)
            spice_tools = sclient.list_tools_sync()
            if not spice_tools:
                hint = (
                    " --spice-tool matches the server's own names WITHOUT the 'spice_' "
                    "prefix (run_ac, not spice_run_ac)."
                    if args.spice_tool
                    else ""
                )
                print(f"spice mcp: server exposed 0 tools.{hint}", file=sys.stderr)
                return 2
            extra_tools.extend(spice_tools)
            print(f"spice mcp: {len(spice_tools)} tools", file=sys.stderr)

        if args.openems:
            try:
                eclient = mcp_openems.openems_mcp_client(args.openems, allowed_tools=args.openems_tool)
            except mcp_openems.OpenemsServerError as err:
                print(f"openems mcp: {err}", file=sys.stderr)
                return 2
            stack.enter_context(eclient)
            em_tools = eclient.list_tools_sync()
            if not em_tools:
                hint = (
                    " --openems-tool matches the names exactly as listed (openems_create_patch)."
                    if args.openems_tool
                    else ""
                )
                print(f"openems mcp: server exposed 0 tools.{hint}", file=sys.stderr)
                return 2
            extra_tools.extend(em_tools)
            print(f"openems mcp: {len(em_tools)} tools", file=sys.stderr)

        tracer = ToolTracer() if args.trace_tools else None
        agent, tap = build_agent(
            use_tools=not args.no_tools,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            provider_order=order,
            window=args.window,
            extra_tools=extra_tools,
            tracer=tracer,
        )

        if args.prompt:
            report(agent(" ".join(args.prompt)), tap)
            if tracer:
                print("\n" + tracer.summary(), file=sys.stderr)
            return 0

        print(f"Ripple 65B via OpenRouter, sandbox root {SANDBOX_ROOT}")
        print("Ctrl-D or 'exit' to quit. Conversation state persists across turns.\n")
        while True:
            try:
                line = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if line in {"exit", "quit"}:
                return 0
            if not line:
                continue
            try:
                report(agent(line), tap)
                if tracer:
                    print(tracer.summary(), file=sys.stderr)
            except Exception as e:
                print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
            print()


if __name__ == "__main__":
    raise SystemExit(main())
