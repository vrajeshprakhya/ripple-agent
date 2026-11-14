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
import datetime as _dt
import operator
import os
import pathlib
import sys

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
# As of 2026-11-13 PrimeIntellect returns a provider error on every request, so
# in practice GMICloud or Together serves these calls. PrimeIntellect is kept
# first because it is both the cheapest and a 1M-context bf16 endpoint when
# healthy; allow_fallbacks means its being down costs a retry, not a failure.
PROVIDER_ORDER = ["PrimeIntellect", "GMICloud", "Together"]

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
    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        conversation_manager=SlidingWindowConversationManager(window_size=window),
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
    args = ap.parse_args()

    order = [] if args.any_provider else args.provider
    agent, tap = build_agent(
        use_tools=not args.no_tools,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        provider_order=order,
        window=args.window,
    )

    if args.prompt:
        report(agent(" ".join(args.prompt)), tap)
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
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        print()


if __name__ == "__main__":
    raise SystemExit(main())
