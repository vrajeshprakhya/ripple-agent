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
# in practice GMICloud serves these calls. It is kept first because it is both
# the cheapest and a 1M-context bf16 endpoint when healthy; allow_fallbacks
# means its being down costs a retry, not a failure.
PROVIDER_ORDER = ["PrimeIntellect", "GMICloud", "Together"]

# Per-Mtok rates used only for the local cost estimate. GMICloud and Together
# both charge this; PrimeIntellect is half, so the estimate is an upper bound.
RATE_IN, RATE_OUT = 0.20, 0.60

SYSTEM_PROMPT = """You are a concise technical assistant running inside a Strands agent loop.

You have tools for reading the local filesystem and doing arithmetic. Use them
rather than guessing: if a question is about a file or directory, read it first.
State plainly when a tool returns an error or when something is outside the
sandbox root, and never invent file contents you did not read."""

# --- Tools ----------------------------------------------------------------
# File tools are confined to SANDBOX_ROOT so the agent cannot wander the disk.
SANDBOX_ROOT = pathlib.Path(
    os.environ.get("RIPPLE_AGENT_ROOT", pathlib.Path.cwd())
).resolve()


def _resolve(path: str) -> pathlib.Path:
    raw = pathlib.Path(path)
    p = (raw if raw.is_absolute() else SANDBOX_ROOT / raw).resolve()
    if p != SANDBOX_ROOT and SANDBOX_ROOT not in p.parents:
        raise ValueError(f"path escapes sandbox root {SANDBOX_ROOT}: {path}")
    return p


@tool
def list_directory(path: str = ".") -> str:
    """List the entries in a directory, with sizes.

    Args:
        path: Directory to list, relative to the sandbox root.
    """
    try:
        d = _resolve(path)
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


@tool
def read_text_file(path: str, max_bytes: int = 20000) -> str:
    """Read a UTF-8 text file and return its contents.

    Args:
        path: File to read, relative to the sandbox root.
        max_bytes: Truncate the file after this many bytes.
    """
    try:
        f = _resolve(path)
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


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Numbers and operators only, no names or calls.

    Args:
        expression: For example "(0.10 * 23 + 0.30 * 38) / 1000000".
    """
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"error: {e}"


@tool
def current_time() -> str:
    """Return the current local date and time in ISO 8601 form."""
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# --- Wiring ---------------------------------------------------------------
def load_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    if KEY_FILE.is_file():
        key = KEY_FILE.read_text().strip()
        if key:
            return key
    sys.exit(
        f"No OpenRouter key. Set OPENROUTER_API_KEY or write one to {KEY_FILE}.\n"
        "Create a key at https://openrouter.ai/keys"
    )


def build_agent(
    *,
    use_tools: bool = True,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    provider_order=None,
) -> Agent:
    params = {"max_tokens": max_tokens, "temperature": temperature}
    order = provider_order if provider_order is not None else PROVIDER_ORDER
    if order:
        # OpenRouter reads routing preferences from a non-OpenAI body field, so it
        # rides along in extra_body rather than as a normal inference parameter.
        params["extra_body"] = {
            "provider": {"order": order, "allow_fallbacks": True}
        }

    model = OpenAIModel(
        client_args={"api_key": load_key(), "base_url": BASE_URL},
        model_id=MODEL_ID,
        params=params,
    )
    tools = [list_directory, read_text_file, calculate, current_time] if use_tools else []
    return Agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)


def report(result) -> None:
    """Print token usage, which is what OpenRouter bills on."""
    try:
        u = result.metrics.accumulated_usage
        tin = u.get("inputTokens", 0)
        tout = u.get("outputTokens", 0)
        # Upper bound: Strands does not surface which provider served the call,
        # so this assumes the dearer of the two working endpoints.
        cost = tin / 1e6 * RATE_IN + tout / 1e6 * RATE_OUT
        print(
            f"\n[tokens in={tin} out={tout} total={u.get('totalTokens', 0)}  <=${cost:.6f}]",
            file=sys.stderr,
        )
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
    agent = build_agent(
        use_tools=not args.no_tools,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        provider_order=order,
    )

    if args.prompt:
        report(agent(" ".join(args.prompt)))
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
            report(agent(line))
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        print()


if __name__ == "__main__":
    raise SystemExit(main())
