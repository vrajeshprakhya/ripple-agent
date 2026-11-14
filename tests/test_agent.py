"""Offline tests. No network, no OpenRouter key required."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import agent as A  # noqa: E402


# --- Sandbox containment --------------------------------------------------
@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "inside.txt").write_text("hello inside\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.txt").write_text("deep\n")
    return tmp_path


def test_resolves_paths_inside_root(sandbox):
    assert A.resolve_in_sandbox("inside.txt", sandbox) == sandbox / "inside.txt"
    assert A.resolve_in_sandbox("sub/deep.txt", sandbox) == sandbox / "sub" / "deep.txt"
    assert A.resolve_in_sandbox(".", sandbox) == sandbox


@pytest.mark.parametrize(
    "escape",
    [
        "..",
        "../..",
        "../outside.txt",
        "sub/../../outside.txt",
        "/etc/passwd",
        "/",
    ],
)
def test_refuses_paths_outside_root(sandbox, escape):
    with pytest.raises(ValueError, match="escapes sandbox root"):
        A.resolve_in_sandbox(escape, sandbox)


def test_refuses_symlink_pointing_outside(sandbox, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    secret = outside / "secret.txt"
    secret.write_text("should not be readable\n")
    link = sandbox / "escape_link"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    # The spelling looks contained; the resolved target is not.
    with pytest.raises(ValueError, match="escapes sandbox root"):
        A.resolve_in_sandbox("escape_link", sandbox)


def test_tools_report_escape_as_error_not_exception(sandbox):
    assert A.read_text_file_impl("/etc/passwd", root=sandbox).startswith("error:")
    assert A.list_directory_impl("..", root=sandbox).startswith("error:")


# --- File tools -----------------------------------------------------------
def test_read_text_file_reads_contents(sandbox):
    assert A.read_text_file_impl("inside.txt", root=sandbox) == "hello inside\n"


def test_read_text_file_truncates(sandbox):
    big = sandbox / "big.txt"
    big.write_text("x" * 5000)
    out = A.read_text_file_impl("big.txt", max_bytes=100, root=sandbox)
    assert out.startswith("x" * 100)
    assert "truncated at 100 of 5000 bytes" in out


def test_read_text_file_rejects_directory(sandbox):
    assert A.read_text_file_impl("sub", root=sandbox).startswith("error: not a file")


def test_read_text_file_survives_invalid_utf8(sandbox):
    (sandbox / "binary.bin").write_bytes(b"\xff\xfe\x00ok")
    out = A.read_text_file_impl("binary.bin", root=sandbox)
    assert "ok" in out  # replaced, not raised


def test_list_directory_lists_entries(sandbox):
    out = A.list_directory_impl(".", root=sandbox)
    assert "inside.txt" in out
    assert "dir " in out and "sub" in out


def test_list_directory_empty(sandbox):
    (sandbox / "empty").mkdir()
    assert A.list_directory_impl("empty", root=sandbox) == "(empty)"


def test_list_directory_rejects_file(sandbox):
    assert A.list_directory_impl("inside.txt", root=sandbox).startswith(
        "error: not a directory"
    )


# --- Calculator -----------------------------------------------------------
@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2 + 3", "5"),
        ("2 ** 40", "1099511627776"),
        ("7 // 2", "3"),
        ("7 % 2", "1"),
        ("-5 + 2", "-3"),
        ("101*0.10/1000000 + 33*0.30/1000000", "2e-05"),
    ],
)
def test_calculate_arithmetic(expr, expected):
    assert A.calculate_impl(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "os.getcwd()",
        "[1,2,3]",
        "lambda: 1",
        "1 if True else 2",
        "'a' * 3",
    ],
)
def test_calculate_refuses_non_arithmetic(expr):
    assert A.calculate_impl(expr).startswith("error:")


def test_calculate_refuses_runaway_exponent():
    # Cheap to type, catastrophic to evaluate.
    out = A.calculate_impl("9 ** 9 ** 9")
    assert out.startswith("error:")
    assert "exponent" in out


def test_calculate_reports_division_by_zero():
    assert A.calculate_impl("1/0").startswith("error:")


# --- Key loading ----------------------------------------------------------
def test_env_key_wins_over_file(tmp_path):
    kf = tmp_path / "key"
    kf.write_text("from-file")
    assert A.load_key({"OPENROUTER_API_KEY": "from-env"}, kf) == "from-env"


def test_falls_back_to_key_file(tmp_path):
    kf = tmp_path / "key"
    kf.write_text("  from-file\n")
    assert A.load_key({}, kf) == "from-file"


def test_blank_env_falls_through_to_file(tmp_path):
    kf = tmp_path / "key"
    kf.write_text("from-file")
    assert A.load_key({"OPENROUTER_API_KEY": "   "}, kf) == "from-file"


def test_exits_when_no_key_anywhere(tmp_path):
    with pytest.raises(SystemExit) as e:
        A.load_key({}, tmp_path / "missing")
    assert "No OpenRouter key" in str(e.value)


def test_exits_when_key_file_is_empty(tmp_path):
    kf = tmp_path / "key"
    kf.write_text("\n")
    with pytest.raises(SystemExit):
        A.load_key({}, kf)


# --- Usage reporting ------------------------------------------------------
class _Chunk:
    """Mimics an openai chunk carrying OpenRouter's unmodelled extras."""

    def __init__(self, model_extra=None, usage=None):
        self.model_extra = model_extra or {}
        self.usage = usage


class _Usage:
    def __init__(self, model_extra):
        self.model_extra = model_extra


def test_tap_captures_provider_and_cost():
    tap = A.UsageTap()
    tap.observe(_Chunk(model_extra={"provider": "GMICloud"}))
    tap.observe(_Chunk(usage=_Usage({"cost": 0.000123})))
    assert tap.provider == "GMICloud"
    assert tap.cost == 0.000123


def test_tap_tolerates_chunks_without_extras():
    tap = A.UsageTap()
    tap.observe(_Chunk())
    tap.observe(object())  # no attributes at all
    assert tap.provider is None and tap.cost is None


def test_report_uses_exact_cost_when_known():
    tap = A.UsageTap()
    tap.provider, tap.cost = "GMICloud", 0.000274
    line = A.format_report({"inputTokens": 101, "outputTokens": 33, "totalTokens": 134}, tap)
    assert "$0.000274" in line
    assert "via GMICloud" in line
    assert "est" not in line


def test_report_falls_back_to_estimate():
    tap = A.UsageTap()  # nothing captured
    line = A.format_report({"inputTokens": 1_000_000, "outputTokens": 0, "totalTokens": 1_000_000}, tap)
    assert "<=$0.200000 est" in line  # RATE_IN on a full Mtok
    assert "via" not in line


# --- System prompt composition -------------------------------------------
def test_base_prompt_unchanged_without_mcp_tools():
    assert A.compose_system_prompt(False) == A.SYSTEM_PROMPT
    assert "MCP" not in A.compose_system_prompt(False)


def test_prompt_describes_mcp_tools_when_attached():
    p = A.compose_system_prompt(True)
    assert p.startswith(A.SYSTEM_PROMPT)
    assert "MCP server" in p


def test_addendum_exempts_mcp_tools_from_the_sandbox_rule():
    # A model told only about its sandbox applied it to tools it does not cover.
    assert "NOT covered by the sandbox" in A.compose_system_prompt(True)
