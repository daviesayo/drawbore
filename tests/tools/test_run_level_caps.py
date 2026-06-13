"""Run-level circuit-breaker cap tests.

Tests for the two always-on run-level caps added to ToolProxy:
  - max_tool_calls_per_run  (total admitted calls in a run)
  - max_distinct_tools_per_run  (distinct tool refs admitted in a run)

Both caps reuse CircuitBreakerError / denied:breaker / circuit_breaker halt code.
"""

from __future__ import annotations

import types

import pytest

from drawbore.tools import ToolProxy, ToolRegistry, TokenIssuer
from drawbore.tools.errors import CircuitBreakerError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ctx(run_id: str = "r1", step: int = 0):
    return types.SimpleNamespace(run_id=run_id, step=step)


async def _echo(args):
    return {"echo": args}


def _setup(
    *,
    max_calls_per_tool: int = 3,
    max_tool_calls_per_run: int = 500,
    max_distinct_tools_per_run: int = 50,
    tool_names: tuple[str, ...] = ("tool.a",),
):
    """Return (registry, issuer, proxy) with all requested tools registered."""
    reg = ToolRegistry()
    for name in tool_names:
        reg.register_tool(name, _echo, allowed_operations=["invoke"])
    issuer = TokenIssuer()
    proxy = ToolProxy(
        reg,
        issuer,
        max_calls_per_tool=max_calls_per_tool,
        max_tool_calls_per_run=max_tool_calls_per_run,
        max_distinct_tools_per_run=max_distinct_tools_per_run,
    )
    return reg, issuer, proxy


async def _admit(proxy, issuer, tool: str, *, run_id: str = "r1", step: int = 0):
    """Issue a token and make one admitted call. Returns the result."""
    tok = issuer.issue(tool, run_id)
    return await proxy.invoke(tool, {}, tok, _ctx(run_id=run_id, step=step))


# ---------------------------------------------------------------------------
# max_tool_calls_per_run tests
# ---------------------------------------------------------------------------


async def test_run_total_cap_first_two_succeed(run_id="r1"):
    """cap=2: the 1st and 2nd admitted calls succeed."""
    _, issuer, proxy = _setup(max_tool_calls_per_run=2)
    await _admit(proxy, issuer, "tool.a")
    await _admit(proxy, issuer, "tool.a", step=1)
    assert proxy._run_total == 2


async def test_run_total_cap_third_call_raises():
    """cap=2: the 3rd admitted call raises CircuitBreakerError."""
    _, issuer, proxy = _setup(max_tool_calls_per_run=2)
    await _admit(proxy, issuer, "tool.a", step=0)
    await _admit(proxy, issuer, "tool.a", step=1)
    tok = issuer.issue("tool.a", "r1")
    with pytest.raises(CircuitBreakerError) as exc_info:
        await proxy.invoke("tool.a", {}, tok, _ctx(step=2))
    assert "max-tool-calls-per-run" in str(exc_info.value)
    assert "2" in str(exc_info.value)


async def test_run_total_cap_denied_entry_is_breaker():
    """A run-level total cap denial logs denied:breaker."""
    _, issuer, proxy = _setup(max_tool_calls_per_run=2)
    await _admit(proxy, issuer, "tool.a", step=0)
    await _admit(proxy, issuer, "tool.a", step=1)
    tok = issuer.issue("tool.a", "r1")
    with pytest.raises(CircuitBreakerError):
        await proxy.invoke("tool.a", {}, tok, _ctx(step=2))
    # The 3rd log entry is the denied call
    assert proxy.log[2]["result"] == "denied:breaker"


async def test_run_total_cap_does_not_advance_on_breach():
    """A run-level denial must NOT advance _run_total (check-before-commit)."""
    _, issuer, proxy = _setup(max_tool_calls_per_run=2)
    await _admit(proxy, issuer, "tool.a", step=0)
    await _admit(proxy, issuer, "tool.a", step=1)
    tok = issuer.issue("tool.a", "r1")
    with pytest.raises(CircuitBreakerError):
        await proxy.invoke("tool.a", {}, tok, _ctx(step=2))
    assert proxy._run_total == 2  # unchanged from before the breach


# ---------------------------------------------------------------------------
# max_distinct_tools_per_run tests
# ---------------------------------------------------------------------------


async def test_distinct_cap_trips_on_third_distinct_tool():
    """cap=2: the 3rd DISTINCT tool raises CircuitBreakerError."""
    _, issuer, proxy = _setup(
        max_distinct_tools_per_run=2,
        tool_names=("tool.a", "tool.b", "tool.c"),
    )
    await _admit(proxy, issuer, "tool.a", step=0)
    await _admit(proxy, issuer, "tool.b", step=1)
    tok = issuer.issue("tool.c", "r1")
    with pytest.raises(CircuitBreakerError) as exc_info:
        await proxy.invoke("tool.c", {}, tok, _ctx(step=2))
    assert "max-distinct-tools-per-run" in str(exc_info.value)
    assert "tool.c" in str(exc_info.value)


async def test_distinct_cap_repeated_calls_to_seen_tool_dont_trip():
    """Repeated calls to an ALREADY-SEEN tool do not trip the distinct cap."""
    _, issuer, proxy = _setup(
        max_distinct_tools_per_run=2,
        tool_names=("tool.a", "tool.b"),
    )
    await _admit(proxy, issuer, "tool.a", step=0)
    await _admit(proxy, issuer, "tool.b", step=1)
    # Calling tool.a again: already seen, should not trip distinct cap.
    # (It will trip the per-step breaker at the 4th call to tool.a in one step,
    #  but here we use different steps to avoid per-step limits.)
    await _admit(proxy, issuer, "tool.a", step=2)
    assert len(proxy._run_tools) == 2


async def test_distinct_cap_does_not_advance_on_breach():
    """A distinct-cap denial must NOT add the tool to _run_tools (check-before-commit)."""
    _, issuer, proxy = _setup(
        max_distinct_tools_per_run=2,
        tool_names=("tool.a", "tool.b", "tool.c"),
    )
    await _admit(proxy, issuer, "tool.a", step=0)
    await _admit(proxy, issuer, "tool.b", step=1)
    tok = issuer.issue("tool.c", "r1")
    with pytest.raises(CircuitBreakerError):
        await proxy.invoke("tool.c", {}, tok, _ctx(step=2))
    assert "tool.c" not in proxy._run_tools
    assert proxy._run_total == 2  # total also unchanged (distinct breach raised before commit)


# ---------------------------------------------------------------------------
# Independence tests
# ---------------------------------------------------------------------------


async def test_per_step_breaker_still_independent():
    """Per-step breaker still trips even with generous run-level caps.

    4th call to one tool in one step trips max_calls_per_tool=3 breaker,
    regardless of the run-level caps.
    """
    _, issuer, proxy = _setup(
        max_calls_per_tool=3,
        max_tool_calls_per_run=500,
        max_distinct_tools_per_run=50,
    )
    # 3 admitted calls to tool.a in step 0
    for _ in range(3):
        await _admit(proxy, issuer, "tool.a", step=0)
    # 4th call to tool.a in step 0 must trip per-step breaker
    tok = issuer.issue("tool.a", "r1")
    with pytest.raises(CircuitBreakerError) as exc_info:
        await proxy.invoke("tool.a", {}, tok, _ctx(step=0))
    # per-step message (not run-level)
    assert "max-tool-calls-per-run" not in str(exc_info.value)
    assert "max-distinct-tools-per-run" not in str(exc_info.value)


async def test_run_level_only_trips_three_different_tools_once_each():
    """Run-level total cap=2 trips when 3 distinct tools are each called once.

    Every per-step count is 1, so the per-step breaker never fires.
    The 3rd call trips run-level.
    """
    _, issuer, proxy = _setup(
        max_calls_per_tool=3,
        max_tool_calls_per_run=2,
        max_distinct_tools_per_run=50,
        tool_names=("tool.a", "tool.b", "tool.c"),
    )
    await _admit(proxy, issuer, "tool.a", step=0)
    await _admit(proxy, issuer, "tool.b", step=1)
    tok = issuer.issue("tool.c", "r1")
    with pytest.raises(CircuitBreakerError) as exc_info:
        await proxy.invoke("tool.c", {}, tok, _ctx(step=2))
    assert "max-tool-calls-per-run" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Denied calls don't count
# ---------------------------------------------------------------------------


async def test_per_step_denied_call_does_not_advance_run_total():
    """A per-step-denied call does NOT increment _run_total.

    Drive a per-step denial (4th call to same tool in same step with cap=3),
    then assert the run-level total reflects only admitted calls.
    """
    _, issuer, proxy = _setup(
        max_calls_per_tool=3,
        max_tool_calls_per_run=500,
        max_distinct_tools_per_run=50,
    )
    # 3 admitted calls
    for _ in range(3):
        await _admit(proxy, issuer, "tool.a", step=0)
    assert proxy._run_total == 3

    # 4th call: per-step breaker fires BEFORE the run-level block
    tok = issuer.issue("tool.a", "r1")
    with pytest.raises(CircuitBreakerError):
        await proxy.invoke("tool.a", {}, tok, _ctx(step=0))

    # run_total must still be 3 (the denied call did not count)
    assert proxy._run_total == 3


# ---------------------------------------------------------------------------
# Generous defaults don't false-halt a normal run
# ---------------------------------------------------------------------------


async def test_generous_defaults_allow_many_calls():
    """Default caps (500 / 50) don't trip on a realistic number of calls."""
    # 10 distinct tools, 5 calls each = 50 total — well under 500/50
    tool_names = tuple(f"tool.{i}" for i in range(10))
    _, issuer, proxy = _setup(tool_names=tool_names)  # defaults 500/50
    for step_idx, name in enumerate(tool_names):
        for call_n in range(5):
            tok = issuer.issue(name, "r1")
            await proxy.invoke(name, {}, tok, _ctx(step=step_idx * 10 + call_n))
    assert proxy._run_total == 50
    assert len(proxy._run_tools) == 10


# ---------------------------------------------------------------------------
# Optional: ValueError if cap < 1
# ---------------------------------------------------------------------------


def test_value_error_if_max_tool_calls_per_run_less_than_one():
    reg = ToolRegistry()
    issuer = TokenIssuer()
    with pytest.raises(ValueError, match="max_tool_calls_per_run"):
        ToolProxy(reg, issuer, max_tool_calls_per_run=0)


def test_value_error_if_max_distinct_tools_per_run_less_than_one():
    reg = ToolRegistry()
    issuer = TokenIssuer()
    with pytest.raises(ValueError, match="max_distinct_tools_per_run"):
        ToolProxy(reg, issuer, max_distinct_tools_per_run=0)
