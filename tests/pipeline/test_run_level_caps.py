"""Pipeline-level tests for run-level circuit-breaker cap kwargs.

Verifies that max_calls_per_tool, max_tool_calls_per_run, and
max_distinct_tools_per_run are accepted as runtime kwargs on Pipeline.run and
TestPipeline.run, and that the ToolProxy constructed inside _run_inner honours
them.
"""

from __future__ import annotations

from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.tools import ToolContext, ToolRegistry


# ---------------------------------------------------------------------------
# Shared models + registry
# ---------------------------------------------------------------------------


class Inp(BaseModel):
    value: int


class Out(BaseModel):
    result: int


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    async def _noop(args):
        return {"ok": True}

    reg.register_tool("svc.call", _noop, allowed_operations=["invoke"])
    return reg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_tightened_total_cap_halts_circuit_breaker():
    """Pipeline.run with max_tool_calls_per_run=1 halts on the 2nd tool call."""
    reg = _make_registry()

    @agent(name="double_caller", input=Inp, output=Out, tools=["svc.call"])
    async def double_caller(v: Inp, tools: ToolContext) -> Out:
        await tools.call("svc.call", {})  # 1st — admitted
        await tools.call("svc.call", {})  # 2nd — must be blocked by run-level cap
        return Out(result=v.value)

    p = Pipeline(name="cap-test", registry=reg)
    p.add(double_caller)
    result = await p.run(Inp(value=1), max_tool_calls_per_run=1)
    assert result.status == "halted"
    assert result.halt_code == "circuit_breaker"


async def test_default_caps_allow_normal_run():
    """Default caps (500/50) do not false-halt a normal one-step pipeline."""
    reg = _make_registry()

    @agent(name="simple", input=Inp, output=Out)
    async def simple(v: Inp) -> Out:
        return Out(result=v.value * 2)

    p = Pipeline(name="normal", registry=reg)
    p.add(simple)
    result = await p.run(Inp(value=3))
    assert result.status == "completed"
    assert result.outputs["simple"].result == 6


async def test_tightened_cap_via_test_mode():
    """TestPipeline.run forwards max_tool_calls_per_run to Pipeline.run."""
    reg = _make_registry()

    @agent(name="tm_double", input=Inp, output=Out, tools=["svc.call"])
    async def tm_double(v: Inp, tools: ToolContext) -> Out:
        await tools.call("svc.call", {})  # 1st — admitted
        await tools.call("svc.call", {})  # 2nd — blocked
        return Out(result=v.value)

    p = Pipeline(name="tm-cap-test", registry=reg)
    p.add(tm_double)

    async with p.test_mode(mock_tools={"svc.call": {"ok": True}}) as test:
        result = await test.run(Inp(value=1), max_tool_calls_per_run=1)

    assert result.status == "halted"
    assert result.halt_code == "circuit_breaker"
