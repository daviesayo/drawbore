"""Tests for TestEngine — agent-name routing, delegate to ADKEngine."""

import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.orchestration import ToolLoopBundle
from drawbore.testing import TestingError, call, final
from drawbore.testing.engine import TestEngine
from drawbore.tools import ToolRegistry, ToolProxy, TokenIssuer, RunContext


class In(BaseModel):
    x: int


class Out(BaseModel):
    x: int


async def test_deterministic_agent_runs_its_real_fn():
    @agent(name="d", input=In, output=Out)
    async def d(v: In) -> Out:
        return Out(x=v.x + 1)
    eng = TestEngine(model_responses={}, loop_scripts={})
    out = await eng.run_step(d.spec, In(x=1))
    # A deterministic agent's raw fn result is returned unwrapped (the pipeline,
    # not the engine, normalizes it to a StepExecution).
    assert out == Out(x=2)


async def test_one_shot_model_agent_uses_the_agent_scoped_gateway():
    @agent(name="scorer", input=In, output=Out, model="fake")
    async def scorer(v: In) -> Out:
        raise AssertionError("one-shot model agents do not run fn")
    eng = TestEngine(model_responses={"scorer": {"x": 99}}, loop_scripts={})
    out = await eng.run_step(scorer.spec, In(x=1))
    assert out.output == {"x": 99}


async def test_two_one_shot_agents_sharing_model_route_by_name():
    @agent(name="a", input=In, output=Out, model="fake")
    async def a(v: In) -> Out: ...
    @agent(name="b", input=In, output=Out, model="fake")
    async def b(v: In) -> Out: ...
    eng = TestEngine(model_responses={"a": {"x": 1}, "b": {"x": 2}}, loop_scripts={})
    assert (await eng.run_step(a.spec, In(x=0))).output == {"x": 1}
    assert (await eng.run_step(b.spec, In(x=0))).output == {"x": 2}


async def test_missing_model_response_fails_closed():
    @agent(name="scorer", input=In, output=Out, model="fake")
    async def scorer(v: In) -> Out: ...
    eng = TestEngine(model_responses={}, loop_scripts={})
    with pytest.raises(TestingError, match="no mock model response"):
        await eng.run_step(scorer.spec, In(x=1))


async def test_model_plus_tools_agent_runs_the_scripted_loop():
    reg = ToolRegistry()
    async def lookup(args):
        return {"hit": True}
    reg.register_tool("lookup", lookup, allowed_operations=("invoke",),
                      schema={"type": "object"})
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer)
    bundle = ToolLoopBundle(
        proxy=proxy, issuer=issuer, registry=reg, declared=("lookup",),
        run_ctx=RunContext(run_id="r1", step=0),
    )

    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In, tools) -> Out: ...

    eng = TestEngine(
        model_responses={},
        loop_scripts={"solver": [call("lookup", {}), final({"x": 5})]},
    )
    out = await eng.run_step(solver.spec, In(x=1), tool_loop=bundle)
    assert out.output == {"x": 5}
    assert any(e["tool"] == "lookup" for e in proxy.log)


async def test_missing_loop_script_fails_closed():
    reg = ToolRegistry()
    reg.register_tool("lookup", lambda a: {}, allowed_operations=("invoke",), schema={"type": "object"})
    issuer = TokenIssuer(); proxy = ToolProxy(reg, issuer)
    bundle = ToolLoopBundle(proxy=proxy, issuer=issuer, registry=reg, declared=("lookup",),
                            run_ctx=RunContext(run_id="r1", step=0))

    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In, tools) -> Out: ...
    eng = TestEngine(model_responses={}, loop_scripts={})
    with pytest.raises(TestingError, match="no mock loop script"):
        await eng.run_step(solver.spec, In(x=1), tool_loop=bundle)
