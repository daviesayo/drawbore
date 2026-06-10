import pytest
from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.tools import ToolRegistry, ToolContext, ToolAccessError


class In(BaseModel):
    id: int


class Out(BaseModel):
    value: int


def _registry():
    reg = ToolRegistry()

    async def read(args):
        return {"n": args["id"] * 10}

    async def write(args):
        return {"ok": True}

    reg.register_tool("db.read", read)
    reg.register_tool("db.write", write)
    return reg


async def test_agent_uses_tool_through_proxy_end_to_end():
    reg = _registry()

    @agent(input=In, output=Out, tools=["db.read"])
    async def fetch(value: In, tools: ToolContext) -> Out:
        row = await tools.call("db.read", {"id": value.id})
        return Out(value=row["n"])

    p = Pipeline(name="t", registry=reg)
    p.add(fetch)
    result = await p.run(In(id=4))
    assert result.status == "completed"
    assert result.outputs["fetch"] == Out(value=40)


async def test_undeclared_tool_call_halts():
    reg = _registry()

    @agent(input=In, output=Out, tools=["db.read"])
    async def bad(value: In, tools: ToolContext) -> Out:
        await tools.call("db.write", {})  # registered, but NOT declared by this agent
        return Out(value=0)

    p = Pipeline(name="t", registry=reg)
    p.add(bad)
    result = await p.run(In(id=1))
    assert result.status == "halted"
    assert result.halted_at == "bad"
    assert "tool_access" in result.reason


async def test_circuit_breaker_halts_the_run():
    reg = _registry()

    @agent(input=In, output=Out, tools=["db.read"])
    async def spammer(value: In, tools: ToolContext) -> Out:
        for _ in range(4):  # default max is 3
            await tools.call("db.read", {"id": value.id})
        return Out(value=0)

    p = Pipeline(name="t", registry=reg)
    p.add(spammer)
    result = await p.run(In(id=1))
    assert result.status == "halted"
    assert "circuit_breaker" in result.reason


def test_declaring_unregistered_tool_raises_at_add():
    reg = ToolRegistry()

    @agent(input=In, output=Out, tools=["ghost.tool"])
    async def a(value: In, tools: ToolContext) -> Out:
        return Out(value=0)

    p = Pipeline(name="t", registry=reg)
    with pytest.raises(ToolAccessError):
        p.add(a)
