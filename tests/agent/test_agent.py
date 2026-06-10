import pytest
from pydantic import BaseModel
from drawbore.agent import agent, Agent, AgentSpec


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def test_decorator_produces_agent_with_spec():
    @agent(name="doubler", input=In, output=Out)
    async def doubler(value: In) -> Out:
        return Out(y=value.x * 2)

    assert isinstance(doubler, Agent)
    assert isinstance(doubler.spec, AgentSpec)
    assert doubler.spec.name == "doubler"
    assert doubler.spec.input is In
    assert doubler.spec.output is Out
    assert doubler.spec.context_access == "none"


def test_name_defaults_to_function_name():
    @agent(input=In, output=Out)
    async def my_agent(value: In) -> Out:
        return Out(y=value.x)

    assert my_agent.spec.name == "my_agent"


async def test_agent_is_callable():
    @agent(input=In, output=Out)
    async def doubler(value: In) -> Out:
        return Out(y=value.x * 2)

    result = await doubler(In(x=3))
    assert result == Out(y=6)


def test_spec_is_frozen():
    @agent(input=In, output=Out)
    async def a(value: In) -> Out:
        return Out(y=value.x)

    with pytest.raises(Exception):
        a.spec.name = "changed"
