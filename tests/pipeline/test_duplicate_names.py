import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.pipeline import Pipeline


class In(BaseModel):
    x: int


class Out(BaseModel):
    x: int


def _echo(name):
    @agent(name=name, input=In, output=Out)
    async def fn(v: In) -> Out:
        return Out(x=v.x)
    return fn


def test_add_rejects_a_duplicate_agent_name():
    # The runtime keys steps/outputs by name; a duplicate would overwrite state.
    p = Pipeline(name="dup")
    p.add(_echo("step"))
    with pytest.raises(ValueError, match="already has a node named 'step'"):
        p.add(_echo("step"))


def test_add_allows_distinct_agent_names():
    p = Pipeline(name="ok")
    p.add(_echo("a")).add(_echo("b"))
    assert [s.agent.name for s in p.steps] == ["a", "b"]


def test_rejected_duplicate_does_not_corrupt_pipeline_state():
    # The guard must fire before any state mutation: after a rejected add the
    # pipeline is exactly as it was — one step, the original agent, _by_name in sync.
    original = _echo("step")
    intruder = _echo("step")
    p = Pipeline(name="dup")
    p.add(original)

    with pytest.raises(ValueError):
        p.add(intruder)

    assert len(p.steps) == 1, "steps list must not grow on a rejected add"
    assert p.steps[0].agent is original, "original step must not be replaced"
    assert list(p._by_name.keys()) == ["step"], "_by_name must mirror steps"
    assert p._by_name["step"].agent is original, "_by_name entry must not be replaced"
