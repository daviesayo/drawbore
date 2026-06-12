# tests/pipeline/test_on_failure_property.py
import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.escalation import EscalationPolicy


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


@agent(name="echoer", input=In, output=Out)
async def echoer(v: In) -> Out:
    return Out(y=v.x)


def test_on_failure_defaults_to_none():
    p = Pipeline("p").add(echoer)
    assert p.on_failure is None


def test_on_failure_returns_the_configured_policy():
    policy = EscalationPolicy(channel="human", target="ops", mode="sync")
    p = Pipeline("p", on_failure=policy).add(echoer)
    assert p.on_failure is policy


def test_on_failure_is_read_only():
    p = Pipeline("p").add(echoer)
    with pytest.raises(AttributeError):
        p.on_failure = EscalationPolicy(channel="human", target="ops", mode="sync")
