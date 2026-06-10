from pydantic import BaseModel
from drawbore.agent import agent


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def test_risk_tier_and_version_default():
    @agent(input=In, output=Out)
    async def a(v: In) -> Out:
        return Out(y=v.x)

    assert a.spec.risk_tier == "low"
    assert a.spec.version == "0.0.0"


def test_risk_tier_and_version_declared():
    @agent(input=In, output=Out, risk_tier="critical", version="2.1.0")
    async def b(v: In) -> Out:
        return Out(y=v.x)

    assert b.spec.risk_tier == "critical"
    assert b.spec.version == "2.1.0"
