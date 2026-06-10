from pydantic import BaseModel
from drawbore.agent import agent


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def test_requires_human_approval_defaults_false():
    @agent(input=In, output=Out)
    async def a(v: In) -> Out:
        return Out(y=v.x)

    assert a.spec.requires_human_approval is False


def test_requires_human_approval_can_be_declared():
    @agent(input=In, output=Out, requires_human_approval=True)
    async def gated(v: In) -> Out:
        return Out(y=v.x)

    assert gated.spec.requires_human_approval is True
