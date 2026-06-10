from pydantic import BaseModel
from drawbore.agent import agent


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def test_tools_default_empty():
    @agent(input=In, output=Out)
    async def a(v: In) -> Out:
        return Out(y=v.x)

    assert a.spec.tools == ()


def test_tools_declared_stored_as_tuple():
    @agent(input=In, output=Out, tools=["db.read", "db.write"])
    async def a(v: In, tools) -> Out:
        return Out(y=v.x)

    assert a.spec.tools == ("db.read", "db.write")
