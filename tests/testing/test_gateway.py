import pytest
from pydantic import BaseModel

from drawbore.llm import ModelRequest, ModelResponse
from drawbore.testing import TestingError
from drawbore.testing.gateway import FakeGateway


def _req(chain=("fake",)):
    return ModelRequest(system="s", user="u", model_chain=chain)


class Out(BaseModel):
    risk: str


async def test_dict_response_becomes_model_response():
    gw = FakeGateway(agent_name="scorer", response={"risk": "low"})
    resp = await gw.complete(_req())
    assert isinstance(resp, ModelResponse)
    assert resp.output == {"risk": "low"}
    assert resp.model_used == "fake"            # chain[0]; never inferred from prompt


async def test_pydantic_model_response_is_dumped():
    gw = FakeGateway(agent_name="scorer", response=Out(risk="high"))
    resp = await gw.complete(_req())
    assert resp.output == {"risk": "high"}


async def test_sequence_is_consumed_in_order_then_fails_closed():
    gw = FakeGateway(agent_name="scorer", response=[{"risk": "low"}, {"risk": "high"}])
    assert (await gw.complete(_req())).output == {"risk": "low"}
    assert (await gw.complete(_req())).output == {"risk": "high"}
    with pytest.raises(TestingError, match="exhausted"):
        await gw.complete(_req())


async def test_callable_receives_the_model_request():
    seen = {}

    def script(request: ModelRequest):
        seen["chain"] = request.model_chain
        return {"risk": "med"}

    gw = FakeGateway(agent_name="scorer", response=script)
    resp = await gw.complete(_req(chain=("primary", "fallback")))
    assert resp.output == {"risk": "med"}
    assert seen["chain"] == ("primary", "fallback")   # can branch on the chain (test 12)


async def test_missing_response_fails_closed():
    gw = FakeGateway(agent_name="scorer", response=None)
    with pytest.raises(TestingError, match="no mock model response"):
        await gw.complete(_req())


async def test_exception_value_is_raised():
    gw = FakeGateway(agent_name="scorer", response=RuntimeError("provider down"))
    with pytest.raises(RuntimeError, match="provider down"):
        await gw.complete(_req())
