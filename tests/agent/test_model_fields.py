from pydantic import BaseModel
from drawbore.agent import agent


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def test_model_fields_default_to_none_for_deterministic_agents():
    @agent(input=In, output=Out)
    async def a(v: In) -> Out:
        return Out(y=v.x)

    assert a.spec.model is None
    assert a.spec.fallback_model is None
    assert a.spec.instructions is None


def test_model_fields_can_be_declared():
    @agent(
        input=In, output=Out,
        model="openai/gpt-4o", fallback_model="anthropic/claude-3-5-sonnet",
        instructions="Score the input.",
    )
    async def m(v: In) -> Out:
        return Out(y=v.x)

    assert m.spec.model == "openai/gpt-4o"
    assert m.spec.fallback_model == "anthropic/claude-3-5-sonnet"
    assert m.spec.instructions == "Score the input."
