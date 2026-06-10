import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.orchestration import ADKEngine
from drawbore.llm import LLMGateway, ModelResponse
from drawbore.observability import semconv


class In(BaseModel):
    text: str


class Out(BaseModel):
    text: str


class FakeGateway(LLMGateway):
    """Returns a fixed structured output without any network call."""

    def __init__(self, model_used="gpt-4o-mini"):
        self._model_used = model_used

    async def complete(self, request):
        return ModelResponse(output={"text": "from-model"}, model_used=self._model_used, raw_text="{}")


async def test_model_backed_step_emits_a_chat_span_nested_under_invoke_agent(captured_spans):
    @agent(name="writer", input=In, output=Out, model="gpt-4o", fallback_model="gpt-4o-mini")
    async def writer(v: In) -> Out:  # body not called on the model path
        raise AssertionError("model-backed agent fn must not run")

    p = Pipeline(name="t")
    p.add(writer)
    result = await p.run(In(text="hi"), engine=ADKEngine(gateway=FakeGateway()))
    assert result.status == "completed"

    spans = {s.name: s for s in captured_spans.get_finished_spans()}
    assert "chat gpt-4o" in spans
    chat = spans["chat gpt-4o"]
    assert chat.attributes[semconv.GEN_AI_OPERATION_NAME] == "chat"
    assert chat.attributes[semconv.GEN_AI_REQUEST_MODEL] == "gpt-4o"
    assert chat.attributes[semconv.GEN_AI_RESPONSE_MODEL] == "gpt-4o-mini"
    # The one-shot chat span carries the declared model ref (a direct string here).
    assert chat.attributes[semconv.DRAWBORE_DECLARED_MODEL] == "gpt-4o"
    # Nested under the agent invocation.
    agent_span = spans["invoke_agent writer"]
    assert chat.parent is not None
    assert chat.parent.span_id == agent_span.context.span_id


async def test_deterministic_step_emits_no_chat_span(captured_spans):
    @agent(name="echo", input=In, output=Out)
    async def echo(v: In) -> Out:
        return Out(text=v.text)

    p = Pipeline(name="t")
    p.add(echo)
    await p.run(In(text="hi"), engine=ADKEngine(gateway=FakeGateway()))
    assert not any(s.name.startswith("chat") for s in captured_spans.get_finished_spans())
