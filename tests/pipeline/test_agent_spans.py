import pytest
from pydantic import BaseModel

from opentelemetry.trace import StatusCode

from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.tools import ToolContext, ToolRegistry
from drawbore.observability import semconv


class In(BaseModel):
    text: str


class Out(BaseModel):
    text: str


async def test_each_step_emits_an_invoke_agent_span(captured_spans):
    @agent(name="echo", input=In, output=Out, version="1.2.3", risk_tier="medium")
    async def echo(v: In) -> Out:
        return Out(text=v.text)

    p = Pipeline(name="t")
    p.add(echo)
    await p.run(In(text="hi"), tenant_id="acme")

    spans = [s for s in captured_spans.get_finished_spans() if s.name.startswith("invoke_agent")]
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent echo"
    assert span.attributes[semconv.GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert span.attributes[semconv.GEN_AI_AGENT_NAME] == "echo"
    assert span.attributes[semconv.GEN_AI_AGENT_VERSION] == "1.2.3"
    assert span.attributes[semconv.DRAWBORE_RISK_TIER] == "medium"
    assert span.attributes[semconv.DRAWBORE_TENANT_ID] == "acme"
    assert span.attributes[semconv.DRAWBORE_STEP] == 0
    assert semconv.DRAWBORE_INPUT_HASH in span.attributes


async def test_a_failing_step_marks_its_invoke_agent_span_error(captured_spans):
    @agent(name="boom", input=In, output=Out)
    async def boom(v: In) -> Out:
        raise RuntimeError("kaboom")

    p = Pipeline(name="t")
    p.add(boom)
    result = await p.run(In(text="hi"))
    assert result.status == "halted"
    span = [s for s in captured_spans.get_finished_spans() if s.name == "invoke_agent boom"][0]
    assert span.status.status_code == StatusCode.ERROR
    assert "kaboom" in (span.status.description or "")


async def test_tool_span_nests_under_its_invoke_agent_span(captured_spans):
    registry = ToolRegistry()

    async def _tool(args):
        return {"ok": True}

    registry.register_tool("noop", _tool)

    @agent(name="caller", input=In, output=Out, tools=["noop"])
    async def caller(v: In, tools: ToolContext) -> Out:
        await tools.call("noop", {"a": 1})
        return Out(text=v.text)

    p = Pipeline(name="t", registry=registry)
    p.add(caller)
    await p.run(In(text="hi"))

    spans = {s.name: s for s in captured_spans.get_finished_spans()}
    agent_span = spans["invoke_agent caller"]
    tool_span = spans["execute_tool noop"]
    # The tool call happened inside the agent invocation: same trace, child span.
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span.context.span_id
    assert tool_span.context.trace_id == agent_span.context.trace_id
