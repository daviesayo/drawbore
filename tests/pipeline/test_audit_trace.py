import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.tools import ToolContext, ToolRegistry
from drawbore.audit import InMemoryAuditSink


class In(BaseModel):
    text: str


class Out(BaseModel):
    text: str


async def test_completed_run_has_the_acceptance_audit_shape():
    @agent(name="a", input=In, output=Out)
    async def a(v: In) -> Out:
        return Out(text=v.text)

    @agent(name="b", input=Out, output=Out)
    async def b(v: Out) -> Out:
        return Out(text=v.text + "!")

    p = Pipeline(name="p", version="1.0.0")
    p.add(a)
    p.add(b)
    result = await p.run(In(text="hi"))
    assert result.status == "completed"
    assert result.audit_trace is not None
    assert result.audit_trace.steps == 2
    assert result.audit_trace.escalations == 0
    assert result.audit_trace.schema_violations == 0
    assert result.audit_trace.pipeline == "p"
    assert [s.agent for s in result.audit_trace.step_records] == ["a", "b"]


async def test_audit_trace_is_written_to_an_injected_sink():
    @agent(name="a", input=In, output=Out)
    async def a(v: In) -> Out:
        return Out(text=v.text)

    sink = InMemoryAuditSink()
    p = Pipeline(name="p")
    p.add(a)
    await p.run(In(text="hi"), run_id="run-1", audit=sink, tenant_id="acme")
    assert sink.by_run_id("run-1") is not None
    assert sink.by_run_id("run-1").tenant_id == "acme"


async def test_schema_violation_is_counted_and_halt_recorded():
    class Strict(BaseModel):
        n: int

    @agent(name="bad", input=In, output=Strict)
    async def bad(v: In) -> Strict:
        return {"n": "not-an-int"}  # a dict (not a Strict instance) → strict-mode output schema violation

    p = Pipeline(name="p")
    p.add(bad)
    result = await p.run(In(text="hi"))
    assert result.status == "halted"
    assert result.audit_trace.steps == 0
    assert result.audit_trace.schema_violations == 1
    assert result.audit_trace.reason.startswith("schema_violation:")


async def test_step_record_captures_tool_calls():
    registry = ToolRegistry()

    async def _tool(args):
        return {"ok": True}

    registry.register_tool("noop", _tool)

    @agent(name="caller", input=In, output=Out, tools=["noop"])
    async def caller(v: In, tools: ToolContext) -> Out:
        await tools.call("noop", {"a": 1})
        return Out(text=v.text)

    p = Pipeline(name="p", registry=registry)
    p.add(caller)
    result = await p.run(In(text="hi"))
    step = result.audit_trace.step_records[0]
    assert step.agent == "caller"
    assert any("noop" in tc for tc in step.tool_calls)
    assert result.audit_trace.legible()  # renders without error
