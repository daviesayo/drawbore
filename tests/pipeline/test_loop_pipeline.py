import json
import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.orchestration import ADKEngine
from drawbore.llm import LLMGateway, ModelResponse
from drawbore.tools import ToolRegistry


class In(BaseModel):
    task: str


class Out(BaseModel):
    answer: str


class _Gw(LLMGateway):
    async def complete(self, request):
        return ModelResponse(output={"answer": "x"}, model_used="m", raw_text="{}")


def _registry():
    reg = ToolRegistry()

    async def _lookup(args):
        return {"v": 42}

    reg.register_tool("lookup", _lookup, allowed_operations=("invoke",),
                      schema={"type": "object", "properties": {"key": {"type": "string"}}})
    return reg


async def test_loop_step_runs_end_to_end_with_audit_and_spans(captured_spans, fake_adk_model):
    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    script = [("call", "lookup", {"key": "k"}), ("final", json.dumps({"answer": "looped:42"}))]
    p = Pipeline(name="aml", registry=_registry())
    p.add(solver)
    r = await p.run(In(task="t"),
                    engine=ADKEngine(gateway=_Gw(), model_factory=lambda n: fake_adk_model(script)))
    assert r.status == "completed"
    assert r.outputs["solver"].answer == "looped:42"
    step = r.audit_trace.step_records[0]
    assert step.model_turns == 2                       # two model turns audited
    assert any("lookup" in tc for tc in step.tool_calls)   # tool call audited
    # spans: chat (x2) + execute_tool (x1) under invoke_agent
    names = [s.name for s in captured_spans.get_finished_spans()]
    assert sum(1 for n in names if n.startswith("chat")) == 2
    assert any(n == "execute_tool lookup" for n in names)


async def test_loop_tool_denial_halts_and_is_in_the_failed_step_audit(fake_adk_model):
    # 'lookup' denies 'invoke' -> the in-loop call is denied -> immediate abort -> halt,
    # and the denied call is in the failed step's audit (not just spans/log).
    reg = ToolRegistry()

    async def _l(args):
        return {"v": 1}

    reg.register_tool("lookup", _l, allowed_operations=("read",),   # denies invoke
                      schema={"type": "object", "properties": {}})

    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    script = [("call", "lookup", {}), ("final", json.dumps({"answer": "x"}))]
    p = Pipeline(name="aml", registry=reg)
    p.add(solver)
    r = await p.run(In(task="t"),
                    engine=ADKEngine(gateway=_Gw(), model_factory=lambda n: fake_adk_model(script)))
    assert r.status == "halted"
    failed = [s for s in r.audit_trace.step_records if s.status == "failed"]
    assert len(failed) == 1
    assert any("lookup" in tc and "denied" in tc for tc in failed[0].tool_calls)


async def test_one_shot_model_step_records_one_turn():
    @agent(name="writer", input=In, output=Out, model="fake")   # no tools
    async def writer(v: In) -> Out:
        raise AssertionError("must not run")

    p = Pipeline(name="t")
    p.add(writer)
    r = await p.run(In(task="t"), engine=ADKEngine(gateway=_Gw()))
    assert r.status == "completed"
    assert r.audit_trace.step_records[0].model_turns == 1


async def test_deterministic_step_records_zero_turns():
    @agent(name="echo", input=In, output=Out)
    async def echo(v: In) -> Out:
        return Out(answer=v.task)

    p = Pipeline(name="t")
    p.add(echo)
    r = await p.run(In(task="hi"))     # LocalEngine
    assert r.audit_trace.step_records[0].model_turns == 0
