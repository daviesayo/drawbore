import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.testing import call, final, multi_call
from drawbore.tools import ToolRegistry


class In(BaseModel):
    x: int


class Out(BaseModel):
    answer: str


def _reg(*, allowed=("invoke",)):
    reg = ToolRegistry()
    async def lookup(args):
        return {"hit": True}
    reg.register_tool("lookup", lookup, allowed_operations=allowed, schema={"type": "object"})
    return reg


def _solver(name="solver"):
    @agent(name=name, input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In, tools) -> Out: ...
    return solver


def _step(trace, agent_name):
    return [s for s in trace.step_records if s.agent == agent_name][0]


async def test_loop_happy_path_calls_a_tool_and_returns_final_json():   # spec test 13
    p = Pipeline(name="aml", registry=_reg())
    p.add(_solver())
    async with p.test_mode(
        mock_tools={"lookup": {"hit": True}},
        mock_loop_scripts={"solver": [call("lookup", {"id": "1"}), final({"answer": "ok"})]},
    ) as test:
        result = await test.run(In(x=1))
    assert result.status == "completed"
    assert result.outputs["solver"].answer == "ok"
    step = _step(result.audit_trace, "solver")
    assert any("lookup (invoke) -> ok" in tc for tc in step.tool_calls)
    assert step.model_turns == 2     # the call turn + the final turn


async def test_loop_tool_denial_halts_with_a_failed_step_audit():   # spec test 14
    # original allows only 'read'; the loop calls the default 'invoke' -> denied.
    p = Pipeline(name="aml", registry=_reg(allowed=("read",)))
    p.add(_solver())
    async with p.test_mode(
        mock_tools={"lookup": {"hit": True}},
        mock_loop_scripts={"solver": [call("lookup", {}), final({"answer": "x"})]},
    ) as test:
        result = await test.run(In(x=1))
    assert result.status == "halted"
    assert result.reason.startswith("tool_access")
    step = _step(result.audit_trace, "solver")
    assert step.status != "ok"
    assert any("denied:scope" in tc for tc in step.tool_calls)


async def test_loop_multi_call_turn_fails_closed():   # spec test 15
    p = Pipeline(name="aml", registry=_reg())
    p.add(_solver())
    async with p.test_mode(
        mock_tools={"lookup": {"hit": True}},
        mock_loop_scripts={"solver": [multi_call(call("lookup", {}), call("lookup", {}))]},
    ) as test:
        result = await test.run(In(x=1))
    assert result.status == "halted"
    assert result.reason.startswith("model_error")   # LLMError on multi-call turn
    assert "solver" not in result.outputs             # loop aborted before any output


async def test_loop_exhaustion_halts_at_the_loop_bound():   # spec test 16
    # No final answer; a low max_llm_calls (2) hits the loop bound BEFORE the
    # per-tool breaker (3) would trip.
    p = Pipeline(name="aml", registry=_reg())
    p.add(_solver())
    async with p.test_mode(
        mock_tools={"lookup": {"hit": True}},
        mock_loop_scripts={"solver": [call("lookup", {}), call("lookup", {}), call("lookup", {})]},
        max_llm_calls=2,
    ) as test:
        result = await test.run(In(x=1))
    assert result.status == "halted"
    assert result.reason.startswith("model_error")   # loop bound -> LLMError
    assert "loop" in result.reason                    # tighten: loop-bound path, not {}-fallback
    step = [s for s in result.audit_trace.step_records if s.agent == "solver"][0]
    assert step.model_turns == 3                      # before_model counted the 3rd (limit) call


async def test_two_loop_agents_same_model_route_by_name():   # spec test 11
    from drawbore.pipeline.binding import From

    class Mid(BaseModel):
        answer: str

    reg = _reg()
    @agent(name="a", input=In, output=Mid, model="fake", tools=["lookup"])
    async def a(v: In, tools) -> Mid: ...
    @agent(name="b", input=Mid, output=Out, model="fake", tools=["lookup"])
    async def b(v: Mid, tools) -> Out: ...

    p = Pipeline(name="aml", registry=reg)
    p.add(a)
    p.add(b, inputs={}, depends_on=["a"])   # b is fed nothing structural; mocked output stands alone

    async with p.test_mode(
        mock_tools={"lookup": {"hit": True}},
        mock_loop_scripts={
            "a": [call("lookup", {}), final({"answer": "from-a"})],
            "b": [call("lookup", {}), final({"answer": "from-b"})],
        },
    ) as test:
        result = await test.run(In(x=1))
    assert result.status == "completed"
    assert result.outputs["a"].answer == "from-a"
    assert result.outputs["b"].answer == "from-b"
