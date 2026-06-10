import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.escalation import EscalationPolicy, HasConfidence
from drawbore.pipeline import Pipeline
from drawbore.tools import ToolRegistry


class In(BaseModel):
    x: int


class Out(BaseModel):
    x: int


class Risk(BaseModel, HasConfidence):
    risk: str
    confidence: float


async def test_low_confidence_output_triggers_escalation():   # spec test 7
    @agent(name="scorer", input=In, output=Risk, model="fake")
    async def scorer(v: In) -> Risk: ...
    p = Pipeline(
        name="aml", registry=ToolRegistry(), confidence_threshold=0.8,
        on_failure=EscalationPolicy(channel="human_review", target="ops", mode="sync"),
    )
    p.add(scorer)
    async with p.test_mode(
        mock_model_responses={"scorer": {"risk": "low", "confidence": 0.2}},
    ) as test:
        result = await test.run(In(x=1))
    assert result.status == "escalated"
    assert "confidence_below_threshold" in result.reason
    assert len(result.escalations) == 1


async def test_human_approval_gate_halts_even_on_success():   # spec test 8
    @agent(name="approver", input=In, output=Out, model="fake", requires_human_approval=True)
    async def approver(v: In) -> Out: ...
    p = Pipeline(name="aml", registry=ToolRegistry())
    p.add(approver)
    async with p.test_mode(mock_model_responses={"approver": {"x": 1}}) as test:
        result = await test.run(In(x=1))
    assert result.status == "halted"
    assert result.reason == "requires_human_approval"


async def test_missing_model_response_halts_with_testing_error():   # spec test 9
    @agent(name="scorer", input=In, output=Out, model="fake")
    async def scorer(v: In) -> Out: ...
    p = Pipeline(name="aml", registry=ToolRegistry())
    p.add(scorer)
    async with p.test_mode(mock_model_responses={}) as test:
        result = await test.run(In(x=1))
    assert result.status == "halted"
    assert result.reason.startswith("testing_error")


async def test_two_agents_same_model_route_by_agent_name():   # spec test 10
    @agent(name="a", input=In, output=Out, model="shared-model")
    async def a(v: In) -> Out: ...
    @agent(name="b", input=Out, output=Out, model="shared-model")
    async def b(v: Out) -> Out: ...
    from drawbore.pipeline.binding import From
    p = Pipeline(name="aml", registry=ToolRegistry())
    p.add(a)
    p.add(b, inputs={"x": From("a.x")}, depends_on=["a"])
    async with p.test_mode(
        mock_model_responses={"a": {"x": 11}, "b": {"x": 22}},   # keyed by spec.name
    ) as test:
        result = await test.run(In(x=0))
    assert result.status == "completed"
    assert result.outputs["a"].x == 11 and result.outputs["b"].x == 22


async def test_fallback_chain_is_exercised_without_a_live_provider():   # spec test 12
    from drawbore.llm import ModelRequest

    @agent(name="scorer", input=In, output=Out, model="primary", fallback_model="backup")
    async def scorer(v: In) -> Out: ...

    captured: list = []

    def script(request: ModelRequest):
        captured.append(request.model_chain)
        return {"x": 7}

    p = Pipeline(name="aml", registry=ToolRegistry())
    p.add(scorer)
    async with p.test_mode(mock_model_responses={"scorer": script}) as test:
        result = await test.run(In(x=1))
    assert result.status == "completed" and result.outputs["scorer"].x == 7
    # The LLMRuntime resolves the declared chain (primary -> backup) and walks it ONE
    # provider attempt at a time, calling the gateway per attempt with a single-model
    # request. The first attempt succeeds, so the script is invoked exactly once for
    # "primary"; the backup is only reached if an attempt fails with a fallback-eligible reason.
    assert len(captured) == 1, "model-response script was never called"
    assert captured[0] == ("primary",), f"unexpected model_chain: {captured[0]}"

    # And a missing scripted response for a fallback-chain agent fails closed.
    p2 = Pipeline(name="aml2", registry=ToolRegistry())
    p2.add(scorer)
    async with p2.test_mode(mock_model_responses={}) as test:
        result = await test.run(In(x=1))
    assert result.status == "halted" and result.reason.startswith("testing_error")
