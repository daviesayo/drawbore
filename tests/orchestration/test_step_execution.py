import json

import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.llm import LLMGateway, LLMRuntime, ModelResponse
from drawbore.orchestration import ADKEngine, StepExecution


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


class _OneShotGateway(LLMGateway):
    async def complete(self, request):
        return ModelResponse(output={"y": 7}, model_used=request.model_chain[0],
                             raw_text=json.dumps({"y": 7}))


def _det_spec():
    @agent(name="det", input=In, output=Out)
    async def det(v: In) -> Out:
        return Out(y=v.x + 1)
    return det.spec


def _model_spec():
    @agent(name="m", input=In, output=Out, model="gpt-4o")
    async def m(v: In) -> Out: ...
    return m.spec


async def test_one_shot_returns_step_execution_with_audit_and_one_turn():
    engine = ADKEngine(llm_runtime=LLMRuntime.from_gateway(_OneShotGateway()))
    out = await engine.run_step(_model_spec(), In(x=1))
    assert isinstance(out, StepExecution)
    assert out.output == {"y": 7}
    assert out.model_turns == 1
    assert out.model_audit is not None
    assert out.model_audit.selected_model == "gpt-4o"


async def test_gateway_keyword_still_constructs_engine():
    # back-compat: ADKEngine(gateway=...) wraps into a from_gateway runtime
    engine = ADKEngine(gateway=_OneShotGateway())
    out = await engine.run_step(_model_spec(), In(x=1))
    assert out.output == {"y": 7}


async def test_deterministic_step_returns_raw_value_for_normalization():
    engine = ADKEngine(gateway=_OneShotGateway())
    out = await engine.run_step(_det_spec(), In(x=4))
    # deterministic agents return their raw value; the pipeline normalizes it
    assert isinstance(out, Out)
    assert out.y == 5


def test_engine_rejects_both_runtime_and_gateway():
    with pytest.raises(ValueError):
        ADKEngine(llm_runtime=LLMRuntime.from_gateway(_OneShotGateway()), gateway=_OneShotGateway())


def test_engine_rejects_neither_runtime_nor_gateway():
    with pytest.raises(ValueError):
        ADKEngine()
