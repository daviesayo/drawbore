import json

from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.llm import LLMGateway, LLMRuntime, ModelResponse
from drawbore.orchestration import ADKEngine
from drawbore.pipeline import Pipeline


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


class _GW(LLMGateway):
    async def complete(self, request):
        return ModelResponse(output={"y": 1}, model_used=request.model_chain[0],
                             raw_text=json.dumps({"y": 1}))


@agent(name="scorer", input=In, output=Out, model="openrouter/anthropic/claude-3-5-sonnet")
async def scorer(v: In) -> Out: ...


async def test_pipeline_records_model_audit_on_the_step():
    p = Pipeline(name="p", version="1.0.0")
    p.add(scorer)
    engine = ADKEngine(llm_runtime=LLMRuntime.from_gateway(_GW()))
    result = await p.run(In(x=1), engine=engine)
    assert result.status == "completed"
    step = result.audit_trace.step_records[0]
    assert step.model is not None
    assert step.model.selected_model == "anthropic/claude-3-5-sonnet"
    assert step.model_turns == 1
    # legible() renders the model summary
    text = result.audit_trace.legible()
    assert "model openrouter/anthropic/claude-3-5-sonnet" in text
    assert "selected" in text
