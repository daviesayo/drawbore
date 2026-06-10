import pytest
from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.orchestration import OrchestratorEngine, LocalEngine


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def test_engine_is_abstract():
    with pytest.raises(TypeError):
        OrchestratorEngine()  # cannot instantiate ABC with abstract run_step


async def test_local_engine_runs_a_step():
    @agent(input=In, output=Out)
    async def doubler(value: In) -> Out:
        return Out(y=value.x * 2)

    engine = LocalEngine()
    result = await engine.run_step(doubler.spec, In(x=4))
    assert result == Out(y=8)
