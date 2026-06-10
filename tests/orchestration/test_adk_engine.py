import pytest
from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.llm import LLMGateway, ModelResponse, ModelUnavailableError
from drawbore.orchestration import ADKEngine


class In(BaseModel):
    amount: int


class Out(BaseModel):
    risk: str


class _FakeGateway(LLMGateway):
    def __init__(self, output, *, model_used="m1"):
        self._output = output
        self._model_used = model_used
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(output=self._output, model_used=self._model_used,
                             raw_text="")


class _FailingGateway(LLMGateway):
    async def complete(self, request):
        raise ModelUnavailableError("all down")


async def test_deterministic_agent_runs_in_process_like_local_engine():
    @agent(name="d", input=In, output=Out)
    async def d(v: In) -> Out:
        return Out(risk="low")

    eng = ADKEngine(gateway=_FakeGateway({"risk": "ignored"}))
    raw = await eng.run_step(d.spec, In(amount=1), tools=None)
    # Deterministic agents do NOT touch the gateway — the fn output is returned.
    assert isinstance(raw, Out) and raw.risk == "low"


async def test_adk_engine_accepts_tool_loop_kwarg():
    # The ABC's run_step now declares a keyword-only `tool_loop`; ADKEngine must
    # accept it (it consumes the bundle for model+tools agents).
    @agent(name="d", input=In, output=Out)
    async def d(v: In) -> Out:
        return Out(risk="low")

    eng = ADKEngine(gateway=_FakeGateway({"risk": "ignored"}))
    raw = await eng.run_step(d.spec, In(amount=1), tool_loop=None)
    assert isinstance(raw, Out) and raw.risk == "low"


async def test_deterministic_agent_with_tools_receives_them_and_skips_gateway():
    sentinel = object()

    @agent(name="t", input=In, output=Out, tools=["x"])
    async def t(v: In, tools) -> Out:
        assert tools is sentinel          # the exact tools object is threaded in
        return Out(risk="low")

    gw = _FakeGateway({"risk": "ignored"})
    eng = ADKEngine(gateway=gw)
    raw = await eng.run_step(t.spec, In(amount=1), tools=sentinel)
    assert isinstance(raw, Out) and raw.risk == "low"
    assert gw.requests == []              # deterministic path never calls the gateway


async def test_model_backed_agent_completes_via_the_gateway():
    @agent(name="m", input=In, output=Out, model="m1", instructions="score")
    async def m(v: In) -> Out:
        raise AssertionError("model-backed agent fn must not be called by the engine")

    gw = _FakeGateway({"risk": "high"})
    eng = ADKEngine(gateway=gw)
    out = await eng.run_step(m.spec, In(amount=999), tools=None)
    assert out.output == {"risk": "high"}       # carrier output; the pipeline validates it
    assert out.model_turns == 1
    assert len(gw.requests) == 1                # the gateway was the model path
    assert gw.requests[0].model_chain == ("m1",)


async def test_model_backed_chain_includes_the_fallback():
    @agent(name="m", input=In, output=Out, model="m1", fallback_model="m2")
    async def m(v: In) -> Out:
        raise AssertionError

    gw = _FakeGateway({"risk": "low"})
    eng = ADKEngine(gateway=gw)
    out = await eng.run_step(m.spec, In(amount=1), tools=None)
    # The runtime resolves the full chain (declared refs carry both models) and
    # calls the gateway once per attempt; the first attempt succeeds, so the single
    # request targets m1 while the declared chain still records the fallback m2.
    assert out.model_audit.declared_refs == ("m1", "m2")
    assert gw.requests[0].model_chain == ("m1",)


async def test_model_unavailable_propagates_for_the_pipeline_to_halt():
    @agent(name="m", input=In, output=Out, model="m1")
    async def m(v: In) -> Out:
        raise AssertionError

    eng = ADKEngine(gateway=_FailingGateway())
    with pytest.raises(ModelUnavailableError):
        await eng.run_step(m.spec, In(amount=1), tools=None)
