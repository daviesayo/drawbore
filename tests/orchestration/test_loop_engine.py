import json
import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.orchestration import ADKEngine, ToolLoopBundle
from drawbore.llm import LLMGateway, ModelResponse
from drawbore.tools import ToolRegistry, ToolProxy, TokenIssuer, RunContext


class In(BaseModel):
    task: str


class Out(BaseModel):
    answer: str


class _Gw(LLMGateway):
    async def complete(self, request):
        return ModelResponse(output={"answer": "one-shot"}, model_used="m", raw_text="{}")


def _bundle(declared=("lookup",)):
    reg = ToolRegistry()

    async def _l(args):
        return {"v": 1}

    reg.register_tool("lookup", _l, allowed_operations=("invoke",),
                      schema={"type": "object", "properties": {}})
    issuer = TokenIssuer()
    return ToolLoopBundle(proxy=ToolProxy(reg, issuer), issuer=issuer, registry=reg,
                          declared=declared, run_ctx=RunContext(run_id="r1", step=0))


async def test_engine_routes_model_plus_tools_to_the_loop(fake_adk_model):
    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    script = [("call", "lookup", {}), ("final", json.dumps({"answer": "looped"}))]
    eng = ADKEngine(gateway=_Gw(), model_factory=lambda name: fake_adk_model(script))
    out = await eng.run_step(solver.spec, In(task="t"), tool_loop=_bundle())
    assert out.output == {"answer": "looped"}


async def test_engine_keeps_one_shot_model_on_the_gateway_path():
    @agent(name="writer", input=In, output=Out, model="fake")  # no tools
    async def writer(v: In) -> Out:
        raise AssertionError("must not run")

    eng = ADKEngine(gateway=_Gw())
    out = await eng.run_step(writer.spec, In(task="t"))
    assert out.output == {"answer": "one-shot"}      # the gateway, not the loop


async def test_fallback_model_on_a_loop_agent_resolves_a_two_attempt_chain(fake_adk_model):
    # A model+tools agent may declare a fallback_model — the loop resolves
    # through the runtime and the chain driver runs the primary first (the safe-fallback
    # semantics are covered in test_loop_fallback.py). Here the primary succeeds, so the
    # fallback is never reached; the run produces the model's final output and an audit.
    @agent(name="solver", input=In, output=Out, model="fake", fallback_model="fb", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    script = [("final", json.dumps({"answer": "looped"}))]
    eng = ADKEngine(gateway=_Gw(), model_factory=lambda name: fake_adk_model(script))
    out = await eng.run_step(solver.spec, In(task="t"), tool_loop=_bundle())
    assert out.output == {"answer": "looped"}
    assert out.model_audit is not None
    assert out.model_audit.attempts[0].outcome == "success"


def test_max_llm_calls_is_validated():
    with pytest.raises(ValueError):
        ADKEngine(gateway=_Gw(), max_llm_calls=0)
    assert ADKEngine(gateway=_Gw())._max_llm_calls == 8     # default
