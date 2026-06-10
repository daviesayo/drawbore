import json
import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.orchestration.adk_loop import run_agentic_loop
from drawbore.orchestration.engine import ToolLoopBundle
from drawbore.tools import ToolRegistry, ToolProxy, TokenIssuer, RunContext


class In(BaseModel):
    task: str


class Out(BaseModel):
    answer: str


def _wiring():
    reg = ToolRegistry()

    async def _lookup(args):
        return {"value": 42, "for": args.get("key")}

    reg.register_tool("lookup", _lookup, allowed_operations=("invoke",),
                      schema={"type": "object", "properties": {"key": {"type": "string"}},
                              "required": ["key"]})
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer)
    bundle = ToolLoopBundle(proxy=proxy, issuer=issuer, registry=reg,
                            declared=("lookup",), run_ctx=RunContext(run_id="r1", step=0))
    return reg, proxy, bundle


async def test_loop_calls_a_tool_then_returns_final_json(captured_spans, fake_adk_model):
    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("model-backed fn must not run")

    script = [("call", "lookup", {"key": "x"}), ("final", json.dumps({"answer": "done:42"}))]
    reg, proxy, bundle = _wiring()
    output, model_turns = await run_agentic_loop(
        solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
        model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
    )
    assert output == {"answer": "done:42"}
    assert model_turns == 2
    # the tool call went through the proxy (logged ok)
    assert any(e["tool"] == "lookup" and e["result"] == "ok" for e in proxy.log)
    # model turns traced as chat spans
    assert sum(1 for s in captured_spans.get_finished_spans() if s.name.startswith("chat")) == 2


async def test_loop_handles_two_sequential_tool_calls(fake_adk_model):
    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    script = [("call", "lookup", {"key": "a"}), ("call", "lookup", {"key": "b"}),
              ("final", json.dumps({"answer": "two"}))]
    reg, proxy, bundle = _wiring()
    output, model_turns = await run_agentic_loop(
        solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
        model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
    )
    assert output == {"answer": "two"}
    assert model_turns == 3
    assert sum(1 for e in proxy.log if e["tool"] == "lookup" and e["result"] == "ok") == 2
