"""Tests for make_scripted_model_factory.

The scripted loop model drives the real ADK Runner loop with scripted
turns (no network). Lives under ``drawbore.orchestration`` because it imports
google.adk (containment invariant).
"""

import json
import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.orchestration import ADKEngine, ToolLoopBundle
from drawbore.orchestration.scripted_model import make_scripted_model_factory
from drawbore.tools import ToolRegistry, ToolProxy, TokenIssuer, RunContext


class In(BaseModel):
    task: str


class Out(BaseModel):
    answer: str


def _spec():
    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In, tools) -> Out:
        raise AssertionError("model+tools agents run via the loop, not fn")
    return solver.spec


def _bundle():
    reg = ToolRegistry()
    async def lookup(args):
        return {"hit": args.get("id")}
    reg.register_tool("lookup", lookup, allowed_operations=("invoke",),
                      schema={"type": "object", "properties": {"id": {"type": "string"}}})
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer)
    return ToolLoopBundle(
        proxy=proxy, issuer=issuer, registry=reg, declared=("lookup",),
        run_ctx=RunContext(run_id="r1", step=0),
    ), proxy


def test_factory_returns_a_callable_taking_a_model_name():
    factory = make_scripted_model_factory([("text", "{}")])
    model = factory("fake")          # the loop calls model_factory(chain[0])
    assert model is not None


async def test_scripted_model_drives_the_real_loop_to_a_final_answer():
    # A call turn (invokes the proxy-backed tool) then a text turn with final JSON.
    bundle, proxy = _bundle()
    factory = make_scripted_model_factory([
        ("call", "lookup", {"id": "txn_1"}),
        ("text", json.dumps({"answer": "done"})),
    ])
    engine = ADKEngine(gateway=_RaisingGateway(), model_factory=factory, max_llm_calls=8)
    out = await engine.run_step(_spec(), In(task="t"), tool_loop=bundle)
    assert out.output == {"answer": "done"}
    # the tool call really went through the proxy
    assert any(e["tool"] == "lookup" and e["result"] == "ok" for e in proxy.log)


async def test_scripted_multicall_turn_fails_closed_through_the_loop():
    from drawbore.llm import LLMError
    bundle, proxy = _bundle()
    factory = make_scripted_model_factory([("multicall", [("lookup", {}), ("lookup", {})])])
    engine = ADKEngine(gateway=_RaisingGateway(), model_factory=factory, max_llm_calls=8)
    with pytest.raises(LLMError):
        await engine.run_step(_spec(), In(task="t"), tool_loop=bundle)


class _RaisingGateway:
    async def complete(self, request):
        raise AssertionError("model+tools loop must not use the gateway")
