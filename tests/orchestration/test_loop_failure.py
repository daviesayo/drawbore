import json
import logging
import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.orchestration.adk_loop import run_agentic_loop_chain
from drawbore.orchestration.engine import ToolLoopBundle
from drawbore.llm import LLMError
from drawbore.llm.resolution import ModelAttempt, ResolvedModelChain
from drawbore.tools import ToolRegistry, ToolProxy, TokenIssuer, RunContext
from drawbore.tools.errors import ToolAccessError


class In(BaseModel):
    task: str


class Out(BaseModel):
    answer: str


def _direct_chain(model: str) -> ResolvedModelChain:
    """Build a minimal single-attempt ResolvedModelChain for a direct model string."""
    attempt = ModelAttempt(
        provider=None,
        model=model,
        request_model=model,
        declared_ref=model,
        source="direct",
        fallback_on=("timeout", "rate_limit", "server_error", "provider_unavailable"),
        credential_env=None,
        credential_required=False,
    )
    return ResolvedModelChain(declared=(model,), attempts=(attempt,))


def _wiring(declared=("a", "b"), allow_a=("invoke",), allow_b=("invoke",)):
    reg = ToolRegistry()

    async def _h(args):
        return {"ok": True}

    reg.register_tool("a", _h, allowed_operations=allow_a, schema={"type": "object", "properties": {}})
    reg.register_tool("b", _h, allowed_operations=allow_b, schema={"type": "object", "properties": {}})
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer)
    bundle = ToolLoopBundle(proxy=proxy, issuer=issuer, registry=reg,
                            declared=declared, run_ctx=RunContext(run_id="r1", step=0))
    return reg, proxy, bundle


def _spec():
    @agent(name="solver", input=In, output=Out, model="fake", tools=["a", "b"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")
    return solver.spec


async def test_tool_failure_aborts_immediately_second_tool_never_runs(fake_adk_model):
    # Turn 1 calls tool 'a' (DENIED — 'invoke' not allowed). Turn 2 WOULD call 'b'.
    # Immediate abort must mean 'b' is never invoked and the run raises a's error.
    reg, proxy, bundle = _wiring(allow_a=("read",))  # 'a' denies 'invoke'
    script = [("call", "a", {}), ("call", "b", {}), ("final", json.dumps({"answer": "x"}))]
    with pytest.raises(ToolAccessError):
        await run_agentic_loop_chain(_spec(), In(task="t"), tool_loop=bundle, run_id="r1",
                                     chain=_direct_chain("fake"),
                                     model_factory=lambda name: fake_adk_model(script),
                                     max_llm_calls=8)
    # 'b' was NEVER called through the proxy (no ok entry for 'b')
    assert not any(e["tool"] == "b" and e["result"] == "ok" for e in proxy.log)


async def test_loop_bound_halts_a_runaway_loop(fake_adk_model):
    # A script that always calls a tool, never finishes -> exceeds max_llm_calls -> LLMError.
    reg, proxy, bundle = _wiring()
    script = [("call", "a", {})] * 100      # never a "final"
    with pytest.raises(LLMError):
        await run_agentic_loop_chain(_spec(), In(task="t"), tool_loop=bundle, run_id="r1",
                                     chain=_direct_chain("fake"),
                                     model_factory=lambda name: fake_adk_model(script),
                                     max_llm_calls=3)


async def test_non_json_final_answer_fails_closed(fake_adk_model):
    # A non-JSON turn triggers a single bounded reprompt; when the reprompt is ALSO
    # non-JSON, the step fails closed (model_error).
    reg, proxy, bundle = _wiring()
    script = [("final", "not json at all"), ("text", "still not json")]
    with pytest.raises(LLMError):
        await run_agentic_loop_chain(_spec(), In(task="t"), tool_loop=bundle, run_id="r1",
                                     chain=_direct_chain("fake"),
                                     model_factory=lambda name: fake_adk_model(script),
                                     max_llm_calls=8)


async def test_multiple_function_calls_in_one_turn_fail_closed(fake_adk_model):
    # A turn that emits two function calls at once is out of scope: fail closed.
    reg, proxy, bundle = _wiring()
    script = [("multicall", [("a", {}), ("b", {})])]
    with pytest.raises(LLMError):
        await run_agentic_loop_chain(_spec(), In(task="t"), tool_loop=bundle, run_id="r1",
                                     chain=_direct_chain("fake"),
                                     model_factory=lambda name: fake_adk_model(script),
                                     max_llm_calls=8)
    # The fail-CLOSED guarantee: the guard fires on the model-output event,
    # before ADK dispatches the calls, so NEITHER requested tool reaches the proxy.
    assert proxy.log == []


def test_adk_and_otel_loggers_suppressed_to_critical():
    """Importing adk_loop must pin google_adk and opentelemetry.context to CRITICAL so
    ADK-internal execution-failure / cancellation noise never leaks through the engine
    boundary (principle 6 / ADK-hiding invariant)."""
    import drawbore.orchestration.adk_loop as _adk_loop  # noqa: F401
    assert logging.getLogger("google_adk").level == logging.CRITICAL
    assert logging.getLogger("opentelemetry.context").level == logging.CRITICAL
