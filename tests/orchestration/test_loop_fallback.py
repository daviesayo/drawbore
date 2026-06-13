"""Loop provider fallback: the model+tools agentic loop resolves through the
``LLMRuntime`` and may advance to the next model attempt — but ONLY before any
tool has run. The load-bearing invariants (principle 7):

  * (test 16) pre-tool transport failure may advance to the next model attempt;
  * (test 17) once any tool has run, a provider failure FAILS CLOSED (no replay);
  * (test 18) a tool failure/denial aborts immediately with the tool error, never a
    provider fallback (a denied tool must not let the model retry on another provider
    to route around a tripped control);
  * (test 19) the loop is bounded across attempts by ``max_llm_calls``.

These drive the loop with the scripted fake loop model so no provider is called,
plus a fake ``model_factory`` that RAISES a classified transport error for the first
model to simulate a provider failure before any tool call. The runtime is given a
2-attempt chain (``model``/``fallback_model`` direct strings; the empty-config runtime
resolves direct strings, deduped, and a direct ref falls back on transport reasons).
"""

import json

import litellm.exceptions as _e
import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.llm import LLMRuntime, ModelUnavailableError
from drawbore.orchestration import ADKEngine, ToolLoopBundle
from drawbore.orchestration.scripted_model import make_scripted_model_factory
from drawbore.tools import RunContext, TokenIssuer, ToolProxy, ToolRegistry


class In(BaseModel):
    task: str


class Out(BaseModel):
    answer: str


def _registry(tool_name="lookup"):
    reg = ToolRegistry()

    async def _l(args):
        return {"v": 1}

    reg.register_tool(
        tool_name, _l, allowed_operations=("invoke",),
        schema={"type": "object", "properties": {}},
    )
    return reg


def _bundle(reg, declared=("lookup",)):
    issuer = TokenIssuer()
    return ToolLoopBundle(
        proxy=ToolProxy(reg, issuer), issuer=issuer, registry=reg,
        declared=declared, run_ctx=RunContext(run_id="r1", step=0),
    )


def _two_target_factory(*, fail_first_with, second_turns, calls):
    """A ``model_factory(model_name) -> BaseLlm`` over a 2-attempt direct chain.

    ``calls`` is a mutable list recording each model_name the factory is built for, so
    a test can assert how many attempts were actually constructed. The FIRST attempt's
    model raises ``fail_first_with`` from its model turn (a provider/transport failure);
    the SECOND attempt returns a scripted loop model running ``second_turns``."""
    second_factory = make_scripted_model_factory(second_turns)

    def factory(model_name):
        calls.append(model_name)
        if model_name == "m1":
            return _RaisingLoopModel(fail_first_with)
        return second_factory(model_name)

    return factory


def _make_raising_loop_model_cls():
    # Import inside orchestration's contained namespace: BaseLlm is an ADK type, so we
    # build the raising fake by subclassing the scripted model's base. This module is a
    # TEST (tests/ is not under drawbore.orchestration) but ADK is a core dep, so the
    # import is allowed in tests exactly as conftest.py imports BaseLlm.
    from google.adk.models import BaseLlm

    class RaisingLoopModel(BaseLlm):
        """A BaseLlm whose model turn RAISES, simulating a provider transport failure
        before the model emits any function call (so no tool runs)."""

        def __init__(self, exc):
            super().__init__(model="drawbore-test-raising")
            object.__setattr__(self, "_exc", exc)

        async def generate_content_async(self, llm_request, stream: bool = False):
            raise self._exc
            yield  # pragma: no cover - makes this an async generator

    return RaisingLoopModel


_RaisingLoopModel = _make_raising_loop_model_cls()


def _loop_spec(**kw):
    @agent(name="solver", input=In, output=Out, tools=["lookup"], **kw)
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    return solver.spec


def _engine(factory, *, max_llm_calls=8):
    # An empty-config runtime resolves direct model strings (no profiles); a 2-target
    # chain comes from model="m1", fallback_model="m2". The runtime's model_factory
    # slot is what the loop reads (the engine prefers it over its default).
    runtime = LLMRuntime.from_gateway(_DummyGateway(), model_factory=factory)
    return ADKEngine(llm_runtime=runtime, max_llm_calls=max_llm_calls)


class _DummyGateway:
    # Never called on the loop path (the loop drives the model via model_factory, not
    # the gateway); present only to satisfy LLMRuntime.from_gateway.
    async def complete(self, request):  # pragma: no cover
        raise AssertionError("gateway must not be called on the loop path")


# --- test 16: pre-tool fallback selects the next model -----------------------------

async def test_pre_tool_transport_failure_advances_to_next_model():
    reg = _registry()
    calls: list[str] = []
    factory = _two_target_factory(
        fail_first_with=_e.Timeout(
            message="boom", model="m1", llm_provider="openai"
        ),
        second_turns=[("text", json.dumps({"answer": "looped"}))],
        calls=calls,
    )
    spec = _loop_spec(model="m1", fallback_model="m2")
    execution = await _engine(factory).run_step(spec, In(task="t"), tool_loop=_bundle(reg))

    assert execution.output == {"answer": "looped"}
    audit = execution.model_audit
    assert audit is not None
    assert audit.loop_fallback_phase == "before_tools"
    assert [a.outcome for a in audit.attempts] == ["fallback", "success"]
    assert audit.attempts[0].reason == "timeout"
    assert calls == ["m1", "m2"]  # both attempts were built; the loop advanced


# --- test 17: post-tool failure halts (no fallback) --------------------------------

async def test_post_tool_provider_failure_fails_closed():
    reg = _registry()
    calls: list[str] = []
    # The FIRST model runs a tool turn (which executes the proxy-backed tool), then its
    # SECOND turn raises a transport error. Because a tool already ran, the loop must
    # fail closed — attempt 2's model is NEVER built.
    # The ``("__raise__",)`` turn is the SINGLE source of truth for WHEN the post-tool
    # raise fires: the second model turn raises the configured ``raise_with`` exception.
    # ``raise_on_index`` is left None so the script alone drives the raise (no duplicate
    # index-based mechanism).
    first_turns = [
        ("call", "lookup", {}),
        ("__raise__",),
    ]
    factory = _two_target_with_first_script(
        first_script=first_turns,
        raise_with=_e.Timeout(message="boom", model="m1", llm_provider="openai"),
        second_turns=[("text", json.dumps({"answer": "should-never-be-used"}))],
        calls=calls,
    )
    spec = _loop_spec(model="m1", fallback_model="m2")

    with pytest.raises(ModelUnavailableError):
        await _engine(factory).run_step(spec, In(task="t"), tool_loop=_bundle(reg))

    # Invariant 1: no post-tool fallback — the second attempt's model was never built.
    assert calls == ["m1"]


# --- test 18: tool denial aborts, no fallback --------------------------------------

async def test_tool_denial_aborts_with_no_provider_fallback():
    from drawbore.tools.errors import ToolAccessError

    calls: list[str] = []
    # A DECLARED tool whose 'invoke' operation is NOT allowed: the proxy denies it at
    # invocation (the proxy-backed tool records the denial in tool_loop.failures and
    # re-raises). A denied control MUST abort immediately with the tool error and MUST
    # NOT trigger a provider fallback — otherwise the model could route around a tripped
    # control by retrying on another provider.
    reg = ToolRegistry()

    async def _h(args):
        return {"ok": True}

    reg.register_tool(
        "lookup", _h, allowed_operations=("read",),  # 'invoke' is DENIED
        schema={"type": "object", "properties": {}},
    )
    first_turns = [("call", "lookup", {})]
    factory = _two_target_with_first_script(
        first_script=first_turns,
        raise_with=None,
        second_turns=[("text", json.dumps({"answer": "should-never-be-used"}))],
        calls=calls,
    )
    spec = _loop_spec(model="m1", fallback_model="m2")
    bundle = _bundle(reg, declared=("lookup",))

    with pytest.raises(ToolAccessError):  # the tool error, NOT a provider fallback
        await _engine(factory).run_step(spec, In(task="t"), tool_loop=bundle)

    # Invariant 2: the denial aborted; the second attempt's model was NEVER built.
    assert calls == ["m1"]
    # The denial was recorded as a tool failure (not classified as a provider failure).
    assert bundle.failures
    assert isinstance(bundle.failures[0], ToolAccessError)


# --- test 19: budget across attempts -----------------------------------------------

async def test_budget_bounds_runaway_attempts():
    from drawbore.llm import LLMError

    reg = _registry()
    calls: list[str] = []
    # A runaway model that keeps calling the tool and never emits a final answer. With a
    # low max_llm_calls the loop is bounded and exhaustion fails closed (an LLMError); it
    # must never blow past the cap into an unbounded run. Both attempts are scripted
    # call-only loops so neither terminates on its own.
    runaway_turns = [("call", "lookup", {})] * 20
    scripted = make_scripted_model_factory(runaway_turns)

    def runaway_factory(model_name):
        calls.append(model_name)
        return scripted(model_name)

    spec = _loop_spec(model="m1", fallback_model="m2")

    with pytest.raises(LLMError):  # exhaustion → model_unavailable / LLMError subclass
        await _engine(runaway_factory, max_llm_calls=2).run_step(
            spec, In(task="t"), tool_loop=_bundle(reg)
        )


def _two_target_with_first_script(
    *, first_script, raise_with, second_turns, calls
):
    """A ``model_factory`` whose FIRST attempt runs ``first_script`` and whose
    SECOND attempt runs ``second_turns``."""
    from google.adk.models import BaseLlm, LlmResponse
    from google.genai import types

    second_factory = make_scripted_model_factory(second_turns)

    class _ScriptOrRaise(BaseLlm):
        def __init__(self, turns, exc):
            super().__init__(model="drawbore-test-script-or-raise")
            object.__setattr__(self, "_turns", list(turns))
            object.__setattr__(self, "_exc", exc)
            object.__setattr__(self, "_i", 0)

        async def generate_content_async(self, llm_request, stream: bool = False):
            i = self._i
            object.__setattr__(self, "_i", i + 1)
            if i >= len(self._turns):
                yield LlmResponse(content=types.Content(
                    role="model", parts=[types.Part(text="{}")]))
                return
            turn = self._turns[i]
            kind = turn[0]
            if kind == "call":
                _, name, args = turn
                yield LlmResponse(content=types.Content(
                    role="model",
                    parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
                ))
            elif kind == "__raise__":
                raise self._exc
            else:  # "text"
                _, text = turn
                yield LlmResponse(content=types.Content(
                    role="model", parts=[types.Part(text=text)]))

    def factory(model_name):
        calls.append(model_name)
        if model_name == "m1":
            return _ScriptOrRaise(first_script, raise_with)
        return second_factory(model_name)

    return factory
