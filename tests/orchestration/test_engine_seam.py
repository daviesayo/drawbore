import pytest

from drawbore.agent import agent
from drawbore.orchestration import LocalEngine, ToolLoopBundle, EngineError
from drawbore.errors import DrawboreError, halt_reason_for
from drawbore.tools import ToolRegistry, ToolProxy, TokenIssuer, RunContext
from pydantic import BaseModel


class In(BaseModel):
    text: str


class Out(BaseModel):
    text: str


def _bundle():
    reg = ToolRegistry()
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer)
    return ToolLoopBundle(
        proxy=proxy, issuer=issuer, registry=reg,
        declared=("echo",), run_ctx=RunContext(run_id="r1", step=0),
    )


def test_bundle_carries_what_the_engine_needs_and_starts_empty():
    b = _bundle()
    assert b.declared == ("echo",)
    assert b.run_ctx.run_id == "r1"
    assert b.failures == []      # mutable, append-able
    assert b.turns == []         # model-turn markers; count == len(turns)


def test_engine_error_is_a_drawbore_error_with_legible_reason():
    assert issubclass(EngineError, DrawboreError)
    assert halt_reason_for(EngineError("x")) == "engine_error"


async def test_local_engine_runs_deterministic_agents():
    @agent(name="echo", input=In, output=Out)
    async def echo(v: In) -> Out:
        return Out(text=v.text)

    out = await LocalEngine().run_step(echo.spec, In(text="hi"))
    assert out.text == "hi"


async def test_local_engine_fails_closed_for_a_model_backed_agent():
    # A model-backed agent on an engine with no model path must NOT silently run
    # spec.fn — it raises an engine-agnostic error.
    @agent(name="writer", input=In, output=Out, model="gpt-4o")
    async def writer(v: In) -> Out:
        raise AssertionError("model-backed fn must not run")

    with pytest.raises(EngineError) as ei:
        await LocalEngine().run_step(writer.spec, In(text="hi"))
    msg = str(ei.value)
    assert "model path" in msg                 # engine-agnostic wording
    assert "model-capable" in msg
    assert "ADK" not in msg                     # MUST NOT name ADK (principle 6)


async def test_local_engine_accepts_tool_loop_kwarg_and_ignores_it():
    @agent(name="echo", input=In, output=Out)
    async def echo(v: In) -> Out:
        return Out(text=v.text)

    out = await LocalEngine().run_step(echo.spec, In(text="hi"), tool_loop=_bundle())
    assert out.text == "hi"
