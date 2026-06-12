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


async def test_loop_non_json_final_error_includes_bounded_excerpt(fake_adk_model):
    from drawbore.llm import LLMError

    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    # A loop whose FINAL turn is raw, non-JSON text: it halts model_error, and the
    # halt reason must now carry a bounded excerpt of the offending final content so
    # the failure is diagnosable from the audit trail. (The loop is NOT retried — a
    # post-tool re-run would replay tool side-effects; only the one-shot path retries.)
    script = [("text", "<not json> " + "y" * 5000)]
    reg, proxy, bundle = _wiring()
    with pytest.raises(LLMError) as ei:
        await run_agentic_loop(
            solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
            model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
        )
    msg = str(ei.value)
    assert "content excerpt" in msg
    assert "<not json>" in msg
    assert len(msg) < 1000                   # bounded: the 5000-char body is not dumped


async def test_loop_recovers_prose_wrapped_json_final_answer(fake_adk_model):
    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    # A real-world failure mode: the model narrates before answering, so its FINAL
    # turn is a single JSON object wrapped in prose. Drawbore deterministically
    # recovers the sole top-level JSON object and the step COMPLETES.
    prose = (
        "Sure! The user wants the answer. Here is the result:\n"
        + json.dumps({"answer": "done:42"})
        + "\nLet me know if you need anything else."
    )
    script = [("call", "lookup", {"key": "x"}), ("text", prose)]
    reg, proxy, bundle = _wiring()
    output, model_turns = await run_agentic_loop(
        solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
        model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
    )
    assert output == {"answer": "done:42"}
    # NO side-effect replay during recovery: the tool ran exactly once (recovery is a
    # pure parse — it invokes no tool and makes no further model call).
    assert sum(1 for e in proxy.log if e["tool"] == "lookup" and e["result"] == "ok") == 1
    assert model_turns == 2                  # the recovery makes no extra model turn


async def test_loop_unrecoverable_prose_still_halts_model_error(fake_adk_model):
    from drawbore.llm import LLMError

    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    # Pure narration with no JSON object at all: nothing to recover -> fail closed.
    script = [("call", "lookup", {"key": "x"}),
              ("text", "The user wants this. I need to: 1. Call the tool. 2. Answer.")]
    reg, proxy, bundle = _wiring()
    with pytest.raises(LLMError) as ei:
        await run_agentic_loop(
            solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
            model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
        )
    assert "content excerpt" in str(ei.value)
    # The tool ran once (the loop turn), and recovery added nothing.
    assert sum(1 for e in proxy.log if e["tool"] == "lookup" and e["result"] == "ok") == 1


async def test_loop_ambiguous_multiple_json_objects_fails_closed(fake_adk_model):
    from drawbore.llm import LLMError

    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    # Two top-level JSON objects in the prose: Drawbore never guesses which to trust,
    # so it fails closed rather than picking one.
    prose = 'First {"answer": "a"} and then also {"answer": "b"}.'
    script = [("text", prose)]
    reg, proxy, bundle = _wiring()
    with pytest.raises(LLMError) as ei:
        await run_agentic_loop(
            solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
            model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
        )
    assert "content excerpt" in str(ei.value)
