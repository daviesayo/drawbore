import json
import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.orchestration.adk_loop import run_agentic_loop_chain
from drawbore.orchestration.engine import ToolLoopBundle
from drawbore.llm.resolution import ModelAttempt, ResolvedModelChain
from drawbore.tools import ToolRegistry, ToolProxy, TokenIssuer, RunContext


class In(BaseModel):
    task: str


class Out(BaseModel):
    answer: str


def _direct_chain(model: str) -> ResolvedModelChain:
    """Build a minimal single-attempt ResolvedModelChain for a direct model string
    (no profile expansion needed in unit tests)."""
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
    result = await run_agentic_loop_chain(
        solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
        chain=_direct_chain("fake"),
        model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
    )
    assert result.output == {"answer": "done:42"}
    assert result.model_turns == 2
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
    result = await run_agentic_loop_chain(
        solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
        chain=_direct_chain("fake"),
        model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
    )
    assert result.output == {"answer": "two"}
    assert result.model_turns == 3
    assert sum(1 for e in proxy.log if e["tool"] == "lookup" and e["result"] == "ok") == 2


async def test_loop_non_json_final_error_includes_bounded_excerpt(fake_adk_model):
    from drawbore.llm import LLMError

    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    # A loop whose turn is raw, non-JSON text triggers a single bounded reprompt; when
    # the reprompt ALSO returns non-JSON prose, the step halts model_error and the halt
    # reason carries a bounded excerpt of the offending content so the failure is
    # diagnosable from the audit trail. (The loop is never re-run wholesale — a post-tool
    # re-run would replay tool side-effects; the reprompt is a pure model turn.)
    big = "<not json> " + "y" * 5000
    script = [("text", big), ("text", big)]  # prose on the original turn AND the reprompt
    reg, proxy, bundle = _wiring()
    with pytest.raises(LLMError) as ei:
        await run_agentic_loop_chain(
            solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
            chain=_direct_chain("fake"),
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
    result = await run_agentic_loop_chain(
        solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
        chain=_direct_chain("fake"),
        model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
    )
    assert result.output == {"answer": "done:42"}
    # NO side-effect replay during recovery: the tool ran exactly once (recovery is a
    # pure parse — it invokes no tool and makes no further model call).
    assert sum(1 for e in proxy.log if e["tool"] == "lookup" and e["result"] == "ok") == 1
    assert result.model_turns == 2                  # the recovery makes no extra model turn


async def test_loop_unrecoverable_prose_still_halts_model_error(fake_adk_model):
    from drawbore.llm import LLMError

    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    # Pure narration with no JSON object at all: nothing to recover. The model narrates
    # on the original turn AND on the single reprompt, so the step fails closed after the
    # one bounded reprompt.
    script = [("call", "lookup", {"key": "x"}),
              ("text", "The user wants this. I need to: 1. Call the tool. 2. Answer."),
              ("text", "Still just narrating; no tool call and no JSON here either.")]
    reg, proxy, bundle = _wiring()
    with pytest.raises(LLMError) as ei:
        await run_agentic_loop_chain(
            solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
            chain=_direct_chain("fake"),
            model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
        )
    assert "content excerpt" in str(ei.value)
    # The tool ran once (the loop turn), and recovery added nothing.
    assert sum(1 for e in proxy.log if e["tool"] == "lookup" and e["result"] == "ok") == 1


async def test_loop_reprompts_once_when_model_narrates_a_tool_call(captured_spans, fake_adk_model):
    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    # The reasoning-class failure mode: on a turn where the model should EMIT a
    # structured tool call, it instead NARRATES its intent as prose (no function call).
    # ADK surfaces that as a non-JSON final response. Drawbore must NOT parse the
    # narrated call out of prose; it reprompts ONCE for a structured response, and the
    # corrected second turn (a real tool call, then a JSON answer) completes the run.
    # Realistic narration carries SEVERAL embedded JSON fragments (not one recoverable
    # object), so the #22 sole-object recovery does not apply and the reprompt path runs.
    narration = (
        "The user wants me to enrich this. I need to: 1. Call `lookup` with "
        '{"key": "x"} to fetch the record, then 2. validate it with {"strict": true}.'
    )
    script = [
        ("text", narration),                            # pass 1: prose, NO tool call
        ("call", "lookup", {"key": "y"}),               # pass 2 (after reprompt): real call
        ("text", json.dumps({"answer": "done:42"})),    # pass 2: JSON final answer
    ]
    reg, proxy, bundle = _wiring()
    result = await run_agentic_loop_chain(
        solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
        chain=_direct_chain("fake"),
        model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
    )
    assert result.output == {"answer": "done:42"}
    # The tool ran EXACTLY ONCE — from the structured call in the reprompted turn, never
    # from the narrated prose. (Had the narrated "call lookup" been executed too, lookup
    # would appear twice.)
    assert sum(1 for e in proxy.log if e["tool"] == "lookup" and e["result"] == "ok") == 1
    # Three model turns total: the prose turn + the reprompt's call turn + its final turn.
    assert result.model_turns == 3
    assert len(bundle.turns) == 3


async def test_loop_narrates_again_on_reprompt_halts_model_error(fake_adk_model):
    from drawbore.llm import LLMError

    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    # The model narrates on BOTH the original turn and the single reprompt. Recovery is
    # bounded to exactly one reprompt, so the step then fails closed (model_error); it
    # never loops unbounded, and no tool is executed from either prose turn.
    script = [
        # Two embedded fragments (ambiguous) — not a recoverable sole object.
        ("text", 'I will call `lookup` with {"key": "x"} then verify {"id": 7}.'),
        ("text", "Okay, calling `lookup` now and returning the result."),
        # A third turn exists but must NEVER be reached (only ONE reprompt is allowed).
        ("text", json.dumps({"answer": "should-not-be-used"})),
    ]
    reg, proxy, bundle = _wiring()
    with pytest.raises(LLMError) as ei:
        await run_agentic_loop_chain(
            solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
            chain=_direct_chain("fake"),
            model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
        )
    assert "content excerpt" in str(ei.value)
    # Bounded: exactly TWO model turns happened (original + one reprompt), proving the
    # loop did not keep reprompting and never reached the third scripted turn.
    assert len(bundle.turns) == 2
    # No tool was executed from either prose turn.
    assert not any(e["tool"] == "lookup" for e in proxy.log)


async def test_loop_no_reprompt_when_no_model_budget_remains(fake_adk_model):
    from drawbore.llm import LLMError

    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    # max_llm_calls=1: the prose turn consumes the only budgeted model turn, so there is
    # no budget left to reprompt — the step fails closed immediately, with no second turn.
    script = [
        # Ambiguous narration (two fragments) — not a recoverable sole object.
        ("text", 'I will call `lookup` with {"key": "x"} then {"strict": true}.'),
        ("text", json.dumps({"answer": "unreached"})),
    ]
    reg, proxy, bundle = _wiring()
    with pytest.raises(LLMError):
        await run_agentic_loop_chain(
            solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
            chain=_direct_chain("fake"),
            model_factory=lambda name: fake_adk_model(script), max_llm_calls=1,
        )
    assert len(bundle.turns) == 1  # the reprompt was NOT attempted (budget exhausted)


async def test_loop_reprompts_on_schema_invalid_final_then_recovers(fake_adk_model):
    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    # The model's final answer is a VALID JSON object but it does not match the agent's
    # output schema (missing the required 'answer' field). The loop reprompts ONCE,
    # seeded with the field errors; the corrected answer matches the schema and the run
    # completes. The reprompt is a pure model turn — no tool runs from it.
    script = [
        ("call", "lookup", {"key": "x"}),               # turn 1: structured tool call
        ("text", json.dumps({"wrong": "x"})),           # turn 2: object, schema-INVALID
        ("text", json.dumps({"answer": "fixed"})),      # turn 3 (after reprompt): valid
    ]
    reg, proxy, bundle = _wiring()
    result = await run_agentic_loop_chain(
        solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
        chain=_direct_chain("fake"),
        model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
    )
    assert result.output == {"answer": "fixed"}
    # The tool ran EXACTLY ONCE — from the structured call, never from a reprompt turn.
    assert sum(1 for e in proxy.log if e["tool"] == "lookup" and e["result"] == "ok") == 1
    assert result.model_turns == 3                  # call + invalid-final + corrected-final


async def test_loop_schema_reprompt_budget_is_shared_not_stacked(fake_adk_model):
    from drawbore.orchestration.adk_loop import _run_one_attempt

    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    # A schema-invalid object earns the single bounded reprompt; if the corrected answer
    # is STILL schema-invalid the object is RETURNED (reprompts == 1) for the pipeline's
    # authoritative schema gate to halt — the loop never reprompts a second time.
    script = [
        ("text", json.dumps({"wrong": "x"})),           # turn 1: object, schema-INVALID
        ("text", json.dumps({"still": "bad"})),         # turn 2 (reprompt): STILL invalid
        ("text", json.dumps({"answer": "unreached"})),  # turn 3 must never be reached
    ]
    reg, proxy, bundle = _wiring()
    output, model_turns, reprompts = await _run_one_attempt(
        solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
        model="fake", model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
    )
    assert output == {"still": "bad"}        # returned, NOT corrected — gate halts later
    assert reprompts == 1                    # exactly one reprompt, never two
    assert model_turns == 2                  # the third scripted turn was never reached


async def test_loop_ambiguous_multiple_json_objects_fails_closed(fake_adk_model):
    from drawbore.llm import LLMError

    @agent(name="solver", input=In, output=Out, model="fake", tools=["lookup"])
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    # Two top-level JSON objects in the prose: Drawbore never guesses which to trust, so
    # it reprompts once for a single structured response. The reprompt is ALSO ambiguous,
    # so the step fails closed rather than picking one.
    prose = 'First {"answer": "a"} and then also {"answer": "b"}.'
    script = [("text", prose), ("text", prose)]  # ambiguous on the original turn AND the reprompt
    reg, proxy, bundle = _wiring()
    with pytest.raises(LLMError) as ei:
        await run_agentic_loop_chain(
            solver.spec, In(task="solve"), tool_loop=bundle, run_id="r1",
            chain=_direct_chain("fake"),
            model_factory=lambda name: fake_adk_model(script), max_llm_calls=8,
        )
    assert "content excerpt" in str(ei.value)
