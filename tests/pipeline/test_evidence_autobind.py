"""Tests: evidence retrieval auto-bind via Pipeline.run / test_mode.

When a model+tools agent declares evidence://retrieve in its tools and the
caller passes evidence_store=store to test_mode / Pipeline.run, the pipeline
must auto-bind the retrieval handler to that same store — the caller's single
evidence_store= argument is the only wiring point required.

Cases covered:
- Single wiring point: no manual register_evidence_tool call; retrieval succeeds.
- Idempotency: two pipeline steps both declaring evidence://retrieve; no duplicate
  registration error when evidence_store is provided.
- Existing workaround still works: manual register_evidence_tool + evidence_store
  to run() is still accepted.
"""

from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.evidence import (
    EVIDENCE_TOOL_REF,
    EvidenceHandle,
    EvidencePolicy,
    InMemoryEvidenceStore,
    register_evidence_tool,
)
from drawbore.pipeline import Pipeline, From
from drawbore.testing import call, final
from drawbore.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Shared models / helpers
# ---------------------------------------------------------------------------

class Row(BaseModel):
    id: int
    amount: int


class In(BaseModel):
    records: list[Row]


class Out(BaseModel):
    answer: str


def _seed_store(store: InMemoryEvidenceStore, handle_id: str = "h1") -> InMemoryEvidenceStore:
    """Pre-seed store with a retrievable original under a known handle id."""
    original = {"records": [{"id": i, "amount": i * 10} for i in range(200)]}
    store.put(
        EvidenceHandle(
            handle_id=handle_id,
            run_id="r0",
            step=0,
            source_agent="screen",
            content_type="json_rows",
            original_hash="o" * 16,
            compressed_hash="c" * 16,
            original_tokens=4000,
            compressed_tokens=900,
            transform="json_rows",
        ),
        original=original,
        compressed={"records": original["records"][:2]},
    )
    store.set_policy(handle_id, allow_full=False, allow_search=True)
    return store


# ---------------------------------------------------------------------------
# Test: single wiring point (the bug fix)
# ---------------------------------------------------------------------------

async def test_evidence_store_to_run_autobinds_retrieval_tool():
    """Passing evidence_store=store to test_mode is the ONLY wiring point needed.
    No manual register_evidence_tool call. On current main this fails at
    Pipeline.add() with ToolAccessError because EVIDENCE_TOOL_REF is not
    pre-registered; after the fix the retrieval succeeds through the proxy."""
    store = InMemoryEvidenceStore()
    _seed_store(store)

    @agent(name="screen", input=In, output=Out, model="fake", tools=[EVIDENCE_TOOL_REF])
    async def screen(v: In) -> Out:
        raise AssertionError("must not run — model-backed agent")

    # No registry argument: uses the default (empty) registry. No manual
    # register_evidence_tool call. This is the single-wiring-point contract.
    p = Pipeline(name="aml")
    p.add(screen, evidence=EvidencePolicy(name="aml", enabled=True, min_tokens=1))

    script = [
        call(EVIDENCE_TOOL_REF, {"handle_id": "h1", "mode": "search", "query": ""}),
        final({"answer": "done"}),
    ]
    async with p.test_mode(
        mock_loop_scripts={"screen": script},
        evidence_store=store,
    ) as test:
        result = await test.run(In(records=[Row(id=i, amount=i * 10) for i in range(50)]))

    assert result.status == "completed", (
        f"expected completed; got {result.status!r}; reason={result.reason!r}"
    )
    assert result.outputs["screen"].answer == "done"
    step = result.audit_trace.step_records[0]
    # The retrieval call reached the proxy and was dispatched successfully.
    assert any(
        EVIDENCE_TOOL_REF in tc and "ok" in tc for tc in step.tool_calls
    ), f"expected an ok evidence tool call in {step.tool_calls!r}"


# ---------------------------------------------------------------------------
# Test: idempotency — two steps declaring EVIDENCE_TOOL_REF, no double-reg error
# ---------------------------------------------------------------------------

async def test_two_steps_declare_evidence_tool_no_duplicate_registration():
    """Multiple pipeline steps declaring evidence://retrieve must not raise a
    duplicate-registration error when evidence_store is provided. The handler is
    bound exactly once for the run."""
    store = InMemoryEvidenceStore()

    class SimpleIn(BaseModel):
        x: int

    class SimpleOut(BaseModel):
        y: int

    @agent(name="step_a", input=SimpleIn, output=SimpleOut, tools=[EVIDENCE_TOOL_REF])
    async def step_a(v: SimpleIn, tools) -> SimpleOut:
        return SimpleOut(y=v.x + 1)

    @agent(name="step_b", input=SimpleOut, output=SimpleOut, tools=[EVIDENCE_TOOL_REF])
    async def step_b(v: SimpleOut, tools) -> SimpleOut:
        return SimpleOut(y=v.y + 1)

    # Both steps declare EVIDENCE_TOOL_REF. After the fix, add() accepts them.
    # Running with evidence_store must not raise a duplicate-registration error.
    p = Pipeline(name="multi")
    p.add(step_a)
    p.add(step_b, inputs={"y": From("step_a.y")})

    # The fact that it completes without raising is the assertion.
    async with p.test_mode(evidence_store=store) as test:
        result = await test.run(SimpleIn(x=0))

    assert result.status == "completed", (
        f"expected completed; got {result.status!r}; reason={result.reason!r}"
    )
    assert result.outputs["step_b"].y == 2


# ---------------------------------------------------------------------------
# Test: existing workaround still works
# ---------------------------------------------------------------------------

async def test_manual_register_plus_evidence_store_still_works():
    """The two-call workaround (manual register_evidence_tool + evidence_store to
    run) must remain accepted — the auto-bind must not break it."""
    store = InMemoryEvidenceStore()
    _seed_store(store)

    @agent(name="screen", input=In, output=Out, model="fake", tools=[EVIDENCE_TOOL_REF])
    async def screen(v: In) -> Out:
        raise AssertionError("must not run — model-backed agent")

    # Workaround: explicit pre-registration with the same store.
    reg = ToolRegistry()
    register_evidence_tool(reg, store=store)

    p = Pipeline(name="aml", registry=reg)
    p.add(screen, evidence=EvidencePolicy(name="aml", enabled=True, min_tokens=1))

    script = [
        call(EVIDENCE_TOOL_REF, {"handle_id": "h1", "mode": "search", "query": ""}),
        final({"answer": "done"}),
    ]
    async with p.test_mode(
        mock_loop_scripts={"screen": script},
        evidence_store=store,
    ) as test:
        result = await test.run(In(records=[Row(id=i, amount=i * 10) for i in range(50)]))

    assert result.status == "completed", (
        f"workaround path failed; status={result.status!r}; reason={result.reason!r}"
    )
    assert result.outputs["screen"].answer == "done"
