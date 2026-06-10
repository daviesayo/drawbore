"""Acceptance tests: evidence modes (disabled/simulate/compress), retrieval
(allow/deny), and test-store rebind (spec tests 17, 18, 19).

Key facts verified against the real evidence API:
- EvidencePolicy params: name, enabled, mode ("simulate"/"compress"), allowed_transforms,
  min_tokens, allow_search_retrieval (not "allow_search").
- Evidence is attached via Pipeline.add(agent, evidence=policy), NOT @agent().
- StepAuditRecord.evidence is decision.legible():
    disabled   -> None            (policy gate skipped, no legible() call)
    simulate   -> contains "compressed" but NOT "handle" (handle_id=None in decision)
    compress   -> contains "compressed" AND "handle" (handle_id=<id> in decision)
- compress_for_model only runs for model-backed steps (spec.model is not None).
- InMemoryEvidenceStore.put(handle, *, original, compressed, ttl_seconds=None).
- InMemoryEvidenceStore.set_policy(handle_id, *, allow_full, allow_search).
- EVIDENCE_TOOL_REF = "evidence://retrieve"; retrieve args: handle_id, mode, query.
- Search returns a list (so len() works); a denied call raises EvidenceRetrievalError
  which is not a ToolError -> proxy logs "error" -> pipeline halts (step status "failed").
- test_mode(evidence_store=store) rebinds evidence://retrieve to store, replacing
  any original-store binding in the pipeline registry.
"""

import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.evidence import (
    EVIDENCE_TOOL_REF,
    EvidenceHandle,
    EvidencePolicy,
    InMemoryEvidenceStore,
    register_evidence_tool,
)
from drawbore.pipeline import Pipeline
from drawbore.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Shared models and helpers
# ---------------------------------------------------------------------------

class RowsIn(BaseModel):
    rows: list[dict]


class RowsOut(BaseModel):
    summary: str


def _evidence_pipeline(policy: EvidencePolicy) -> Pipeline:
    """A single-step model-backed pipeline with an evidence policy on the step.

    Evidence is declared on Pipeline.add (not @agent). The agent must have a
    model so the pipeline's evidence-compression gate fires (gate: policy.enabled
    AND spec.model is not None).
    """
    @agent(name="score", input=RowsIn, output=RowsOut, model="fake")
    async def score(v: RowsIn) -> RowsOut: ...

    p = Pipeline(name="aml", registry=ToolRegistry())
    p.add(score, evidence=policy)
    return p


def _big_rows() -> RowsIn:
    """40-row input whose model_dump() -> {"rows": [list[dict]]} satisfies the
    json_rows transform's _has_compressible_list check.  With min_tokens=1 the
    token gate always fires, so compression (or simulate-compression) runs."""
    return RowsIn(rows=[
        {"id": i, "amount": i * 10, "note": "x" * 20}
        for i in range(40)
    ])


def _step(trace, name: str):
    """Return the StepAuditRecord for the named agent."""
    return next(s for s in trace.step_records if s.agent == name)


# ---------------------------------------------------------------------------
# Spec test 17 — evidence modes
# ---------------------------------------------------------------------------

async def test_evidence_disabled_records_no_decision():   # spec test 17a
    """policy.enabled=False -> evidence gate is skipped -> StepAuditRecord.evidence is None."""
    p = _evidence_pipeline(EvidencePolicy(name="off", enabled=False))
    async with p.test_mode(mock_model_responses={"score": {"summary": "ok"}}) as test:
        result = await test.run(_big_rows())
    assert result.status == "completed"
    assert _step(result.audit_trace, "score").evidence is None


async def test_evidence_simulate_records_decision_without_a_handle():   # spec test 17b
    """simulate mode -> decision.legible() has "compressed" but handle_id=None -> no "handle" substring."""
    p = _evidence_pipeline(EvidencePolicy(
        name="sim", enabled=True, mode="simulate",
        allowed_transforms=("json_rows",), min_tokens=1,
    ))
    async with p.test_mode(mock_model_responses={"score": {"summary": "ok"}}) as test:
        result = await test.run(_big_rows())
    assert result.status == "completed"
    summary = _step(result.audit_trace, "score").evidence
    assert summary is not None, "simulate should record a decision"
    assert "compressed" in summary, f"expected 'compressed' in {summary!r}"
    assert "handle" not in summary, f"simulate must NOT store a handle; got {summary!r}"


async def test_evidence_compress_stores_original_and_records_handle():   # spec test 17c
    """compress mode -> decision has handle_id -> legible() contains both 'compressed' and 'handle',
    AND the handle is retrievable from the injected store via metadata() (non-policy-gated)."""
    p = _evidence_pipeline(EvidencePolicy(
        name="rows", enabled=True, mode="compress",
        allowed_transforms=("json_rows",), min_tokens=1,
        allow_search_retrieval=True,
    ))
    store = InMemoryEvidenceStore()
    async with p.test_mode(
        mock_model_responses={"score": {"summary": "ok"}},
        evidence_store=store,
    ) as test:
        result = await test.run(_big_rows())
    assert result.status == "completed"
    summary = _step(result.audit_trace, "score").evidence
    assert summary is not None, "compress should record a decision"
    assert "compressed" in summary, f"expected 'compressed' in {summary!r}"
    assert "handle" in summary, f"expected 'handle' in {summary!r}"
    # Direct store check: parse the handle id from legible() ("... handle <id>") and
    # confirm the store holds it via metadata() — the one non-policy-gated existence
    # accessor (store.py:96).  This proves put() was called with the correct handle_id,
    # i.e. the original was actually persisted, not merely referenced in the audit trail.
    handle_id = summary.split("handle ")[-1].split(";")[0].split(" ")[0].strip()
    stored_handle = store.metadata(handle_id)
    assert stored_handle.handle_id == handle_id, (
        f"store.metadata({handle_id!r}) returned a handle with a different id: "
        f"{stored_handle.handle_id!r}"
    )


# ---------------------------------------------------------------------------
# Helpers for spec tests 18 and 19 — retrieval via the proxy
# ---------------------------------------------------------------------------

def _handle(hid: str) -> EvidenceHandle:
    """Construct a minimal EvidenceHandle for seeding a store."""
    return EvidenceHandle(
        handle_id=hid,
        run_id="seed",
        step=0,
        source_agent="seed",
        content_type="application/json",
        original_hash="sha256:x",
        compressed_hash="sha256:y",
        original_tokens=100,
        compressed_tokens=10,
        transform="json_rows",
    )


class In(BaseModel):
    x: int


class Out(BaseModel):
    found: int


def _reader_pipeline(registry: ToolRegistry) -> Pipeline:
    """A deterministic agent that calls evidence://retrieve and returns the count
    of search results. Tests that the proxy routes the call and handles errors."""

    @agent(name="reader", input=In, output=Out, tools=[EVIDENCE_TOOL_REF])
    async def reader(v: In, tools) -> Out:
        rows = await tools.call(
            EVIDENCE_TOOL_REF, {"handle_id": "h1", "mode": "search", "query": ""}
        )
        return Out(found=len(rows))

    p = Pipeline(name="ret", registry=registry)
    p.add(reader)
    return p


# ---------------------------------------------------------------------------
# Spec test 18 — retrieval allowed and denied
# ---------------------------------------------------------------------------

async def test_evidence_retrieval_allowed_through_the_proxy():   # spec test 18a
    """Allowed search retrieval succeeds, result flows through proxy, step audit
    records the 'ok' call."""
    base = ToolRegistry()
    register_evidence_tool(base, store=InMemoryEvidenceStore())   # original store A
    p = _reader_pipeline(base)

    store_b = InMemoryEvidenceStore()
    store_b.put(_handle("h1"), original=[{"k": 1}, {"k": 2}], compressed=[])
    store_b.set_policy("h1", allow_full=False, allow_search=True)

    async with p.test_mode(evidence_store=store_b) as test:   # rebind to B
        result = await test.run(In(x=1))

    assert result.status == "completed"
    assert result.outputs["reader"].found == 2
    step = _step(result.audit_trace, "reader")
    assert any(
        f"{EVIDENCE_TOOL_REF} (invoke) -> ok" in tc
        for tc in step.tool_calls
    ), f"expected ok tool-call entry; got {step.tool_calls!r}"


async def test_evidence_retrieval_denied_halts_like_any_tool_denial():   # spec test 18b
    """Denied search (allow_search=False) raises EvidenceRetrievalError, which is
    not a ToolError: the proxy logs 'error', the pipeline halts, and the step
    record shows the failed call."""
    base = ToolRegistry()
    register_evidence_tool(base, store=InMemoryEvidenceStore())
    p = _reader_pipeline(base)

    store_b = InMemoryEvidenceStore()
    store_b.put(_handle("h1"), original=[{"k": 1}], compressed=[])
    store_b.set_policy("h1", allow_full=False, allow_search=False)   # search denied

    async with p.test_mode(evidence_store=store_b) as test:
        result = await test.run(In(x=1))

    assert result.status == "halted", f"expected halted; got {result.status!r}"
    step = _step(result.audit_trace, "reader")
    assert step.status != "ok", f"failed step should not be 'ok'; got {step.status!r}"


# ---------------------------------------------------------------------------
# Spec test 19 — test store shadows original binding
# ---------------------------------------------------------------------------

async def test_evidence_retrieve_is_rebound_to_the_test_store():   # spec test 19
    """The original registry binds evidence://retrieve to store A (empty — fails
    closed on unknown handle).  test_mode(evidence_store=store_b) must rebind it
    to B (which has 'h1').  Retrieval succeeding with found==3 proves B was reached,
    not A (A would fail closed on the unknown handle and halt the pipeline)."""
    store_a = InMemoryEvidenceStore()   # empty — has no 'h1'
    base = ToolRegistry()
    register_evidence_tool(base, store=store_a)
    p = _reader_pipeline(base)

    store_b = InMemoryEvidenceStore()
    store_b.put(
        _handle("h1"),
        original=[{"k": 1}, {"k": 2}, {"k": 3}],
        compressed=[],
    )
    store_b.set_policy("h1", allow_full=False, allow_search=True)

    async with p.test_mode(evidence_store=store_b) as test:
        result = await test.run(In(x=1))

    assert result.status == "completed", (
        f"expected completed (store B reached); got {result.status!r}; "
        f"reason={result.reason!r}"
    )
    assert result.outputs["reader"].found == 3, (
        f"expected 3 rows from store B; got {result.outputs['reader'].found}"
    )
