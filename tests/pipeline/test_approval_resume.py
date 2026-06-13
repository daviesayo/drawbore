# tests/pipeline/test_approval_resume.py
import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.escalation import ApprovalDecision
from drawbore.state import InMemoryCheckpointStore


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


@agent(name="gated", input=In, output=Out, requires_human_approval=True)
async def gated(v: In) -> Out:
    return Out(y=v.x + 1)


@agent(name="plain", input=In, output=Out)
async def plain(v: In) -> Out:
    return Out(y=v.x)


async def test_gate_populates_approval_request_and_records_it():
    p = Pipeline("approval-demo").add(gated)
    store = InMemoryCheckpointStore()
    result = await p.run(In(x=1), run_id="r-1", checkpoints=store)
    assert result.status in ("halted", "escalated")
    assert result.halt_code == "requires_human_approval"
    req = result.approval_request
    assert req is not None and req.step == "gated"
    assert req.proposed_output == {"y": 2}
    assert store.approval_request_of("r-1") == req


async def test_approve_resumes_and_completes_with_full_success_block():
    p = Pipeline("approval-demo").add(gated)
    store = InMemoryCheckpointStore()
    first = await p.run(In(x=1), run_id="r-1", checkpoints=store)
    decision = ApprovalDecision(
        request_id=first.approval_request.request_id, verdict="approved",
        reviewer_id="rev-1",
    )
    second = await p.run(In(x=1), run_id="r-1", checkpoints=store, approval=decision)
    assert second.status == "completed"
    assert second.outputs["gated"].y == 2
    step = second.audit_trace.step_records[0]
    assert step.human_decision == "approved" and step.reviewer_id == "rev-1"
    assert store.approval_request_of("r-1") is None
    third = await p.run(In(x=1), run_id="r-1", checkpoints=store)
    assert third.status == "completed"


async def test_amendment_passes_the_real_schema_gate():
    p = Pipeline("approval-demo").add(gated)
    store = InMemoryCheckpointStore()
    first = await p.run(In(x=1), run_id="r-1", checkpoints=store)
    good = ApprovalDecision(
        request_id=first.approval_request.request_id, verdict="amended",
        reviewer_id="rev-1", amended_output={"y": 99},
    )
    second = await p.run(In(x=1), run_id="r-1", checkpoints=store, approval=good)
    assert second.status == "completed"
    assert second.outputs["gated"].y == 99
    step = second.audit_trace.step_records[0]
    assert step.amendment_original_hash is not None
    assert step.amendment_applied_hash is not None
    assert step.amendment_original_hash != step.amendment_applied_hash


async def test_invalid_amendment_halts_schema_violation():
    p = Pipeline("approval-demo").add(gated)
    store = InMemoryCheckpointStore()
    first = await p.run(In(x=1), run_id="r-1", checkpoints=store)
    bad = ApprovalDecision(
        request_id=first.approval_request.request_id, verdict="amended",
        reviewer_id="rev-1", amended_output={"wrong": "shape"},
    )
    second = await p.run(In(x=1), run_id="r-1", checkpoints=store, approval=bad)
    assert second.status in ("halted", "escalated")
    assert second.halt_code == "schema_violation"


async def test_rejection_halts_and_a_rerun_mints_a_fresh_request():
    p = Pipeline("approval-demo").add(gated)
    store = InMemoryCheckpointStore()
    first = await p.run(In(x=1), run_id="r-1", checkpoints=store)
    rid = first.approval_request.request_id
    rejected = await p.run(
        In(x=1), run_id="r-1", checkpoints=store,
        approval=ApprovalDecision(
            request_id=rid, verdict="rejected", reviewer_id="rev-1",
            rationale="not convincing",
        ),
    )
    assert rejected.halt_code == "approval_rejected"
    assert "not convincing" in (rejected.reason or "")
    again = await p.run(In(x=1), run_id="r-1", checkpoints=store)
    assert again.halt_code == "requires_human_approval"
    assert again.approval_request.request_id != rid
    stale = await p.run(
        In(x=1), run_id="r-1", checkpoints=store,
        approval=ApprovalDecision(request_id=rid, verdict="approved", reviewer_id="rev-1"),
    )
    assert stale.halt_code == "approval_error"


async def test_decisionless_resume_keeps_the_stored_request():
    p = Pipeline("approval-demo").add(gated)
    store = InMemoryCheckpointStore()
    first = await p.run(In(x=1), run_id="r-1", checkpoints=store)
    second = await p.run(In(x=1), run_id="r-1", checkpoints=store)
    assert second.halt_code == "requires_human_approval"
    assert second.approval_request.request_id == first.approval_request.request_id


async def test_approval_without_store_is_a_value_error():
    p = Pipeline("approval-demo").add(gated)
    with pytest.raises(ValueError, match="approval"):
        await p.run(In(x=1), approval=ApprovalDecision(
            request_id="x", verdict="approved", reviewer_id="rev-1",
        ))


async def test_unconsumed_decision_halts_instead_of_silently_ignoring():
    p = Pipeline("no-gate").add(plain)
    store = InMemoryCheckpointStore()
    result = await p.run(
        In(x=1), run_id="r-2", checkpoints=store,
        approval=ApprovalDecision(request_id="x", verdict="approved", reviewer_id="rev-1"),
    )
    assert result.halt_code == "approval_error"
