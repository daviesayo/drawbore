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
    assert store.approval_request_of("r-1") == req.model_dump(mode="json")


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


# ---------------------------------------------------------------------------
# Fix 1 & Fix 2 tests
# ---------------------------------------------------------------------------

_poll_counter = 0


async def test_decisionless_poll_does_not_re_execute_the_gated_agent():
    """A decision-less resume must re-surface the pending request WITHOUT
    re-executing the gated agent (no double-fire)."""
    global _poll_counter
    _poll_counter = 0

    @agent(name="counted_gated", input=In, output=Out, requires_human_approval=True)
    async def counted_gated(v: In) -> Out:
        global _poll_counter
        _poll_counter += 1
        return Out(y=v.x + 1)

    p = Pipeline("poll-counter").add(counted_gated)
    store = InMemoryCheckpointStore()

    # First run: executes the agent and halts at the gate.
    first = await p.run(In(x=1), run_id="r-poll", checkpoints=store)
    assert first.halt_code == "requires_human_approval"
    assert _poll_counter == 1, "agent must execute exactly once at mint"
    first_request_id = first.approval_request.request_id

    # Decision-less poll 1: must NOT re-execute the agent.
    second = await p.run(In(x=1), run_id="r-poll", checkpoints=store)
    assert second.halt_code == "requires_human_approval"
    assert _poll_counter == 1, "agent must NOT re-execute on a decision-less poll"
    assert second.approval_request.request_id == first_request_id

    # Decision-less poll 2: still no re-execution.
    third = await p.run(In(x=1), run_id="r-poll", checkpoints=store)
    assert third.halt_code == "requires_human_approval"
    assert _poll_counter == 1, "agent must NOT re-execute on repeated decision-less polls"
    assert third.approval_request.request_id == first_request_id


async def test_approval_preserves_untrusted_output_trust():
    """Approval must not launder an UNTRUSTED output trust to TRUSTED.

    Setup: a gated model+tools agent calls an UNTRUSTED-source tool during
    execution, widening its trust scope to UNTRUSTED.  A downstream step is
    guarded by a When condition on the gated agent's output.

    Expected: after the original run halts requires_human_approval and the
    reviewer approves, the downstream When gate reads the restored UNTRUSTED
    trust label and halts condition_tainted — proving taint was never laundered.

    Without Fix 1 the approval path would record TRUSTED (derived from the
    seed) and the downstream step would be reached, which would make this test
    fail (condition_tainted not raised).
    """
    from drawbore import From, When
    from drawbore.testing import call, final
    from drawbore.tools import ToolRegistry, TrustLabel

    UNTRUSTED_SRC = "ar:untrusted_src"

    async def _handler(args):
        return {"data": "external"}

    reg = ToolRegistry()
    reg.register_mcp_tool(UNTRUSTED_SRC, _handler, source_trust=TrustLabel.UNTRUSTED)

    class TxIn(BaseModel):
        val: int

    class TxMid(BaseModel):
        result: int

    class TxOut(BaseModel):
        ok: bool

    @agent(
        name="taint_gated",
        input=TxIn,
        output=TxMid,
        model="m-1.0",
        tools=[UNTRUSTED_SRC],
        requires_human_approval=True,
    )
    async def taint_gated(v: TxIn, tools) -> TxMid: ...

    @agent(name="downstream", input=TxMid, output=TxOut)
    async def downstream(v: TxMid) -> TxOut:
        return TxOut(ok=True)

    p = (
        Pipeline("taint-approval", registry=reg)
        .add(taint_gated)
        .add(
            downstream,
            inputs={"result": From("taint_gated.result")},
            when=When("taint_gated.result", equals=42),
        )
    )
    store = InMemoryCheckpointStore()

    # Run 1: taint_gated executes (touching the UNTRUSTED source), widens trust
    # to UNTRUSTED, then halts at the approval gate.
    async with p.test_mode(
        mock_loop_scripts={
            "taint_gated": [call(UNTRUSTED_SRC), final({"result": 42})],
        },
        mock_tools={UNTRUSTED_SRC: {"data": "external"}},
    ) as tp:
        first = await tp.run(TxIn(val=1), run_id="r-taint", checkpoints=store)

    assert first.halt_code == "requires_human_approval", (
        f"expected requires_human_approval, got {first.halt_code}: {first.reason}"
    )
    # The minted ApprovalRequest must carry the UNTRUSTED label (Fix 1 field).
    assert first.approval_request.proposed_output_trust == "untrusted", (
        f"expected proposed_output_trust='untrusted', "
        f"got {first.approval_request.proposed_output_trust!r}"
    )

    decision = ApprovalDecision(
        request_id=first.approval_request.request_id,
        verdict="approved",
        reviewer_id="rev-compliance",
    )

    # Run 2 (approval resume): the downstream step reads the gated output via
    # a When condition.  Because trust is restored to UNTRUSTED (Fix 1), the
    # branch-gate must halt condition_tainted — not proceed to the downstream step.
    async with p.test_mode(
        mock_loop_scripts={},  # taint_gated must NOT re-run
        mock_tools={},
    ) as tp:
        second = await tp.run(
            TxIn(val=1), run_id="r-taint", checkpoints=store, approval=decision
        )

    assert second.halt_code == "condition_tainted", (
        f"expected condition_tainted (taint preserved), "
        f"got {second.halt_code!r}: {second.reason}"
    )
