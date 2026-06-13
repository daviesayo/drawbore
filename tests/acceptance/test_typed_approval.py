# tests/acceptance/test_typed_approval.py
"""Acceptance: typed approval cycle end to end through test_mode.

A two-step remittance-flavoured pipeline:
  step 1 — ``retriever``: deterministic one-shot model agent, produces a value.
  step 2 — ``gated_writer``: one-shot model agent with ``requires_human_approval``,
            bound to step 1's output via ``From``.
  step 3 — ``reporter``: deterministic agent (downstream consumer), bound to step 2's
            output, used to confirm that an AMENDED value flows downstream.

The pipeline is built fresh in each test so no state leaks between cases. The two-phase
cycle (fire → approve) is handled by opening the test_mode context twice, each backed
by the SAME ``InMemoryCheckpointStore`` and ``run_id``, matching the idiom in
``tests/pipeline/test_approval_resume.py::test_approval_preserves_untrusted_output_trust``.

The REAL safety layer decides every outcome; only externals are mocked.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.escalation import ApprovalDecision, EscalationPolicy, RecordingDispatcher
from drawbore.pipeline.binding import From
from drawbore.state import InMemoryCheckpointStore


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TxIn(BaseModel):
    amount: int


class TxRecord(BaseModel):
    amount: int
    currency: str


class Decision(BaseModel):
    approved: bool
    note: str


class Report(BaseModel):
    final_note: str


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


@agent(name="retriever", input=TxIn, output=TxRecord, model="m-1.0")
async def retriever(v: TxIn) -> TxRecord: ...  # one-shot model


@agent(
    name="gated_writer",
    input=TxRecord,
    output=Decision,
    model="m-1.0",
    requires_human_approval=True,
)
async def gated_writer(v: TxRecord) -> Decision: ...  # one-shot model, GATED


@agent(name="reporter", input=Decision, output=Report)
async def reporter(v: Decision) -> Report:
    return Report(final_note=v.note)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POLICY = EscalationPolicy(channel="human", target="ops", mode="sync")

RETRIEVER_RESPONSE = {"amount": 5000, "currency": "USD"}
GATED_RESPONSE = {"approved": True, "note": "looks good"}


def _pipeline(*, dispatcher: RecordingDispatcher | None = None) -> Pipeline:
    """Fresh two-step pipeline each time so no accumulated state."""
    d = dispatcher or RecordingDispatcher()
    p = (
        Pipeline("typed-approval-acceptance", on_failure=POLICY, dispatcher=d)
        .add(retriever)
        .add(
            gated_writer,
            inputs={
                "amount": From("retriever.amount"),
                "currency": From("retriever.currency"),
            },
            depends_on=["retriever"],
        )
    )
    return p


def _pipeline_with_reporter(*, dispatcher: RecordingDispatcher | None = None) -> Pipeline:
    """Three-step pipeline used only when we need downstream consumption."""
    d = dispatcher or RecordingDispatcher()
    p = (
        Pipeline("typed-approval-downstream", on_failure=POLICY, dispatcher=d)
        .add(retriever)
        .add(
            gated_writer,
            inputs={
                "amount": From("retriever.amount"),
                "currency": From("retriever.currency"),
            },
            depends_on=["retriever"],
        )
        .add(
            reporter,
            inputs={"approved": From("gated_writer.approved"), "note": From("gated_writer.note")},
            depends_on=["gated_writer"],
        )
    )
    return p


MOCKS = dict(
    mock_model_responses={
        "retriever": RETRIEVER_RESPONSE,
        "gated_writer": GATED_RESPONSE,
    }
)


# ---------------------------------------------------------------------------
# Case 1: full cycle — gate fires, approval given, audit records decision
# ---------------------------------------------------------------------------


async def test_full_cycle_gate_fires_then_approved():
    d = RecordingDispatcher()
    p = _pipeline(dispatcher=d)
    store = InMemoryCheckpointStore()
    run_id = "ta-cycle-1"

    # Phase 1: first run — gate fires, halts for approval.
    async with p.test_mode(**MOCKS) as tp:
        first = await tp.run(TxIn(amount=5000), run_id=run_id, checkpoints=store)

    assert first.status in ("halted", "escalated")
    assert first.halt_code == "requires_human_approval"
    req = first.approval_request
    assert req is not None
    assert req.step == "gated_writer"
    assert req.proposed_output == GATED_RESPONSE
    # The RecordingDispatcher on the pipeline should have recorded a dispatch.
    assert len(d.sent) == 1

    # Phase 2: resume with approval decision.
    decision = ApprovalDecision(
        request_id=req.request_id,
        verdict="approved",
        reviewer_id="compliance-1",
    )
    async with p.test_mode(**MOCKS) as tp:
        second = await tp.run(
            TxIn(amount=5000), run_id=run_id, checkpoints=store, approval=decision
        )

    assert second.status == "completed"
    assert second.outputs["gated_writer"].approved is True

    # Audit trace must carry the human decision.
    step_record = next(
        (s for s in second.audit_trace.step_records if s.agent == "gated_writer"), None
    )
    assert step_record is not None
    assert step_record.human_decision == "approved"
    assert step_record.reviewer_id == "compliance-1"

    # Approval request should be consumed (cleared from store).
    assert store.approval_request_of(run_id) is None


# ---------------------------------------------------------------------------
# Case 2: amendment consumed downstream
# ---------------------------------------------------------------------------


async def test_amendment_consumed_downstream():
    p = _pipeline_with_reporter()
    store = InMemoryCheckpointStore()
    run_id = "ta-amend-downstream"

    # Phase 1: gate fires.
    async with p.test_mode(**MOCKS) as tp:
        first = await tp.run(TxIn(amount=5000), run_id=run_id, checkpoints=store)

    assert first.halt_code == "requires_human_approval"
    req = first.approval_request

    # Phase 2: reviewer amends the output with a different note.
    amended = ApprovalDecision(
        request_id=req.request_id,
        verdict="amended",
        reviewer_id="compliance-2",
        amended_output={"approved": True, "note": "reviewer-amended"},
    )
    async with p.test_mode(**MOCKS) as tp:
        second = await tp.run(
            TxIn(amount=5000), run_id=run_id, checkpoints=store, approval=amended
        )

    assert second.status == "completed"
    # The amended value must have been recorded on the gated step.
    assert second.outputs["gated_writer"].note == "reviewer-amended"
    # The downstream reporter must have received the amended value.
    assert second.outputs["reporter"].final_note == "reviewer-amended"

    step_record = next(
        (s for s in second.audit_trace.step_records if s.agent == "gated_writer"), None
    )
    assert step_record is not None
    assert step_record.amendment_original_hash is not None
    assert step_record.amendment_applied_hash is not None
    assert step_record.amendment_original_hash != step_record.amendment_applied_hash


# ---------------------------------------------------------------------------
# Case 3: invalid amendment → schema_violation; downstream never ran
# ---------------------------------------------------------------------------


async def test_invalid_amendment_schema_violation():
    p = _pipeline_with_reporter()
    store = InMemoryCheckpointStore()
    run_id = "ta-invalid-amend"

    async with p.test_mode(**MOCKS) as tp:
        first = await tp.run(TxIn(amount=5000), run_id=run_id, checkpoints=store)

    assert first.halt_code == "requires_human_approval"

    bad = ApprovalDecision(
        request_id=first.approval_request.request_id,
        verdict="amended",
        reviewer_id="compliance-3",
        amended_output={"totally": "wrong_shape"},
    )
    async with p.test_mode(**MOCKS) as tp:
        second = await tp.run(
            TxIn(amount=5000), run_id=run_id, checkpoints=store, approval=bad
        )

    assert second.status in ("halted", "escalated")
    assert second.halt_code == "schema_violation"
    # Reporter must not have executed.
    assert "reporter" not in second.outputs
    # Invalid amendment consumes/clears the pending request.
    assert store.approval_request_of(run_id) is None


# ---------------------------------------------------------------------------
# Case 4: rejection → approval_rejected; rationale appears in result
# ---------------------------------------------------------------------------


async def test_rejection_halts_with_rationale():
    p = _pipeline()
    store = InMemoryCheckpointStore()
    run_id = "ta-reject"

    async with p.test_mode(**MOCKS) as tp:
        first = await tp.run(TxIn(amount=5000), run_id=run_id, checkpoints=store)

    assert first.halt_code == "requires_human_approval"

    rejected = ApprovalDecision(
        request_id=first.approval_request.request_id,
        verdict="rejected",
        reviewer_id="compliance-4",
        rationale="amount exceeds threshold",
    )
    async with p.test_mode(**MOCKS) as tp:
        second = await tp.run(
            TxIn(amount=5000), run_id=run_id, checkpoints=store, approval=rejected
        )

    assert second.halt_code == "approval_rejected"
    # The rationale must surface directly in the result reason.
    assert second.reason is not None
    assert "amount exceeds threshold" in second.reason


# ---------------------------------------------------------------------------
# Case 5: wrong request_id → approval_error; replayed decision → approval_error
# ---------------------------------------------------------------------------


async def test_wrong_request_id_and_replay_both_halt_approval_error():
    p = _pipeline()
    store = InMemoryCheckpointStore()
    run_id = "ta-binding"

    # Gate fires.
    async with p.test_mode(**MOCKS) as tp:
        first = await tp.run(TxIn(amount=5000), run_id=run_id, checkpoints=store)

    assert first.halt_code == "requires_human_approval"
    real_request_id = first.approval_request.request_id

    # Wrong request_id → approval_error.
    wrong_id = ApprovalDecision(
        request_id="definitely-not-the-right-id",
        verdict="approved",
        reviewer_id="compliance-5",
    )
    async with p.test_mode(**MOCKS) as tp:
        wrong = await tp.run(
            TxIn(amount=5000), run_id=run_id, checkpoints=store, approval=wrong_id
        )
    assert wrong.halt_code == "approval_error"

    # The gate is still pending (wrong_id was refused without consuming the request).
    assert store.approval_request_of(run_id) is not None

    # Now approve correctly to consume the request.
    good = ApprovalDecision(
        request_id=real_request_id,
        verdict="approved",
        reviewer_id="compliance-5",
    )
    async with p.test_mode(**MOCKS) as tp:
        approved = await tp.run(
            TxIn(amount=5000), run_id=run_id, checkpoints=store, approval=good
        )
    assert approved.status == "completed"
    assert store.approval_request_of(run_id) is None

    # Replay the same decision → approval_error (request already consumed).
    async with p.test_mode(**MOCKS) as tp:
        replayed = await tp.run(
            TxIn(amount=5000), run_id=run_id, checkpoints=store, approval=good
        )
    assert replayed.halt_code == "approval_error"


# ---------------------------------------------------------------------------
# Case 6: decision-less resume re-escalates with the SAME request_id
# ---------------------------------------------------------------------------


async def test_decisionless_resume_same_request_id():
    p = _pipeline()
    store = InMemoryCheckpointStore()
    run_id = "ta-poll"

    # Gate fires — mint request.
    async with p.test_mode(**MOCKS) as tp:
        first = await tp.run(TxIn(amount=5000), run_id=run_id, checkpoints=store)

    assert first.halt_code == "requires_human_approval"
    original_request_id = first.approval_request.request_id

    # Decision-less resume — must re-surface the SAME request_id.
    async with p.test_mode(**MOCKS) as tp:
        polled = await tp.run(TxIn(amount=5000), run_id=run_id, checkpoints=store)

    assert polled.halt_code == "requires_human_approval"
    assert polled.approval_request.request_id == original_request_id

    # Another decision-less poll — still the same.
    async with p.test_mode(**MOCKS) as tp:
        polled2 = await tp.run(TxIn(amount=5000), run_id=run_id, checkpoints=store)

    assert polled2.halt_code == "requires_human_approval"
    assert polled2.approval_request.request_id == original_request_id

    # Eventually approve, then confirm completed.
    async with p.test_mode(**MOCKS) as tp:
        final = await tp.run(
            TxIn(amount=5000),
            run_id=run_id,
            checkpoints=store,
            approval=ApprovalDecision(
                request_id=original_request_id,
                verdict="approved",
                reviewer_id="compliance-6",
            ),
        )
    assert final.status == "completed"


# ---------------------------------------------------------------------------
# Case 7: re-resume after approval completes → status=completed (no re-run)
# ---------------------------------------------------------------------------


async def test_re_resume_after_completion_returns_completed():
    p = _pipeline()
    store = InMemoryCheckpointStore()
    run_id = "ta-reruns"

    # Phase 1: gate fires.
    async with p.test_mode(**MOCKS) as tp:
        first = await tp.run(TxIn(amount=5000), run_id=run_id, checkpoints=store)

    assert first.halt_code == "requires_human_approval"

    # Phase 2: approve.
    async with p.test_mode(**MOCKS) as tp:
        approved = await tp.run(
            TxIn(amount=5000),
            run_id=run_id,
            checkpoints=store,
            approval=ApprovalDecision(
                request_id=first.approval_request.request_id,
                verdict="approved",
                reviewer_id="compliance-7",
            ),
        )

    assert approved.status == "completed"

    # Phase 3: a third run with the same run_id — all steps already sealed; must
    # return completed without re-executing the gated agent.
    async with p.test_mode(**MOCKS) as tp:
        third = await tp.run(TxIn(amount=5000), run_id=run_id, checkpoints=store)

    assert third.status == "completed"
