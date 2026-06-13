# tests/acceptance/test_low_confidence_approval.py
"""Acceptance: resumable low-confidence approval cycle through test_mode.

A ``HasConfidence`` step whose ``confidence`` is below the declared threshold no
longer dies terminally in sync mode: when a checkpoint store is present it mints a
resumable ``ApprovalRequest`` and halts ``confidence_approval_pending``, resolvable
via the same ``pipeline.run(..., approval=ApprovalDecision)`` approve/amend/reject
cycle as the explicit ``requires_human_approval`` gate.

These tests mirror ``tests/acceptance/test_typed_approval.py``: fresh pipeline per
test, the two-phase cycle handled by opening ``test_mode`` twice over the SAME
``InMemoryCheckpointStore`` + ``run_id``. The REAL safety layer decides every
outcome.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.escalation import (
    ApprovalDecision,
    EscalationPolicy,
    HasConfidence,
    RecordingDispatcher,
)
from drawbore.pipeline.binding import From
from drawbore.state import InMemoryCheckpointStore


# ---------------------------------------------------------------------------
# Schemas + a counter-backed scorer (to prove the agent is not re-run on poll)
# ---------------------------------------------------------------------------


class Seed(BaseModel):
    value: int


class Score(BaseModel, HasConfidence):
    score: int
    confidence: float


_scorer_calls = 0


@agent(name="scorer", input=Seed, output=Score)
async def scorer(v: Seed) -> Score:
    global _scorer_calls
    _scorer_calls += 1
    return Score(score=v.value, confidence=0.3)  # below the 0.5 threshold


# A model+tools scorer used only by the taint test (touches an UNTRUSTED source).
class TaintIn(BaseModel):
    val: int


class TaintScore(BaseModel, HasConfidence):
    result: int
    confidence: float


class TaintOut(BaseModel):
    ok: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sync_pipeline() -> Pipeline:
    """Sync pipeline (no on_failure policy → a plain halt), confidence-checked."""
    return Pipeline("low-conf-sync", confidence_threshold=0.5).add(scorer)


# ---------------------------------------------------------------------------
# Case 1: sync + store mints a resumable request; decision-less poll re-surfaces
#          the SAME request and does NOT re-run the agent.
# ---------------------------------------------------------------------------


async def test_low_confidence_mints_resumable_request_and_poll_does_not_rerun():
    global _scorer_calls
    _scorer_calls = 0
    p = _sync_pipeline()
    store = InMemoryCheckpointStore()
    run_id = "lc-mint"

    async with p.test_mode() as tp:
        first = await tp.run(Seed(value=7), run_id=run_id, checkpoints=store)

    assert first.status == "halted"
    assert first.halt_code == "confidence_approval_pending"
    req = first.approval_request
    assert req is not None
    assert req.step == "scorer"
    assert req.reason == "confidence_below_threshold"
    assert req.proposed_output == {"score": 7, "confidence": 0.3}
    assert _scorer_calls == 1

    # Decision-less re-poll: must re-surface the SAME request with the SAME code,
    # WITHOUT re-running the agent.
    async with p.test_mode() as tp:
        polled = await tp.run(Seed(value=7), run_id=run_id, checkpoints=store)

    assert polled.halt_code == "confidence_approval_pending"
    assert polled.approval_request.request_id == req.request_id
    assert _scorer_calls == 1, "agent must NOT re-run on a decision-less poll"


# ---------------------------------------------------------------------------
# Case 2: approve → completes with the ORIGINAL low-confidence output.
# ---------------------------------------------------------------------------


async def test_approve_completes_with_original_output():
    p = _sync_pipeline()
    store = InMemoryCheckpointStore()
    run_id = "lc-approve"

    async with p.test_mode() as tp:
        first = await tp.run(Seed(value=7), run_id=run_id, checkpoints=store)
    assert first.halt_code == "confidence_approval_pending"

    decision = ApprovalDecision(
        request_id=first.approval_request.request_id,
        verdict="approved",
        reviewer_id="compliance-1",
    )
    async with p.test_mode() as tp:
        second = await tp.run(
            Seed(value=7), run_id=run_id, checkpoints=store, approval=decision
        )

    assert second.status == "completed"
    assert second.outputs["scorer"].confidence == 0.3
    assert second.outputs["scorer"].score == 7
    assert store.approval_request_of(run_id) is None


# ---------------------------------------------------------------------------
# Case 3: amend to a HIGHER-confidence output → completes with the amended output
#          (the trigger does not re-fire).
# ---------------------------------------------------------------------------


async def test_amend_to_higher_confidence_completes():
    p = _sync_pipeline()
    store = InMemoryCheckpointStore()
    run_id = "lc-amend-high"

    async with p.test_mode() as tp:
        first = await tp.run(Seed(value=7), run_id=run_id, checkpoints=store)
    assert first.halt_code == "confidence_approval_pending"

    amended = ApprovalDecision(
        request_id=first.approval_request.request_id,
        verdict="amended",
        reviewer_id="compliance-2",
        amended_output={"score": 7, "confidence": 0.95},
    )
    async with p.test_mode() as tp:
        second = await tp.run(
            Seed(value=7), run_id=run_id, checkpoints=store, approval=amended
        )

    assert second.status == "completed"
    assert second.outputs["scorer"].confidence == 0.95


# ---------------------------------------------------------------------------
# Case 4: amend to a STILL-below-threshold output → completes (human accepted it;
#          NO re-mint loop — the trigger does not re-fire on the decision path).
# ---------------------------------------------------------------------------


async def test_amend_still_below_threshold_completes_no_remint():
    p = _sync_pipeline()
    store = InMemoryCheckpointStore()
    run_id = "lc-amend-low"

    async with p.test_mode() as tp:
        first = await tp.run(Seed(value=7), run_id=run_id, checkpoints=store)
    assert first.halt_code == "confidence_approval_pending"

    amended = ApprovalDecision(
        request_id=first.approval_request.request_id,
        verdict="amended",
        reviewer_id="compliance-3",
        amended_output={"score": 7, "confidence": 0.1},  # still below 0.5
    )
    async with p.test_mode() as tp:
        second = await tp.run(
            Seed(value=7), run_id=run_id, checkpoints=store, approval=amended
        )

    assert second.status == "completed"
    assert second.outputs["scorer"].confidence == 0.1
    assert store.approval_request_of(run_id) is None


# ---------------------------------------------------------------------------
# Case 5: reject → halts approval_rejected.
# ---------------------------------------------------------------------------


async def test_reject_halts_approval_rejected():
    p = _sync_pipeline()
    store = InMemoryCheckpointStore()
    run_id = "lc-reject"

    async with p.test_mode() as tp:
        first = await tp.run(Seed(value=7), run_id=run_id, checkpoints=store)
    assert first.halt_code == "confidence_approval_pending"

    rejected = ApprovalDecision(
        request_id=first.approval_request.request_id,
        verdict="rejected",
        reviewer_id="compliance-4",
        rationale="confidence too low to accept",
    )
    async with p.test_mode() as tp:
        second = await tp.run(
            Seed(value=7), run_id=run_id, checkpoints=store, approval=rejected
        )

    assert second.halt_code == "approval_rejected"
    assert "confidence too low to accept" in (second.reason or "")


# ---------------------------------------------------------------------------
# Case 6: async mode → dispatches review and CONTINUES (unchanged; no mint).
# ---------------------------------------------------------------------------


async def test_async_mode_dispatches_and_continues():
    global _scorer_calls
    _scorer_calls = 0
    d = RecordingDispatcher()
    p = Pipeline(
        "low-conf-async",
        on_failure=EscalationPolicy("email", "ops", mode="async"),
        dispatcher=d,
        confidence_threshold=0.5,
    ).add(scorer)
    store = InMemoryCheckpointStore()
    run_id = "lc-async"

    async with p.test_mode() as tp:
        result = await tp.run(Seed(value=7), run_id=run_id, checkpoints=store)

    assert result.status == "completed"          # async review does not block
    assert result.approval_request is None        # no resumable mint in async mode
    assert len(result.escalations) == 1
    assert len(d.sent) == 1
    # No pending request was recorded.
    assert store.approval_request_of(run_id) is None


# ---------------------------------------------------------------------------
# Case 7: sync, NO store → terminal confidence_below_threshold halt (no request).
# ---------------------------------------------------------------------------


async def test_sync_no_store_terminal_halt():
    p = _sync_pipeline()

    async with p.test_mode() as tp:
        result = await tp.run(Seed(value=7))  # no checkpoints

    assert result.status == "halted"
    assert result.halt_code == "confidence_below_threshold"
    assert result.approval_request is None


# ---------------------------------------------------------------------------
# Case 8 (CRITICAL anti-bypass): a step with requires_human_approval=True AND a
#   below-threshold output reaches the EXPLICIT approval gate, NOT a
#   confidence_below_threshold / confidence_approval_pending halt.
# ---------------------------------------------------------------------------


async def test_both_triggers_reach_explicit_gate_not_confidence_halt():
    @agent(
        name="dual",
        input=Seed,
        output=Score,
        requires_human_approval=True,
    )
    async def dual(v: Seed) -> Score:
        return Score(score=v.value, confidence=0.2)  # below threshold AND gated

    p = Pipeline("both-triggers", confidence_threshold=0.5).add(dual)
    store = InMemoryCheckpointStore()
    run_id = "lc-both"

    async with p.test_mode() as tp:
        first = await tp.run(Seed(value=7), run_id=run_id, checkpoints=store)

    assert first.halt_code == "requires_human_approval", (
        f"both-triggers step must reach the explicit gate, got {first.halt_code}"
    )
    req = first.approval_request
    assert req is not None
    assert req.reason == "requires_human_approval"

    # And it resolves through the explicit gate as a normal approval.
    decision = ApprovalDecision(
        request_id=req.request_id, verdict="approved", reviewer_id="compliance-8"
    )
    async with p.test_mode() as tp:
        second = await tp.run(
            Seed(value=7), run_id=run_id, checkpoints=store, approval=decision
        )
    assert second.status == "completed"
    assert second.outputs["dual"].confidence == 0.2


# ---------------------------------------------------------------------------
# Case 9: taint preserved — an approved low-confidence output keeps its pinned
#   UNTRUSTED trust (UNTRUSTED stays UNTRUSTED downstream).
# ---------------------------------------------------------------------------


async def test_taint_preserved_across_approved_low_confidence():
    from drawbore import From as _From, When
    from drawbore.testing import call, final
    from drawbore.tools import ToolRegistry, TrustLabel

    UNTRUSTED_SRC = "ar:untrusted_src"

    async def _handler(args):
        return {"data": "external"}

    reg = ToolRegistry()
    reg.register_mcp_tool(UNTRUSTED_SRC, _handler, source_trust=TrustLabel.UNTRUSTED)

    @agent(
        name="taint_scorer",
        input=TaintIn,
        output=TaintScore,
        model="m-1.0",
        tools=[UNTRUSTED_SRC],
    )
    async def taint_scorer(v: TaintIn, tools) -> TaintScore: ...

    @agent(name="downstream", input=TaintScore, output=TaintOut)
    async def downstream(v: TaintScore) -> TaintOut:
        return TaintOut(ok=True)

    p = (
        Pipeline("taint-low-conf", registry=reg, confidence_threshold=0.5)
        .add(taint_scorer)
        .add(
            downstream,
            inputs={"result": _From("taint_scorer.result"), "confidence": _From("taint_scorer.confidence")},
            when=When("taint_scorer.result", equals=42),
        )
    )
    store = InMemoryCheckpointStore()
    run_id = "lc-taint"

    # Run 1: taint_scorer touches the UNTRUSTED source, widens trust to UNTRUSTED,
    # produces a below-threshold confidence, and halts at the resumable mint.
    async with p.test_mode(
        mock_loop_scripts={
            "taint_scorer": [call(UNTRUSTED_SRC), final({"result": 42, "confidence": 0.3})],
        },
        mock_tools={UNTRUSTED_SRC: {"data": "external"}},
    ) as tp:
        first = await tp.run(TaintIn(val=1), run_id=run_id, checkpoints=store)

    assert first.halt_code == "confidence_approval_pending", (
        f"expected confidence_approval_pending, got {first.halt_code}: {first.reason}"
    )
    assert first.approval_request.proposed_output_trust == "untrusted"

    decision = ApprovalDecision(
        request_id=first.approval_request.request_id,
        verdict="approved",
        reviewer_id="rev-compliance",
    )

    # Run 2 (approval resume): the downstream When gate reads the restored
    # UNTRUSTED trust and halts condition_tainted — proving taint was preserved.
    async with p.test_mode(mock_loop_scripts={}, mock_tools={}) as tp:
        second = await tp.run(
            TaintIn(val=1), run_id=run_id, checkpoints=store, approval=decision
        )

    assert second.halt_code == "condition_tainted", (
        f"expected condition_tainted (taint preserved), got {second.halt_code!r}: {second.reason}"
    )
