# tests/acceptance/test_effect_ledger_resume.py
"""Acceptance: exactly-once effectful-tool resume through test_mode.

A pipeline with one effectful agent exercises the durable effect ledger end to
end through ``pipeline.test_mode``:

1. ``test_exactly_once_effect_resume_through_test_mode`` — a single step fires
   effect DEBIT (recorded succeeded), then crashes mid-step BEFORE calling
   RECEIPT. The step does NOT complete so it is not checkpointed; on resume it
   re-enters the proxy: DEBIT is replayed from the ledger (handler not re-called)
   and RECEIPT fires fresh. Across both attempts each handler fires exactly once —
   the ledger replay path is actively exercised by the resume.

2. ``test_effect_divergence_halts_through_test_mode`` — the ledger is pre-seeded
   with a SUCCEEDED entry for a step's first effect; on resume the step makes a
   DIFFERENT effectful call at that position; the run halts ``effect_divergence``.

The REAL safety layer (proxy, ledger, checkpoint) decides every outcome; only
externals (tool handlers, model responses) are mocked through test_mode. The
intra-step crash mechanism mirrors ``tests/pipeline/test_effect_resume.py``
(a crash-flag dict that causes the agent to raise after the first effectful call
on its first attempt), driven here through ``pipeline.test_mode``.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.state import (
    InMemoryCheckpointStore,
    InMemoryEffectLedger,
)
from drawbore.state.effect_ledger import (
    EffectEntry,
    EffectStatus,
    ledger_args_hash,
)
from drawbore.escalation import EscalationPolicy


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Request(BaseModel):
    tx_id: str


class EffectResult(BaseModel):
    recorded: bool


# ---------------------------------------------------------------------------
# Tool refs
# ---------------------------------------------------------------------------

DEBIT = "ledger-acc:debit"
RECEIPT = "ledger-acc:receipt"

# ---------------------------------------------------------------------------
# Module-level call counter (reset by fixture)
# ---------------------------------------------------------------------------

HANDLER_CALLS: dict[str, int] = {"debit": 0, "receipt": 0}


@pytest.fixture(autouse=True)
def _reset_calls():
    HANDLER_CALLS["debit"] = 0
    HANDLER_CALLS["receipt"] = 0


# ---------------------------------------------------------------------------
# Shared policy
# ---------------------------------------------------------------------------

POLICY = EscalationPolicy(channel="human", target="ops", mode="sync")


# ---------------------------------------------------------------------------
# Case 1: exactly-once on resume through test_mode (intra-step crash)
# ---------------------------------------------------------------------------


async def test_exactly_once_effect_resume_through_test_mode():
    """A single step fires effect DEBIT (recorded), then crashes mid-step BEFORE
    calling RECEIPT. The step does NOT complete so it is not checkpointed.

    On resume the step re-enters the proxy: DEBIT is replayed from the ledger
    (handler not re-called) and RECEIPT fires fresh. Across both attempts each
    handler fires exactly once — the assertion on DEBIT is load-bearing on the
    ledger: without ``effect_ledger=`` the resumed step would call ``debit_handler``
    a second time, making ``HANDLER_CALLS["debit"] == 2``.
    """
    crash_flag = {"crashed": False}

    from drawbore.tools import ToolRegistry

    reg = ToolRegistry()

    async def debit_handler(args):
        HANDLER_CALLS["debit"] += 1
        return {"debited": True}

    async def receipt_handler(args):
        HANDLER_CALLS["receipt"] += 1
        return {"receipt": True}

    reg.register_tool(DEBIT, debit_handler)
    reg.register_tool(RECEIPT, receipt_handler)

    @agent(name="two-effect-step", input=Request, output=EffectResult, tools=[DEBIT, RECEIPT])
    async def two_effect_step(v: Request, tools) -> EffectResult:
        await tools.call(DEBIT, {"tx_id": v.tx_id, "amount": 100})
        # Simulate an intra-step crash after DEBIT is recorded succeeded but
        # before RECEIPT is attempted. The step does not complete (it raises),
        # so it is not checkpointed — on resume the proxy re-enters it and
        # DEBIT must replay from the ledger, not re-fire the handler.
        if not crash_flag["crashed"]:
            crash_flag["crashed"] = True
            raise RuntimeError("simulated crash after DEBIT, before RECEIPT")
        await tools.call(RECEIPT, {"tx_id": v.tx_id})
        return EffectResult(recorded=True)

    pipe = Pipeline("effect-acc-replay", on_failure=POLICY, registry=reg).add(two_effect_step)

    store = InMemoryCheckpointStore()
    ledger = InMemoryEffectLedger()
    run_id = "acc-effect-1"

    # Phase 1: DEBIT fires and is recorded succeeded, then the step raises.
    # allow_real_tools so the real debit_handler and receipt_handler run through
    # the proxy and ledger — mocking them would bypass the call counters and make
    # the exactly-once assertion meaningless.
    async with pipe.test_mode(
        allow_real_tools=[DEBIT, RECEIPT],
    ) as tp:
        first = await tp.run(
            Request(tx_id="tx-1"),
            run_id=run_id,
            checkpoints=store,
            effect_ledger=ledger,
        )

    assert first.status in ("halted", "escalated"), f"expected halt, got {first.status}"
    # Step raised before completing — it must NOT be checkpointed.
    assert not store.is_completed(run_id, 0), (
        "step must not be checkpointed after an intra-step crash"
    )
    # The ledger has the DEBIT entry recorded as SUCCEEDED.
    entry = ledger.entry_at(run_id, 0, 0)
    assert entry is not None, "DEBIT should be recorded in the ledger"
    assert entry.status == EffectStatus.SUCCEEDED
    # Only DEBIT fired; RECEIPT was never reached.
    assert HANDLER_CALLS["debit"] == 1
    assert HANDLER_CALLS["receipt"] == 0

    # Phase 2: resume on the SAME store + ledger + run_id.
    # The step is not checkpointed, so it re-runs. The proxy replays DEBIT from
    # the ledger (handler not called again); RECEIPT fires fresh.
    async with pipe.test_mode(
        allow_real_tools=[DEBIT, RECEIPT],
    ) as tp:
        second = await tp.run(
            Request(tx_id="tx-1"),
            run_id=run_id,
            checkpoints=store,
            effect_ledger=ledger,
        )

    assert second.status == "completed", f"expected completed, got {second.status}"
    # DEBIT: handler called exactly once total (replayed on second attempt).
    # This assertion is load-bearing on the ledger: without effect_ledger= the
    # resumed step would call debit_handler again, making HANDLER_CALLS["debit"] == 2.
    assert HANDLER_CALLS["debit"] == 1, (
        f"DEBIT handler fired {HANDLER_CALLS['debit']} times; expected exactly 1 "
        "(should replay from ledger on resume, not re-call the handler)"
    )
    # RECEIPT: handler called exactly once (only on the second attempt).
    assert HANDLER_CALLS["receipt"] == 1, (
        f"RECEIPT handler fired {HANDLER_CALLS['receipt']} times; expected exactly 1"
    )
    # Smoke: audit trace is legible.
    legible = second.audit_trace.legible()
    assert isinstance(legible, str) and len(legible) > 0


# ---------------------------------------------------------------------------
# Case 2: divergence halts through test_mode
# ---------------------------------------------------------------------------


async def test_effect_divergence_halts_through_test_mode():
    """Pre-seed the ledger with a SUCCEEDED entry for a step's first effectful
    call (DEBIT with specific args); resume with a step that calls DEBIT with
    DIFFERENT args at that position — the run must halt effect_divergence.
    """
    from drawbore.tools import ToolRegistry

    reg = ToolRegistry()

    async def debit_handler(args):
        HANDLER_CALLS["debit"] += 1
        return {"debited": True}

    reg.register_tool(DEBIT, debit_handler)

    @agent(name="diverging-step", input=Request, output=EffectResult, tools=[DEBIT])
    async def diverging_step(v: Request, tools) -> EffectResult:
        # Always calls DEBIT with a DIFFERENT amount than what was pre-seeded.
        await tools.call(DEBIT, {"tx_id": v.tx_id, "amount": 999})
        return EffectResult(recorded=True)

    pipe = (
        Pipeline("effect-acc-diverge", on_failure=POLICY, registry=reg)
        .add(diverging_step)
    )

    # Pre-seed the ledger with a SUCCEEDED entry for (run_id, step=0, pos=0)
    # that recorded DEBIT called with amount=100 — NOT 999.
    ledger = InMemoryEffectLedger()
    run_id = "acc-effect-diverge"
    original_args = {"tx_id": "tx-div", "amount": 100}
    original_hash = ledger_args_hash(original_args)
    idempotency_key = (
        __import__("hashlib")
        .sha256(
            __import__("json")
            .dumps([run_id, "0", "0", DEBIT, original_hash])
            .encode()
        )
        .hexdigest()
    )
    ledger.record_pending(
        EffectEntry(
            run_id=run_id,
            step=0,
            position=0,
            tool_ref=DEBIT,
            input_hash=original_hash,
            idempotency_key=idempotency_key,
            status=EffectStatus.PENDING,
            output=None,
        )
    )
    ledger.record_succeeded(run_id, 0, 0, {"debited": True})

    # Resume: diverging_step calls DEBIT with amount=999 at position 0,
    # which does not match the recorded amount=100.
    async with pipe.test_mode(
        mock_tools={DEBIT: {"debited": True}},
    ) as tp:
        result = await tp.run(
            Request(tx_id="tx-div"),
            run_id=run_id,
            effect_ledger=ledger,
        )

    assert result.status in ("halted", "escalated"), (
        f"expected halt on divergence, got {result.status}"
    )
    assert result.halt_code == "effect_divergence", (
        f"expected effect_divergence, got {result.halt_code!r}"
    )
    # The divergence is caught by the proxy before any handler runs: the mock
    # for DEBIT is never invoked. (The HANDLER_CALLS counter is not asserted
    # here because mock_tools overrides the registered debit_handler — only the
    # halt_code check above is load-bearing on the divergence detection.)
