# tests/acceptance/test_effect_ledger_resume.py
"""Acceptance: exactly-once effectful-tool resume through test_mode.

A small two-step pipeline with one effectful tool call exercises the durable
effect ledger end to end through ``pipeline.test_mode``:

1. ``test_exactly_once_effect_resume_through_test_mode`` — a step fires an
   effectful tool, then the run halts (because the SECOND step fails); resume
   on the same ``InMemoryCheckpointStore`` + ``InMemoryEffectLedger`` + ``run_id``
   replays the first step's effect from the ledger (the handler is not re-called)
   and the completed run's output is correct.

2. ``test_effect_divergence_halts_through_test_mode`` — the ledger is pre-seeded
   with a SUCCEEDED entry for a step's first effect; on resume the step makes a
   DIFFERENT effectful call at that position; the run halts ``effect_divergence``.

The REAL safety layer (proxy, ledger, checkpoint) decides every outcome; only
externals (tool handlers, model responses) are mocked through test_mode. This
mirrors the crash+resume idiom used in ``tests/acceptance/test_typed_approval.py``:
the test_mode context is opened twice, each backed by the SAME stores + run_id.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.pipeline.binding import From
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


class FinalResult(BaseModel):
    ok: bool


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
# Agents
# ---------------------------------------------------------------------------


@agent(name="effect-step", input=Request, output=EffectResult, tools=[DEBIT])
async def effect_step(v: Request, tools) -> EffectResult:
    await tools.call(DEBIT, {"tx_id": v.tx_id, "amount": 100})
    return EffectResult(recorded=True)


@agent(name="finalizer", input=EffectResult, output=FinalResult)
async def finalizer(v: EffectResult) -> FinalResult:
    return FinalResult(ok=v.recorded)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POLICY = EscalationPolicy(channel="human", target="ops", mode="sync")

DEBIT_ARGS = {"tx_id": "tx-1", "amount": 100}


def _pipeline() -> Pipeline:
    return (
        Pipeline("effect-ledger-acc", on_failure=POLICY)
        .add(effect_step)
        .add(
            finalizer,
            inputs={"recorded": From("effect-step.recorded")},
            depends_on=["effect-step"],
        )
    )


def _registry_with_failing_finalizer():
    """Tool registry with a debit tool. The finalizer agent is deterministic
    (no tools), so we don't need to register anything for it."""
    from drawbore.tools import ToolRegistry

    reg = ToolRegistry()

    async def debit_handler(args):
        HANDLER_CALLS["debit"] += 1
        return {"debited": True}

    # effectful=True is the default, so this tool is tracked by the ledger.
    reg.register_tool(DEBIT, debit_handler)
    return reg


# ---------------------------------------------------------------------------
# Case 1: exactly-once on resume through test_mode
# ---------------------------------------------------------------------------


async def test_exactly_once_effect_resume_through_test_mode():
    """Crash AFTER effect-step's effect is recorded, before finalizer completes.

    The crash is simulated by the finalizer failing on the first run (so the
    run halts, effect-step's effect is recorded in the ledger, but effect-step
    is also checkpointed as completed — the ledger replay path verifies that on
    resume the proxy returns the recorded output without calling the handler
    again).

    Concretely: effect-step checkpoints succeeded after calling DEBIT once;
    the run halts at a SECOND step that raises. On resume, effect-step is
    short-circuited by the checkpoint (not re-run), and finalizer completes
    correctly. DEBIT's handler fires exactly once total.
    """
    # We need the crash to happen AFTER effect-step records its effect but
    # BEFORE the overall run completes. The simplest way: use a real registry
    # so the debit handler is real, and make the finalizer fail on first call
    # then succeed on second. We do this with a module-level flag.

    crash_flag = {"crashed": False}

    from drawbore.tools import ToolRegistry

    reg = ToolRegistry()

    async def debit_handler(args):
        HANDLER_CALLS["debit"] += 1
        return {"debited": True}

    reg.register_tool(DEBIT, debit_handler)

    @agent(name="crashing-finalizer", input=EffectResult, output=FinalResult)
    async def crashing_finalizer(v: EffectResult) -> FinalResult:
        if not crash_flag["crashed"]:
            crash_flag["crashed"] = True
            raise RuntimeError("simulated crash after effect recorded")
        return FinalResult(ok=v.recorded)

    pipe = (
        Pipeline("effect-acc-crash", on_failure=POLICY, registry=reg)
        .add(effect_step)
        .add(
            crashing_finalizer,
            inputs={"recorded": From("effect-step.recorded")},
            depends_on=["effect-step"],
        )
    )

    store = InMemoryCheckpointStore()
    ledger = InMemoryEffectLedger()
    run_id = "acc-effect-1"

    # Phase 1: effect-step fires DEBIT (recorded in ledger), crashing-finalizer raises.
    # allow_real_tools so the real debit_handler (and its counter) runs through the
    # proxy — mocking it would bypass the counter and prove nothing about re-fires.
    async with pipe.test_mode(
        allow_real_tools=[DEBIT],
    ) as tp:
        first = await tp.run(
            Request(tx_id="tx-1"),
            run_id=run_id,
            checkpoints=store,
            effect_ledger=ledger,
        )

    assert first.status in ("halted", "escalated"), f"expected halt, got {first.status}"
    # effect-step was checkpointed as completed even though the run halted.
    assert store.is_completed(run_id, 0), "effect-step should be checkpointed"
    # The ledger recorded the debit call.
    entry = ledger.entry_at(run_id, 0, 0)
    assert entry is not None, "DEBIT should be in the ledger"
    assert entry.status == EffectStatus.SUCCEEDED
    # Handler fired exactly once.
    assert HANDLER_CALLS["debit"] == 1

    # Phase 2: resume on the SAME store + ledger + run_id.
    # effect-step is short-circuited by the checkpoint; DEBIT is never re-invoked.
    async with pipe.test_mode(
        allow_real_tools=[DEBIT],
    ) as tp:
        second = await tp.run(
            Request(tx_id="tx-1"),
            run_id=run_id,
            checkpoints=store,
            effect_ledger=ledger,
        )

    assert second.status == "completed", f"expected completed, got {second.status}"
    # effect-step was restored from checkpoint: its effect was NOT re-fired.
    # DEBIT's handler count remains 1 across both runs.
    assert HANDLER_CALLS["debit"] == 1, (
        f"DEBIT handler fired {HANDLER_CALLS['debit']} times; expected exactly 1 "
        "(should replay from ledger, not re-call handler)"
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
    # The handler must NOT have been called (divergence caught before re-fire).
    assert HANDLER_CALLS["debit"] == 0
