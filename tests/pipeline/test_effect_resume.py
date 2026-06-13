"""Exactly-once effectful tool calls across a crash+resume.

These tests exercise the durable effect ledger through the real ``Pipeline.run``
path: a step that fires an effectful tool then crashes is resumed on the same
``EffectLedger``; the already-fired effect is replayed from the ledger (never
re-fired) and forward progress continues freshly. The step-end orphan check
halts ``effect_divergence`` when a resumed step consumes fewer effectful calls
than were recorded for it.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.state import InMemoryCheckpointStore
from drawbore.state.effect_ledger import (
    EffectEntry,
    EffectStatus,
    InMemoryEffectLedger,
    ledger_args_hash,
)
from drawbore.tools import ToolRegistry

TOOL_A = "eff:a"
TOOL_B = "eff:b"


class In(BaseModel):
    id: str


class Out(BaseModel):
    ok: bool


@pytest.fixture(autouse=True)
def _reset_counts():
    COUNTS.clear()
    COUNTS.update({"a": 0, "b": 0})
    STATE["crashed"] = False


COUNTS: dict[str, int] = {}
STATE: dict[str, bool] = {"crashed": False}


def _registry() -> ToolRegistry:
    async def handler_a(args):
        COUNTS["a"] += 1
        return {"from": "a"}

    async def handler_b(args):
        COUNTS["b"] += 1
        return {"from": "b"}

    reg = ToolRegistry()
    reg.register_tool(TOOL_A, handler_a)   # effectful=True by default
    reg.register_tool(TOOL_B, handler_b)
    return reg


async def test_exactly_once_resume_prevents_double_fire():
    """A step fires effect A (recorded succeeded) then crashes BEFORE effect B
    is recorded; the resume replays A from the ledger (A's handler is NOT
    re-called) and freshly fires B, completing the run. A double debit prevented.
    """

    @agent(name="worker", input=In, output=Out, tools=[TOOL_A, TOOL_B])
    async def worker(payload: In, tools) -> Out:
        await tools.call(TOOL_A, {"x": 1})
        # Simulate a crash after A succeeded but before B is even attempted:
        # the first attempt raises here, so B never records a pending entry.
        if not STATE["crashed"]:
            STATE["crashed"] = True
            raise RuntimeError("boom after effect A, before effect B")
        await tools.call(TOOL_B, {"y": 2})
        return Out(ok=True)

    reg = _registry()
    pipe = Pipeline("eff-once", registry=reg).add(worker)
    checkpoints = InMemoryCheckpointStore()
    led = InMemoryEffectLedger()

    # --- attempt 1: crashes after A; A recorded succeeded, B never recorded ---
    first = await pipe.run(
        In(id="r1"), run_id="run-1", checkpoints=checkpoints, effect_ledger=led
    )
    assert first.status in ("halted", "escalated")
    assert COUNTS == {"a": 1, "b": 0}

    # --- resume on the SAME ledger + checkpoints + run_id ---
    second = await pipe.run(
        In(id="r1"), run_id="run-1", checkpoints=checkpoints, effect_ledger=led
    )
    assert second.status == "completed"
    # A replayed (handler still called exactly once total); B fired once.
    assert COUNTS == {"a": 1, "b": 1}
    assert second.outputs["worker"] == Out(ok=True)


async def test_orphaned_recorded_effect_halts_divergence():
    """A ledger pre-seeded with two SUCCEEDED effects for a step, resumed by a
    step that makes only ONE effectful call, halts ``effect_divergence`` at the
    step-end orphan check (recorded(2) > consumed(1))."""

    @agent(name="solo", input=In, output=Out, tools=[TOOL_A])
    async def solo(payload: In, tools) -> Out:
        await tools.call(TOOL_A, {"x": 1})   # replays the pos-0 entry
        return Out(ok=True)                  # never calls the orphaned pos-1 effect

    reg = _registry()
    pipe = Pipeline("eff-orphan", registry=reg).add(solo)

    led = InMemoryEffectLedger()
    a_hash = ledger_args_hash({"x": 1})
    led.record_pending(
        EffectEntry(
            run_id="run-2", step=0, position=0, tool_ref=TOOL_A,
            input_hash=a_hash, idempotency_key="k0",
            status=EffectStatus.PENDING, output=None,
        )
    )
    led.record_succeeded("run-2", 0, 0, {"from": "a"})
    led.record_pending(
        EffectEntry(
            run_id="run-2", step=0, position=1, tool_ref=TOOL_B,
            input_hash=ledger_args_hash({"y": 2}), idempotency_key="k1",
            status=EffectStatus.PENDING, output=None,
        )
    )
    led.record_succeeded("run-2", 0, 1, {"from": "b"})

    result = await pipe.run(In(id="r2"), run_id="run-2", effect_ledger=led)

    assert result.status in ("halted", "escalated")
    assert result.halt_code == "effect_divergence"
    # The orphaned effect is named legibly.
    assert TOOL_B in result.reason
    # A's handler was never re-fired (it replayed from the ledger).
    assert COUNTS["a"] == 0
