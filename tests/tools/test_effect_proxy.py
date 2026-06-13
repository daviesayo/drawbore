"""Effect-ledger wiring in ToolProxy.invoke: exactly-once replay, divergence,
unresolved, and ledger-write halts; idempotency-key injection; cursor accounting.
"""

import hashlib
import json
import types

import pytest

from drawbore.tools import ToolProxy, ToolRegistry, TokenIssuer
from drawbore.tools.access import current_idempotency_key
from drawbore.state.effect_ledger import (
    EffectDivergenceError,
    EffectEntry,
    EffectLedgerWriteError,
    EffectStatus,
    EffectUnresolvedError,
    InMemoryEffectLedger,
    ledger_args_hash,
)


def _ctx(run_id="r1", step=0):
    return types.SimpleNamespace(run_id=run_id, step=step)


def _ikey(run_id, step, pos, tool_ref, ihash):
    return hashlib.sha256(
        json.dumps([run_id, str(step), str(pos), tool_ref, ihash]).encode()
    ).hexdigest()


class Counter:
    def __init__(self):
        self.n = 0

    async def __call__(self, args):
        self.n += 1
        return {"calls": self.n, "args": args}


class RecordingLedger(InMemoryEffectLedger):
    """In-memory ledger that records the order of mutating calls."""

    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    def record_pending(self, entry):
        self.calls.append("record_pending")
        super().record_pending(entry)

    def record_succeeded(self, run_id, step, position, output):
        self.calls.append("record_succeeded")
        super().record_succeeded(run_id, step, position, output)


def _seed_succeeded(ledger, run_id, step, pos, tool_ref, args, output):
    ihash = ledger_args_hash(args)
    ledger.record_pending(
        EffectEntry(
            run_id=run_id,
            step=step,
            position=pos,
            tool_ref=tool_ref,
            input_hash=ihash,
            idempotency_key=_ikey(run_id, step, pos, tool_ref, ihash),
            status=EffectStatus.PENDING,
            output=None,
        )
    )
    ledger.record_succeeded(run_id, step, pos, output)


# ---------------------------------------------------------------------------
# 1. Fresh effectful call: PENDING then SUCCEEDED, output recorded.
# ---------------------------------------------------------------------------


async def test_fresh_records_pending_then_succeeded():
    reg = ToolRegistry()
    counter = Counter()
    reg.register_tool("acct.debit", counter, allowed_operations=["invoke"])
    issuer = TokenIssuer()
    ledger = RecordingLedger()
    proxy = ToolProxy(reg, issuer, effect_ledger=ledger)

    tok = issuer.issue("acct.debit", "r1")
    out = await proxy.invoke("acct.debit", {"amt": 10}, tok, _ctx("r1", 0))

    assert out == {"calls": 1, "args": {"amt": 10}}
    assert ledger.calls == ["record_pending", "record_succeeded"]
    entry = ledger.entry_at("r1", 0, 0)
    assert entry is not None
    assert entry.status == EffectStatus.SUCCEEDED
    assert entry.position == 0
    assert entry.output == out
    assert ledger.recorded_count("r1", 0) == 1


# ---------------------------------------------------------------------------
# 2. Replay: pre-seeded SUCCEEDED entry returns output, handler NOT called.
# ---------------------------------------------------------------------------


async def test_replay_returns_recorded_output_without_calling_handler():
    reg = ToolRegistry()
    counter = Counter()
    reg.register_tool("acct.debit", counter, allowed_operations=["invoke"])
    issuer = TokenIssuer()
    ledger = InMemoryEffectLedger()
    args = {"amt": 10}
    _seed_succeeded(ledger, "r1", 0, 0, "acct.debit", args, {"recorded": True})

    # A NEW proxy instance, same run/step, fresh cursor.
    proxy = ToolProxy(reg, issuer, effect_ledger=ledger)
    tok = issuer.issue("acct.debit", "r1")
    out = await proxy.invoke("acct.debit", args, tok, _ctx("r1", 0))

    assert out == {"recorded": True}
    assert counter.n == 0
    # Replay is still observable in the proxy log.
    assert proxy.log[-1]["result"] == "replay"


# ---------------------------------------------------------------------------
# 3. Divergence: different args at a recorded position halts, no re-fire.
# ---------------------------------------------------------------------------


async def test_divergence_on_arg_mismatch():
    reg = ToolRegistry()
    counter = Counter()
    reg.register_tool("acct.debit", counter, allowed_operations=["invoke"])
    issuer = TokenIssuer()
    ledger = InMemoryEffectLedger()
    _seed_succeeded(ledger, "r1", 0, 0, "acct.debit", {"amt": 10}, {"recorded": True})

    proxy = ToolProxy(reg, issuer, effect_ledger=ledger)
    tok = issuer.issue("acct.debit", "r1")
    with pytest.raises(EffectDivergenceError):
        await proxy.invoke("acct.debit", {"amt": 999}, tok, _ctx("r1", 0))
    assert counter.n == 0


# ---------------------------------------------------------------------------
# 4. Unresolved: a PENDING entry at the position halts.
# ---------------------------------------------------------------------------


async def test_unresolved_on_pending_entry():
    reg = ToolRegistry()
    counter = Counter()
    reg.register_tool("acct.debit", counter, allowed_operations=["invoke"])
    issuer = TokenIssuer()
    ledger = InMemoryEffectLedger()
    args = {"amt": 10}
    ihash = ledger_args_hash(args)
    ledger.record_pending(
        EffectEntry(
            run_id="r1",
            step=0,
            position=0,
            tool_ref="acct.debit",
            input_hash=ihash,
            idempotency_key=_ikey("r1", 0, 0, "acct.debit", ihash),
            status=EffectStatus.PENDING,
            output=None,
        )
    )

    proxy = ToolProxy(reg, issuer, effect_ledger=ledger)
    tok = issuer.issue("acct.debit", "r1")
    with pytest.raises(EffectUnresolvedError):
        await proxy.invoke("acct.debit", args, tok, _ctx("r1", 0))
    assert counter.n == 0


# ---------------------------------------------------------------------------
# 5. record_pending failure halts BEFORE the handler fires.
# ---------------------------------------------------------------------------


async def test_record_pending_failure_halts_before_handler():
    class FailPendingLedger(InMemoryEffectLedger):
        def record_pending(self, entry):
            raise RuntimeError("disk full")

    reg = ToolRegistry()
    counter = Counter()
    reg.register_tool("acct.debit", counter, allowed_operations=["invoke"])
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer, effect_ledger=FailPendingLedger())

    tok = issuer.issue("acct.debit", "r1")
    with pytest.raises(EffectLedgerWriteError):
        await proxy.invoke("acct.debit", {"amt": 10}, tok, _ctx("r1", 0))
    assert counter.n == 0


# ---------------------------------------------------------------------------
# 6. Read-only (effectful=False) tool: never ledgered, cursor not advanced.
# ---------------------------------------------------------------------------


async def test_readonly_tool_bypasses_ledger():
    reg = ToolRegistry()
    counter = Counter()
    reg.register_tool(
        "acct.read", counter, allowed_operations=["invoke"], effectful=False
    )
    issuer = TokenIssuer()
    ledger = InMemoryEffectLedger()
    proxy = ToolProxy(reg, issuer, effect_ledger=ledger)

    tok = issuer.issue("acct.read", "r1")
    out = await proxy.invoke("acct.read", {"id": 1}, tok, _ctx("r1", 0))

    assert out == {"calls": 1, "args": {"id": 1}}
    assert ledger.recorded_count("r1", 0) == 0
    assert proxy.effects_consumed("r1", 0) == 0


# ---------------------------------------------------------------------------
# 7. Idempotency key: visible + stable inside the handler, None outside.
# ---------------------------------------------------------------------------


async def test_idempotency_key_visible_and_stable():
    seen = {}

    async def handler(args):
        seen.setdefault("keys", []).append(current_idempotency_key())
        return {}

    reg = ToolRegistry()
    reg.register_tool("acct.debit", handler, allowed_operations=["invoke"])
    issuer = TokenIssuer()
    args = {"amt": 10}

    # Two independent fresh proxies (fresh cursor + fresh ledger) at pos 0.
    proxy1 = ToolProxy(reg, issuer, effect_ledger=InMemoryEffectLedger())
    await proxy1.invoke("acct.debit", args, issuer.issue("acct.debit", "r1"), _ctx("r1", 0))
    proxy2 = ToolProxy(reg, issuer, effect_ledger=InMemoryEffectLedger())
    await proxy2.invoke("acct.debit", args, issuer.issue("acct.debit", "r1"), _ctx("r1", 0))

    k1, k2 = seen["keys"]
    assert k1 is not None
    assert len(k1) == 64
    int(k1, 16)  # is hex
    assert k1 == k2  # same five-tuple -> same key
    # No key leaks after the call returns.
    assert current_idempotency_key() is None


# ---------------------------------------------------------------------------
# 8. Positions: two effectful calls get 0 and 1; a read-only call between
#    them does NOT advance the cursor.
# ---------------------------------------------------------------------------


async def test_positions_advance_only_for_effectful_calls():
    reg = ToolRegistry()
    eff = Counter()
    ro = Counter()
    reg.register_tool("acct.debit", eff, allowed_operations=["invoke"])
    reg.register_tool("acct.read", ro, allowed_operations=["invoke"], effectful=False)
    issuer = TokenIssuer()
    ledger = InMemoryEffectLedger()
    proxy = ToolProxy(reg, issuer, effect_ledger=ledger)

    await proxy.invoke("acct.debit", {"amt": 1}, issuer.issue("acct.debit", "r1"), _ctx("r1", 0))
    await proxy.invoke("acct.read", {"id": 1}, issuer.issue("acct.read", "r1"), _ctx("r1", 0))
    await proxy.invoke("acct.debit", {"amt": 2}, issuer.issue("acct.debit", "r1"), _ctx("r1", 0))

    assert ledger.entry_at("r1", 0, 0).tool_ref == "acct.debit"
    assert ledger.entry_at("r1", 0, 1).tool_ref == "acct.debit"
    assert ledger.entry_at("r1", 0, 2) is None
    assert ledger.recorded_count("r1", 0) == 2


# ---------------------------------------------------------------------------
# 9. effects_consumed reflects the number of effectful calls processed.
# ---------------------------------------------------------------------------


async def test_effects_consumed_counts_effectful_calls():
    reg = ToolRegistry()
    eff = Counter()
    ro = Counter()
    reg.register_tool("acct.debit", eff, allowed_operations=["invoke"])
    reg.register_tool("acct.read", ro, allowed_operations=["invoke"], effectful=False)
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer, effect_ledger=InMemoryEffectLedger())

    assert proxy.effects_consumed("r1", 0) == 0
    await proxy.invoke("acct.debit", {"amt": 1}, issuer.issue("acct.debit", "r1"), _ctx("r1", 0))
    await proxy.invoke("acct.read", {"id": 1}, issuer.issue("acct.read", "r1"), _ctx("r1", 0))
    await proxy.invoke("acct.debit", {"amt": 2}, issuer.issue("acct.debit", "r1"), _ctx("r1", 0))
    assert proxy.effects_consumed("r1", 0) == 2
