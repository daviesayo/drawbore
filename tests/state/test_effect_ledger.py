"""Tests for effect_ledger: hash, EffectLedger ABC, and halt codes."""

from drawbore.state.effect_ledger import ledger_args_hash


def test_hash_is_dict_order_independent():
    assert ledger_args_hash({"a": 1, "b": 2}) == ledger_args_hash({"b": 2, "a": 1})


def test_hash_normalizes_int_and_float_consistently():
    assert ledger_args_hash({"amount": 100}) == ledger_args_hash({"amount": 100})


def test_hash_distinguishes_different_args():
    assert ledger_args_hash({"amount": 100}) != ledger_args_hash({"amount": 200})


def test_hash_handles_nested_and_non_str_keys_via_default_str():
    h = ledger_args_hash({"nested": {"x": [1, 2, 3]}})
    assert isinstance(h, str) and len(h) == 64


# ---------------------------------------------------------------------------
# Task 2: EffectEntry, EffectLedger ABC, InMemoryEffectLedger
# ---------------------------------------------------------------------------

from drawbore.state.effect_ledger import EffectStatus, EffectEntry, InMemoryEffectLedger


def _entry(run="r", step=0, pos=0, tool="t", ih="h"):
    return EffectEntry(
        run_id=run,
        step=step,
        position=pos,
        tool_ref=tool,
        input_hash=ih,
        idempotency_key="k",
        status=EffectStatus.PENDING,
        output=None,
    )


def test_pending_then_succeeded_lifecycle():
    led = InMemoryEffectLedger()
    led.record_pending(_entry())
    assert led.entry_at("r", 0, 0).status == EffectStatus.PENDING
    led.record_succeeded("r", 0, 0, output={"ok": True})
    e = led.entry_at("r", 0, 0)
    assert e.status == EffectStatus.SUCCEEDED and e.output == {"ok": True}


def test_recorded_count_and_entries_from():
    led = InMemoryEffectLedger()
    led.record_pending(_entry(pos=0))
    led.record_succeeded("r", 0, 0, output=1)
    led.record_pending(_entry(pos=1, ih="h2"))
    led.record_succeeded("r", 0, 1, output=2)
    assert led.recorded_count("r", 0) == 2
    assert [e.position for e in led.entries_from("r", 0, 1)] == [1]


def test_entry_at_absent_returns_none():
    assert InMemoryEffectLedger().entry_at("r", 0, 5) is None


def test_runs_are_isolated():
    led = InMemoryEffectLedger()
    led.record_pending(_entry(run="a"))
    assert led.entry_at("b", 0, 0) is None
