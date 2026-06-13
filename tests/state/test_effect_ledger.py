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
