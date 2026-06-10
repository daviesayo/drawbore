from drawbore.evidence import EvidencePolicy
from drawbore.evidence.transforms import get_transform, json_rows


def _policy(**kw):
    return EvidencePolicy(enabled=True, **kw)


def _rows(n):
    return {"records": [{"id": i, "amount": i, "status": "ok"} for i in range(n)]}


def test_registry_resolves_json_rows_by_name():
    assert get_transform("json_rows") is json_rows


def test_should_apply_only_above_threshold():
    big = _rows(500)
    small = _rows(2)
    assert json_rows.should_apply(big, _policy(min_tokens=100)) is True
    assert json_rows.should_apply(small, _policy(min_tokens=100_000)) is False


def test_compress_is_deterministic_byte_identical():
    value = _rows(500)
    out1, w1 = json_rows.compress(value, _policy())
    out2, w2 = json_rows.compress(value, _policy())
    import json
    assert json.dumps(out1, sort_keys=True) == json.dumps(out2, sort_keys=True)
    assert w1 == w2


def test_compress_shrinks_and_preserves_order_and_endpoints():
    value = _rows(500)
    out, _ = json_rows.compress(value, _policy())
    kept = out["records"]
    assert len(kept) < 500
    assert kept[0]["id"] == 0
    assert kept[-1]["id"] == 499
    ids = [r["id"] for r in kept]
    assert ids == sorted(ids)


def test_notable_rows_are_kept_by_structured_signal_not_prose():
    value = {"records": (
        [{"id": i, "status": "ok", "note": "no error here"} for i in range(400)]
        + [{"id": 400, "status": "flagged", "note": "review"}]
    )}
    out, _ = json_rows.compress(value, _policy())
    kept_ids = {r["id"] for r in out["records"]}
    assert 400 in kept_ids


def test_adversarial_every_row_flagged_is_capped_with_warning():
    value = {"records": [{"id": i, "status": "flagged"} for i in range(1000)]}
    out, warnings = json_rows.compress(value, _policy(max_output_tokens=2000))
    assert len(out["records"]) < 1000
    assert any("cap" in w.lower() or "drop" in w.lower() for w in warnings)
    # Legibility: when notable/signal rows are dropped by the cap, the warning says so
    # (an auditor must see signal loss, not just a row count).
    assert any("notable" in w.lower() for w in warnings)


def test_output_budget_is_shared_across_multiple_compressible_lists():
    # Two record lists under one budget: the TOTAL kept rows honour max_output_tokens
    # (the budget is divided per list), not each list independently.
    value = {
        "alpha": [{"id": i, "amount": i} for i in range(500)],
        "beta": [{"id": i, "amount": i} for i in range(500)],
    }
    out, _ = json_rows.compress(value, _policy(max_output_tokens=800))
    total_rows = 800 // 40  # 20 across both lists
    assert len(out["alpha"]) + len(out["beta"]) <= total_rows


def test_adversarial_malformed_json_string_field_passes_through_untouched():
    value = {"records": [{"id": i} for i in range(300)], "blob": "{not valid json"}
    out, _ = json_rows.compress(value, _policy())
    assert out["blob"] == "{not valid json"


def test_non_list_payload_is_returned_unchanged():
    value = {"summary": "no lists here", "count": 5}
    out, warnings = json_rows.compress(value, _policy())
    assert out == value
