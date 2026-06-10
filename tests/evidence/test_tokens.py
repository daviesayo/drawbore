from drawbore.evidence import estimate_tokens


def test_estimate_tokens_is_deterministic():
    # Determinism that matters: insertion order must not change the estimate
    # (sort_keys=True), and the formula is pinned so a regression is caught.
    assert estimate_tokens({"a": 2, "b": 1}) == estimate_tokens({"b": 1, "a": 2})
    assert estimate_tokens({"rows": [{"id": i} for i in range(50)]}) == 150


def test_estimate_tokens_grows_with_size():
    assert estimate_tokens([1, 2, 3]) < estimate_tokens(list(range(1000)))


def test_estimate_tokens_handles_non_json_native_values():
    import datetime
    assert estimate_tokens({"when": datetime.datetime(2026, 6, 3)}) >= 1


def test_estimate_tokens_is_at_least_one():
    assert estimate_tokens({}) >= 1
    assert estimate_tokens(None) >= 1
