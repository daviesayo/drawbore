from drawbore.observability import payload_hash


def test_payload_hash_is_deterministic_and_short():
    h = payload_hash({"a": 1})
    assert h == payload_hash({"a": 1})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_payload_hash_distinguishes_values():
    assert payload_hash({"a": 1}) != payload_hash({"a": 2})


def test_payload_hash_matches_the_proxy_log_scheme():
    # One identity scheme everywhere. The proxy historically hashed with
    # sha256(repr(value))[:16]; payload_hash must produce the same digest so a
    # span tag, a proxy-log entry, and an audit record are comparable.
    import hashlib
    value = {"text": "hi"}
    expected = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:16]
    assert payload_hash(value) == expected
