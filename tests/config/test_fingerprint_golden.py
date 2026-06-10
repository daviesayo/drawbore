"""Golden byte-identity tests: the config fingerprints must never change bytes
when their hashing is refactored, and the shared canonical-hash leaf must
reproduce the exact same algorithm."""

from pydantic import BaseModel

from drawbore._canon import canonical_fingerprint, text_fingerprint
from drawbore.config.fingerprint import schema_fingerprint, footprint_fingerprint


class _Order(BaseModel):
    order_id: str
    amount: float


def test_schema_fingerprint_golden_bytes():
    # Recorded from the pre-refactor implementation. If this fails after a
    # refactor, the canonical form changed — that is a regression, not a
    # test to update.
    import hashlib, json

    canonical = json.dumps(
        _Order.model_json_schema(), sort_keys=True, separators=(",", ":")
    )
    expected = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert schema_fingerprint(_Order) == expected


def test_footprint_fingerprint_golden_bytes():
    import hashlib, json

    facts = [("a", "tool", "t1", "read"), ("b", "tool", "t2", "write")]
    rows = sorted([list(t) for t in facts])
    canonical = json.dumps(rows, separators=(",", ":"))
    expected = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert footprint_fingerprint(facts) == expected


def test_canonical_fingerprint_matches_schema_fingerprint_algorithm():
    assert canonical_fingerprint(_Order.model_json_schema()) == schema_fingerprint(_Order)


def test_canonical_fingerprint_key_order_insensitive():
    assert canonical_fingerprint({"b": 1, "a": 2}) == canonical_fingerprint({"a": 2, "b": 1})


def test_text_fingerprint_none_is_distinct_sentinel():
    # None must NOT hash like the empty string: a None -> "" change is drift.
    assert text_fingerprint(None) == "none"
    assert text_fingerprint("") != "none"
    assert text_fingerprint("") != text_fingerprint(None)
    assert text_fingerprint("abc").startswith("sha256:")


def test_canon_is_stdlib_only():
    import drawbore._canon as canon

    source = open(canon.__file__).read()
    assert "pydantic" not in source
    assert "drawbore." not in source.replace("drawbore._canon", "")
