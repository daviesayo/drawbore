import json

import pytest
from drawbore.context import sanitize
from drawbore.errors import SanitizationError


def test_normal_value_passes_through_unchanged():
    v = {"id": "abc", "amount": 10.0}
    assert sanitize(v) is v


def test_oversize_value_raises():
    with pytest.raises(SanitizationError):
        sanitize("x" * 10, max_bytes=4)


def test_too_deep_value_raises():
    deep = {"a": {"b": {"c": {"d": 1}}}}
    with pytest.raises(SanitizationError):
        sanitize(deep, max_depth=2)


def test_within_depth_passes():
    ok = {"a": {"b": 1}}
    assert sanitize(ok, max_depth=3) == ok


def test_depth_boundary_is_exact():
    # Depth counting: root=1, a-value=2, b-value=3, c-value(scalar)=4.
    # max_depth=4 is exactly at the limit (scalar reached at depth 4 does not exceed it).
    # max_depth=3 is one short — the scalar at depth 4 triggers the error.
    three_deep = {"a": {"b": {"c": 1}}}
    assert sanitize(three_deep, max_depth=4) == three_deep      # exactly at limit: ok
    with pytest.raises(SanitizationError):
        sanitize(three_deep, max_depth=3)                        # one short: raises


def test_multibyte_payload_measured_in_json_bytes():
    """CJK text: repr().encode() undercounts vs json.dumps().encode().

    Each CJK character is 3 UTF-8 bytes via repr(), but 6 bytes via
    json.dumps() (ensure_ascii=True escapes as \\uXXXX).  A payload whose
    repr-byte length is UNDER the threshold but whose JSON-byte length is
    OVER it must raise SanitizationError.

    This test FAILS against the repr()-based implementation and PASSES after
    the json.dumps() fix.
    """
    # Use a CJK character: U+4E2D (中). repr encodes it as 3 UTF-8 bytes;
    # json.dumps (ensure_ascii=True) escapes it to the 6-ASCII-byte sequence
    # backslash-u-4-e-2-d.
    cjk_char = "中"

    # Choose a repetition count so:
    #   repr byte count  < threshold  (test precondition: old code would pass)
    #   json byte count  > threshold  (test expectation:  new code must raise)
    #
    # For N repetitions in a dict {"t": "<N chars>"}:
    #   repr bytes ≈ N * 3  (UTF-8 direct encoding, no escaping)
    #   json bytes ≈ N * 6  (\\uXXXX escaping)
    #
    # Choose threshold = 200 bytes, N = 50:
    #   repr bytes = len(repr({"t": cjk_char * 50}).encode("utf-8")) ≈ 50*3 + overhead
    #   json bytes = len(json.dumps({"t": cjk_char * 50}).encode("utf-8")) ≈ 50*6 + overhead
    n = 50
    threshold = 200
    payload = {"t": cjk_char * n}

    repr_bytes = len(repr(payload).encode("utf-8"))
    json_bytes = len(json.dumps(payload).encode("utf-8"))

    # Assert the precondition so the test is self-documenting:
    assert repr_bytes < threshold, (
        f"precondition failed: repr bytes {repr_bytes} should be < {threshold}"
    )
    assert json_bytes > threshold, (
        f"precondition failed: json bytes {json_bytes} should be > {threshold}"
    )

    # The size gate must fire (JSON bytes exceed the limit).
    with pytest.raises(SanitizationError):
        sanitize(payload, max_bytes=threshold)


def test_non_json_serialisable_value_raises_sanitization_error():
    """A value json.dumps cannot serialise (a contract violation, e.g. a raw
    model_dump() with a datetime) is surfaced as SanitizationError, not a bare
    TypeError escaping the sanitisation boundary."""
    import datetime

    with pytest.raises(SanitizationError):
        sanitize({"when": datetime.datetime(2026, 1, 1)})
