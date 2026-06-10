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
