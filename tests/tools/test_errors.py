import pytest
from drawbore.tools.errors import (
    ToolError, ToolAccessError, TokenError, CircuitBreakerError,
)


def test_hierarchy():
    for cls in (ToolAccessError, TokenError, CircuitBreakerError):
        assert issubclass(cls, ToolError)
    assert issubclass(ToolError, Exception)


def test_raisable():
    with pytest.raises(ToolAccessError):
        raise ToolAccessError("nope")
