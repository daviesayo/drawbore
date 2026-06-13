"""Tests for effectful tool classification, idempotency-key accessor, and CI helper."""

import pytest
from drawbore.tools.registry import Tool, ToolRegistry
from drawbore.tools.access import (
    current_idempotency_key,
    _set_idempotency_key,
    _reset_idempotency_key,
)
from drawbore.tools import unclassified_effectful_tools


async def _handler(args):
    return {}


# ---------------------------------------------------------------------------
# Tool.effectful field
# ---------------------------------------------------------------------------


def test_tool_defaults_effectful_true():
    r = ToolRegistry()
    tool = r.register_tool("svc.write", _handler)
    assert tool.effectful is True


def test_register_tool_effectful_false():
    r = ToolRegistry()
    tool = r.register_tool("svc.read", _handler, effectful=False)
    assert tool.effectful is False


def test_register_tool_effectful_true_explicit():
    r = ToolRegistry()
    tool = r.register_tool("svc.write2", _handler, effectful=True)
    assert tool.effectful is True


def test_register_builtin_honors_effectful_false():
    r = ToolRegistry()
    tool = r.register_builtin("builtin.read", _handler, effectful=False)
    assert tool.effectful is False


def test_register_builtin_defaults_effectful_true():
    r = ToolRegistry()
    tool = r.register_builtin("builtin.write", _handler)
    assert tool.effectful is True


def test_register_mcp_tool_honors_effectful_false():
    r = ToolRegistry()
    tool = r.register_mcp_tool("mcp.pure", _handler, effectful=False)
    assert tool.effectful is False


def test_register_mcp_tool_defaults_effectful_true():
    r = ToolRegistry()
    tool = r.register_mcp_tool("mcp.write", _handler)
    assert tool.effectful is True


def test_tool_dataclass_field_present():
    """Tool dataclass has the effectful field at the expected default."""
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(Tool)}
    assert "effectful" in fields
    assert fields["effectful"].default is True


# ---------------------------------------------------------------------------
# evidence://retrieve is registered with effectful=False
# ---------------------------------------------------------------------------


def test_evidence_retrieve_registered_not_effectful():
    from drawbore.evidence.retrieval import register_evidence_tool
    from drawbore.evidence.store import InMemoryEvidenceStore

    r = ToolRegistry()
    store = InMemoryEvidenceStore()
    register_evidence_tool(r, store=store)
    tool = r.get("evidence://retrieve")
    assert tool.effectful is False


# ---------------------------------------------------------------------------
# current_idempotency_key accessor
# ---------------------------------------------------------------------------


def test_current_idempotency_key_returns_none_outside_call():
    assert current_idempotency_key() is None


def test_set_idempotency_key_makes_it_visible():
    token = _set_idempotency_key("abc-123")
    try:
        assert current_idempotency_key() == "abc-123"
    finally:
        _reset_idempotency_key(token)


def test_reset_idempotency_key_restores_none():
    token = _set_idempotency_key("abc-123")
    _reset_idempotency_key(token)
    assert current_idempotency_key() is None


def test_idempotency_key_nested_set_reset():
    """Inner set/reset does not leak into the outer scope."""
    outer_token = _set_idempotency_key("outer")
    try:
        inner_token = _set_idempotency_key("inner")
        assert current_idempotency_key() == "inner"
        _reset_idempotency_key(inner_token)
        assert current_idempotency_key() == "outer"
    finally:
        _reset_idempotency_key(outer_token)
    assert current_idempotency_key() is None


# ---------------------------------------------------------------------------
# unclassified_effectful_tools CI helper
# ---------------------------------------------------------------------------


def test_unclassified_effectful_tools_returns_default_effectful():
    r = ToolRegistry()
    r.register_tool("a.write", _handler)            # default effectful=True
    r.register_tool("b.read", _handler, effectful=False)
    r.register_tool("c.write", _handler)            # default effectful=True
    result = unclassified_effectful_tools(r)
    assert isinstance(result, tuple)
    assert "a.write" in result
    assert "c.write" in result
    assert "b.read" not in result


def test_unclassified_effectful_tools_sorted():
    r = ToolRegistry()
    r.register_tool("z.write", _handler)
    r.register_tool("a.write", _handler)
    result = unclassified_effectful_tools(r)
    assert result == tuple(sorted(result))


def test_unclassified_effectful_tools_empty_registry():
    r = ToolRegistry()
    result = unclassified_effectful_tools(r)
    assert result == ()


def test_unclassified_effectful_tools_all_classified():
    r = ToolRegistry()
    r.register_tool("a.read", _handler, effectful=False)
    r.register_builtin("b.read", _handler, effectful=False)
    result = unclassified_effectful_tools(r)
    assert result == ()


def test_unclassified_effectful_tools_includes_builtins_and_mcp():
    r = ToolRegistry()
    r.register_builtin("bi.write", _handler)         # default True
    r.register_mcp_tool("mcp.write", _handler)       # default True
    r.register_tool("custom.read", _handler, effectful=False)
    result = unclassified_effectful_tools(r)
    assert "bi.write" in result
    assert "mcp.write" in result
    assert "custom.read" not in result
