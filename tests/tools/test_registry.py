import pytest
from drawbore.tools.registry import Tool, ToolRegistry, registry as default_registry
from drawbore.tools.errors import ToolAccessError


async def _handler(args):
    return {"ok": True}


def test_register_and_get():
    r = ToolRegistry()
    r.register_tool("db.read", _handler, allowed_operations=["invoke"])
    tool = r.get("db.read")
    assert isinstance(tool, Tool)
    assert tool.name == "db.read"
    assert tool.allowed_operations == ("invoke",)


def test_duplicate_name_raises():
    r = ToolRegistry()
    r.register_tool("db.read", _handler)
    with pytest.raises(ValueError):
        r.register_tool("db.read", _handler)


def test_get_unknown_raises_tool_access_error():
    r = ToolRegistry()
    with pytest.raises(ToolAccessError):
        r.get("ghost")


def test_has():
    r = ToolRegistry()
    assert r.has("x") is False
    r.register_builtin("x", _handler)
    assert r.has("x") is True


def test_default_registry_exists():
    assert isinstance(default_registry, ToolRegistry)


def test_register_tool_accepts_and_stores_schema():
    from pydantic import BaseModel

    class Q(BaseModel):
        id: str

    r = ToolRegistry()
    tool = r.register_tool("db.read", _handler, allowed_operations=("read",), schema=Q)
    assert tool.schema is Q
    assert tool.kind == "custom"


def test_register_builtin_records_builtin_provenance():
    r = ToolRegistry()
    tool = r.register_builtin("http.read", _handler)
    assert tool.kind == "builtin"


# --- Item A: tools() and clone() ---

def test_tools_view_returns_all_registered():
    r = ToolRegistry()
    r.register_tool("a", _handler)
    r.register_builtin("b", _handler)
    view = r.tools()
    assert set(view.keys()) == {"a", "b"}


def test_tools_view_is_read_only():
    from types import MappingProxyType

    r = ToolRegistry()
    r.register_tool("x", _handler)
    view = r.tools()
    assert isinstance(view, MappingProxyType)
    with pytest.raises(TypeError):
        view["y"] = None  # type: ignore[index]


def test_clone_returns_independent_registry():
    r = ToolRegistry()
    r.register_tool("a", _handler)
    c = r.clone()
    # same tools present
    assert c.has("a")
    assert c.get("a") is r.get("a")  # same frozen Tool object
    # mutations on clone do not affect original
    c.register_tool("b", _handler)
    assert not r.has("b")


# --- Item C: Tool.kind validation ---

def test_tool_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        Tool(name="t", handler=_handler, kind="unknown")
