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
