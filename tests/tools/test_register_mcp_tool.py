import pytest

from drawbore.tools import ToolRegistry


async def _h(args):
    return args


def test_register_mcp_tool_stamps_kind_mcp():
    r = ToolRegistry()
    tool = r.register_mcp_tool("mcp://slack/send_message", _h, schema={"type": "object"})
    assert tool.kind == "mcp"
    assert r.get("mcp://slack/send_message").kind == "mcp"
    assert r.get("mcp://slack/send_message").schema == {"type": "object"}


def test_register_mcp_tool_rejects_duplicate():
    r = ToolRegistry()
    r.register_mcp_tool("mcp://slack/send_message", _h)
    with pytest.raises(ValueError):
        r.register_mcp_tool("mcp://slack/send_message", _h)
