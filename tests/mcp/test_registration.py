import pytest

from drawbore.tools import ToolRegistry
from drawbore.mcp import (
    FakeMCPClient, MCPServerConfig, MCPToolSpec, OAuthConfig,
    MCPToolNotFoundError, register_mcp_server,
)


def _client():
    return FakeMCPClient(tools={
        "send_message": MCPToolSpec(
            name="send_message",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            description="post a message",
        ),
        "delete_channel": MCPToolSpec(name="delete_channel", input_schema={}, description="danger"),
    }, results={"send_message": {"ok": True}})


async def test_registers_only_the_declared_tools_not_the_whole_server():
    # Registering a server with allowed_tools=["send_message"]
    # does NOT grant access to delete_channel.
    r = ToolRegistry()
    refs = await register_mcp_server(
        r, name="slack", url="mcp://slack.com/mcp",
        allowed_tools=["send_message"],
        auth=OAuthConfig(client_id="cid", scopes=("chat:write",)),
        client=_client(),
    )
    assert refs == ("mcp://slack/send_message",)
    assert r.has("mcp://slack/send_message")
    assert not r.has("mcp://slack/delete_channel")   # never registered → unreachable


async def test_registered_tool_has_mcp_kind_and_the_servers_schema():
    r = ToolRegistry()
    await register_mcp_server(
        r, name="slack", url="mcp://slack.com/mcp",
        allowed_tools=["send_message"], auth=None, client=_client(),
    )
    tool = r.get("mcp://slack/send_message")
    assert tool.kind == "mcp"
    assert tool.schema["properties"]["text"]["type"] == "string"   # captured from the server


async def test_declaring_a_tool_the_server_does_not_advertise_raises():
    r = ToolRegistry()
    with pytest.raises(MCPToolNotFoundError):
        await register_mcp_server(
            r, name="slack", url="mcp://slack.com/mcp",
            allowed_tools=["phantom_tool"], auth=None, client=_client(),
        )


async def test_registered_handler_routes_to_the_clients_call_tool():
    r = ToolRegistry()
    client = _client()
    await register_mcp_server(
        r, name="slack", url="mcp://slack.com/mcp",
        allowed_tools=["send_message"], auth=None, client=client,
    )
    # Invoke the registered handler the way the proxy would (a single args dict).
    handler = r.get("mcp://slack/send_message").handler
    out = await handler({"text": "hi"})
    assert out == {"ok": True}
    assert client.calls == [("send_message", {"text": "hi"})]   # routed to the real tool name


async def test_server_auth_is_consumed_at_registration_only():
    r = ToolRegistry()
    client = _client()
    auth = OAuthConfig(client_id="cid", scopes=("chat:write",))
    await register_mcp_server(
        r, name="slack", url="mcp://slack.com/mcp",
        allowed_tools=["send_message"], auth=auth, client=client,
    )
    # The server-level auth was used to connect; it is NOT stored on the Tool.
    assert any(cfg.auth is auth for cfg in client.connected_with.values())
    assert getattr(r.get("mcp://slack/send_message"), "auth", None) is None
