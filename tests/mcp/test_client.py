import pytest

from drawbore.mcp import (
    MCPClient, FakeMCPClient, MCPServerConfig, MCPToolSpec, OAuthConfig,
    MCPToolNotFoundError,
)


def _server(auth=None):
    return MCPServerConfig(name="slack", url="mcp://slack.com/mcp", auth=auth)


def _fake():
    # The fake advertises two tools; the scoping is the registry's job (Task 3).
    return FakeMCPClient(tools={
        "send_message": MCPToolSpec(
            name="send_message",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            description="post a message",
        ),
        "delete_channel": MCPToolSpec(name="delete_channel", input_schema={}, description="danger"),
    }, results={"send_message": {"ok": True}})


def test_mcpclient_is_abstract():
    with pytest.raises(TypeError):
        MCPClient()


async def test_fake_connect_records_the_server_auth():
    auth = OAuthConfig(client_id="cid", scopes=("chat:write",))
    client = _fake()
    session = await client.connect(_server(auth=auth))
    # Server-level auth is consumed at connect (registration), not per call.
    assert client.connected_with[session].auth is auth


async def test_fake_list_tools_returns_the_advertised_specs():
    client = _fake()
    session = await client.connect(_server())
    specs = await client.list_tools(session)
    assert {s.name for s in specs} == {"send_message", "delete_channel"}
    by_name = {s.name: s for s in specs}
    assert by_name["send_message"].input_schema["properties"]["text"]["type"] == "string"


async def test_fake_call_tool_runs_and_records_the_invocation():
    client = _fake()
    session = await client.connect(_server())
    out = await client.call_tool(session, "send_message", {"text": "hi"})
    assert out == {"ok": True}
    assert client.calls == [("send_message", {"text": "hi"})]


async def test_fake_call_tool_unknown_tool_raises_mcp_error():
    client = _fake()
    session = await client.connect(_server())
    with pytest.raises(MCPToolNotFoundError):
        await client.call_tool(session, "not_a_tool", {})
