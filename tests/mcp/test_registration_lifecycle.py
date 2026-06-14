"""Tests for the mcp_server async context manager (session lifecycle / teardown).

These tests use FakeMCPClient exclusively — no live server, no `mcp` SDK.
They verify:
  - The session is OPEN during the block and CLOSED after exit.
  - The returned refs match what register_mcp_server returns.
  - Idempotent close: exercising the close path twice does NOT double-close the
    underlying session.
  - Error propagation: an exception raised inside the block still triggers teardown.
  - register_mcp_server's existing behavior is unchanged (it does NOT close the
    session on success — that's the known leak; this context manager is the fix).
"""
import pytest

from drawbore.tools import ToolRegistry
from drawbore.mcp import (
    FakeMCPClient,
    MCPToolNotFoundError,
    MCPToolSpec,
    mcp_server,
    register_mcp_server,
)


def _client():
    return FakeMCPClient(
        tools={
            "send_message": MCPToolSpec(
                name="send_message",
                input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                description="post a message",
            ),
        },
        results={"send_message": {"ok": True}},
    )


async def test_mcp_server_yields_refs_matching_register_mcp_server():
    """The context manager returns the same tuple of refs as register_mcp_server."""
    r = ToolRegistry()
    async with mcp_server(
        r,
        name="slack",
        url="mcp://slack.com/mcp",
        allowed_tools=["send_message"],
        auth=None,
        client=_client(),
    ) as refs:
        assert refs == ("mcp://slack/send_message",)
        assert r.has("mcp://slack/send_message")


async def test_session_is_open_inside_block_and_closed_after():
    """Inside the block the session is connected; after exit it is in .closed."""
    r = ToolRegistry()
    client = _client()
    async with mcp_server(
        r,
        name="slack",
        url="mcp://slack.com/mcp",
        allowed_tools=["send_message"],
        auth=None,
        client=client,
    ) as refs:
        # At least one session is live (in connected_with)
        assert len(client.connected_with) == 1
        # Not yet closed
        assert len(client.closed) == 0

    # After the block, the session should have been closed exactly once
    assert len(client.closed) == 1
    # The closed session is one that was open
    closed_session = client.closed[0]
    assert closed_session in client.connected_with


async def test_session_closed_even_when_block_raises():
    """An exception raised inside the block still triggers session teardown."""
    r = ToolRegistry()
    client = _client()
    with pytest.raises(RuntimeError, match="boom"):
        async with mcp_server(
            r,
            name="slack",
            url="mcp://slack.com/mcp",
            allowed_tools=["send_message"],
            auth=None,
            client=client,
        ):
            raise RuntimeError("boom")

    # Teardown must still have run
    assert len(client.closed) == 1


async def test_idempotent_close_context_manager_does_not_close_twice():
    """The context manager is idempotent: calling __aexit__ a second time after the
    block has already exited does NOT close the underlying session a second time.
    Idempotency is provided by the asynccontextmanager protocol — the generator is
    exhausted after the first exit, so a subsequent __aexit__ is a no-op and
    client.close is never called twice.
    """
    r = ToolRegistry()
    client = _client()

    # Normal exit path: session is closed exactly once by the CM.
    sessions_seen = []
    cm = mcp_server(
        r,
        name="slack",
        url="mcp://slack.com/mcp",
        allowed_tools=["send_message"],
        auth=None,
        client=client,
    )
    refs = await cm.__aenter__()
    sessions_seen.extend(list(client.connected_with.keys()))
    await cm.__aexit__(None, None, None)

    # First exit closed the session
    assert len(client.closed) == 1
    # The closed session is the one that was live inside the block
    assert client.closed[0] in sessions_seen

    # Calling __aexit__ again (the CM is exhausted / _closed=True) must be a no-op
    await cm.__aexit__(None, None, None)
    assert len(client.closed) == 1  # still exactly one, no double close


async def test_register_mcp_server_still_does_not_close_on_success():
    """register_mcp_server must remain unchanged — it does NOT close the session on
    success (the caller is responsible for lifecycle). This test documents the
    known pre-existing behavior and ensures we did not accidentally change it."""
    r = ToolRegistry()
    client = _client()
    refs = await register_mcp_server(
        r,
        name="slack",
        url="mcp://slack.com/mcp",
        allowed_tools=["send_message"],
        auth=None,
        client=client,
    )
    assert refs == ("mcp://slack/send_message",)
    # No close was called on success — that is the documented leak this CM fixes
    assert len(client.closed) == 0


async def test_mcp_server_registers_tool_callable_through_registry():
    """Tools registered via mcp_server are fully functional inside the block."""
    r = ToolRegistry()
    client = _client()
    async with mcp_server(
        r,
        name="slack",
        url="mcp://slack.com/mcp",
        allowed_tools=["send_message"],
        auth=None,
        client=client,
    ) as refs:
        handler = r.get("mcp://slack/send_message").handler
        result = await handler({"text": "hello"})
        assert result == {"ok": True}
        assert client.calls == [("send_message", {"text": "hello"})]


async def test_mcp_server_closes_session_when_registration_fails():
    """If a declared tool is not advertised by the server, registration raises
    MCPToolNotFoundError. The session must still be closed exactly once and the
    error must propagate out of the async with."""
    r = ToolRegistry()
    client = _client()  # server advertises only "send_message"

    with pytest.raises(MCPToolNotFoundError):
        async with mcp_server(
            r,
            name="slack",
            url="mcp://slack.com/mcp",
            allowed_tools=["ghost_tool"],  # not advertised — registration fails
            auth=None,
            client=client,
        ):
            pass  # unreachable: exception fires before the block body runs

    # A session was opened during connect(), then closed by the error handler
    assert len(client.closed) == 1
