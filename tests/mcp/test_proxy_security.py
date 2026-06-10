import pytest

from drawbore.tools import (
    ToolRegistry, ToolProxy, TokenIssuer, RunContext, build_tool_context,
    ToolAccessError, CircuitBreakerError, set_run_context, reset_run_context,
)
from drawbore.mcp import (
    FakeMCPClient, MCPToolSpec, OAuthConfig, register_mcp_server,
)


def _client():
    return FakeMCPClient(tools={
        "send_message": MCPToolSpec(name="send_message", input_schema={}, description=""),
        "delete_channel": MCPToolSpec(name="delete_channel", input_schema={}, description=""),
    }, results={"send_message": {"ok": True}})


async def _wire(allowed=("send_message",)):
    registry = ToolRegistry()
    client = _client()
    auth = OAuthConfig(client_id="cid", scopes=("chat:write",))
    await register_mcp_server(
        registry, name="slack", url="mcp://slack.com/mcp",
        allowed_tools=list(allowed), auth=auth, client=client,
    )
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer)
    return registry, issuer, proxy, client


async def test_declared_mcp_tool_is_reached_only_through_the_proxy_with_a_jit_token():
    registry, issuer, proxy, client = await _wire()
    ctx = build_tool_context(["mcp://slack/send_message"], proxy, issuer)
    token = set_run_context(RunContext(run_id="r1", step=0))
    try:
        out = await ctx.call("mcp://slack/send_message", {"text": "hi"})
    finally:
        reset_run_context(token)
    assert out == {"ok": True}
    assert client.calls == [("send_message", {"text": "hi"})]  # reached the server tool
    # The call went through the proxy chokepoint (it logged the invocation).
    assert [e["tool"] for e in proxy.log] == ["mcp://slack/send_message"]
    assert proxy.log[0]["result"] == "ok"


async def test_agent_cannot_reach_an_undeclared_tool_on_the_same_server():
    # delete_channel was never registered (not in allowed_tools), so it is blocked
    # at BOTH layers: the agent's ToolContext (scope) and the proxy/registry
    # (registration scope) — lateral movement is impossible by any path.
    registry, issuer, proxy, client = await _wire(allowed=("send_message",))
    assert not registry.has("mcp://slack/delete_channel")  # never registered

    token = set_run_context(RunContext(run_id="r1", step=0))
    try:
        # Layer 1 — an agent's ToolContext exposes ONLY its declared tools.
        ctx = build_tool_context(["mcp://slack/send_message"], proxy, issuer)
        with pytest.raises(ToolAccessError):
            await ctx.call("mcp://slack/delete_channel", {})

        # Layer 2 — even bypassing the ToolContext and minting a token directly,
        # the proxy rejects an unregistered ref: the tool simply does not exist in
        # the registry, so server access never implies tool access.
        forged = issuer.issue("mcp://slack/delete_channel", "r1", "invoke")
        with pytest.raises(ToolAccessError):
            await proxy.invoke(
                "mcp://slack/delete_channel", {}, forged,
                RunContext(run_id="r1", step=0),
            )
    finally:
        reset_run_context(token)
    assert client.calls == []  # the server tool was never touched, by any path


async def test_server_auth_never_reaches_the_invocation_path():
    # Server-level OAuth is consumed at registration (connect); the per-invocation
    # path carries only the JIT token — no server credential rides through it.
    registry, issuer, proxy, client = await _wire()
    auth = next(cfg.auth for cfg in client.connected_with.values() if cfg.auth)
    assert auth is not None  # the credential WAS used to connect

    # Drive a real proxied invocation and inspect what actually reached the wire.
    ctx = build_tool_context(["mcp://slack/send_message"], proxy, issuer)
    token = set_run_context(RunContext(run_id="r1", step=0))
    try:
        await ctx.call("mcp://slack/send_message", {"text": "hi"})
    finally:
        reset_run_context(token)

    (tool_name, sent_args), = client.calls
    assert tool_name == "send_message"
    # The OAuthConfig object never appears in the arguments routed to the tool.
    assert auth not in sent_args.values()
    assert "auth" not in sent_args
    # The proxy's audit log records refs/hashes/outcomes — never a credential key.
    assert all("auth" not in entry for entry in proxy.log)
    # The agent-facing registry Tool exposes no auth attribute, either.
    assert getattr(registry.get("mcp://slack/send_message"), "auth", None) is None


async def test_mcp_tool_respects_the_per_step_circuit_breaker():
    registry, issuer, proxy, client = await _wire()
    ctx = build_tool_context(["mcp://slack/send_message"], proxy, issuer)
    token = set_run_context(RunContext(run_id="r1", step=0))
    try:
        # Default breaker is 3 calls/tool/step; the 4th trips it.
        for _ in range(3):
            await ctx.call("mcp://slack/send_message", {"text": "x"})
        with pytest.raises(CircuitBreakerError):
            await ctx.call("mcp://slack/send_message", {"text": "x"})
    finally:
        reset_run_context(token)
