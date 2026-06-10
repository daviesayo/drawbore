"""End-to-end proof: a deterministic agent uses an MCP tool through a real
``Pipeline`` with every Drawbore guarantee intact.

An MCP-backed tool is just a proxy-backed ``Tool`` in the ``ToolRegistry``,
so the agent reaches it through the same ``ToolContext.call`` path as any other
tool — ``pipeline.py`` needs no change. A failed MCP call self-declares
``halt_reason="mcp_error"``, so the halt/escalation reason is legible to a
regulator, and declaring an MCP tool the server never granted is rejected at
``Pipeline.add`` time (the declared-tool check).
"""

import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.tools import ToolContext, ToolRegistry, ToolAccessError
from drawbore.escalation import EscalationPolicy, RecordingDispatcher
from drawbore.mcp import FakeMCPClient, MCPError, MCPToolSpec, OAuthConfig, register_mcp_server


class In(BaseModel):
    text: str


class Out(BaseModel):
    delivered: bool


def _client(fail=False):
    results = {} if fail else {"send_message": {"ok": True}}
    c = FakeMCPClient(
        tools={"send_message": MCPToolSpec(name="send_message", input_schema={}, description="")},
        results=results,
    )
    if fail:
        # Replace the instance attribute; the registration handler does dynamic
        # dispatch (`await client.call_tool(session, tool_name, args)`), so the
        # patch takes effect on every call. Three positionals, no `self`.
        async def boom(session, tool_name, arguments):
            raise MCPError("server exploded")

        c.call_tool = boom  # type: ignore[method-assign]
    return c


async def _registry(client):
    r = ToolRegistry()
    await register_mcp_server(
        r,
        name="slack",
        url="mcp://slack.com/mcp",
        allowed_tools=["send_message"],
        auth=OAuthConfig(client_id="cid", scopes=("chat:write",)),
        client=client,
    )
    return r


async def test_deterministic_agent_calls_an_mcp_tool_end_to_end():
    registry = await _registry(_client())

    @agent(name="notifier", input=In, output=Out, tools=["mcp://slack/send_message"])
    async def notifier(value: In, tools: ToolContext) -> Out:
        res = await tools.call("mcp://slack/send_message", {"text": value.text})
        return Out(delivered=bool(res.get("ok")))

    p = Pipeline(name="t", registry=registry)
    p.add(notifier)
    r = await p.run(In(text="hi"))
    assert r.status == "completed"
    assert r.outputs["notifier"].delivered is True


async def test_mcp_tool_failure_halts_and_escalates_with_mcp_error_reason():
    registry = await _registry(_client(fail=True))

    @agent(name="notifier", input=In, output=Out, tools=["mcp://slack/send_message"])
    async def notifier(value: In, tools: ToolContext) -> Out:
        await tools.call("mcp://slack/send_message", {"text": value.text})
        return Out(delivered=True)

    d = RecordingDispatcher()
    p = Pipeline(name="t", registry=registry, on_failure=EscalationPolicy("slack", "q"), dispatcher=d)
    p.add(notifier)
    r = await p.run(In(text="hi"))
    assert r.status == "escalated"
    assert r.halted_at == "notifier"
    # MCPError self-declares halt_reason="mcp_error", so the package carries a
    # legible reason, not the generic "agent_error".
    assert "mcp_error" in r.escalations[0].reason
    assert len(d.sent) == 1


async def test_declaring_an_unregistered_mcp_tool_is_rejected_at_pipeline_add():
    registry = await _registry(_client())

    @agent(name="bad", input=In, output=Out, tools=["mcp://slack/delete_channel"])
    async def bad(value: In, tools: ToolContext) -> Out:
        return Out(delivered=False)

    p = Pipeline(name="t", registry=registry)
    # delete_channel was never registered (only send_message was declared), so the
    # declared-tool check rejects it at add time — it is unreachable.
    with pytest.raises(ToolAccessError):
        p.add(bad)
