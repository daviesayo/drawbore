import pytest
from drawbore.orchestration.adk_tools import as_adk_tool, make_before_tool_callback
from drawbore.tools import ToolRegistry, ToolProxy, TokenIssuer, RunContext


async def _echo(args):
    return {"echoed": args}


def _wiring():
    registry = ToolRegistry()
    registry.register_tool("echo", _echo, allowed_operations=("invoke",))
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer)
    ctx = RunContext(run_id="r1", step=0)
    return registry, issuer, proxy, ctx


async def test_as_adk_tool_routes_through_the_proxy(monkeypatch):
    # The ADK FunctionTool's callable must reach the tool ONLY via proxy.invoke
    # (the authoritative proxy.invoke path) — never the raw handler.
    registry, issuer, proxy, ctx = _wiring()
    seen = {}
    real_invoke = proxy.invoke

    async def spy_invoke(*args, **kwargs):
        seen["called"] = True
        return await real_invoke(*args, **kwargs)

    monkeypatch.setattr(proxy, "invoke", spy_invoke)

    tool = as_adk_tool("echo", proxy=proxy, issuer=issuer, run_ctx=ctx)
    assert tool.name == "echo"
    # Invoke the wrapped callable the way ADK would (kwargs from the model).
    result = await tool.func(value=7)
    assert seen.get("called") is True
    assert result == {"echoed": {"value": 7}}


def test_before_tool_callback_blocks_undeclared_tools():
    cb = make_before_tool_callback(declared=("echo",))

    class _Tool:
        name = "secret"

    blocked = cb(_Tool(), {"x": 1}, None)
    assert isinstance(blocked, dict)          # a dict return SKIPS/blocks the tool (proxy callback guardrail)
    assert "secret" in str(blocked)


def test_before_tool_callback_allows_declared_tools():
    cb = make_before_tool_callback(declared=("echo",))

    class _Tool:
        name = "echo"

    assert cb(_Tool(), {"x": 1}, None) is None   # None runs the (already proxy-wrapped) tool
