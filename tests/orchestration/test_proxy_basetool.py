import pytest

from drawbore.orchestration.adk_tools import proxy_backed_tool
from drawbore.tools import ToolRegistry, ToolProxy, TokenIssuer, RunContext
from drawbore.tools.errors import ToolAccessError

_SCHEMA = {
    "type": "object",
    "properties": {
        "handle_id": {"type": "string"},
        "mode": {"type": "string"},
        "query": {"type": "string"},
    },
    "required": ["handle_id"],
}


def _wiring(allowed=("invoke",)):
    reg = ToolRegistry()

    async def _echo(args):
        return {"echoed": args}

    reg.register_tool("retrieve", _echo, allowed_operations=allowed, schema=_SCHEMA)
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer)
    return reg, issuer, proxy


def test_tool_advertises_the_registry_schema_params():
    reg, issuer, proxy = _wiring()
    failures: list = []
    tool = proxy_backed_tool(
        "retrieve", proxy=proxy, issuer=issuer,
        run_ctx=RunContext(run_id="r1", step=0), schema=_SCHEMA, failures=failures,
    )
    assert tool.name == "retrieve"
    decl = tool._get_declaration()
    assert decl is not None
    # the model sees the declared parameters (exact genai shape verified in Step 0)
    rendered = str(decl)
    assert "handle_id" in rendered and "mode" in rendered and "query" in rendered


async def test_run_async_routes_through_the_proxy_with_a_jit_token(monkeypatch):
    reg, issuer, proxy = _wiring()
    failures: list = []
    real_invoke = proxy.invoke
    seen = {}

    async def spy(*a, **k):
        seen["called"] = True
        return await real_invoke(*a, **k)

    monkeypatch.setattr(proxy, "invoke", spy)
    tool = proxy_backed_tool(
        "retrieve", proxy=proxy, issuer=issuer,
        run_ctx=RunContext(run_id="r1", step=0), schema=_SCHEMA, failures=failures,
    )
    out = await tool.run_async(args={"handle_id": "h1", "mode": "search"}, tool_context=None)
    assert seen["called"] is True
    assert out == {"echoed": {"handle_id": "h1", "mode": "search"}}
    assert failures == []


async def test_a_denied_call_is_recorded_in_failures_and_reraised():
    # 'invoke' not permitted -> the proxy raises ToolAccessError; the tool records
    # it in `failures` (drives immediate abort) AND re-raises so ADK sees it.
    reg, issuer, proxy = _wiring(allowed=("read",))
    failures: list = []
    tool = proxy_backed_tool(
        "retrieve", proxy=proxy, issuer=issuer,
        run_ctx=RunContext(run_id="r1", step=0), schema=_SCHEMA, failures=failures,
    )
    with pytest.raises(ToolAccessError):
        await tool.run_async(args={"handle_id": "h1"}, tool_context=None)
    assert len(failures) == 1
    assert isinstance(failures[0], ToolAccessError)
