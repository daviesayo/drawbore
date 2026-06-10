import pytest
from drawbore.tools.registry import ToolRegistry
from drawbore.tools.tokens import TokenIssuer
from drawbore.tools.proxy import ToolProxy
from drawbore.tools.access import (
    RunContext, ToolContext, build_tool_context,
    set_run_context, reset_run_context, get_run_context,
)
from drawbore.tools.errors import ToolAccessError


async def _echo(args):
    return {"echo": args}


def _objs():
    reg = ToolRegistry()
    reg.register_tool("db.read", _echo)
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer)
    return proxy, issuer


async def test_call_declared_tool_routes_through_proxy():
    proxy, issuer = _objs()
    tools = build_tool_context({"db.read"}, proxy, issuer)
    token = set_run_context(RunContext(run_id="r1"))
    try:
        out = await tools.call("db.read", {"id": 7})
    finally:
        reset_run_context(token)
    assert out == {"echo": {"id": 7}}
    assert len(proxy.log) == 1


async def test_call_undeclared_tool_raises():
    proxy, issuer = _objs()
    tools = build_tool_context({"db.read"}, proxy, issuer)
    token = set_run_context(RunContext(run_id="r1"))
    try:
        with pytest.raises(ToolAccessError):
            await tools.call("db.write", {})
    finally:
        reset_run_context(token)


def test_get_run_context_outside_run_raises():
    with pytest.raises(ToolAccessError):
        get_run_context()


# --- C1 adversarial tests: an agent must not be able to bypass the proxy ---

def _wire():
    reg = ToolRegistry()

    async def safe(args):
        return {"ok": args}

    async def danger(args):
        return {"DANGER": args}

    reg.register_tool("svc.safe", safe)
    reg.register_tool("svc.danger", danger)  # registered but NOT declared by the agent
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer)
    ctx = build_tool_context({"svc.safe"}, proxy, issuer)
    return reg, issuer, proxy, ctx


def test_toolcontext_exposes_no_proxy_issuer_or_registry():
    _, _, _, ctx = _wire()
    assert not hasattr(ctx, "_proxy")
    assert not hasattr(ctx, "_issuer")
    assert not hasattr(ctx, "_registry")
    assert not hasattr(ctx, "_declared")
    leaked = [
        a for a in dir(ctx)
        if any(k in a.lower() for k in ("proxy", "issuer", "registry"))
    ]
    assert leaked == []


async def test_agent_cannot_call_undeclared_tool_via_context():
    _, _, _, ctx = _wire()
    tok = set_run_context(RunContext(run_id="r", step=0))
    try:
        with pytest.raises(ToolAccessError):
            await ctx.call("svc.danger", {"x": 1})
    finally:
        reset_run_context(tok)


async def test_declared_tool_still_works_and_is_logged():
    _, _, proxy, ctx = _wire()
    tok = set_run_context(RunContext(run_id="r", step=0))
    try:
        result = await ctx.call("svc.safe", {"x": 1})
    finally:
        reset_run_context(tok)
    assert result == {"ok": {"x": 1}}
    assert proxy.log and proxy.log[-1]["tool"] == "svc.safe"
    assert proxy.log[-1]["result"] == "ok"
