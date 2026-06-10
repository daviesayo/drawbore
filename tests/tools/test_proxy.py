import types

import pytest
from drawbore.tools import ToolProxy, ToolRegistry, TokenIssuer
from drawbore.tools.proxy import ToolProxy  # noqa: F811 (same symbol)
from drawbore.tools.errors import ToolAccessError, TokenError, CircuitBreakerError


# ---------------------------------------------------------------------------
# Helpers shared by the original tests
# ---------------------------------------------------------------------------

def _ctx(run_id="r1", step=0):
    return types.SimpleNamespace(run_id=run_id, step=step)


async def _echo(args):
    return {"echo": args}


def _setup(max_calls=3):
    reg = ToolRegistry()
    reg.register_tool("db.read", _echo, allowed_operations=["invoke"])
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer, max_calls_per_tool=max_calls)
    return reg, issuer, proxy


# ---------------------------------------------------------------------------
# Original proxy tests
# ---------------------------------------------------------------------------

async def test_invoke_happy_path_logs():
    reg, issuer, proxy = _setup()
    tok = issuer.issue("db.read", "r1")
    out = await proxy.invoke("db.read", {"id": 1}, tok, _ctx())
    assert out == {"echo": {"id": 1}}
    assert len(proxy.log) == 1
    rec = proxy.log[0]
    assert rec["tool"] == "db.read" and rec["result"] == "ok"
    assert rec["input_hash"] and rec["output_hash"]


async def test_token_is_single_use_via_proxy():
    reg, issuer, proxy = _setup()
    tok = issuer.issue("db.read", "r1")
    await proxy.invoke("db.read", {}, tok, _ctx())
    with pytest.raises(TokenError):
        await proxy.invoke("db.read", {}, tok, _ctx())


async def test_disallowed_operation_raises():
    reg = ToolRegistry()
    reg.register_tool("db.read", _echo, allowed_operations=["read"])
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer)
    tok = issuer.issue("db.read", "r1", operation="write")
    with pytest.raises(ToolAccessError):
        await proxy.invoke("db.read", {}, tok, _ctx(), operation="write")


async def test_unknown_tool_raises():
    reg = ToolRegistry()
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer)
    tok = issuer.issue("ghost", "r1")
    with pytest.raises(ToolAccessError):
        await proxy.invoke("ghost", {}, tok, _ctx())


async def test_circuit_breaker_trips():
    reg, issuer, proxy = _setup(max_calls=2)
    for _ in range(2):
        tok = issuer.issue("db.read", "r1")
        await proxy.invoke("db.read", {}, tok, _ctx())
    tok = issuer.issue("db.read", "r1")
    with pytest.raises(CircuitBreakerError):
        await proxy.invoke("db.read", {}, tok, _ctx())


# ---------------------------------------------------------------------------
# Audit H1-H3 hardening tests (Task 2)
# ---------------------------------------------------------------------------

def _setup2(max_calls=3):
    reg = ToolRegistry()

    async def handler(args):
        if args == {"boom": True}:
            raise ValueError("handler blew up")
        return {"echo": args}

    reg.register_tool("svc.do", handler, allowed_operations=("invoke", "read"))
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer, max_calls_per_tool=max_calls)
    return reg, issuer, proxy


def _ctx2(run_id="r", step=0):
    return types.SimpleNamespace(run_id=run_id, step=step)


async def test_breaker_is_per_agent_step_not_per_run():
    _, issuer, proxy = _setup2(max_calls=2)
    for step in (0, 1):
        for _ in range(2):
            tok = issuer.issue("svc.do", "r", "invoke")
            await proxy.invoke("svc.do", {"x": 1}, tok, _ctx2(step=step), "invoke")


async def test_breaker_trips_for_a_single_step_over_limit():
    _, issuer, proxy = _setup2(max_calls=2)
    with pytest.raises(CircuitBreakerError):
        for _ in range(3):
            tok = issuer.issue("svc.do", "r", "invoke")
            await proxy.invoke("svc.do", {"x": 1}, tok, _ctx2(step=0), "invoke")


async def test_invalid_tokens_do_not_count_toward_breaker():
    _, issuer, proxy = _setup2(max_calls=2)
    for _ in range(3):
        tok = issuer.issue("svc.do", "r", "invoke")
        issuer.consume(tok, "svc.do", "r", "invoke")  # pre-consume -> now invalid
        with pytest.raises(TokenError):
            await proxy.invoke("svc.do", {"x": 1}, tok, _ctx2(step=0), "invoke")
    tok = issuer.issue("svc.do", "r", "invoke")
    assert await proxy.invoke("svc.do", {"x": 1}, tok, _ctx2(step=0), "invoke") == {"echo": {"x": 1}}


async def test_denied_and_failed_calls_are_logged():
    _, issuer, proxy = _setup2(max_calls=2)
    bad = issuer.issue("svc.do", "r", "invoke")
    issuer.consume(bad, "svc.do", "r", "invoke")
    with pytest.raises(TokenError):
        await proxy.invoke("svc.do", {"x": 1}, bad, _ctx2(), "invoke")
    tok = issuer.issue("svc.do", "r", "delete")
    with pytest.raises(ToolAccessError):
        await proxy.invoke("svc.do", {"x": 1}, tok, _ctx2(), "delete")
    tok = issuer.issue("svc.do", "r", "invoke")
    with pytest.raises(ValueError):
        await proxy.invoke("svc.do", {"boom": True}, tok, _ctx2(), "invoke")
    labels = [e["result"] for e in proxy.log]
    assert "denied:token" in labels
    assert "denied:scope" in labels
    assert "error" in labels


async def test_breaker_trip_is_logged_as_denied_breaker():
    _, issuer, proxy = _setup2(max_calls=1)
    tok = issuer.issue("svc.do", "r", "invoke")
    await proxy.invoke("svc.do", {"x": 1}, tok, _ctx2(step=0), "invoke")  # 1st: ok
    tok = issuer.issue("svc.do", "r", "invoke")
    with pytest.raises(CircuitBreakerError):
        await proxy.invoke("svc.do", {"x": 1}, tok, _ctx2(step=0), "invoke")  # 2nd: trips
    assert proxy.log[-1]["result"] == "denied:breaker"
