"""Tests for the taint sink-gate and UNTRUSTED-source bump in ToolProxy.invoke."""
import pytest

from drawbore.tools import TaintError, TaintLedger, ToolRegistry, TrustLabel
from drawbore.tools.access import RunContext
from drawbore.tools.proxy.proxy import ToolProxy
from drawbore.tools.tokens import TokenIssuer


async def _ok(args):
    return {"ok": True}


def _wire(registry, ledger):
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer, ledger=ledger)
    return issuer, proxy


async def test_sink_gate_blocks_exfil_under_untrusted_scope():
    reg = ToolRegistry()
    reg.register_tool("sink", _ok, exfil_capable=True)
    led = TaintLedger(managed=True)
    led.seed("r", 0, TrustLabel.UNTRUSTED)
    issuer, proxy = _wire(reg, led)
    tok = issuer.issue("sink", "r", "invoke")
    with pytest.raises(TaintError):
        await proxy.invoke("sink", {}, tok, RunContext("r", 0), "invoke")
    # gate-first: the token was NOT consumed and the breaker did not count it
    assert issuer._records[tok._secret].used is False
    assert proxy._counts.get(("r", 0, "sink"), 0) == 0


async def test_sink_allowed_under_trusted_scope():
    reg = ToolRegistry()
    reg.register_tool("sink", _ok, exfil_capable=True)
    led = TaintLedger(managed=True)
    led.seed("r", 0, TrustLabel.TRUSTED)
    issuer, proxy = _wire(reg, led)
    tok = issuer.issue("sink", "r", "invoke")
    assert await proxy.invoke("sink", {}, tok, RunContext("r", 0), "invoke") == {"ok": True}


async def test_untrusted_source_bumps_scope_then_gates_a_later_sink():
    reg = ToolRegistry()
    reg.register_mcp_tool("fetch", _ok)              # source_trust UNTRUSTED by default
    reg.register_tool("sink", _ok, exfil_capable=True)
    led = TaintLedger(managed=True)
    led.seed("r", 0, TrustLabel.TRUSTED)             # step starts trusted
    issuer, proxy = _wire(reg, led)
    # fetch is allowed (trusted scope, not a sink) and bumps the step to UNTRUSTED
    await proxy.invoke("fetch", {}, issuer.issue("fetch", "r", "invoke"), RunContext("r", 0))
    assert led.scope("r", 0) is TrustLabel.UNTRUSTED
    # the subsequent sink is now gated
    with pytest.raises(TaintError):
        await proxy.invoke("sink", {}, issuer.issue("sink", "r", "invoke"), RunContext("r", 0))


# ── Finding 5: additional proxy-layer regression guards ──────────────────────

async def test_taint_denial_is_logged():
    reg = ToolRegistry()
    reg.register_tool("sink", _ok, exfil_capable=True)
    led = TaintLedger(managed=True)
    led.seed("r", 0, TrustLabel.UNTRUSTED)
    issuer, proxy = _wire(reg, led)
    with pytest.raises(TaintError):
        await proxy.invoke("sink", {}, issuer.issue("sink", "r", "invoke"), RunContext("r", 0))
    assert proxy.log[-1]["result"] == "denied:taint"


async def test_same_tool_untrusted_exfil_first_allowed_then_gated():
    # A tool that is BOTH untrusted-source and exfil-capable: first call passes the
    # gate (scope still trusted), bumps the step; second call is gated.
    reg = ToolRegistry()
    reg.register_mcp_tool("both", _ok, exfil_capable=True)
    led = TaintLedger(managed=True)
    led.seed("r", 0, TrustLabel.TRUSTED)
    issuer, proxy = _wire(reg, led)
    assert await proxy.invoke("both", {}, issuer.issue("both", "r", "invoke"), RunContext("r", 0)) == {"ok": True}
    with pytest.raises(TaintError):
        await proxy.invoke("both", {}, issuer.issue("both", "r", "invoke"), RunContext("r", 0))


async def test_none_step_fails_closed_on_managed_ledger():
    reg = ToolRegistry()
    reg.register_tool("sink", _ok, exfil_capable=True)
    led = TaintLedger(managed=True)          # nothing seeded for (r, None)
    issuer, proxy = _wire(reg, led)
    ctx = RunContext("r", None)
    with pytest.raises(TaintError):
        await proxy.invoke("sink", {}, issuer.issue("sink", "r", "invoke"), ctx)


async def test_denied_call_still_taints_the_step():
    # Intentional: a call with a bad token to an untrusted-source tool still taints
    # the step (over-taint on a denial is fail-safe).
    from drawbore.tools.errors import TokenError
    reg = ToolRegistry()
    reg.register_mcp_tool("fetch", _ok)
    led = TaintLedger(managed=True)
    led.seed("r", 0, TrustLabel.TRUSTED)
    issuer, proxy = _wire(reg, led)
    wrong = issuer.issue("fetch", "OTHER-RUN", "invoke")     # token for the wrong run
    with pytest.raises(TokenError):
        await proxy.invoke("fetch", {}, wrong, RunContext("r", 0))
    assert led.scope("r", 0) is TrustLabel.UNTRUSTED
