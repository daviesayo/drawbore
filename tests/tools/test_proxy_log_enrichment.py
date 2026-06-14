"""Tests for the enriched proxy log entries (step, scope, kind, exfil_capable).

Each log entry must carry four new fields alongside the original seven:
  step          - the int | None step index from run_ctx
  scope         - "trusted" | "untrusted" (the taint scope at call time)
  kind          - the resolved Tool.kind, or None for an unresolved-tool denial
  exfil_capable - the resolved Tool.exfil_capable, or False for an unresolved-tool denial
"""
import types
import pytest

from drawbore.tools import TaintLedger, ToolRegistry, TrustLabel
from drawbore.tools.access import RunContext
from drawbore.tools.errors import TaintError, ToolAccessError, TokenError
from drawbore.tools.proxy.proxy import ToolProxy
from drawbore.tools.tokens import TokenIssuer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _ok(args):
    return {"ok": True}


def _ctx(run_id="r1", step=0):
    return RunContext(run_id, step)


def _wire(registry, *, ledger=None, max_calls=3):
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer, max_calls_per_tool=max_calls, ledger=ledger)
    return issuer, proxy


# ---------------------------------------------------------------------------
# Successful call carries all four new fields
# ---------------------------------------------------------------------------

async def test_successful_call_log_entry_has_all_enriched_fields():
    reg = ToolRegistry()
    reg.register_tool("svc.read", _ok, allowed_operations=["invoke"])
    led = TaintLedger()
    issuer, proxy = _wire(reg, ledger=led)
    tok = issuer.issue("svc.read", "r1", "invoke")
    await proxy.invoke("svc.read", {}, tok, _ctx("r1", 2), "invoke")

    entry = proxy.log[-1]
    # Existing fields untouched
    assert entry["tool"] == "svc.read"
    assert entry["result"] == "ok"
    assert entry["run_id"] == "r1"
    assert entry["operation"] == "invoke"
    assert "input_hash" in entry
    assert "output_hash" in entry
    assert "duration" in entry
    # New fields
    assert entry["step"] == 2
    assert entry["scope"] == led.scope("r1", 2).value   # "trusted"
    assert entry["kind"] == "custom"
    assert entry["exfil_capable"] is False


async def test_successful_call_scope_is_untrusted_when_step_is_tainted():
    reg = ToolRegistry()
    reg.register_tool("svc.read", _ok, allowed_operations=["invoke"])
    led = TaintLedger(managed=True)
    led.seed("r1", 0, TrustLabel.UNTRUSTED)
    # svc.read is NOT exfil_capable so the call goes through even under UNTRUSTED scope
    issuer, proxy = _wire(reg, ledger=led)
    tok = issuer.issue("svc.read", "r1", "invoke")
    await proxy.invoke("svc.read", {}, tok, _ctx("r1", 0), "invoke")

    entry = proxy.log[-1]
    assert entry["scope"] == "untrusted"
    assert entry["result"] == "ok"
    assert entry["kind"] == "custom"
    assert entry["exfil_capable"] is False


async def test_exfil_capable_tool_log_entry_reflects_true():
    reg = ToolRegistry()
    reg.register_tool("sink", _ok, exfil_capable=True, allowed_operations=["invoke"])
    led = TaintLedger()
    issuer, proxy = _wire(reg, ledger=led)
    tok = issuer.issue("sink", "r1", "invoke")
    await proxy.invoke("sink", {}, tok, _ctx("r1", 1), "invoke")

    entry = proxy.log[-1]
    assert entry["exfil_capable"] is True
    assert entry["result"] == "ok"
    assert entry["kind"] == "custom"
    assert entry["scope"] == "trusted"
    assert entry["step"] == 1


async def test_mcp_tool_log_entry_kind_is_mcp():
    reg = ToolRegistry()
    reg.register_mcp_tool("fetch", _ok)
    led = TaintLedger()
    issuer, proxy = _wire(reg, ledger=led)
    tok = issuer.issue("fetch", "r1", "invoke")
    await proxy.invoke("fetch", {}, tok, _ctx("r1", 0), "invoke")

    entry = proxy.log[-1]
    assert entry["kind"] == "mcp"
    assert entry["result"] == "ok"


# ---------------------------------------------------------------------------
# Denied calls (taint denial) also carry the new fields
# ---------------------------------------------------------------------------

async def test_taint_denial_log_entry_has_enriched_fields():
    reg = ToolRegistry()
    reg.register_tool("sink", _ok, exfil_capable=True, allowed_operations=["invoke"])
    led = TaintLedger(managed=True)
    led.seed("r1", 0, TrustLabel.UNTRUSTED)
    issuer, proxy = _wire(reg, ledger=led)
    tok = issuer.issue("sink", "r1", "invoke")
    with pytest.raises(TaintError):
        await proxy.invoke("sink", {}, tok, _ctx("r1", 0), "invoke")

    entry = proxy.log[-1]
    assert entry["result"] == "denied:taint"
    assert entry["step"] == 0
    assert entry["scope"] == "untrusted"
    assert entry["kind"] == "custom"
    assert entry["exfil_capable"] is True


async def test_token_denial_log_entry_has_enriched_fields():
    reg = ToolRegistry()
    reg.register_tool("svc.read", _ok, allowed_operations=["invoke"])
    # Explicit unmanaged ledger: this test verifies that the logged scope value is
    # "trusted" for an unseeded step — the documented opt-out contract.
    issuer, proxy = _wire(reg, ledger=TaintLedger(managed=False))
    bad = issuer.issue("svc.read", "r1", "invoke")
    issuer.consume(bad, "svc.read", "r1", "invoke")   # pre-consume → now invalid
    with pytest.raises(TokenError):
        await proxy.invoke("svc.read", {}, bad, _ctx("r1", 3), "invoke")

    entry = proxy.log[-1]
    assert entry["result"] == "denied:token"
    assert entry["step"] == 3
    assert entry["scope"] == "trusted"   # explicit unmanaged ledger → unseeded step is TRUSTED
    assert entry["kind"] == "custom"
    assert entry["exfil_capable"] is False


# ---------------------------------------------------------------------------
# Unresolved-tool denial: registry.get raises → kind=None, exfil_capable=False
# ---------------------------------------------------------------------------

async def test_unresolved_tool_denial_logs_kind_none_and_exfil_false():
    """When registry.get itself raises ToolAccessError (tool not registered),
    the log entry must have kind=None and exfil_capable=False (the tool was
    never resolved), plus the correct step and scope."""
    reg = ToolRegistry()   # empty — "ghost" is not registered
    # Explicit unmanaged ledger: this test verifies logged scope for an
    # unresolved-tool denial with the opt-out documented.
    issuer, proxy = _wire(reg, ledger=TaintLedger(managed=False))
    tok = issuer.issue("ghost", "r1", "invoke")
    with pytest.raises(ToolAccessError):
        await proxy.invoke("ghost", {}, tok, _ctx("r1", 7), "invoke")

    entry = proxy.log[-1]
    assert entry["result"] == "denied:scope"
    assert entry["step"] == 7
    assert entry["scope"] == "trusted"   # explicit unmanaged ledger → unseeded step is TRUSTED
    assert entry["kind"] is None
    assert entry["exfil_capable"] is False


# ---------------------------------------------------------------------------
# ToolCallMetric.from_log_entry still works on enriched entries (regression)
# ---------------------------------------------------------------------------

async def test_tool_call_metric_from_enriched_log_entry():
    from drawbore.audit.metrics import ToolCallMetric

    reg = ToolRegistry()
    reg.register_tool("svc.read", _ok, allowed_operations=["invoke"])
    issuer, proxy = _wire(reg)
    tok = issuer.issue("svc.read", "r1", "invoke")
    await proxy.invoke("svc.read", {}, tok, _ctx("r1", 0), "invoke")

    entry = proxy.log[-1]
    # Extra keys must be harmless — from_log_entry reads only tool/operation/duration/result
    metric = ToolCallMetric.from_log_entry(entry)
    assert metric.tool == "svc.read"
    assert metric.operation == "invoke"
    assert metric.result == "ok"
    assert metric.duration_seconds >= 0
