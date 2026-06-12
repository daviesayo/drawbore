"""Per-tool trust/exfil declarations on register_mcp_server.

An outward-writing MCP tool (a notifier, an email relay) must be declarable as
exfil-capable through the public MCP registration path, so the taint sink-gate
protects it with the same honesty a custom `register_tool(..., exfil_capable=True)`
gets. Without the declaration the gate cannot fire — that is the gap these tests
close.
"""
import pytest

from drawbore.tools import TaintError, TaintLedger, TokenIssuer, TrustLabel
from drawbore.tools.access import RunContext
from drawbore.tools.proxy.proxy import ToolProxy
from drawbore.tools import ToolRegistry
from drawbore.mcp import (
    FakeMCPClient, MCPToolSpec, register_mcp_server,
)


def _client():
    return FakeMCPClient(tools={
        "send_message": MCPToolSpec(name="send_message", input_schema={}, description=""),
        "read_status": MCPToolSpec(name="read_status", input_schema={}, description=""),
    }, results={"send_message": {"ok": True}, "read_status": {"ok": True}})


async def _register(**kwargs):
    registry = ToolRegistry()
    await register_mcp_server(
        registry, name="notify", url="mcp://notify.example/mcp",
        allowed_tools=["send_message"], client=_client(), **kwargs,
    )
    return registry


def _wire(registry, ledger):
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer, ledger=ledger)
    return issuer, proxy


async def test_exfil_declaration_on_server_makes_mcp_tool_a_gated_sink():
    # Declare the notifier exfil-capable through register_mcp_server itself.
    registry = await _register(exfil_capable={"send_message": True})
    assert registry.get("mcp://notify/send_message").exfil_capable is True

    led = TaintLedger(managed=True)
    led.seed("r", 0, TrustLabel.UNTRUSTED)            # the step is handling untrusted data
    issuer, proxy = _wire(registry, led)
    tok = issuer.issue("mcp://notify/send_message", "r", "invoke")
    with pytest.raises(TaintError):
        await proxy.invoke("mcp://notify/send_message", {}, tok, RunContext("r", 0))


async def test_without_declaration_the_exfil_gate_cannot_fire():
    # The gap: with no exfil declaration the SAME outward-writing tool is not a
    # sink, so the taint gate never refuses it even under untrusted scope.
    registry = await _register()
    assert registry.get("mcp://notify/send_message").exfil_capable is False

    led = TaintLedger(managed=True)
    led.seed("r", 0, TrustLabel.UNTRUSTED)
    issuer, proxy = _wire(registry, led)
    tok = issuer.issue("mcp://notify/send_message", "r", "invoke")
    assert await proxy.invoke("mcp://notify/send_message", {}, tok, RunContext("r", 0)) == {"ok": True}


async def test_source_trust_override_on_server_stops_tainting_the_step():
    # An internal, trusted read-only server: declaring it TRUSTED means invoking it
    # does not taint the step (the MCP default stays UNTRUSTED for everything else).
    registry = ToolRegistry()
    await register_mcp_server(
        registry, name="kb", url="mcp://kb.internal/mcp",
        allowed_tools=["read_status"], client=FakeMCPClient(
            tools={"read_status": MCPToolSpec(name="read_status", input_schema={}, description="")},
            results={"read_status": {"ok": True}},
        ),
        source_trust={"read_status": TrustLabel.TRUSTED},
    )
    assert registry.get("mcp://kb/read_status").source_trust is TrustLabel.TRUSTED

    led = TaintLedger(managed=True)
    led.seed("r", 0, TrustLabel.TRUSTED)
    issuer, proxy = _wire(registry, led)
    tok = issuer.issue("mcp://kb/read_status", "r", "invoke")
    await proxy.invoke("mcp://kb/read_status", {}, tok, RunContext("r", 0))
    assert led.scope("r", 0) is TrustLabel.TRUSTED     # untrusted-by-default override held


async def test_mcp_default_stays_untrusted_and_non_exfil():
    # Fail-safe default: no declaration → untrusted source, not a sink.
    registry = await _register()
    tool = registry.get("mcp://notify/send_message")
    assert tool.source_trust is TrustLabel.UNTRUSTED
    assert tool.exfil_capable is False


async def test_flag_for_a_tool_not_in_allowed_tools_is_rejected():
    # Cannot scope a flag to a phantom tool — fail closed before connecting.
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        await register_mcp_server(
            registry, name="notify", url="mcp://notify.example/mcp",
            allowed_tools=["send_message"], client=_client(),
            exfil_capable={"read_status": True},          # not declared in allowed_tools
        )


async def test_source_trust_flag_for_undeclared_tool_is_rejected():
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        await register_mcp_server(
            registry, name="notify", url="mcp://notify.example/mcp",
            allowed_tools=["send_message"], client=_client(),
            source_trust={"read_status": TrustLabel.TRUSTED},
        )
