"""Pipeline run-loop taint integration tests.

Verifies that trust labels propagate across steps and that a step seeded
UNTRUSTED (because its predecessor called an UNTRUSTED-source tool) cannot
invoke an exfil-capable tool.
"""
import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.testing import call, final
from drawbore.tools import ToolRegistry, TrustLabel

SRC = "ti:src"
SINK = "ti:sink"


class In(BaseModel):
    id: str


class Mid(BaseModel):
    note: str


class Out(BaseModel):
    ok: bool


@agent(name="producer", input=In, output=Mid, model="m-1.0", tools=[SRC])
async def producer(v: In, tools) -> Mid: ...


@agent(name="sender", input=Mid, output=Out, model="m-1.0", tools=[SINK])
async def sender(v: Mid, tools) -> Out: ...


def _untrusted_registry():
    async def _h(args):
        return {"ok": True}
    reg = ToolRegistry()
    reg.register_mcp_tool(SRC, _h)                      # UNTRUSTED source (MCP default)
    reg.register_tool(SINK, _h, exfil_capable=True)     # exfil-capable sink
    return reg


async def test_untrusted_output_gates_a_downstream_sink():
    """A step whose predecessor touched an UNTRUSTED source must not be able to
    call an exfil-capable tool (taint-gated lethal-trifecta breaker)."""
    reg = _untrusted_registry()
    # unbound second step -> predecessor binding (sender reads from producer)
    pipe = Pipeline("ti", registry=reg).add(producer).add(sender)
    async with pipe.test_mode(
        mock_loop_scripts={
            "producer": [call(SRC), final({"note": "x"})],   # touches the untrusted source
            "sender": [call(SINK), final({"ok": True})],      # tries to exfil downstream
        },
        mock_tools={SRC: {"ok": True}, SINK: {"ok": True}},
    ) as tp:
        result = await tp.run(In(id="t1"))
    assert result.status in ("halted", "escalated")
    assert result.reason.startswith("taint_violation:")
    assert "denied:taint" in result.audit_trace.legible()


async def test_trusted_source_lets_the_sink_through():
    """A step whose predecessor only touched TRUSTED sources can invoke an
    exfil-capable tool without restriction."""
    async def _h(args):
        return {"ok": True}
    reg = ToolRegistry()
    reg.register_tool(SRC, _h)                           # TRUSTED source (custom default)
    reg.register_tool(SINK, _h, exfil_capable=True)
    pipe = Pipeline("ti2", registry=reg).add(producer).add(sender)
    async with pipe.test_mode(
        mock_loop_scripts={
            "producer": [call(SRC), final({"note": "x"})],
            "sender": [call(SINK), final({"ok": True})],
        },
        mock_tools={SRC: {"ok": True}, SINK: {"ok": True}},
    ) as tp:
        result = await tp.run(In(id="t2"))
    assert result.status == "completed"
