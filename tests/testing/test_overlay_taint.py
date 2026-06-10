"""Test that the test-mode overlay preserves source_trust and exfil_capable.

Regression guard: _register_like previously dropped taint facts, so the proxy
sink-gate would never fire under test_mode.
"""

import dataclasses

from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.evidence import InMemoryEvidenceStore
from drawbore.tools import ToolRegistry, TrustLabel
from drawbore.testing.tools import build_scoped_registry

FETCH = "ovl:fetch"
SINK = "ovl:sink"
BLT = "ovl:blt"


class In(BaseModel):
    id: str


class Out(BaseModel):
    ok: bool


@agent(name="rev", input=In, output=Out, model="m-1.0", tools=[FETCH, SINK, BLT])
async def rev(v: In, tools) -> Out: ...


async def _h(args):
    return {"ok": True}


def test_overlay_preserves_source_trust_and_exfil_capable():
    reg = ToolRegistry()
    reg.register_mcp_tool(FETCH, _h)                                       # kind=mcp, source_trust=UNTRUSTED
    reg.register_tool(SINK, _h, exfil_capable=True)                        # kind=custom, sink
    reg.register_builtin(BLT, _h, source_trust=TrustLabel.UNTRUSTED, exfil_capable=True)  # kind=builtin
    pipeline = Pipeline("ovl", registry=reg).add(rev)
    overlay = build_scoped_registry(
        pipeline,
        mock_tools={FETCH: {"ok": True}, SINK: {"ok": True}, BLT: {"ok": True}},
        allow_real_tools=(),
        evidence_store=InMemoryEvidenceStore(),
    )
    # Spot-check the original taint assertions
    assert overlay.get(FETCH).source_trust is TrustLabel.UNTRUSTED
    assert overlay.get(SINK).exfil_capable is True

    # Structural parity: every Tool field except name/handler must be copied verbatim.
    # Derived from the dataclass so a future field is auto-covered (field-drop regression).
    copied = [
        f.name for f in dataclasses.fields(type(reg.get(FETCH)))
        if f.name not in ("name", "handler")
    ]
    assert copied, "Tool dataclass shape changed unexpectedly"
    for ref in (FETCH, SINK, BLT):
        for field_name in copied:
            assert getattr(overlay.get(ref), field_name) == getattr(reg.get(ref), field_name), field_name
