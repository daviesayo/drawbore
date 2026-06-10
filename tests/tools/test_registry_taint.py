"""TaintError + denied:taint label + public exports."""

import pytest

from drawbore.errors import halt_reason_for
from drawbore.tools import TaintError, TrustLabel, TaintLedger, join, ToolRegistry
from drawbore.tools.proxy.proxy import _denial_label


def test_taint_error_classifies_as_taint_violation():
    assert halt_reason_for(TaintError("blocked")) == "taint_violation"


def test_denial_label_for_taint():
    assert _denial_label(TaintError("blocked")) == "denied:taint"


def test_taint_surface_is_exported_from_drawbore_tools():
    assert TrustLabel.UNTRUSTED.value == "untrusted"
    assert join() is TrustLabel.TRUSTED
    assert isinstance(TaintLedger(), TaintLedger)
    import drawbore.tools as tools_pkg
    for name in ("TaintError", "TrustLabel", "TaintLedger", "join"):
        assert name in tools_pkg.__all__


# ── Task 3: source_trust / exfil_capable tool facts ──────────────────────────

async def _h(args):
    return args


def test_tool_facts_default_safe_but_inert():
    reg = ToolRegistry()
    t = reg.register_tool("a", _h)
    assert t.source_trust is TrustLabel.TRUSTED and t.exfil_capable is False


def test_mcp_tool_defaults_untrusted_source():
    reg = ToolRegistry()
    t = reg.register_mcp_tool("m", _h)
    assert t.source_trust is TrustLabel.UNTRUSTED and t.exfil_capable is False


def test_facts_are_overridable_on_every_kind():
    reg = ToolRegistry()
    sink = reg.register_tool("sink", _h, exfil_capable=True)
    assert sink.exfil_capable is True and sink.source_trust is TrustLabel.TRUSTED
    b = reg.register_builtin("b", _h, source_trust=TrustLabel.UNTRUSTED)
    assert b.source_trust is TrustLabel.UNTRUSTED
    trusted_mcp = reg.register_mcp_tool("tm", _h, source_trust=TrustLabel.TRUSTED)
    assert trusted_mcp.source_trust is TrustLabel.TRUSTED
    em = reg.register_mcp_tool("em", _h, exfil_capable=True)
    assert em.exfil_capable is True and em.source_trust is TrustLabel.UNTRUSTED


# ── Task 4: __post_init__ coercion — structural, not name-based ──────────────

def test_tool_coerces_string_source_trust():
    from drawbore.tools.registry.registry import Tool
    t = Tool(name="raw", handler=_h, source_trust="untrusted")
    assert t.source_trust is TrustLabel.UNTRUSTED
    with pytest.raises(ValueError):
        Tool(name="bad", handler=_h, source_trust="not-a-label")
