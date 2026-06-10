# tests/testing/test_footprint_soundness.py
from types import SimpleNamespace

import pytest

from drawbore.config import PipelineConfig
from drawbore.testing import assert_footprint_sound
from drawbore.testing.authority import exercised_refs


def _cfg(tools):
    a = {
        "name": "risk_scorer", "ref": "x.r", "version": "1.0", "risk_tier": "low",
        "requires_human_approval": False, "context_access": "none", "tools": tools,
        "model": "m-1.0", "fallback_model": None, "instructions": None,
        "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
        "input_schema_hash": "sha256:0", "output_schema_hash": "sha256:0",
    }
    return PipelineConfig.model_validate({
        "schema_version": 1, "pipeline": {"name": "p", "version": "1.0"},
        "agents": [a], "steps": [{"agent": "risk_scorer", "input": {"source": "initial"}}],
    })


def _trace(*tool_calls):
    # The audit render format (executor._tool_calls_since): "<tool> (<op>) -> <result>".
    step = SimpleNamespace(tool_calls=tuple(tool_calls))
    return SimpleNamespace(step_records=(step,))


def test_exercised_refs_parses_the_audit_render_format():
    t = _trace("mcp://x (invoke) -> ok", "evidence://retrieve (full) -> ok")
    assert exercised_refs(t) == frozenset({"mcp://x", "evidence://retrieve"})


def test_soundness_passes_when_exercised_subset_of_footprint():
    cfg = _cfg(["mcp://x"])
    assert_footprint_sound(cfg, _trace("mcp://x (invoke) -> ok"))  # no raise


def test_soundness_tripwire_fires_on_out_of_footprint_ref():
    cfg = _cfg(["mcp://x"])  # footprint has mcp://x only
    with pytest.raises(AssertionError) as ei:
        assert_footprint_sound(cfg, _trace("ghost://tool (invoke) -> ok"))
    assert "ghost://tool" in str(ei.value)
