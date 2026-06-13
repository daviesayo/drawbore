# tests/config/test_authority.py
from drawbore.config import PipelineConfig
from drawbore.config.authority import CapabilityFact, effective_authority


def _manifest(*, risk_tools, evidence_full, evidence_enabled=True, risk_model="risk-1.0"):
    """A minimal 2-step manifest: an initial agent, then a model agent 'risk_scorer'
    with declared tools and an optional evidence policy."""
    def agent(name, tools, model):
        return {
            "name": name, "ref": f"x.{name}", "version": "1.0", "risk_tier": "low",
            "requires_human_approval": False, "context_access": "none",
            "tools": tools, "model": model, "fallback_model": None, "instructions": None,
            "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
            "input_schema_hash": "sha256:0", "output_schema_hash": "sha256:0",
        }
    steps = [
        {"agent": "loader", "input": {"source": "initial"}, "depends_on": []},
        {
            "agent": "risk_scorer",
            "input": {"source": "previous", "agent": "loader"},
            "depends_on": ["loader"],
            "evidence": {
                "enabled": evidence_enabled,
                "allow_search_retrieval": True,
                "allow_full_retrieval": evidence_full,
            },
        },
    ]
    return PipelineConfig.model_validate({
        "schema_version": 1,
        "pipeline": {"name": "p", "version": "1.0"},
        "agents": [agent("loader", [], None), agent("risk_scorer", risk_tools, risk_model)],
        "steps": steps,
    })


def test_tool_and_evidence_facts_derived():
    cfg = _manifest(risk_tools=["mcp://identity/verify"], evidence_full=False)
    facts = effective_authority(cfg).facts
    assert CapabilityFact("risk_scorer", "tool", "mcp://identity/verify", "*") in facts
    assert CapabilityFact("risk_scorer", "evidence", "evidence://retrieve", "search") in facts
    assert CapabilityFact("risk_scorer", "evidence", "evidence://retrieve", "full") not in facts


def test_full_retrieval_adds_full_fact():
    facts = effective_authority(_manifest(risk_tools=[], evidence_full=True)).facts
    assert CapabilityFact("risk_scorer", "evidence", "evidence://retrieve", "full") in facts


def test_disabled_or_modelless_evidence_yields_no_evidence_facts():
    # enabled=False → no evidence facts
    off = effective_authority(_manifest(risk_tools=[], evidence_full=True, evidence_enabled=False)).facts
    assert not any(f.kind == "evidence" for f in off)
    # model is None → the retrieval builtin never binds (mirrors executor.py:90)
    no_model = effective_authority(_manifest(risk_tools=[], evidence_full=True, risk_model=None)).facts
    assert not any(f.kind == "evidence" for f in no_model)


# ---------------------------------------------------------------------------
# Task 2: footprint fingerprint
# ---------------------------------------------------------------------------
from drawbore.config.authority import effective_authority as _ea  # noqa: E402


def test_footprint_fingerprint_is_stable():
    cfg = _manifest(risk_tools=["b://t", "a://t"], evidence_full=True)
    fp1 = _ea(cfg).fingerprint()
    fp2 = _ea(cfg).fingerprint()
    assert fp1 == fp2
    assert fp1.startswith("sha256:")


def test_footprint_fingerprint_is_order_independent():
    from drawbore.config.fingerprint import footprint_fingerprint
    facts_ab = [("a", "tool", "ref1", "*"), ("b", "tool", "ref2", "*")]
    facts_ba = [("b", "tool", "ref2", "*"), ("a", "tool", "ref1", "*")]
    assert footprint_fingerprint(facts_ab) == footprint_fingerprint(facts_ba)


def test_footprint_fingerprint_changes_when_authority_changes():
    narrow = _ea(_manifest(risk_tools=[], evidence_full=False)).fingerprint()
    wide = _ea(_manifest(risk_tools=[], evidence_full=True)).fingerprint()
    assert narrow != wide


# ---------------------------------------------------------------------------
# Task 3: authority diff + certificate
# ---------------------------------------------------------------------------
import json as _json  # noqa: E402
from drawbore.config.authority import authority_diff, CapabilityFact as F  # noqa: E402


def test_diff_flags_added_authority_and_passes_narrowing():
    old = _manifest(risk_tools=["mcp://x"], evidence_full=False)
    wider = _manifest(risk_tools=["mcp://x"], evidence_full=True)
    d = authority_diff(old, wider)
    assert d.ok is False
    assert d.added == frozenset({F("risk_scorer", "evidence", "evidence://retrieve", "full")})
    assert d.removed == frozenset()

    # narrowing: drop the tool -> ok, added empty, removed names the tool fact
    narrower = _manifest(risk_tools=[], evidence_full=False)
    n = authority_diff(old, narrower)
    assert n.ok is True
    assert n.added == frozenset()
    assert F("risk_scorer", "tool", "mcp://x", "*") in n.removed

    # identity diff is empty/ok
    same = authority_diff(old, old)
    assert same.ok is True and not same.added and not same.removed


def test_certificate_is_legible_and_carries_fingerprints():
    d = authority_diff(_manifest(risk_tools=[], evidence_full=False),
                       _manifest(risk_tools=[], evidence_full=True))
    cert = d.certificate()
    assert "FAILED" in cert
    assert "evidence://retrieve" in cert and "full" in cert
    assert d.old_fingerprint in cert and d.new_fingerprint in cert
    # last line is canonical JSON and round-trips
    payload = _json.loads(cert.splitlines()[-1])
    assert payload["ok"] is False
    assert ["risk_scorer", "evidence", "evidence://retrieve", "full"] in payload["added"]


# ---------------------------------------------------------------------------
# Task 4: CI gate — check_no_new_authority + AuthorityRegressionError
# ---------------------------------------------------------------------------
import pytest  # noqa: E402
from drawbore.config.authority import check_no_new_authority  # noqa: E402
from drawbore.config.errors import AuthorityRegressionError  # noqa: E402


def test_check_raises_on_added_authority_and_is_silent_on_narrowing():
    old = _manifest(risk_tools=["mcp://x"], evidence_full=False)
    wider = _manifest(risk_tools=["mcp://x"], evidence_full=True)
    with pytest.raises(AuthorityRegressionError) as ei:
        check_no_new_authority(old, wider)
    assert "FAILED" in str(ei.value)
    assert ei.value.diff.added  # the diff is attached for the caller to print

    # narrowing returns None (no raise)
    assert check_no_new_authority(old, _manifest(risk_tools=[], evidence_full=False)) is None


# ---------------------------------------------------------------------------
# Task 5: Public exports + config import-boundary check
# ---------------------------------------------------------------------------
def test_public_surface_is_exported_from_drawbore_config():
    from drawbore.config import (
        CapabilityFact as PF, CapabilityFootprint, AuthorityDiff,
        effective_authority as ea, authority_diff as ad, check_no_new_authority as cn,
        AuthorityRegressionError as ARE,
    )
    assert all(x is not None for x in (PF, CapabilityFootprint, AuthorityDiff, ea, ad, cn, ARE))


def test_authority_module_imports_no_runtime_layers():
    # config boundary: config must not pull in tools/orchestration/llm/audit/observability.
    import drawbore.config.authority as mod
    import inspect
    src = inspect.getsource(mod)
    for forbidden in ("drawbore.tools", "drawbore.orchestration", "drawbore.llm",
                      "drawbore.audit", "drawbore.observability", "drawbore.evidence"):
        assert forbidden not in src, f"authority.py must not import {forbidden}"


def test_evidence_retrieve_ref_matches_retrieval_module():
    """EVIDENCE_RETRIEVE_REF in config.authority must equal EVIDENCE_TOOL_REF in
    evidence.retrieval — the config layer duplicates it as a literal to preserve the
    import boundary; this test catches any divergence."""
    from drawbore.config.authority import EVIDENCE_RETRIEVE_REF
    from drawbore.evidence.retrieval import EVIDENCE_TOOL_REF
    assert EVIDENCE_RETRIEVE_REF == EVIDENCE_TOOL_REF
