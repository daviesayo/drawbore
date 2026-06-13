"""Unit tests for the confinement receipt models + offline verifier.

These build receipts as plain data (no pipeline/proxy) and exercise the two
confinement invariants (footprint containment + taint no-exfil), the fail-closed
UNVERIFIABLE rule, tamper-evidence (fingerprint), log-binding, and determinism.
"""

from __future__ import annotations

from drawbore.confinement import (
    ConfinementReceipt,
    ConfinementVerdict,
    ObservedCall,
    verify,
)
from drawbore.confinement.receipt import (
    EVIDENCE_REF,
    _evaluate,
    _fingerprint,
)


def _call(
    step=0,
    agent="A",
    ref="t",
    op="invoke",
    mk="tool",
    result="ok",
    taint="trusted",
    exfil=False,
    ih="h",
    oh="h2",
) -> ObservedCall:
    return ObservedCall(
        step=step,
        agent=agent,
        tool_ref=ref,
        operation=op,
        match_kind=mk,
        result=result,
        taint_scope=taint,
        exfil_capable=exfil,
        input_hash=ih,
        output_hash=oh,
    )


def _receipt(
    calls,
    facts,
    step_agents=("A",),
    run_status="completed",
    run_id="r",
) -> ConfinementReceipt:
    """Build a well-formed receipt: derive the verdict from the calls+facts the
    SAME way the minter does, then fingerprint over the exact positional payload."""
    calls = tuple(calls)
    facts = tuple(facts)
    step_agents = tuple(step_agents)
    verdict = _evaluate(calls, facts)
    # footprint_fingerprint is embedded as data; verify hashes it inside the
    # receipt payload but does not re-derive it from facts. Any stable string works.
    footprint_fp = "sha256:footprint"
    receipt_fp = _fingerprint(
        run_id, run_status, footprint_fp, facts, step_agents, calls, verdict
    )
    return ConfinementReceipt(
        run_id=run_id,
        status=run_status,
        footprint_fingerprint=footprint_fp,
        declared_facts=facts,
        step_agents=step_agents,
        observed_calls=calls,
        verdict=verdict,
        receipt_fingerprint=receipt_fp,
    )


def test_evidence_ref_is_reused_constant():
    assert EVIDENCE_REF == "evidence://retrieve"


def test_confined_receipt_verifies_confined():
    facts = (("A", "tool", "t", "*"),)
    r = _receipt([_call(ref="t")], facts)
    v = verify(r)
    assert v.status == "confined" and v.confined is True


def test_out_of_footprint_executed_call_breaches():
    facts = (("A", "tool", "declared", "*"),)
    r = _receipt([_call(ref="undeclared")], facts)
    v = verify(r)
    assert v.status == "breached" and v.confined is False
    assert any("undeclared" in b for b in v.breaches)


def test_exfil_under_untrusted_breaches():
    facts = (("A", "tool", "sink", "*"),)
    r = _receipt([_call(ref="sink", exfil=True, taint="untrusted")], facts)
    v = verify(r)
    assert v.status == "breached"
    assert any("untrusted" in b for b in v.breaches)


def test_denied_call_is_not_a_breach():
    facts = (("A", "tool", "sink", "*"),)
    r = _receipt(
        [_call(ref="sink", exfil=True, taint="untrusted", result="denied:taint")],
        facts,
    )
    # A denial is evidence a control fired, never a breach.
    assert verify(r).status == "confined"


def test_error_result_is_not_a_breach():
    facts = (("A", "tool", "sink", "*"),)
    r = _receipt(
        [_call(ref="undeclared", result="error")],
        facts,
    )
    assert verify(r).status == "confined"


def test_replay_result_is_an_executed_candidate():
    facts = (("A", "tool", "declared", "*"),)
    r = _receipt([_call(ref="undeclared", result="replay")], facts)
    assert verify(r).status == "breached"


def test_unattributable_call_is_unverifiable():
    facts = (("A", "tool", "t", "*"),)
    r = _receipt([_call(agent=None, ref="t")], facts)
    assert verify(r).status == "unverifiable"


def test_unverifiable_takes_precedence_over_breach():
    # One unattributable executed call + one out-of-footprint call: fail closed.
    facts = (("A", "tool", "declared", "*"),)
    r = _receipt(
        [_call(agent=None, ref="t"), _call(agent="A", ref="undeclared")],
        facts,
    )
    assert verify(r).status == "unverifiable"


def test_tampered_field_fails_integrity():
    facts = (("A", "tool", "t", "*"),)
    r = _receipt([_call(ref="t")], facts)
    bad = r.model_copy(update={"observed_calls": (_call(ref="evil"),)})
    v = verify(bad)
    assert v.status == "unverifiable"
    assert any("tamper" in u.lower() for u in v.unverifiable)


def test_tampered_verdict_fails_integrity():
    facts = (("A", "tool", "declared", "*"),)
    # honest receipt is BREACHED (call to an undeclared tool); forge a "confined"
    # verdict. The fingerprint was taken over the real breached verdict, so the
    # swap is caught at the integrity step.
    r = _receipt([_call(ref="undeclared")], facts)
    assert r.verdict.status == "breached"
    forged = ConfinementVerdict(
        status="confined", confined=True, breaches=(), unverifiable=()
    )
    bad = r.model_copy(update={"verdict": forged})
    assert verify(bad).status == "unverifiable"


def test_log_binding_passes_when_log_matches():
    facts = (("A", "tool", "t", "*"),)
    r = _receipt([_call(ref="t")], facts)
    matching_log = [
        {
            "tool": "t",
            "run_id": "r",
            "operation": "invoke",
            "input_hash": "h",
            "output_hash": "h2",
            "result": "ok",
            "step": 0,
            "scope": "trusted",
            "kind": "custom",
            "exfil_capable": False,
        }
    ]
    assert verify(r, matching_log).status == "confined"


def test_log_binding_detects_swapped_log():
    facts = (("A", "tool", "t", "*"),)
    r = _receipt([_call(ref="t")], facts)
    altered_log = [
        {
            "tool": "evil",
            "run_id": "r",
            "operation": "invoke",
            "input_hash": "h",
            "output_hash": "h2",
            "result": "ok",
            "step": 0,
            "scope": "trusted",
            "kind": "custom",
            "exfil_capable": False,
        }
    ]
    assert verify(r, altered_log).status == "unverifiable"


def test_determinism():
    facts = (("A", "tool", "t", "*"),)
    a = _receipt([_call(ref="t")], facts)
    b = _receipt([_call(ref="t")], facts)
    assert a.receipt_fingerprint == b.receipt_fingerprint
    assert a.receipt_fingerprint.startswith("sha256:")


def test_verdict_validator_rejects_inconsistent_status():
    import pytest

    with pytest.raises(ValueError):
        ConfinementVerdict(
            status="confined", confined=False, breaches=(), unverifiable=()
        )


def test_legible_is_readable_string():
    facts = (("A", "tool", "t", "*"),)
    r = _receipt([_call(ref="t")], facts)
    text = r.legible()
    assert isinstance(text, str)
    assert "CONFINED" in text
    assert r.run_id in text
    assert "t" in text
