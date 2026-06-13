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
from drawbore.config.fingerprint import footprint_fingerprint as _fp_fn


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
    SAME way the minter does, then fingerprint over the exact positional payload.
    footprint_fp is derived from the declared facts so that verify's consistency
    check (footprint_fingerprint must match declared_facts) always holds here."""
    calls = tuple(calls)
    facts = tuple(facts)
    step_agents = tuple(step_agents)
    verdict = _evaluate(calls, facts)
    footprint_fp = _fp_fn(facts)
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


# ---------------------------------------------------------------------------
# FIX 1a — footprint_fingerprint must match declared_facts (unconditional check)
# ---------------------------------------------------------------------------


def test_inconsistent_footprint_fingerprint_is_unverifiable():
    """A receipt whose footprint_fingerprint does not match its declared_facts is
    caught unconditionally by verify — no proxy log or expected fp required.
    This closes the gap where a forger changes declared_facts but leaves the original
    footprint_fingerprint, making an out-of-footprint call look declared."""
    from drawbore.config.fingerprint import footprint_fingerprint as fp_fn

    facts = (("A", "tool", "real_tool", "*"),)
    # Fingerprint of DIFFERENT facts — inconsistent with the declared_facts above.
    wrong_fp = fp_fn((("A", "tool", "other_tool", "*"),))

    calls = (_call(ref="real_tool"),)
    step_agents = ("A",)
    verdict = _evaluate(calls, facts)
    # Build a self-consistent receipt (integrity check passes) but with wrong_fp.
    receipt_fp = _fingerprint("r", "completed", wrong_fp, facts, step_agents, calls, verdict)
    receipt = ConfinementReceipt(
        run_id="r",
        status="completed",
        footprint_fingerprint=wrong_fp,
        declared_facts=facts,
        step_agents=step_agents,
        observed_calls=calls,
        verdict=verdict,
        receipt_fingerprint=receipt_fp,
    )

    v = verify(receipt)
    assert v.status == "unverifiable"
    assert any("footprint fingerprint" in u for u in v.unverifiable)


# ---------------------------------------------------------------------------
# FIX 1b — expected_footprint_fingerprint binds declared footprint to manifest
# ---------------------------------------------------------------------------


def test_inflated_declared_facts_caught_by_expected_footprint():
    """A fully self-consistent forged receipt with inflated declared_facts passes
    verify alone (the documented limitation of keyless SHA-256), but is caught by
    verify(..., expected_footprint_fingerprint=<true manifest fp>).
    This is the adversary who recomputes ALL fingerprints over a self-consistent
    forged receipt — the only defence is binding to the manifest the auditor holds."""
    from drawbore.config.fingerprint import footprint_fingerprint as fp_fn

    original_facts = (("A", "tool", "real_tool", "*"),)
    original_fp = fp_fn(original_facts)

    # Forger inflates declared_facts to cover a tool that was out of footprint.
    forged_facts = (("A", "tool", "extra_tool", "*"), ("A", "tool", "real_tool", "*"))
    forged_fp = fp_fn(forged_facts)  # consistent with forged_facts

    calls = (_call(ref="extra_tool"),)
    step_agents = ("A",)
    verdict = _evaluate(calls, forged_facts)
    assert verdict.status == "confined"  # the forgery makes the call look declared

    # Forger recomputes receipt_fingerprint over all forged data (keyless — possible).
    receipt_fp = _fingerprint(
        "r", "completed", forged_fp, forged_facts, step_agents, calls, verdict
    )
    forged_receipt = ConfinementReceipt(
        run_id="r",
        status="completed",
        footprint_fingerprint=forged_fp,
        declared_facts=forged_facts,
        step_agents=step_agents,
        observed_calls=calls,
        verdict=verdict,
        receipt_fingerprint=receipt_fp,
    )

    # verify alone: forged_fp IS consistent with forged_facts, so FIX 1a passes too.
    # This is the documented limitation of keyless fingerprinting.
    v_alone = verify(forged_receipt)
    assert v_alone.status == "confined", (
        "verify alone cannot detect a fully self-consistent keyless forgery"
    )

    # Binding to the true manifest footprint catches the inflation.
    v_bound = verify(forged_receipt, expected_footprint_fingerprint=original_fp)
    assert v_bound.status == "unverifiable"
    assert any("expected" in u.lower() for u in v_bound.unverifiable)


# ---------------------------------------------------------------------------
# FIX 3 — log-binding must reject a log from a different run_id
# ---------------------------------------------------------------------------


def test_log_binding_rejects_different_run_id():
    """When a proxy log is supplied, each entry's run_id must match the receipt's
    run_id. A log from a different run with otherwise byte-identical calls must
    not pass as binding evidence for this run."""
    facts = (("A", "tool", "t", "*"),)
    r = _receipt([_call(ref="t")], facts)
    wrong_run_log = [
        {
            "tool": "t",
            "run_id": "completely-different-run",
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
    v = verify(r, wrong_run_log)
    assert v.status == "unverifiable"
    assert any("run" in u.lower() for u in v.unverifiable)
