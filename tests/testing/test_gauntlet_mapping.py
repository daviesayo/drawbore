# tests/testing/test_gauntlet_mapping.py
from types import SimpleNamespace

from drawbore.testing.gauntlet.cases import Containment
from drawbore.testing.gauntlet.runner import _verdict


def _audit(*tool_calls_per_step, schema_violations=0):
    steps = [SimpleNamespace(tool_calls=tuple(tc)) for tc in tool_calls_per_step]
    return SimpleNamespace(step_records=tuple(steps), schema_violations=schema_violations)


def _result(status, reason="", audit=None):
    return SimpleNamespace(status=status, reason=reason, audit_trace=audit)


def test_completed_is_not_contained():
    assert _verdict(_result("completed", audit=_audit())) is None


def test_confidence_trigger_under_sync_policy_is_escalated():
    r = _result("escalated", reason="confidence_below_threshold: 0.2 < 0.8", audit=_audit())
    assert _verdict(r) is Containment.ESCALATED


def test_denial_labels_map_regardless_of_status():
    for label, expect in (("denied:breaker", Containment.DENIED_BREAKER),
                          ("denied:scope", Containment.DENIED_SCOPE),
                          ("denied:token", Containment.DENIED_TOKEN)):
        a = _audit([f"t:a (invoke) -> {label}"])
        assert _verdict(_result("halted", reason="circuit_breaker: x", audit=a)) is expect
        assert _verdict(_result("escalated", reason="circuit_breaker: x", audit=a)) is expect


def test_hard_class_wins_over_escalated_status():
    # THE regression guard: a policy wraps every hard halt as status="escalated";
    # the refusal class must still win over the ESCALATED fallback.
    assert _verdict(_result("escalated", reason="schema_violation: output: ...",
                            audit=_audit(schema_violations=1))) is Containment.SCHEMA_REJECT
    assert _verdict(_result("escalated", reason="testing_error: no mock",
                            audit=_audit())) is Containment.FAIL_CLOSED
    assert _verdict(_result("escalated", reason="model_error: non-JSON final",
                            audit=_audit())) is Containment.MODEL_REFUSED


def test_plain_halts_map_by_reason():
    assert _verdict(_result("halted", reason="schema_violation: output: ...",
                            audit=_audit(schema_violations=1))) is Containment.SCHEMA_REJECT
    assert _verdict(_result("halted", reason="testing_error: x", audit=_audit())) is Containment.FAIL_CLOSED
    assert _verdict(_result("halted", reason="model_error: x", audit=_audit())) is Containment.MODEL_REFUSED
    assert _verdict(_result("halted", reason="agent_error: boom", audit=_audit())) is Containment.HALTED


def test_denial_precedence_over_schema_is_synthetic_but_guarded():
    # SYNTHETIC: a denied tool aborts the loop immediately, so a denial label and a
    # schema violation cannot co-occur in a real run. The stub is defensive only.
    a = _audit(["t:a (invoke) -> denied:scope"], schema_violations=1)
    assert _verdict(_result("halted", reason="tool_access: x", audit=a)) is Containment.DENIED_SCOPE
