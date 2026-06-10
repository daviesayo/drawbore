from drawbore.escalation import EscalationPackage, build_escalation


def test_agent_id_defaults_to_none_and_is_omitted_from_legible():
    pkg = build_escalation(
        step="charge", reason="r", received=None, attempted_output=None,
        trace=[], now=lambda: "t",
    )
    assert pkg.agent_id is None
    assert "Agent id" not in pkg.legible()


def test_agent_id_is_shown_in_legible_when_present():
    pkg = build_escalation(
        step="charge", reason="decommissioned", received=None,
        attempted_output=None, trace=[], now=lambda: "t", agent_id="scorer-abc123",
    )
    assert pkg.agent_id == "scorer-abc123"
    assert "scorer-abc123" in pkg.legible()
