from drawbore.escalation import EscalationPackage, build_escalation


def test_build_escalation_stamps_fields_and_trace():
    pkg = build_escalation(
        step="charge",
        reason="circuit_breaker: too many calls",
        received={"amount": 100},
        attempted_output=None,
        trace=["retrieve", "score"],
        now=lambda: "2026-06-03T00:00:00Z",
    )
    assert isinstance(pkg, EscalationPackage)
    assert pkg.step == "charge"
    assert pkg.reason.startswith("circuit_breaker")
    assert pkg.received == {"amount": 100}
    assert pkg.attempted_output is None
    assert pkg.trace == ("retrieve", "score")
    assert pkg.timestamp == "2026-06-03T00:00:00Z"


def test_legible_is_human_readable():
    pkg = build_escalation(
        step="charge",
        reason="requires_human_approval",
        received={"amount": 100},
        attempted_output={"status": "ready"},
        trace=["retrieve"],
        now=lambda: "2026-06-03T00:00:00Z",
    )
    text = pkg.legible()
    assert "charge" in text
    assert "requires_human_approval" in text
    assert "retrieve" in text
    assert "ready" in text  # the attempted output is shown


def test_legible_omits_attempted_output_when_none():
    pkg = build_escalation(
        step="x", reason="schema_violation",
        received={"amount": 5}, attempted_output=None,
        trace=["a", "b"], now=lambda: "t",
    )
    text = pkg.legible()
    assert "schema_violation" in text
    assert "tried to produce" not in text   # omitted when attempted_output is None
    assert "a, b" in text                    # trace rendered


def test_legible_shows_none_for_empty_trace():
    pkg = build_escalation(
        step="ingest", reason="sanitization",
        received={"x": 1}, attempted_output=None,
        trace=[], now=lambda: "t",
    )
    text = pkg.legible()
    assert "(none)" in text                  # empty trace renders as (none)
