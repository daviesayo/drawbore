from drawbore.escalation import (
    EscalationPolicy, EscalationDispatcher, RecordingDispatcher, build_escalation,
)
import pytest


def test_policy_defaults_to_sync():
    p = EscalationPolicy(channel="slack", target="compliance_queue")
    assert p.channel == "slack"
    assert p.target == "compliance_queue"
    assert p.mode == "sync"


def test_dispatcher_is_abstract():
    with pytest.raises(TypeError):
        EscalationDispatcher()


def test_recording_dispatcher_records():
    d = RecordingDispatcher()
    policy = EscalationPolicy(channel="email", target="ops", mode="async")
    pkg = build_escalation(step="x", reason="r", received=None,
                           attempted_output=None, trace=[], now=lambda: "t")
    d.dispatch(pkg, policy)
    assert d.sent == [(pkg, policy)]
