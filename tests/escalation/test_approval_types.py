import pytest
from pydantic import ValidationError

from drawbore.escalation import ApprovalDecision, ApprovalRequest


def test_request_is_json_safe():
    req = ApprovalRequest(
        request_id="abc123", run_id="r-1", step="risk_scorer",
        question="Approve the proposed risk score?",
        reason="requires_human_approval",
        package_legible="Run stopped at step 'risk_scorer'.",
        proposed_output={"risk_level": "low", "confidence": 0.95, "factors": []},
        proposed_output_trust="trusted",
    )
    dumped = req.model_dump(mode="json")
    assert dumped["proposed_output"]["confidence"] == 0.95
    assert ApprovalRequest.model_validate(dumped) == req


def test_request_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        ApprovalRequest(
            request_id="x", run_id="r", step="s", question="q", reason="r",
            package_legible="p", proposed_output={}, proposed_output_trust="trusted",
            extra_field=1,
        )


def test_decision_blank_reviewer_fails_closed():
    with pytest.raises(ValidationError, match="reviewer"):
        ApprovalDecision(request_id="abc", verdict="approved", reviewer_id="   ")


def test_decision_amended_requires_amended_output():
    with pytest.raises(ValidationError, match="amended_output"):
        ApprovalDecision(request_id="abc", verdict="amended", reviewer_id="rev-1")


def test_decision_non_amended_forbids_amended_output():
    with pytest.raises(ValidationError, match="amended_output"):
        ApprovalDecision(
            request_id="abc", verdict="approved", reviewer_id="rev-1",
            amended_output={"x": 1},
        )


def test_decision_happy_paths():
    a = ApprovalDecision(request_id="abc", verdict="approved", reviewer_id="rev-1")
    assert a.amended_output is None and a.rationale is None
    m = ApprovalDecision(
        request_id="abc", verdict="amended", reviewer_id="rev-1",
        amended_output={"x": 1}, rationale="fixed the level",
    )
    assert m.amended_output == {"x": 1}
    r = ApprovalDecision(
        request_id="abc", verdict="rejected", reviewer_id="rev-1",
        rationale="not today",
    )
    assert r.verdict == "rejected"
