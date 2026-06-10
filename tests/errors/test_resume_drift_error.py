"""resume_drift joins the closed halt vocabulary."""

from drawbore.errors import ResumeDriftError
from drawbore.errors.errors import HALT_CODES, halt_reason_for


def test_resume_drift_in_halt_codes():
    assert "resume_drift" in HALT_CODES


def test_resume_drift_error_declares_reason():
    assert halt_reason_for(ResumeDriftError("model changed")) == "resume_drift"
