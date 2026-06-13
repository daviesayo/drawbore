"""approval_rejected and approval_error join the closed halt vocabulary."""

from drawbore.errors import HALT_CODES


def test_approval_codes_are_in_the_vocabulary():
    assert "approval_rejected" in HALT_CODES
    assert "approval_error" in HALT_CODES
