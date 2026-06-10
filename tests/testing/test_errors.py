from drawbore.testing import TestingError
from drawbore.errors import DrawboreError, halt_reason_for


def test_testing_error_is_a_drawbore_error():
    assert issubclass(TestingError, DrawboreError)


def test_testing_error_has_a_legible_halt_reason():
    # Self-declared halt_reason; the pipeline classifies a test-mode failure
    # legibly without drawbore.errors importing drawbore.testing (a cycle).
    assert halt_reason_for(TestingError("x")) == "testing_error"


def test_testing_error_carries_its_message():
    assert "no mock provided" in str(TestingError("no mock provided for tool 'db.read'"))
