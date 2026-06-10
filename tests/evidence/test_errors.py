from drawbore.errors import DrawboreError, halt_reason_for
from drawbore.evidence import (
    EvidenceError, EvidenceStoreError, EvidenceRetrievalError, EvidenceTransformError,
)


def test_evidence_errors_are_drawbore_errors():
    assert issubclass(EvidenceError, DrawboreError)
    assert issubclass(EvidenceStoreError, EvidenceError)
    assert issubclass(EvidenceRetrievalError, EvidenceError)
    assert issubclass(EvidenceTransformError, EvidenceError)


def test_evidence_error_has_legible_halt_reason():
    assert halt_reason_for(EvidenceError("x")) == "evidence_error"
    assert halt_reason_for(EvidenceStoreError("x")) == "evidence_error"
    assert halt_reason_for(EvidenceRetrievalError("x")) == "evidence_error"
    assert halt_reason_for(EvidenceTransformError("x")) == "evidence_error"
