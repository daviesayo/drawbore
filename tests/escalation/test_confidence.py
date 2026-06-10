from pydantic import BaseModel
from drawbore.escalation import HasConfidence


def test_marked_model_is_instance_of_marker():
    class RiskScore(BaseModel, HasConfidence):
        score: int
        confidence: float

    rs = RiskScore(score=10, confidence=0.4)
    assert isinstance(rs, HasConfidence)
    assert rs.confidence == 0.4


def test_incidental_confidence_field_is_not_marked():
    # A model with a `confidence` field that does NOT inherit the marker is
    # NOT treated as confidence-bearing.
    class NotMarked(BaseModel):
        confidence: float

    nm = NotMarked(confidence=0.1)
    assert not isinstance(nm, HasConfidence)
