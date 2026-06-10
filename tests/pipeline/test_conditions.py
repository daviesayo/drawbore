# tests/pipeline/test_conditions.py
import pytest
from pydantic import BaseModel
from drawbore.pipeline.conditions import When


class _Out(BaseModel):
    level: str
    amount: int
    flag: bool


def _outputs():
    return {"risk": _Out(level="high", amount=12000, flag=True)}


def test_exactly_one_operator_required():
    with pytest.raises(ValueError):
        When("risk.level")                       # no operator
    with pytest.raises(ValueError):
        When("risk.level", equals="high", gt=1)  # two operators


def test_ref_must_name_a_field():
    with pytest.raises(ValueError):
        When("risk", equals="high")              # whole-object gating not allowed


def test_equals_in_is_true_gt_lt():
    assert When("risk.level", equals="high").evaluate(_outputs()) is True
    assert When("risk.level", equals="low").evaluate(_outputs()) is False
    assert When("risk.level", in_=("low", "high")).evaluate(_outputs()) is True
    assert When("risk.flag", is_true=True).evaluate(_outputs()) is True
    assert When("risk.amount", gt=10000).evaluate(_outputs()) is True
    assert When("risk.amount", lt=10000).evaluate(_outputs()) is False


def test_legible():
    assert When("risk.level", equals="high").legible() == "risk.level == 'high'"
    assert When("risk.amount", gt=10000).legible() == "risk.amount > 10000"
    assert When("risk.level", in_=("low", "high")).legible() == "risk.level in ['low', 'high']"
    assert When("risk.flag", is_true=True).legible() == "risk.flag is true"
    assert When("risk.amount", lt=10000).legible() == "risk.amount < 10000"
    assert When("risk.flag", is_true=False).legible() == "risk.flag is false"
