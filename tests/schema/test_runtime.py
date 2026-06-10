import pytest
from pydantic import BaseModel
from drawbore.schema import validate, SchemaValidationError


class Money(BaseModel):
    amount: float
    currency: str


def test_validate_accepts_conforming_dict():
    result = validate(Money, {"amount": 10.0, "currency": "USD"})
    assert isinstance(result, Money)
    assert result.amount == 10.0


def test_validate_accepts_model_instance():
    m = Money(amount=1.0, currency="GBP")
    assert validate(Money, m) == m


def test_validate_rejects_missing_field():
    with pytest.raises(SchemaValidationError) as exc:
        validate(Money, {"amount": 10.0})
    assert exc.value.model is Money
    assert exc.value.errors  # non-empty list of pydantic errors


def test_validate_is_strict_no_coercion():
    # strict mode: a string is not a valid float
    with pytest.raises(SchemaValidationError):
        validate(Money, {"amount": "10.0", "currency": "USD"})
