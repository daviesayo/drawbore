from typing import Optional

import pytest
from drawbore.schema import is_assignable, check_compatibility, SchemaCompatibilityError


class Animal:
    pass


class Dog(Animal):
    pass


def test_identical_types_assignable():
    assert is_assignable(int, int) is True


def test_subclass_assignable_to_base():
    assert is_assignable(Dog, Animal) is True
    assert is_assignable(Animal, Dog) is False


def test_non_optional_satisfies_optional_target():
    assert is_assignable(int, Optional[int]) is True


def test_optional_source_does_not_satisfy_non_optional_target():
    assert is_assignable(Optional[int], int) is False


def test_any_target_accepts_anything():
    from typing import Any
    assert is_assignable(str, Any) is True


def test_check_compatibility_raises_on_mismatch():
    with pytest.raises(SchemaCompatibilityError):
        check_compatibility(str, int)


def test_check_compatibility_returns_true_on_match():
    assert check_compatibility(int, int) is True


def test_structural_model_width_subtyping():
    from pydantic import BaseModel

    class Wide(BaseModel):
        a: int
        b: str

    class Narrow(BaseModel):
        a: int

    # Wide supplies every required Narrow field with an assignable type (extra 'b' ignored)
    assert is_assignable(Wide, Narrow) is True
    # Narrow lacks 'b' which is required by Wide
    assert is_assignable(Narrow, Wide) is False
