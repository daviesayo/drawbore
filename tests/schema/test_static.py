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


def test_literal_output_assignable_to_base_str_input():
    from typing import Literal

    # A Literal['USD','EUR','GBP'] value is always a valid str: safe narrowing.
    assert is_assignable(Literal["USD", "EUR", "GBP"], str) is True


def test_literal_int_output_assignable_to_int_input():
    from typing import Literal

    assert is_assignable(Literal[1, 2, 3], int) is True


def test_literal_output_assignable_to_optional_base_input():
    from typing import Literal, Optional

    assert is_assignable(Literal["USD", "EUR"], Optional[str]) is True


def test_str_output_not_assignable_to_literal_input():
    from typing import Literal

    # A plain str is not guaranteed to be one of the literals: must stay refused.
    assert is_assignable(str, Literal["USD", "EUR", "GBP"]) is False


def test_literal_with_value_not_in_base_type_refused():
    from typing import Literal

    # 1 is not a str, so the Literal does not uniformly satisfy a str input.
    assert is_assignable(Literal["USD", 1], str) is False


def test_literal_subset_assignable_to_literal_superset():
    from typing import Literal

    assert is_assignable(Literal["USD"], Literal["USD", "EUR", "GBP"]) is True


def test_literal_superset_not_assignable_to_literal_subset():
    from typing import Literal

    assert is_assignable(Literal["USD", "EUR"], Literal["USD"]) is False


def test_check_compatibility_raises_on_str_into_literal():
    from typing import Literal

    with pytest.raises(SchemaCompatibilityError):
        check_compatibility(str, Literal["USD", "EUR"])


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
