from pydantic import BaseModel, create_model
from drawbore.versioning import classify_change


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def test_identical_schemas_are_non_breaking():
    assert classify_change(In, Out, In, Out) == "non_breaking"


def test_adding_an_optional_field_is_non_breaking():
    class Out2(BaseModel):
        y: int
        note: str | None = None  # optional

    assert classify_change(In, Out, In, Out2) == "non_breaking"


def test_adding_a_required_field_is_breaking():
    class Out2(BaseModel):
        y: int
        note: str  # required

    assert classify_change(In, Out, In, Out2) == "breaking"


def test_removing_a_field_is_breaking():
    class Out2(BaseModel):
        pass  # 'y' removed

    assert classify_change(In, Out, In, Out2) == "breaking"


def test_changing_a_field_type_is_breaking():
    class Out2(BaseModel):
        y: str  # was int

    assert classify_change(In, Out, In, Out2) == "breaking"


def test_making_an_optional_field_required_is_breaking():
    class A(BaseModel):
        y: int = 0  # optional

    class B(BaseModel):
        y: int  # now required

    assert classify_change(In, A, In, B) == "breaking"


def test_a_breaking_input_change_dominates_a_clean_output():
    class In2(BaseModel):
        x: int
        extra: str  # added required input field → breaking

    assert classify_change(In, Out, In2, Out) == "breaking"


def test_nested_model_breaking_change_is_breaking():
    # A required field added to a NESTED input model is breaking, but str(annotation)
    # keys on the nested class name "Address" (unchanged) and would miss it.
    Addr1 = create_model("Address", street=(str, ...))
    CustIn1 = create_model("CustomerIn", address=(Addr1, ...))
    Addr2 = create_model("Address", street=(str, ...), city=(str, ...))  # added required
    CustIn2 = create_model("CustomerIn", address=(Addr2, ...))
    assert classify_change(CustIn1, Out, CustIn2, Out) == "breaking"


def test_adding_an_optional_input_field_is_non_breaking():
    class In2(BaseModel):
        x: int
        note: str | None = None  # optional input add → non-breaking (symmetry with output)

    assert classify_change(In, Out, In2, Out) == "non_breaking"
