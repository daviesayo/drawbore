import pytest
from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.pipeline import Pipeline, From
from drawbore.schema import SchemaCompatibilityError


class Seed(BaseModel):
    value: int


class ADoubled(BaseModel):
    doubled: int


class CIn(BaseModel):
    doubled: int
    label: str


@agent(input=Seed, output=ADoubled)
async def a(value: Seed) -> ADoubled:
    return ADoubled(doubled=value.value * 2)


def test_add_first_agent_and_linear_default():
    p = Pipeline(name="t")
    p.add(a)
    assert [s.agent.name for s in p.steps] == ["a"]


def test_add_with_valid_binding():
    @agent(name="c", input=CIn, output=CIn)
    async def c(value: CIn) -> CIn:
        return value

    @agent(name="b", input=ADoubled, output=BLabel)
    async def b(value: ADoubled) -> "BLabel":
        return BLabel(label="x")

    p = Pipeline(name="t")
    p.add(a)
    p.add(b)
    p.add(c, inputs={"doubled": From("a.doubled"), "label": From("b.label")})
    assert [s.agent.name for s in p.steps] == ["a", "b", "c"]


def test_binding_to_unknown_source_agent_raises():
    @agent(name="c", input=CIn, output=CIn)
    async def c(value: CIn) -> CIn:
        return value

    p = Pipeline(name="t")
    p.add(a)
    with pytest.raises(SchemaCompatibilityError):
        p.add(c, inputs={"doubled": From("a.doubled"), "label": From("ghost.label")})


def test_binding_type_mismatch_raises():
    @agent(name="needs_str", input=NeedsStr, output=NeedsStr)
    async def needs_str(value: "NeedsStr") -> "NeedsStr":
        return value

    p = Pipeline(name="t")
    p.add(a)
    # a.doubled is int; target field wants str -> incompatible
    with pytest.raises(SchemaCompatibilityError):
        p.add(needs_str, inputs={"s": From("a.doubled")})


def test_unbound_required_field_raises():
    @agent(name="c", input=CIn, output=CIn)
    async def c(value: CIn) -> CIn:
        return value

    p = Pipeline(name="t")
    p.add(a)
    with pytest.raises(SchemaCompatibilityError):
        p.add(c, inputs={"doubled": From("a.doubled")})  # 'label' unbound


class BLabel(BaseModel):
    label: str


class NeedsStr(BaseModel):
    s: str


def test_binding_to_unknown_source_field_raises():
    class OneField(BaseModel):
        doubled: int

    @agent(name="of", input=OneField, output=OneField)
    async def of(value: OneField) -> OneField:
        return value

    p = Pipeline(name="t")
    p.add(a)
    # 'a' is in the pipeline but its output (ADoubled) has no field 'ghostfield'
    with pytest.raises(SchemaCompatibilityError):
        p.add(of, inputs={"doubled": From("a.ghostfield")})
