"""Static validation of when= branch conditions on Pipeline.add()."""
import pytest
from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.pipeline import Pipeline, From, When
from drawbore.schema.errors import SchemaCompatibilityError


class Seed(BaseModel):
    value: int

class Scored(BaseModel):
    level: str

class Note(BaseModel):
    note: str


@agent(name="score", input=Seed, output=Scored)
async def score(v: Seed) -> Scored:
    return Scored(level="high" if v.value > 10 else "low")

@agent(name="hi", input=Scored, output=Note)
async def hi(v: Scored) -> Note:
    return Note(note="enhanced")


def test_when_ref_must_be_in_pipeline():
    p = Pipeline(name="t")
    p.add(score)
    with pytest.raises(SchemaCompatibilityError, match="When.* 'missing'"):
        p.add(hi, inputs={"level": From("score.level")},
              when=When("missing.level", equals="high"))


def test_when_rejected_on_when_carrying_node():
    # D63c conservative rule: cannot gate on a node that itself carries a `when`.
    p = Pipeline(name="t")
    p.add(score)
    p.add(hi, inputs={"level": From("score.level")}, when=When("score.level", equals="high"))
    @agent(name="hi2", input=Scored, output=Note)
    async def hi2(v: Scored) -> Note:
        return Note(note="x")
    with pytest.raises(SchemaCompatibilityError, match="carries a 'when'"):
        p.add(hi2, inputs={"level": From("score.level")}, when=When("hi.note", equals="enhanced"))


def test_when_field_not_on_output_model():
    p = Pipeline(name="t")
    p.add(score)
    with pytest.raises(SchemaCompatibilityError, match="does not exist"):
        p.add(hi, inputs={"level": From("score.level")},
              when=When("score.nonexistent", equals="high"))
