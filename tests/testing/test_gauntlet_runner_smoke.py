# tests/testing/test_gauntlet_runner_smoke.py
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.testing.gauntlet import Containment, schema_violation, run_containment, assert_contained


class GIn(BaseModel):
    id: str


class Out(BaseModel):
    label: str


@agent(name="scorer", input=GIn, output=Out, model="m-1.0")
async def scorer(v: GIn) -> Out: ...   # one-shot model agent; body unused


async def test_schema_violation_is_contained_end_to_end():
    pipeline = Pipeline("smoke").add(scorer)
    case = schema_violation("scorer", {"wrong": "field"})   # missing required 'label'
    observed = await run_containment(pipeline, case, initial=GIn(id="x"))
    assert observed is Containment.SCHEMA_REJECT
    await assert_contained(pipeline, case, initial=GIn(id="x"))   # no raise


async def test_not_contained_returns_none_and_assert_raises():
    pipeline = Pipeline("smoke").add(scorer)
    # a schema-VALID output completes -> not contained
    valid = schema_violation("scorer", {"label": "ok"})
    assert await run_containment(pipeline, valid, initial=GIn(id="x")) is None
    import pytest
    with pytest.raises(AssertionError):
        await assert_contained(pipeline, valid, initial=GIn(id="x"))
