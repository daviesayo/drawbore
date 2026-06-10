import pytest
from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.pipeline import Pipeline, From, When, Join


class Seed(BaseModel):
    value: int
class Scored(BaseModel):
    level: str
class Review(BaseModel):
    verdict: str
class Final(BaseModel):
    verdict: str
class Merged(BaseModel):
    enhanced_verdict: str
    standard_verdict: str


@agent(name="score", input=Seed, output=Scored)
async def score(v: Seed) -> Scored:
    return Scored(level="high" if v.value > 10 else "low")

@agent(name="enhanced", input=Scored, output=Review)
async def enhanced(v: Scored) -> Review:
    return Review(verdict="enhanced")

@agent(name="standard", input=Scored, output=Review)
async def standard(v: Scored) -> Review:
    return Review(verdict="standard")

@agent(name="writer", input=Review, output=Final)
async def writer(v: Review) -> Final:
    return Final(verdict=v.verdict)


def _pipeline():
    p = Pipeline(name="join")
    p.add(score)
    p.add(enhanced, inputs={"level": From("score.level")}, when=When("score.level", equals="high"))
    p.add(standard, inputs={"level": From("score.level")}, when=When("score.level", in_=("low", "medium")))
    p.add(Join("review", sources=["enhanced", "standard"], policy="exactly_one", output=Review))
    p.add(writer, inputs={"verdict": From("review.verdict")})
    return p


async def test_exactly_one_forwards_the_branch_that_ran():
    r = await _pipeline().run(Seed(value=1))           # low -> standard
    assert r.status == "completed"
    assert r.outputs["review"].verdict == "standard"
    assert r.outputs["writer"].verdict == "standard"
    join_rec = [s for s in r.audit_trace.step_records if s.agent == "review"][0]
    assert join_rec.node_kind == "join"
    assert "exactly_one" in join_rec.join and "standard" in join_rec.join


async def test_exactly_one_halts_when_none_ran():
    p = Pipeline(name="j2")
    p.add(score)
    p.add(enhanced, inputs={"level": From("score.level")}, when=When("score.level", equals="never"))
    p.add(standard, inputs={"level": From("score.level")}, when=When("score.level", equals="alsonever"))
    p.add(Join("review", sources=["enhanced", "standard"], policy="exactly_one", output=Review))
    r = await p.run(Seed(value=1))
    assert r.status == "halted"
    assert r.halted_at == "review"
    assert "join_policy_violation" in r.reason


def test_join_name_collision_rejected():
    p = Pipeline(name="j3")
    p.add(score)
    with pytest.raises(ValueError, match="already has"):
        p.add(Join("score", sources=["score"], policy="all_present", output=Scored))


def test_when_gates_on_join_output_valid_field_accepted():
    """Gating a downstream agent on a join's typed output field is valid."""
    p = Pipeline(name="j4")
    p.add(score)
    p.add(enhanced, inputs={"level": From("score.level")}, when=When("score.level", equals="high"))
    p.add(standard, inputs={"level": From("score.level")}, when=When("score.level", in_=("low", "medium")))
    p.add(Join("review", sources=["enhanced", "standard"], policy="exactly_one", output=Review))
    # verdict is a real field on Review — this must not raise
    p.add(writer, inputs={"verdict": From("review.verdict")}, when=When("review.verdict", equals="standard"))


def test_when_gates_on_join_output_nonexistent_field_raises_schema_error():
    """Gating on a non-existent field of a join's output raises SchemaCompatibilityError (not AttributeError)."""
    from drawbore.schema import SchemaCompatibilityError

    p = Pipeline(name="j5")
    p.add(score)
    p.add(enhanced, inputs={"level": From("score.level")}, when=When("score.level", equals="high"))
    p.add(standard, inputs={"level": From("score.level")}, when=When("score.level", in_=("low", "medium")))
    p.add(Join("review", sources=["enhanced", "standard"], policy="exactly_one", output=Review))
    with pytest.raises(SchemaCompatibilityError, match="does not exist"):
        p.add(writer, inputs={"verdict": From("review.verdict")}, when=When("review.nonexistent", equals="x"))


# --- first_by_priority --------------------------------------------------------

async def test_first_by_priority_forwards_first_in_sources_order_when_overlapping():
    """Two sources both ran (overlap) -> the join forwards the FIRST in `sources`
    order (deterministic priority), regardless of which produced what."""
    p = Pipeline(name="fbp1")
    p.add(score)
    # Both branches run unconditionally (no `when`) -> overlap.
    p.add(enhanced, inputs={"level": From("score.level")})
    p.add(standard, inputs={"level": From("score.level")})
    # `sources` order is [enhanced, standard] -> enhanced wins.
    p.add(Join("review", sources=["enhanced", "standard"], policy="first_by_priority", output=Review))
    r = await p.run(Seed(value=1))
    assert r.status == "completed"
    assert r.outputs["review"].verdict == "enhanced"
    join_rec = [s for s in r.audit_trace.step_records if s.agent == "review"][0]
    assert join_rec.node_kind == "join"
    assert "first_by_priority" in join_rec.join and "enhanced" in join_rec.join


async def test_first_by_priority_halts_when_none_ran():
    p = Pipeline(name="fbp2")
    p.add(score)
    p.add(enhanced, inputs={"level": From("score.level")}, when=When("score.level", equals="never"))
    p.add(standard, inputs={"level": From("score.level")}, when=When("score.level", equals="alsonever"))
    p.add(Join("review", sources=["enhanced", "standard"], policy="first_by_priority", output=Review))
    r = await p.run(Seed(value=1))
    assert r.status == "halted"
    assert r.halted_at == "review"
    assert "join_policy_violation" in r.reason


# --- all_present --------------------------------------------------------------

async def test_all_present_builds_output_from_inputs_when_all_ran():
    """All sources ran -> the join builds its `output` from `inputs` bindings,
    merging two upstream agents into the join's output model."""
    p = Pipeline(name="ap1")
    p.add(score)
    p.add(enhanced, inputs={"level": From("score.level")})
    p.add(standard, inputs={"level": From("score.level")})
    p.add(Join(
        "review",
        sources=["enhanced", "standard"],
        policy="all_present",
        output=Merged,
        inputs={
            "enhanced_verdict": From("enhanced.verdict"),
            "standard_verdict": From("standard.verdict"),
        },
    ))
    r = await p.run(Seed(value=1))
    assert r.status == "completed"
    merged = r.outputs["review"]
    assert merged.enhanced_verdict == "enhanced"
    assert merged.standard_verdict == "standard"
    join_rec = [s for s in r.audit_trace.step_records if s.agent == "review"][0]
    assert join_rec.node_kind == "join"
    assert "all_present" in join_rec.join


async def test_all_present_halts_when_a_source_was_skipped():
    p = Pipeline(name="ap2")
    p.add(score)
    # `enhanced` runs (low -> not high? no): gate enhanced so it is SKIPPED.
    p.add(enhanced, inputs={"level": From("score.level")}, when=When("score.level", equals="high"))
    p.add(standard, inputs={"level": From("score.level")})
    p.add(Join(
        "review",
        sources=["enhanced", "standard"],
        policy="all_present",
        output=Merged,
        inputs={
            "enhanced_verdict": From("enhanced.verdict"),
            "standard_verdict": From("standard.verdict"),
        },
    ))
    r = await p.run(Seed(value=1))  # low -> enhanced skipped
    assert r.status == "halted"
    assert r.halted_at == "review"
    assert "join_policy_violation" in r.reason


def test_all_present_static_check_rejects_bad_source_field():
    """An all_present binding to a non-existent source field fails CLOSED at
    registration time with SchemaCompatibilityError (not a runtime KeyError)."""
    from drawbore.schema import SchemaCompatibilityError

    p = Pipeline(name="ap3")
    p.add(score)
    p.add(enhanced, inputs={"level": From("score.level")})
    p.add(standard, inputs={"level": From("score.level")})
    with pytest.raises(SchemaCompatibilityError, match="does not exist"):
        p.add(Join(
            "review",
            sources=["enhanced", "standard"],
            policy="all_present",
            output=Merged,
            inputs={
                "enhanced_verdict": From("enhanced.nonexistent"),
                "standard_verdict": From("standard.verdict"),
            },
        ))


async def test_join_pipeline_runs_in_test_mode():
    async with _pipeline().test_mode() as test:
        r = await test.run(Seed(value=1))
    assert r.status == "completed"
    assert r.outputs["writer"].verdict == "standard"
