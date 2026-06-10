from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.pipeline import Pipeline, From, When


class Seed(BaseModel):
    value: int

class Scored(BaseModel):
    level: str

class Note(BaseModel):
    note: str

class Final(BaseModel):
    summary: str

class Flagged(BaseModel):
    flag: bool


@agent(name="score", input=Seed, output=Scored)
async def score(v: Seed) -> Scored:
    return Scored(level="high" if v.value > 10 else "low")

@agent(name="enhanced", input=Scored, output=Note)
async def enhanced(v: Scored) -> Note:
    return Note(note="enhanced-ran")

@agent(name="downstream", input=Note, output=Final)
async def downstream(v: Note) -> Final:
    return Final(summary=v.note)


def _branch_pipeline():
    p = Pipeline(name="branch")
    p.add(score)
    p.add(enhanced, inputs={"level": From("score.level")},
          when=When("score.level", equals="high"))
    p.add(downstream, inputs={"note": From("enhanced.note")})
    return p


async def test_branch_taken_runs_node():
    result = await _branch_pipeline().run(Seed(value=50))   # high
    assert result.status == "completed"
    assert result.outputs["enhanced"].note == "enhanced-ran"
    assert result.outputs["downstream"].summary == "enhanced-ran"


async def test_branch_not_taken_skips_node_and_cascades():
    result = await _branch_pipeline().run(Seed(value=1))    # low -> enhanced skipped
    assert result.status == "completed"
    assert "enhanced" not in result.outputs        # skipped, no output
    assert "downstream" not in result.outputs       # cascade: required source skipped
    trace = result.audit_trace
    skipped = [s for s in trace.step_records if s.status == "skipped"]
    names = {s.agent for s in skipped}
    assert "enhanced" in names and "downstream" in names
    enh = [s for s in skipped if s.agent == "enhanced"][0]
    assert "score.level == 'high'" in enh.condition and "false" in enh.condition


async def test_skip_does_not_count_as_step():
    result = await _branch_pipeline().run(Seed(value=1))
    assert result.steps_run == 1                    # only `score` ran
    assert result.audit_trace.steps == 1


async def test_cascade_via_implicit_predecessor_edge():
    # `tail` declares NO inputs — reachability falls back to the positional
    # predecessor (`enhanced`), which is skipped.  This exercises the
    # `else ({predecessor} ...)` arm of the reach_sources logic.
    @agent(name="tail", input=Note, output=Final)
    async def tail(v: Note) -> Final:
        return Final(summary=v.note)

    p = Pipeline(name="implicit")
    p.add(score)                                        # runs; level="low"
    p.add(enhanced, inputs={"level": From("score.level")},
          when=When("score.level", equals="high"))      # skipped (low != high)
    p.add(tail)                                         # no inputs -> predecessor fallback -> cascade-skipped

    r = await p.run(Seed(value=1))
    assert r.status == "completed"
    assert "tail" not in r.outputs

    tail_records = [s for s in r.audit_trace.step_records
                    if s.agent == "tail" and s.status == "skipped"]
    assert tail_records, "tail should be recorded as skipped"
    assert "cascade: enhanced skipped" in tail_records[0].condition


async def test_condition_unevaluable_fails_closed():
    """D64b: a When gate on a cascade-skipped node halts with condition_unevaluable.

    Topology:
    - score      runs; level="low"
    - branch     When(score.level == "high") -> SKIPPED (low); has a `when` so
                 D63c would block others from gating on it directly
    - mid        no `when`; binds From(branch.flag) -> CASCADE-SKIPPED because
                 branch is in skipped; carries no `when` so D63c allows gating on it
    - target     When(mid.flag is_true=True); binds From(score.level) — NOT from mid,
                 so reach_sources={"score"}, cascade does NOT fire; but
                 when.agent="mid" is in skipped -> condition_unevaluable HALT
    """
    @agent(name="cu_branch", input=Scored, output=Flagged)
    async def cu_branch(v: Scored) -> Flagged:
        return Flagged(flag=True)

    @agent(name="cu_mid", input=Flagged, output=Flagged)
    async def cu_mid(v: Flagged) -> Flagged:
        return Flagged(flag=v.flag)

    @agent(name="cu_target", input=Scored, output=Final)
    async def cu_target(v: Scored) -> Final:
        return Final(summary="should not run")

    p = Pipeline(name="unevaluable")
    p.add(score)
    # branch: conditionally run (when score.level == "high") — will be skipped
    p.add(cu_branch, inputs={"level": From("score.level")},
          when=When("score.level", equals="high"))
    # mid: no `when` (D63c allows gating on it); binds From branch -> cascade-skips
    p.add(cu_mid, inputs={"flag": From("cu_branch.flag")})
    # target: gates on mid (which was cascade-skipped), but binds From score
    # (not mid) so reach_sources={"score"} -> cascade check does NOT fire
    p.add(cu_target, inputs={"level": From("score.level")},
          when=When("cu_mid.flag", is_true=True))

    result = await p.run(Seed(value=1))   # level="low" -> branch skipped -> mid cascades -> target halts

    assert result.status == "halted"
    assert result.halted_at == "cu_target"
    assert "condition_unevaluable" in result.reason
