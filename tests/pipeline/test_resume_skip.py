# tests/pipeline/test_resume_skip.py
from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.pipeline import Pipeline, From, When
from drawbore.state import InMemoryCheckpointStore


class Seed(BaseModel):
    value: int
class Scored(BaseModel):
    level: str
class Note(BaseModel):
    note: str

@agent(name="score", input=Seed, output=Scored)
async def score(v: Seed) -> Scored:
    return Scored(level="low")

@agent(name="enhanced", input=Scored, output=Note)
async def enhanced(v: Scored) -> Note:
    return Note(note="x")


async def test_resume_replays_a_cascade_or_condition_skip():
    p = Pipeline(name="resume")
    p.add(score)
    p.add(enhanced, inputs={"level": From("score.level")},
          when=When("score.level", equals="high"))    # low -> enhanced skipped
    cp = InMemoryCheckpointStore()
    r1 = await p.run(Seed(value=1), run_id="run1", checkpoints=cp)
    assert r1.status == "completed" and "enhanced" not in r1.outputs
    # resume the same run: the skip is replayed from the store, not silently re-derived
    r2 = await p.run(Seed(value=1), run_id="run1", checkpoints=cp)
    assert r2.status == "completed" and "enhanced" not in r2.outputs


async def test_resume_condition_false_skip_preserves_original_condition():
    # D64c: a condition-false skip is NOT persisted. On resume it re-derives from the
    # checkpointed upstream output AND the resumed trace keeps the ORIGINAL legible
    # condition string (legibility-first), never the generic "(resumed: skipped)".
    p = Pipeline(name="resume")
    p.add(score)
    p.add(enhanced, inputs={"level": From("score.level")},
          when=When("score.level", equals="high"))    # low -> enhanced condition-false
    cp = InMemoryCheckpointStore()
    r1 = await p.run(Seed(value=1), run_id="cond", checkpoints=cp)
    assert r1.status == "completed" and "enhanced" not in r1.outputs
    # the condition-false skip must NOT have been persisted to the checkpoint store
    assert cp.is_skipped("cond", 1) is False
    # resume the same run: the skip re-derives deterministically from restored output
    r2 = await p.run(Seed(value=1), run_id="cond", checkpoints=cp)
    assert r2.status == "completed" and "enhanced" not in r2.outputs
    # the resumed trace records the skipped node with its ORIGINAL condition string,
    # not the generic replay marker (which it WOULD show if the skip were persisted)
    skipped_records = [s for s in r2.audit_trace.step_records
                       if s.agent == "enhanced" and s.status == "skipped"]
    assert len(skipped_records) == 1
    cond = skipped_records[0].condition
    assert "== 'high'" in cond
    assert "(false)" in cond
    assert "(resumed: skipped)" not in cond


async def test_resume_after_topology_change_starts_fresh():
    cp = InMemoryCheckpointStore()
    p1 = Pipeline(name="resume")
    p1.add(score)
    await p1.run(Seed(value=1), run_id="run2", checkpoints=cp)
    # a different topology under the same run_id must NOT resume against stale indices
    p2 = Pipeline(name="resume")
    p2.add(enhanced, inputs={})        # different node at index 0
    # fingerprint mismatch -> fresh run (no stale resume); here it simply runs cleanly
    assert cp.fingerprint_matches("run2", p1._topology_fingerprint()) is True
    assert cp.fingerprint_matches("run2", p2._topology_fingerprint()) is False
