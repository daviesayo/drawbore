import pytest
from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.escalation import EscalationPolicy, RecordingDispatcher, HasConfidence
from drawbore.pipeline import From


class Seed(BaseModel):
    value: int


class Score(BaseModel, HasConfidence):
    score: int
    confidence: float


class Plain(BaseModel):
    score: int
    confidence: float  # incidental field, NOT the HasConfidence marker


async def test_halt_without_policy_stays_halted():
    @agent(name="boom", input=Seed, output=Score)
    async def boom(v: Seed) -> Score:
        raise RuntimeError("x")

    p = Pipeline(name="t")  # no on_failure
    p.add(boom)
    r = await p.run(Seed(value=1))
    assert r.status == "halted"
    assert r.escalations == []


async def test_halt_with_policy_escalates_and_dispatches():
    @agent(name="boom", input=Seed, output=Score)
    async def boom(v: Seed) -> Score:
        raise RuntimeError("x")

    d = RecordingDispatcher()
    p = Pipeline(name="t", on_failure=EscalationPolicy("slack", "q"), dispatcher=d)
    p.add(boom)
    r = await p.run(Seed(value=1))
    assert r.status == "escalated"
    assert r.halted_at == "boom"
    assert len(r.escalations) == 1
    assert r.escalations[0].reason.startswith("agent_error")
    assert len(d.sent) == 1 and d.sent[0][1].channel == "slack"


async def test_confidence_below_threshold_escalates_sync():
    @agent(name="scorer", input=Seed, output=Score)
    async def scorer(v: Seed) -> Score:
        return Score(score=v.value, confidence=0.3)

    d = RecordingDispatcher()
    p = Pipeline(name="t", on_failure=EscalationPolicy("slack", "q"),
                 dispatcher=d, confidence_threshold=0.5)
    p.add(scorer)
    r = await p.run(Seed(value=1))
    assert r.status == "escalated"
    assert "confidence_below_threshold" in r.escalations[0].reason
    assert len(d.sent) == 1


async def test_confidence_async_dispatches_but_continues():
    @agent(name="scorer", input=Seed, output=Score)
    async def scorer(v: Seed) -> Score:
        return Score(score=v.value, confidence=0.3)

    d = RecordingDispatcher()
    p = Pipeline(name="t", on_failure=EscalationPolicy("email", "ops", mode="async"),
                 dispatcher=d, confidence_threshold=0.5)
    p.add(scorer)
    r = await p.run(Seed(value=1))
    assert r.status == "completed"          # async review does not block
    assert len(r.escalations) == 1          # but the escalation was raised
    assert len(d.sent) == 1


async def test_confidence_above_threshold_does_not_escalate():
    @agent(name="scorer", input=Seed, output=Score)
    async def scorer(v: Seed) -> Score:
        return Score(score=v.value, confidence=0.9)

    p = Pipeline(name="t", on_failure=EscalationPolicy("slack", "q"), confidence_threshold=0.5)
    p.add(scorer)
    r = await p.run(Seed(value=1))
    assert r.status == "completed"
    assert r.escalations == []


async def test_incidental_confidence_field_is_ignored():
    @agent(name="plain", input=Seed, output=Plain)
    async def plain(v: Seed) -> Plain:
        return Plain(score=v.value, confidence=0.1)  # low, but model is NOT marked

    p = Pipeline(name="t", on_failure=EscalationPolicy("slack", "q"), confidence_threshold=0.5)
    p.add(plain)
    r = await p.run(Seed(value=1))
    assert r.status == "completed"  # unmarked model is not confidence-checked
    assert r.escalations == []


async def test_requires_human_approval_is_a_sync_gate_even_under_async_policy():
    @agent(name="pay", input=Seed, output=Score, requires_human_approval=True)
    async def pay(v: Seed) -> Score:
        return Score(score=v.value, confidence=1.0)

    d = RecordingDispatcher()
    p = Pipeline(name="t", on_failure=EscalationPolicy("slack", "q", mode="async"), dispatcher=d)
    p.add(pay)
    r = await p.run(Seed(value=1))
    assert r.status == "escalated"  # approval always blocks, regardless of mode
    assert r.escalations[0].reason == "requires_human_approval"
    assert r.escalations[0].attempted_output == Score(score=1, confidence=1.0)
    assert len(d.sent) == 1


async def test_confidence_below_threshold_without_policy_halts():
    # confidence threshold set, but NO on_failure policy → still halts (no dispatch)
    @agent(name="scorer", input=Seed, output=Score)
    async def scorer(v: Seed) -> Score:
        return Score(score=v.value, confidence=0.3)

    p = Pipeline(name="t", confidence_threshold=0.5)  # no on_failure
    p.add(scorer)
    r = await p.run(Seed(value=1))
    assert r.status == "halted"
    assert "confidence_below_threshold" in r.reason
    assert r.escalations == []


async def test_has_confidence_marker_without_field_halts_not_raises():
    # A model that inherits the HasConfidence marker but omits the required
    # `confidence` field must FAIL CLOSED (halt/escalate), never raise out of run().
    class BadOut(BaseModel, HasConfidence):
        score: int  # marker present, but NO `confidence` field

    @agent(name="bad", input=Seed, output=BadOut)
    async def bad(v: Seed) -> BadOut:
        return BadOut(score=v.value)

    # no policy -> halt (not raise)
    p = Pipeline(name="t", confidence_threshold=0.5)
    p.add(bad)
    r = await p.run(Seed(value=1))
    assert r.status == "halted"
    assert "confidence_marker_without_value" in r.reason

    # with policy -> escalate + dispatch (not raise)
    d = RecordingDispatcher()
    p2 = Pipeline(name="t2", on_failure=EscalationPolicy("slack", "q"),
                  dispatcher=d, confidence_threshold=0.5)
    p2.add(bad)
    r2 = await p2.run(Seed(value=1))
    assert r2.status == "escalated"
    assert "confidence_marker_without_value" in r2.escalations[0].reason
    assert len(d.sent) == 1


async def test_confidence_async_output_flows_to_downstream():
    # async confidence escalation must NOT block: the flagged step's output is
    # still stored and reaches a downstream step.
    class Step2In(BaseModel):
        score: int

    class Final(BaseModel):
        total: int

    @agent(name="scorer", input=Seed, output=Score)
    async def scorer(v: Seed) -> Score:
        return Score(score=v.value, confidence=0.3)

    @agent(name="finaliser", input=Step2In, output=Final)
    async def finaliser(v: Step2In) -> Final:
        return Final(total=v.score + 10)

    d = RecordingDispatcher()
    p = Pipeline(name="t", on_failure=EscalationPolicy("email", "ops", mode="async"),
                 dispatcher=d, confidence_threshold=0.5)
    p.add(scorer)
    p.add(finaliser, inputs={"score": From("scorer.score")})
    r = await p.run(Seed(value=5))
    assert r.status == "completed"
    assert r.outputs["finaliser"].total == 15   # scorer's output reached finaliser
    assert len(r.escalations) == 1              # scorer's low confidence still flagged
    assert len(d.sent) == 1
