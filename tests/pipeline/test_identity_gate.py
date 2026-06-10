import pytest
from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.identity import IdentityRegistry
from drawbore.escalation import EscalationPolicy, RecordingDispatcher
from drawbore.state import InMemoryCheckpointStore


class Seed(BaseModel):
    value: int


class Result(BaseModel):
    value: int


def _scorer():
    @agent(name="scorer", input=Seed, output=Result)
    async def scorer(v: Seed) -> Result:
        return Result(value=v.value + 1)
    return scorer


async def test_unregistered_agent_runs_normally_without_a_registry():
    p = Pipeline(name="t")
    p.add(_scorer())
    r = await p.run(Seed(value=1))
    assert r.status == "completed"
    assert r.outputs["scorer"].value == 2


async def test_registered_active_agent_runs():
    reg = IdentityRegistry()
    scorer = _scorer()
    reg.register(scorer.spec, sponsor="alice", purpose="p", ttl_seconds=60)
    p = Pipeline(name="t")
    p.add(scorer)
    r = await p.run(Seed(value=1), identities=reg)
    assert r.status == "completed"
    assert r.outputs["scorer"].value == 2


async def test_decommissioned_agent_halts_before_running():
    reg = IdentityRegistry()
    scorer = _scorer()
    reg.register(scorer.spec, sponsor="alice", purpose="p", ttl_seconds=60,
                 agent_id="scorer-xyz")
    reg.decommission("scorer")
    p = Pipeline(name="t")
    p.add(scorer)
    r = await p.run(Seed(value=1), identities=reg)
    assert r.status == "halted"
    assert r.halted_at == "scorer"
    assert "identity_decommissioned" in r.reason
    assert r.escalations == []  # no policy → plain halt


async def test_suspended_agent_escalates_with_agent_id_under_policy():
    reg = IdentityRegistry()
    scorer = _scorer()
    reg.register(scorer.spec, sponsor="alice", purpose="p", ttl_seconds=60,
                 agent_id="scorer-xyz")
    reg.suspend("scorer")
    d = RecordingDispatcher()
    p = Pipeline(name="t", on_failure=EscalationPolicy("slack", "q"), dispatcher=d)
    p.add(scorer)
    r = await p.run(Seed(value=1), identities=reg)
    assert r.status == "escalated"
    assert "identity_suspended" in r.reason
    assert r.escalations[0].agent_id == "scorer-xyz"  # escalation package carries agent_id
    assert len(d.sent) == 1


async def test_reattestation_required_agent_halts():
    reg = IdentityRegistry()
    scorer = _scorer()
    reg.register(scorer.spec, sponsor="alice", purpose="p", ttl_seconds=60)

    # Re-define the agent under the SAME name "scorer" but with an escalated risk
    # tier: that moves its attestation surface, so update() blocks the registered
    # "scorer" until its sponsor re-attests — even though the pipeline runs the
    # original spec (the registry is keyed by agent name, not spec instance).
    @agent(name="scorer", input=Seed, output=Result, risk_tier="critical")
    async def scorer_v2(v: Seed) -> Result:
        return Result(value=v.value + 1)

    reg.update(scorer_v2.spec)  # surface changed → needs re-attestation
    p = Pipeline(name="t")
    p.add(scorer)
    r = await p.run(Seed(value=1), identities=reg)
    assert r.status == "halted"
    assert "identity_reattestation_required" in r.reason


async def test_decommissioned_agent_is_not_resumed_from_checkpoint():
    # The identity gate runs BEFORE the checkpoint-resume skip: a step that was
    # checkpoint-completed while the agent was active must NOT be silently resumed
    # once the agent is decommissioned — it must halt on the identity gate.
    reg = IdentityRegistry()
    scorer = _scorer()
    reg.register(scorer.spec, sponsor="alice", purpose="p", ttl_seconds=60,
                 agent_id="scorer-xyz")
    store = InMemoryCheckpointStore()
    p = Pipeline(name="t")
    p.add(scorer)

    # First run: agent active → completes and checkpoints step 0.
    r1 = await p.run(Seed(value=1), identities=reg, run_id="run-1", checkpoints=store)
    assert r1.status == "completed"

    # Decommission, then re-run with the same run_id + checkpoint store.
    reg.decommission("scorer")
    r2 = await p.run(Seed(value=1), identities=reg, run_id="run-1", checkpoints=store)
    assert r2.status == "halted"
    assert "identity_decommissioned" in r2.reason  # gate fired, not a checkpoint skip
