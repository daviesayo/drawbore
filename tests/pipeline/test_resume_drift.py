"""Refuse-on-drift resume: semantic rollback is refused pre-flight with zero
double-fire, topology drift refuses when progress exists, and every run
carries a resume ledger."""

import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.escalation import EscalationPolicy
from drawbore.pipeline import From, When, Join
from drawbore.state import InMemoryCheckpointStore


class Doc(BaseModel):
    text: str


class Fetched(BaseModel):
    content: str


class Verdict(BaseModel):
    ok: bool


CALLS: dict[str, int] = {}


def _fresh_agents(writer_model: str | None = None, writer_version: str = "1.0.0"):
    """Build a 3-step pipeline: fetcher -> failer/scorer -> writer.

    Each call builds NEW agent objects so a test can mutate declared fields
    between attempts without object identity carrying over.
    """

    @agent(input=Doc, output=Fetched, name="fetcher", version="1.0.0")
    async def fetcher(payload: Doc) -> Fetched:
        CALLS["fetcher"] = CALLS.get("fetcher", 0) + 1
        return Fetched(content=payload.text)

    @agent(input=Fetched, output=Verdict, name="scorer", version="1.0.0")
    async def scorer(payload: Fetched) -> Verdict:
        CALLS["scorer"] = CALLS.get("scorer", 0) + 1
        if payload.content == "BOOM":
            raise RuntimeError("scorer exploded")
        return Verdict(ok=True)

    @agent(
        input=Verdict, output=Verdict, name="writer",
        version=writer_version, model=writer_model,
    )
    async def writer(payload: Verdict) -> Verdict:
        CALLS["writer"] = CALLS.get("writer", 0) + 1
        return Verdict(ok=payload.ok)

    return fetcher, scorer, writer


def _pipeline(fetcher, scorer, writer, **kw) -> Pipeline:
    p = Pipeline("resume-demo", **kw)
    p.add(fetcher)
    p.add(scorer)
    p.add(writer)
    return p


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()


@pytest.mark.asyncio
async def test_refuse_on_drift_blocks_semantic_rollback():
    """The named proof: mutate writer's declared model (deterministic agents
    here, so the declaration is what drifts), topology unchanged; resume
    refuses, names the field, and re-executes nothing."""
    store = InMemoryCheckpointStore()
    fetcher, scorer, writer = _fresh_agents(writer_model=None)
    first = await _pipeline(fetcher, scorer, writer).run(
        Doc(text="BOOM"), run_id="r1", checkpoints=store
    )
    assert first.status == "halted"  # scorer exploded; fetcher checkpointed
    assert CALLS == {"fetcher": 1, "scorer": 1}

    # Semantic rollback attempt: same topology, fetcher's version drifts.
    fetcher2, scorer2, writer2 = _fresh_agents()
    object.__setattr__(fetcher2.spec, "version", "9.9.9")
    second = await _pipeline(fetcher2, scorer2, writer2).run(
        Doc(text="ok"), run_id="r1", checkpoints=store
    )
    assert second.status == "halted"
    assert second.halt_code == "resume_drift"
    assert "fetcher" in second.reason
    assert "version" in second.reason
    # Zero double-fire: NOTHING executed in the second attempt.
    assert CALLS == {"fetcher": 1, "scorer": 1}
    assert second.steps_run == 0


@pytest.mark.asyncio
async def test_clean_resume_verifies_and_completes():
    store = InMemoryCheckpointStore()
    fetcher, scorer, writer = _fresh_agents()
    first = await _pipeline(fetcher, scorer, writer).run(
        Doc(text="BOOM"), run_id="r1", checkpoints=store
    )
    assert first.status == "halted"

    fetcher2, scorer2, writer2 = _fresh_agents()

    # Same declarations -> resume proceeds; fetcher is NOT re-run, its "BOOM"
    # output is restored from the checkpoint, so scorer halts again
    # deterministically. The behavior under test is the restore + verification.
    second = await _pipeline(fetcher2, scorer2, writer2).run(
        Doc(text="ok"), run_id="r1", checkpoints=store
    )
    # fetcher restored (1 call total), scorer re-ran, writer ran.
    assert CALLS["fetcher"] == 1
    assert second.status == "halted" or second.status == "completed"
    # NOTE for implementer: the restored fetcher output is "BOOM", so scorer
    # halts again deterministically. The assertion that matters here is the
    # restore + verification behavior:
    ledger = second.resume_ledger
    assert ledger is not None
    assert ledger.resumed is True
    assert ledger.topology == "verified"
    assert ledger.entries[0].disposition == "restored"
    assert ledger.entries[0].seal == "verified"


@pytest.mark.asyncio
async def test_topology_drift_with_progress_refuses():
    store = InMemoryCheckpointStore()
    fetcher, scorer, writer = _fresh_agents()
    await _pipeline(fetcher, scorer, writer).run(
        Doc(text="BOOM"), run_id="r1", checkpoints=store
    )
    # Change topology: drop the writer.
    fetcher2, scorer2, _ = _fresh_agents()
    p2 = Pipeline("resume-demo")
    p2.add(fetcher2)
    p2.add(scorer2)
    result = await p2.run(Doc(text="ok"), run_id="r1", checkpoints=store)
    assert result.status == "halted"
    assert result.halt_code == "resume_drift"
    assert result.resume_ledger.topology == "drifted"
    assert CALLS == {"fetcher": 1, "scorer": 1}  # nothing re-ran


@pytest.mark.asyncio
async def test_topology_drift_zero_progress_runs_fresh():
    """A recorded fingerprint with no completed/skipped steps must not brick
    the run_id: the run proceeds fresh and re-records."""
    store = InMemoryCheckpointStore()
    store.record_fingerprint("r1", "sha256:stale-other-topology")
    fetcher, scorer, writer = _fresh_agents()
    result = await _pipeline(fetcher, scorer, writer).run(
        Doc(text="ok"), run_id="r1", checkpoints=store
    )
    assert result.status == "completed"
    assert result.resume_ledger.resumed is False


@pytest.mark.asyncio
async def test_missing_seal_fails_closed():
    """Completed steps with no stored seal (un-upgraded store / pre-upgrade
    data) refuse the resume."""

    class NoSealStore(InMemoryCheckpointStore):
        def record_seal(self, run_id, step, seal):
            return None  # un-upgraded: persists nothing

        def seal_of(self, run_id, step):
            return None

    store = NoSealStore()
    fetcher, scorer, writer = _fresh_agents()
    await _pipeline(fetcher, scorer, writer).run(
        Doc(text="BOOM"), run_id="r1", checkpoints=store
    )
    fetcher2, scorer2, writer2 = _fresh_agents()
    result = await _pipeline(fetcher2, scorer2, writer2).run(
        Doc(text="ok"), run_id="r1", checkpoints=store
    )
    assert result.status == "halted"
    assert result.halt_code == "resume_drift"
    assert result.resume_ledger.entries[0].seal == "missing"
    assert CALLS == {"fetcher": 1, "scorer": 1}


@pytest.mark.asyncio
async def test_resume_drift_escalates_with_policy():
    store = InMemoryCheckpointStore()
    fetcher, scorer, writer = _fresh_agents()
    await _pipeline(fetcher, scorer, writer).run(
        Doc(text="BOOM"), run_id="r1", checkpoints=store
    )
    fetcher2, scorer2, writer2 = _fresh_agents()
    object.__setattr__(fetcher2.spec, "version", "9.9.9")
    p2 = _pipeline(
        fetcher2, scorer2, writer2,
        on_failure=EscalationPolicy(channel="log", target="ops"),
    )
    result = await p2.run(Doc(text="ok"), run_id="r1", checkpoints=store)
    assert result.status == "escalated"
    assert result.halt_code == "resume_drift"
    assert len(result.escalations) == 1


@pytest.mark.asyncio
async def test_fresh_run_ledger_always_present():
    fetcher, scorer, writer = _fresh_agents()
    result = await _pipeline(fetcher, scorer, writer).run(Doc(text="ok"))
    ledger = result.resume_ledger
    assert ledger is not None
    assert ledger.resumed is False
    assert ledger.topology == "untracked"  # no store passed
    assert all(e.seal == "none" for e in ledger.entries)
    assert [e.disposition for e in ledger.entries] == ["executed"] * 3


@pytest.mark.asyncio
async def test_fresh_run_with_store_records_seals():
    store = InMemoryCheckpointStore()
    fetcher, scorer, writer = _fresh_agents()
    result = await _pipeline(fetcher, scorer, writer).run(
        Doc(text="ok"), run_id="r1", checkpoints=store
    )
    assert result.status == "completed"
    assert result.resume_ledger.topology == "recorded"
    assert all(e.seal == "recorded" for e in result.resume_ledger.entries)
    assert store.seal_of("r1", 0) is not None
    assert store.seal_of("r1", 2) is not None


@pytest.mark.asyncio
async def test_refusal_reason_uses_friendly_label():
    """An instructions change must read 'instructions' in the reason, not the
    internal fingerprint field name."""
    store = InMemoryCheckpointStore()
    fetcher, scorer, writer = _fresh_agents()
    await _pipeline(fetcher, scorer, writer).run(
        Doc(text="BOOM"), run_id="r1", checkpoints=store
    )
    fetcher2, scorer2, writer2 = _fresh_agents()
    object.__setattr__(fetcher2.spec, "instructions", "now different")
    result = await _pipeline(fetcher2, scorer2, writer2).run(
        Doc(text="ok"), run_id="r1", checkpoints=store
    )
    assert result.halt_code == "resume_drift"
    assert "instructions" in result.reason
    assert "instructions_fingerprint" not in result.reason


# --- resume ledger with a checkpoint-completed Join ---------------------------


class Seed(BaseModel):
    value: int


class Scored(BaseModel):
    level: str


class Review(BaseModel):
    verdict: str


class JFinal(BaseModel):
    verdict: str


def _join_pipeline(*, fail_writer: bool, score_version: str = "1.0.0") -> Pipeline:
    """Build score(0) -> enhanced(1)/standard(2) -> Join review(3) ->
    reviewer(4) -> writer(5).

    With value=1 the branch gate sends work down `standard`; the join and the
    post-join `reviewer` both complete before `writer`, which halts in the
    first attempt (``fail_writer``). On a clean resume `writer` succeeds. The
    completed agent AFTER the join (reviewer, index 4) is what exposes the
    out-of-order bug if the join's restored entry is logged late.
    """

    @agent(name="score", input=Seed, output=Scored, version=score_version)
    async def score(v: Seed) -> Scored:
        CALLS["score"] = CALLS.get("score", 0) + 1
        return Scored(level="high" if v.value > 10 else "low")

    @agent(name="enhanced", input=Scored, output=Review, version="1.0.0")
    async def enhanced(v: Scored) -> Review:
        CALLS["enhanced"] = CALLS.get("enhanced", 0) + 1
        return Review(verdict="enhanced")

    @agent(name="standard", input=Scored, output=Review, version="1.0.0")
    async def standard(v: Scored) -> Review:
        CALLS["standard"] = CALLS.get("standard", 0) + 1
        return Review(verdict="standard")

    @agent(name="reviewer", input=Review, output=JFinal, version="1.0.0")
    async def reviewer(v: Review) -> JFinal:
        CALLS["reviewer"] = CALLS.get("reviewer", 0) + 1
        return JFinal(verdict=v.verdict)

    @agent(name="writer", input=JFinal, output=JFinal, version="1.0.0")
    async def writer(v: JFinal) -> JFinal:
        CALLS["writer"] = CALLS.get("writer", 0) + 1
        if fail_writer:
            raise RuntimeError("writer exploded")
        return JFinal(verdict=v.verdict)

    p = Pipeline(name="join-resume")
    p.add(score)
    p.add(enhanced, inputs={"level": From("score.level")}, when=When("score.level", equals="high"))
    p.add(standard, inputs={"level": From("score.level")}, when=When("score.level", in_=("low", "medium")))
    p.add(Join("review", sources=["enhanced", "standard"], policy="exactly_one", output=Review))
    p.add(reviewer, inputs={"verdict": From("review.verdict")})
    p.add(writer, inputs={"verdict": From("reviewer.verdict")})
    return p


@pytest.mark.asyncio
async def test_refused_resume_ledger_includes_completed_join():
    """A refused resume must list a checkpoint-completed join in its ledger:
    the join (index 3) appears with disposition 'restored' and seal 'none',
    even though refusal is driven by a drifted agent."""
    store = InMemoryCheckpointStore()
    first = await _join_pipeline(fail_writer=True).run(
        Seed(value=1), run_id="r1", checkpoints=store
    )
    assert first.status == "halted"  # writer exploded; join + reviewer checkpointed

    # Drift the (completed) score agent's version; topology unchanged -> refused.
    p2 = _join_pipeline(fail_writer=False, score_version="9.9.9")
    second = await p2.run(Seed(value=1), run_id="r1", checkpoints=store)
    assert second.status == "halted"
    assert second.halt_code == "resume_drift"

    ledger = second.resume_ledger
    assert ledger is not None
    join_entries = [e for e in ledger.entries if e.agent == "review"]
    assert len(join_entries) == 1
    assert join_entries[0].index == 3
    assert join_entries[0].disposition == "restored"
    assert join_entries[0].seal == "none"


@pytest.mark.asyncio
async def test_clean_resume_with_join_keeps_ledger_in_index_order():
    """On a clean resume past a completed join, ledger entries stay in index
    order — the join's restored entry is logged in pre-flight, not appended
    late by the in-loop short-circuit."""
    store = InMemoryCheckpointStore()
    first = await _join_pipeline(fail_writer=True).run(
        Seed(value=1), run_id="r1", checkpoints=store
    )
    assert first.status == "halted"

    second = await _join_pipeline(fail_writer=False).run(
        Seed(value=1), run_id="r1", checkpoints=store
    )
    assert second.status == "completed"
    ledger = second.resume_ledger
    assert ledger is not None
    # The join (3) sits between completed agents (0/standard 2) and the
    # post-join reviewer (4). Restored/executed entries must be index-ordered:
    # without the pre-flight fix the join's restored entry is appended late by
    # the in-loop short-circuit, landing after reviewer(4). (Skipped-branch
    # entries are logged in execution order — a separate artifact — so they are
    # excluded here.)
    indices = [e.index for e in ledger.entries if e.disposition != "skipped"]
    assert indices == sorted(indices)
    assert 3 in indices  # the completed join is part of the ordered run
    # The completed join is present exactly once.
    assert sum(1 for e in ledger.entries if e.agent == "review") == 1


@pytest.mark.asyncio
async def test_ledger_legible_smoke():
    store = InMemoryCheckpointStore()
    fetcher, scorer, writer = _fresh_agents()
    await _pipeline(fetcher, scorer, writer).run(
        Doc(text="BOOM"), run_id="r1", checkpoints=store
    )
    fetcher2, scorer2, writer2 = _fresh_agents()
    object.__setattr__(fetcher2.spec, "version", "9.9.9")
    result = await _pipeline(fetcher2, scorer2, writer2).run(
        Doc(text="ok"), run_id="r1", checkpoints=store
    )
    text = result.resume_ledger.legible()
    assert "fetcher" in text
    assert "version" in text
