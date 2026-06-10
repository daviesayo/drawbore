import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.audit import InMemoryAuditSink
from drawbore.evidence import EvidencePolicy, InMemoryEvidenceStore
from drawbore.pipeline import Pipeline
from drawbore.pipeline.binding import From
from drawbore.testing import call, final
from drawbore.tools import ToolRegistry


class TxIn(BaseModel):
    transaction_id: str


class Fetched(BaseModel):
    transaction_id: str
    amount: int


class ScoreIn(BaseModel):
    amount: int


class Scored(BaseModel):
    amount: int
    risk: str


class Looked(BaseModel):
    amount: int
    answer: str


def _build():
    reg = ToolRegistry()
    async def real_db(args):
        raise AssertionError("real tool must not run in test mode")
    reg.register_tool("db.read", real_db, allowed_operations=("invoke",),
                      schema={"type": "object"})
    reg.register_tool("lookup", real_db, allowed_operations=("invoke",),
                      schema={"type": "object"})

    @agent(name="fetch", input=TxIn, output=Fetched, tools=["db.read"])
    async def fetch(v: TxIn, tools) -> Fetched:
        row = await tools.call("db.read", {"id": v.transaction_id})
        return Fetched(transaction_id=v.transaction_id, amount=row["amount"])

    @agent(name="score", input=ScoreIn, output=Scored, model="fake")
    async def score(v: ScoreIn) -> Scored: ...

    @agent(name="solve", input=Scored, output=Looked, model="fake", tools=["lookup"])
    async def solve(v: Scored, tools) -> Looked: ...

    p = Pipeline(name="aml", registry=reg)
    p.add(fetch)
    p.add(score, inputs={"amount": From("fetch.amount")}, depends_on=["fetch"],
          evidence=EvidencePolicy(name="rows", enabled=True, mode="compress",
                                  allowed_transforms=("json_rows",), min_tokens=1))
    p.add(solve, inputs={"amount": From("score.amount"), "risk": From("score.risk")},
          depends_on=["score"])
    return p


async def test_happy_path_runs_all_step_kinds_through_the_real_safety_layer():
    p = _build()
    async with p.test_mode(
        mock_tools={"db.read": {"amount": 100}, "lookup": {"hit": True}},
        mock_model_responses={"score": {"amount": 100, "risk": "low"}},
        mock_loop_scripts={"solve": [call("lookup", {}), final({"amount": 100, "answer": "ok"})]},
    ) as test:
        result = await test.run(TxIn(transaction_id="t1"))

    assert result.status == "completed"
    assert result.steps_run == 3
    # production-shaped audit trace
    trace = result.audit_trace
    assert trace is not None and trace.status == "completed"
    # NOTE: AuditRecord.steps is an INT (success count); per-step detail is
    # AuditRecord.step_records (a tuple of StepAuditRecord) — match the repo pattern.
    assert trace.steps == 3
    assert [s.agent for s in trace.step_records] == ["fetch", "score", "solve"]
    # the mocked tool call appears in the deterministic step's audit
    assert any("db.read" in tc for tc in trace.step_records[0].tool_calls)
    # evidence decision recorded on the compressed model step
    assert trace.step_records[1].evidence is not None
    # model turns counted: one-shot = 1, loop = number of model turns (>=1)
    assert trace.step_records[1].model_turns == 1
    assert trace.step_records[2].model_turns >= 1
    # the harness exposes its stores/sink
    assert isinstance(test.audit_sink, InMemoryAuditSink)
    assert isinstance(test.evidence_store, InMemoryEvidenceStore)


async def test_run_id_is_deterministic_with_a_prefix():
    p = _build()
    async with p.test_mode(
        mock_tools={"db.read": {"amount": 1}, "lookup": {}},
        mock_model_responses={"score": {"amount": 1, "risk": "low"}},
        mock_loop_scripts={"solve": [final({"amount": 1, "answer": "x"})]},
        run_id_prefix="case",
    ) as test:
        r1 = await test.run(TxIn(transaction_id="a"))
        r2 = await test.run(TxIn(transaction_id="b"))
    assert r1.audit_trace.run_id == "case-1"
    assert r2.audit_trace.run_id == "case-2"


async def test_sync_with_is_rejected_clearly():
    p = _build()
    with pytest.raises(TypeError):
        with p.test_mode():   # the real path is async; require `async with`
            pass


async def test_sequence_tool_mock_is_fresh_per_run():
    """Sequence-form tool mocks must reset each run, not be consumed across runs."""
    reg = ToolRegistry()

    async def real(args):
        raise AssertionError("real tool must not run")

    reg.register_tool("db.read", real, allowed_operations=("invoke",), schema={"type": "object"})

    @agent(name="fetch", input=TxIn, output=Fetched, tools=["db.read"])
    async def fetch(v: TxIn, tools) -> Fetched:
        row = await tools.call("db.read", {"id": v.transaction_id})
        return Fetched(transaction_id=v.transaction_id, amount=row["amount"])

    p = Pipeline(name="seq", registry=reg)
    p.add(fetch)

    async with p.test_mode(mock_tools={"db.read": [{"amount": 1}, {"amount": 2}]}) as test:
        r1 = await test.run(TxIn(transaction_id="a"))
        r2 = await test.run(TxIn(transaction_id="b"))

    # Both runs should see the FIRST element — sequence is fresh per run, not consumed
    assert r1.outputs["fetch"].amount == 1
    assert r2.outputs["fetch"].amount == 1
